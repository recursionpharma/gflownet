import copy
from typing import Dict, List, Tuple

import networkx as nx
import torch
import torch_geometric.data as gd
from networkx.algorithms.isomorphism import is_isomorphic as nx_is_isomorphic

from gflownet.envs.graph_building_env import (
    Graph,
    GraphAction,
    GraphActionType,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
    graph_without_edge,
)
from gflownet.utils.graphs import random_walk_probs


def hashg(g):
    return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(g, node_attr="v")


def is_isomorphic(u, v):
    return nx_is_isomorphic(u, v, lambda a, b: a == b, lambda a, b: a == b)


class BasicGraphContext(GraphBuildingEnvContext):
    """
    A basic graph generation context.

    This simple environment context is designed to be used to test implementations. It only allows for AddNode and
    AddEdge actions, and is meant to be used within the BasicGraphTask to generate graphs of up to 7 nodes with
    only two possible node attributes, making the state space a total of ~70k states (which is nicely enumerable
    and allows us to compute p_theta(x) exactly for all x in the state space).
    """

    def __init__(self, max_nodes=7, num_cond_dim=0, graph_data=None, output_gid=False):
        self.max_nodes = max_nodes
        self.output_gid = output_gid
        self.use_graph_cache = False

        self.node_attr_values = {
            "v": [0, 1],  # Imagine this is as colors
        }
        self._num_rw_feat = 8
        self.not_a_molecule_env = True

        self.num_new_node_values = len(self.node_attr_values["v"])
        self.num_node_attr_logits = None
        self.num_node_dim = self.num_new_node_values + 1 + self._num_rw_feat
        self.num_node_attrs = 1
        self.num_edge_attr_logits = None
        self.num_edge_attrs = 0
        self.num_cond_dim = num_cond_dim
        self.num_edge_dim = 1
        self.edges_are_duplicated = True
        self.edges_are_unordered = True

        # Order in which models have to output logits
        self.action_type_order = [
            GraphActionType.Stop,
            GraphActionType.AddNode,
            GraphActionType.AddEdge,
        ]
        self.bck_action_type_order = [
            GraphActionType.RemoveNode,
            GraphActionType.RemoveEdge,
        ]
        self._env = GraphBuildingEnv()
        self.device = torch.device("cpu")
        self.graph_data = graph_data
        self.hash_to_graphs: Dict[str, int] = {}
        if graph_data is not None:
            states_hash = [hashg(i) for i in graph_data]
            for i, h, g in zip(range(len(graph_data)), states_hash, graph_data):
                self.hash_to_graphs[h] = self.hash_to_graphs.get(h, list()) + [(g, i)]
            self._cache = {}

    def relabel(self, g, ga):
        if ga.action != GraphActionType.Stop:
            gp = self._env.step(g, ga)
        ig = self.graph_data[self.get_graph_idx(g)]
        rmap = nx.vf2pp_isomorphism(g, ig, "v")
        ga = copy.copy(ga)
        if rmap is None and not len(g):
            rmap = {0: 0}
        if ga.source is not None:
            ga.source = rmap[ga.source]
        if ga.target is not None:
            ga.target = rmap[ga.target]
        if ga.action != GraphActionType.Stop:
            gp2 = self._env.step(ig, ga)
            if not nx.is_isomorphic(gp2, gp, lambda a, b: a == b):
                raise ValueError()
        return copy.deepcopy(ig), ga

    def get_graph_idx(self, g, default=None):
        h = hashg(g)
        if h not in self.hash_to_graphs and default is not None:
            return default
        bucket = self.hash_to_graphs[h]
        if len(bucket) == 1:
            return bucket[0][1]
        for i in bucket:
            if is_isomorphic(i[0], g):
                return i[1]
        if default is not None:
            return default
        raise ValueError(g)

    def aidx_to_GraphAction(self, g: gd.Data, action_idx: Tuple[int, int, int], fwd: bool = True):
        """Translate an action index (e.g. from a GraphActionCategorical) to a GraphAction"""
        act_type, act_row, act_col = [int(i) for i in action_idx]
        if fwd:
            t = self.action_type_order[act_type]
        else:
            t = self.bck_action_type_order[act_type]

        if t is GraphActionType.Stop:
            return GraphAction(t)
        elif t is GraphActionType.AddNode:
            return GraphAction(t, source=act_row, value=self.node_attr_values["v"][act_col])
        elif t is GraphActionType.AddEdge:
            a, b = g.non_edge_index[:, act_row]
            return GraphAction(t, source=a.item(), target=b.item())
        elif t is GraphActionType.RemoveNode:
            return GraphAction(t, source=act_row)
        elif t is GraphActionType.RemoveEdge:
            a, b = g.edge_index[:, act_row * 2]
            return GraphAction(t, source=a.item(), target=b.item())

    def GraphAction_to_aidx(self, g: gd.Data, action: GraphAction) -> Tuple[int, int, int]:
        """Translate a GraphAction to an index tuple"""
        if action.action is GraphActionType.Stop:
            row = col = 0
            type_idx = self.action_type_order.index(action.action)
        elif action.action is GraphActionType.AddNode:
            row = action.source
            col = self.node_attr_values["v"].index(action.value)
            type_idx = self.action_type_order.index(action.action)
        elif action.action is GraphActionType.AddEdge:
            # Here we have to retrieve the index in non_edge_index of an edge (s,t)
            # that's also possibly in the reverse order (t,s).
            # That's definitely not too efficient, can we do better?
            row = (
                (g.non_edge_index.T == torch.tensor([(action.source, action.target)])).prod(1)
                + (g.non_edge_index.T == torch.tensor([(action.target, action.source)])).prod(1)
            ).argmax()
            col = 0
            type_idx = self.action_type_order.index(action.action)
        elif action.action is GraphActionType.RemoveNode:
            row = action.source
            col = 0
            type_idx = self.bck_action_type_order.index(action.action)
        elif action.action is GraphActionType.RemoveEdge:
            row = ((g.edge_index.T == torch.tensor([(action.source, action.target)])).prod(1)).argmax()
            row = int(row) // 2  # edges are duplicated, but edge logits are not
            col = 0
            type_idx = self.bck_action_type_order.index(action.action)
        return (type_idx, int(row), int(col))

    def graph_to_Data(self, g: Graph, t: int = 0) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance"""
        if self.graph_data is not None and self.use_graph_cache:
            # This caching achieves two things, first we'll speed things up
            gidx = self.get_graph_idx(g)
            if gidx in self._cache:
                return self._cache[gidx]
            # And second we'll always have the same node ordering, which is necessary for the tabular model
            # to work. In the non-tabular case, we're hopefully using a model that's invariant to node ordering, so this
            # shouldn't cause any problems.
            g = self.graph_data[gidx]

        x = torch.zeros((max(1, len(g.nodes)), self.num_node_dim - self._num_rw_feat))
        x[0, -1] = len(g.nodes) == 0
        remove_node_mask = torch.zeros((x.shape[0], 1)) + (1 if len(g) == 0 else 0)
        for i, n in enumerate(g.nodes):
            ad = g.nodes[n]
            x[i, self.node_attr_values["v"].index(ad["v"])] = 1
            if g.degree(n) <= 1:
                remove_node_mask[i] = 1

        remove_edge_mask = torch.zeros((len(g.edges), 1))
        for i, (u, v) in enumerate(g.edges):
            if g.degree(u) > 1 and g.degree(v) > 1:
                if nx.algorithms.is_connected(graph_without_edge(g, (u, v))):
                    remove_edge_mask[i] = 1
        edge_attr = torch.zeros((len(g.edges) * 2, self.num_edge_dim))
        edge_index = (
            torch.tensor([e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=torch.long).reshape((-1, 2)).T
        )
        gc = nx.complement(g)
        non_edge_index = torch.tensor([i for i in gc.edges], dtype=torch.long).reshape((-1, 2)).T
        gid = self.get_graph_idx(g) if self.output_gid else 0

        data = self._preprocess(
            gd.Data(
                x,
                edge_index,
                edge_attr,
                non_edge_index=non_edge_index,
                stop_mask=torch.ones((1, 1)),
                add_node_mask=torch.ones((x.shape[0], self.num_new_node_values)) * (len(g) < self.max_nodes),
                add_edge_mask=torch.ones((non_edge_index.shape[1], 1)),
                remove_node_mask=remove_node_mask,
                remove_edge_mask=remove_edge_mask,
                gid=gid,
            )
        )
        if self.graph_data is not None and self.use_graph_cache:
            self._cache[gidx] = data
        return data

    def _preprocess(self, g: gd.Data) -> gd.Data:
        if self._num_rw_feat > 0:
            g.x = torch.cat([g.x, random_walk_probs(g, self._num_rw_feat, skip_odd=True)], 1)
        return g

    def collate(self, graphs: List[gd.Data]):
        """Batch Data instances"""
        return gd.Batch.from_data_list(graphs, follow_batch=["edge_index", "non_edge_index"])

    def mol_to_graph(self, obj: Graph) -> Graph:
        return obj  # This is already a graph

    def graph_to_mol(self, g: Graph) -> Graph:
        # idem
        return g

    def is_sane(self, g: Graph) -> bool:
        return True

    def get_object_description(self, g: Graph, is_valid: bool) -> str:
        return str(self.get_graph_idx(g, -1))
