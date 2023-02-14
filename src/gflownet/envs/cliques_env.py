from typing import List, Tuple, Dict

import networkx as nx
from networkx.algorithms.isomorphism import is_isomorphic as nx_is_isomorphic
import torch
import torch_geometric.data as gd

from gflownet.envs.graph_building_env import Graph, GraphAction, GraphActionType, GraphBuildingEnvContext
from gflownet.utils.graphs import random_walk_probs


def hashg(g):
    return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(g, node_attr='v')


def is_isomorphic(u, v):
    nx_is_isomorphic(u, v, lambda a, b: a == b, lambda a, b: a == b)


class CliquesEnvContext(GraphBuildingEnvContext):
    """
    Color idea:
    - penalize neighbors of the same color
    - increasing the number of colors makes the task _easier_, but, color here could be an other
      good metric of diversity. Could even be used as a generalization metric. If there is an
      imbalance in the colors seen, what happens to the diversity?
    """
    def __init__(self, clique_size, num_cliques, num_cond_dim=0, graph_data=None):
        # The max reward is achievable with this many nodes and steps
        self.recommended_max_nodes = clique_size * num_cliques - num_cliques + 1
        self.recommended_max_steps = (self.recommended_max_nodes +
                                      (len(nx.complete_graph(clique_size).edges) - clique_size) * num_cliques)

        self.node_attr_values = {
            'v': [0, 1],  # Imagine this is as colors
        }
        self._num_rw_feat = 8

        self.num_new_node_values = len(self.node_attr_values['v'])
        self.num_node_attr_logits = None
        self.num_node_dim = self.num_new_node_values + 1 + self._num_rw_feat
        self.num_edge_attr_logits = None
        self.num_cond_dim = num_cond_dim
        self.num_edge_dim = 1

        # Order in which models have to output logits
        self.action_type_order = [
            GraphActionType.Stop,
            GraphActionType.AddNode,
            GraphActionType.AddEdge,
        ]
        self.action_mask_names = [
            None,
            'add_node_mask',
            'add_edge_mask',
        ]
        self.device = torch.device('cpu')
        self.graph_data = graph_data
        self.hash_to_graphs: Dict[str, int] = {}
        if graph_data is not None:
            states_hash = [hashg(i) for i in graph_data]
            for i, h, g in zip(range(len(graph_data)), states_hash, graph_data):
                self.hash_to_graphs[h] = self.hash_to_graphs.get(h, list()) + [(g, i)]

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

    def aidx_to_GraphAction(self, g: gd.Data, action_idx: Tuple[int, int, int]):
        """Translate an action index (e.g. from a GraphActionCategorical) to a GraphAction"""
        act_type, act_row, act_col = [int(i) for i in action_idx]
        t = self.action_type_order[act_type]
        if t is GraphActionType.Stop:
            return GraphAction(t)
        elif t is GraphActionType.AddNode:
            return GraphAction(t, source=act_row, value=self.node_attr_values['v'][act_col])
        elif t is GraphActionType.AddEdge:
            a, b = g.non_edge_index[:, act_row]
            return GraphAction(t, source=a.item(), target=b.item())

    def GraphAction_to_aidx(self, g: gd.Data, action: GraphAction) -> Tuple[int, int, int]:
        """Translate a GraphAction to an index tuple"""
        if action.action is GraphActionType.Stop:
            row = col = 0
        elif action.action is GraphActionType.AddNode:
            row = action.source
            col = self.node_attr_values['v'].index(action.value)
        elif action.action is GraphActionType.AddEdge:
            # Here we have to retrieve the index in non_edge_index of an edge (s,t)
            # that's also possibly in the reverse order (t,s).
            # That's definitely not too efficient, can we do better?
            row = ((g.non_edge_index.T == torch.tensor([(action.source, action.target)])).prod(1) +
                   (g.non_edge_index.T == torch.tensor([(action.target, action.source)])).prod(1)).argmax()
            col = 0
        type_idx = self.action_type_order.index(action.action)
        return (type_idx, int(row), int(col))

    def graph_to_Data(self, g: Graph) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance"""
        x = torch.zeros((max(1, len(g.nodes)), self.num_node_dim - self._num_rw_feat))
        x[0, -1] = len(g.nodes) == 0
        for i, n in enumerate(g.nodes):
            ad = g.nodes[n]
            x[i, self.node_attr_values['v'].index(ad['v'])] = 1

        edge_attr = torch.zeros((len(g.edges) * 2, self.num_edge_dim))
        edge_index = torch.tensor([e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=torch.long).reshape(
            (-1, 2)).T
        gc = nx.complement(g)
        non_edge_index = torch.tensor([i for i in gc.edges], dtype=torch.long).T.reshape((2, -1))

        return self._preprocess(
            gd.Data(
                x,
                edge_index,
                edge_attr,
                non_edge_index=non_edge_index,
                add_node_mask=torch.ones((x.shape[0], self.num_new_node_values)),
                add_edge_mask=torch.ones((non_edge_index.shape[1], 1)),
            ))

    def _preprocess(self, g: gd.Data) -> gd.Data:
        g.x = torch.cat([g.x, random_walk_probs(g, self._num_rw_feat, skip_odd=True)], 1)
        return g

    def collate(self, graphs: List[gd.Data]):
        """Batch Data instances"""
        return gd.Batch.from_data_list(graphs, follow_batch=['edge_index', 'non_edge_index'])

    def obj_to_graph(self, obj: object) -> Graph:
        return obj  # This is already a graph

    def graph_to_obj(self, g: Graph) -> Graph:
        # idem
        return g

    def is_sane(self, g: Graph) -> bool:
        return True

    def get_object_description(self, g: Graph, is_valid: bool) -> str:
        return str(self.get_graph_idx(g, -1))


