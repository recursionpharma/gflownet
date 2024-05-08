import copy
import enum
import json
import re
from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch_geometric.data as gd
from networkx.algorithms.isomorphism import is_isomorphic
from torch_scatter import scatter, scatter_max


class Graph(nx.Graph):
    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{list(self.nodes)}, {list(self.edges)}, {list(self.nodes[i]["v"] for i in self.nodes)}>'

    def bridges(self):
        return list(nx.bridges(self))

    def relabel_nodes(self, rmap):
        return nx.relabel_nodes(self, rmap)

    def clear_cache(self):
        self._Data_cache = None


def graph_without_edge(g, e):
    gp = g.copy()
    gp.remove_edge(*e)
    return gp


def graph_without_node(g, n):
    gp = g.copy()
    gp.remove_node(n)
    return gp


def graph_without_node_attr(g, n, a):
    gp = g.copy()
    del gp.nodes[n][a]
    return gp


def graph_without_edge_attr(g, e, a):
    gp = g.copy()
    del gp.edges[e][a]
    return gp


class GraphActionType(enum.Enum):
    # Forward actions
    Stop = enum.auto()
    AddNode = enum.auto()
    AddEdge = enum.auto()
    SetNodeAttr = enum.auto()
    SetEdgeAttr = enum.auto()
    # Backward actions
    RemoveNode = enum.auto()
    RemoveEdge = enum.auto()
    RemoveNodeAttr = enum.auto()
    RemoveEdgeAttr = enum.auto()

    @cached_property
    def cname(self):
        return re.sub(r"(?<!^)(?=[A-Z])", "_", self.name).lower()

    @cached_property
    def mask_name(self):
        return self.cname + "_mask"

    @cached_property
    def is_backward(self):
        return self.name.startswith("Remove")


class ActionIndex(NamedTuple):
    """
    Represents an index for an action in the GraphBuildingEnv.

    Different types of actions lead to logit matrices of different shapes,
    for exemple GraphActionType.Stop has a shape of (1, 1), while
    GraphActionType.AddNode has a shape of (n, m) where n is the number of
    nodes in the graph and m is the number of possible node types (idem for edge actions).
    It is thus convenient to represent the action as a tuple of indices.
    """

    action_type: int  # Index of the action type according to GraphActionType
    row_idx: int  # Index of the element the action applies to (e.g. node or edge)
    col_idx: int  # Index of the action variant (e.g. which attribute to set)


class GraphAction:
    def __init__(self, action: GraphActionType, source=None, target=None, value=None, attr=None):
        """A single graph-building action

        Parameters
        ----------
        action: GraphActionType
            the action type
        source: int, optional (e.g. GraphActionType.Stop has no source)
            the source node this action is applied on
        target: int, optional
            the target node (i.e. if specified this is an edge action)
        attr: str, optional
            the set attribute of a node/edge
        value: Any, optional
            the value (e.g. new node type) applied
        """
        self.action: GraphActionType = action
        self.source: Optional[int] = source
        self.target: Optional[int] = target
        self.attr: Optional[str] = attr
        self.value: Optional[Any] = value

    def __repr__(self):
        attrs = ", ".join(str(i) for i in [self.source, self.target, self.attr, self.value] if i is not None)
        return f"<{self.action}, {attrs}>"


class GraphBuildingEnv:
    """
    A Graph building environment which induces a DAG state space, compatible with GFlowNet.
    Supports forward and backward actions, with a `parents` function that list parents of
    forward actions.

    Edges and nodes can have attributes added to them in a key:value style.

    Edges and nodes are created with _implicit_ default attribute
    values (e.g. chirality, single/double bondness) so that:
        - an agent gets to do an extra action to set that attribute, but only
          if it is still default-valued (DAG property preserved)
        - we can generate a legal action for any attribute that isn't a default one.
    """

    def __init__(self, allow_add_edge=True, allow_node_attr=True, allow_edge_attr=True):
        """A graph building environment instance

        Parameters
        ----------
        allow_add_edge: bool
            if True, allows this action and computes AddEdge parents (i.e. if False, this
            env only allows for tree generation)
        allow_node_attr: bool
            if True, allows this action and computes SetNodeAttr parents
        allow_edge_attr: bool
            if True, allows this action and computes SetEdgeAttr parents
        """
        self.allow_add_edge = allow_add_edge
        self.allow_node_attr = allow_node_attr
        self.allow_edge_attr = allow_edge_attr

    def new(self):
        return Graph()

    def step(self, g: Graph, action: GraphAction) -> Graph:
        """Step forward the given graph state with an action

        Parameters
        ----------
        g: Graph
            the graph to be modified
        action: GraphAction
            the action taken on the graph, indices must match

        Returns
        -------
        gp: Graph
            the new graph
        """
        gp = g.copy()
        if action.action is GraphActionType.AddEdge:
            a, b = action.source, action.target
            assert self.allow_add_edge
            assert a in g and b in g
            if a > b:
                a, b = b, a
            assert a != b
            assert not g.has_edge(a, b)
            # Ideally the FA underlying this must only be able to send
            # create_edge actions which respect this a<b property (or
            # its inverse!) , otherwise symmetry will be broken
            # because of the way the parents method is written
            gp.add_edge(a, b)

        elif action.action is GraphActionType.AddNode:
            if len(g) == 0:
                assert action.source == 0  # TODO: this may not be useful
                gp.add_node(0, v=action.value)
            else:
                assert action.source in g.nodes
                e = [action.source, max(g.nodes) + 1]
                # if kw and 'relabel' in kw:
                #     e[1] = kw['relabel']  # for `parent` consistency, allow relabeling
                assert not g.has_edge(*e)
                gp.add_node(e[1], v=action.value)
                gp.add_edge(*e)

        elif action.action is GraphActionType.SetNodeAttr:
            assert self.allow_node_attr
            assert action.source in gp.nodes
            # For some "optional" attributes like wildcard atoms, we indicate that they haven't been
            # chosen by the 'None' value. Here we make sure that either the attribute doesn't
            # exist, or that it's an optional attribute that hasn't yet been set.
            assert action.attr not in gp.nodes[action.source] or gp.nodes[action.source][action.attr] is None
            gp.nodes[action.source][action.attr] = action.value

        elif action.action is GraphActionType.SetEdgeAttr:
            assert self.allow_edge_attr
            assert g.has_edge(action.source, action.target)
            assert action.attr not in gp.edges[(action.source, action.target)]
            gp.edges[(action.source, action.target)][action.attr] = action.value

        elif action.action is GraphActionType.RemoveNode:
            assert g.has_node(action.source)
            gp = graph_without_node(gp, action.source)
        elif action.action is GraphActionType.RemoveNodeAttr:
            assert g.has_node(action.source)
            gp = graph_without_node_attr(gp, action.source, action.attr)
        elif action.action is GraphActionType.RemoveEdge:
            assert g.has_edge(action.source, action.target)
            gp = graph_without_edge(gp, (action.source, action.target))
        elif action.action is GraphActionType.RemoveEdgeAttr:
            assert g.has_edge(action.source, action.target)
            gp = graph_without_edge_attr(gp, (action.source, action.target), action.attr)
        else:
            raise ValueError(f"Unknown action type {action.action}", action.action)

        gp.clear_cache()  # Invalidate cached properties since we've modified the graph
        return gp

    def parents(self, g: Graph):
        """List possible parents of graph `g`

        Parameters
        ----------
        g: Graph
            graph

        Returns
        -------
        parents: List[Pair(GraphAction, Graph)]
            The list of parent-action pairs that lead to `g`.
        """
        parents: List[Tuple[GraphAction, Graph]] = []
        # Count node degrees
        degree: Dict[int, int] = defaultdict(int)
        for a, b in g.edges:
            degree[a] += 1
            degree[b] += 1

        def add_parent(a, new_g):
            # Only add parent if the proposed parent `new_g` is not isomorphic
            # to already identified parents
            for ap, gp in parents:
                # Here we are relying on the dict equality operator for nodes and edges
                if is_isomorphic(new_g, gp, lambda a, b: a == b, lambda a, b: a == b):
                    return
            parents.append((a, new_g))

        for a, b in g.edges:
            if degree[a] > 1 and degree[b] > 1 and len(g.edges[(a, b)]) == 0:
                # Can only remove edges connected to non-leaves and without
                # attributes (the agent has to remove the attrs, then remove
                # the edge)
                new_g = graph_without_edge(g, (a, b))
                if nx.algorithms.is_connected(new_g):
                    add_parent(GraphAction(GraphActionType.AddEdge, source=a, target=b), new_g)
            for k in g.edges[(a, b)]:
                add_parent(
                    GraphAction(GraphActionType.SetEdgeAttr, source=a, target=b, attr=k, value=g.edges[(a, b)][k]),
                    graph_without_edge_attr(g, (a, b), k),
                )

        for i in g.nodes:
            # Can only remove leaf nodes and without attrs (except 'v'),
            # and without edges with attrs.
            if degree[i] == 1 and len(g.nodes[i]) == 1:
                edge = list(g.edges(i))[0]  # There should only be one since deg == 1
                if len(g.edges[edge]) == 0:
                    anchor = edge[0] if edge[1] == i else edge[1]
                    new_g = graph_without_node(g, i)
                    add_parent(
                        GraphAction(GraphActionType.AddNode, source=anchor, value=g.nodes[i]["v"]),
                        new_g,
                    )
            if len(g.nodes) == 1 and len(g.nodes[i]) == 1:
                # The final node is degree 0, need this special case to remove it
                # and end up with S0, the empty graph root (but only if it has no attrs except 'v')
                add_parent(
                    GraphAction(GraphActionType.AddNode, source=0, value=g.nodes[i]["v"]),
                    graph_without_node(g, i),
                )
            for k in g.nodes[i]:
                if k == "v":
                    continue
                add_parent(
                    GraphAction(GraphActionType.SetNodeAttr, source=i, attr=k, value=g.nodes[i][k]),
                    graph_without_node_attr(g, i, k),
                )
        return parents

    def count_backward_transitions(self, g: Graph, check_idempotent: bool = False):
        """Counts the number of parents of g (by default, without checking for isomorphisms)"""
        # We can count actions backwards easily, but only if we don't check that they don't lead to
        # the same parent. To do so, we need to enumerate (unique) parents and count how many there are:
        if check_idempotent:
            return len(self.parents(g))
        c = 0
        deg = [g.degree[i] for i in range(len(g.nodes))]
        for a, b in g.edges:
            if deg[a] > 1 and deg[b] > 1 and len(g.edges[(a, b)]) == 0:
                # Can only remove edges connected to non-leaves and without
                # attributes (the agent has to remove the attrs, then remove
                # the edge). Removal cannot disconnect the graph.
                new_g = graph_without_edge(g, (a, b))
                if nx.algorithms.is_connected(new_g):
                    c += 1
            c += len(g.edges[(a, b)])  # One action per edge attr
        for i in g.nodes:
            if deg[i] == 1 and len(g.nodes[i]) == 1 and len(g.edges[list(g.edges(i))[0]]) == 0:
                c += 1
            c += len(g.nodes[i]) - 1  # One action per node attr, except 'v'
            if len(g.nodes) == 1 and len(g.nodes[i]) == 1:
                # special case if last node in graph
                c += 1
        return c

    def reverse(self, g: Graph, ga: GraphAction):
        if ga.action == GraphActionType.Stop:
            return ga
        elif ga.action == GraphActionType.AddNode:
            return GraphAction(GraphActionType.RemoveNode, source=len(g.nodes))
        elif ga.action == GraphActionType.AddEdge:
            return GraphAction(GraphActionType.RemoveEdge, source=ga.source, target=ga.target)
        elif ga.action == GraphActionType.SetNodeAttr:
            return GraphAction(GraphActionType.RemoveNodeAttr, source=ga.source, attr=ga.attr)
        elif ga.action == GraphActionType.SetEdgeAttr:
            return GraphAction(GraphActionType.RemoveEdgeAttr, source=ga.source, target=ga.target, attr=ga.attr)
        elif ga.action == GraphActionType.RemoveNode:
            # TODO: implement neighbors or something?
            # neighbors = list(g.neighbors(ga.source))
            # source = 0 if not len(neighbors) else neighbors[0]
            neighbors = [i for i in g.edges if i[0] == ga.source or i[1] == ga.source]
            assert len(neighbors) <= 1  # RemoveNode should only be a legal action if the node has one or zero neighbors
            source = 0 if not len(neighbors) else neighbors[0][0] if neighbors[0][0] != ga.source else neighbors[0][1]
            return GraphAction(GraphActionType.AddNode, source=source, value=g.nodes[ga.source]["v"])
        elif ga.action == GraphActionType.RemoveEdge:
            return GraphAction(GraphActionType.AddEdge, source=ga.source, target=ga.target)
        elif ga.action == GraphActionType.RemoveNodeAttr:
            return GraphAction(
                GraphActionType.SetNodeAttr,
                source=ga.source,
                target=ga.target,
                attr=ga.attr,
                value=g.nodes[ga.source][ga.attr],
            )
        elif ga.action == GraphActionType.RemoveEdgeAttr:
            return GraphAction(
                GraphActionType.SetEdgeAttr,
                source=ga.source,
                target=ga.target,
                attr=ga.attr,
                value=g.edges[ga.source, ga.target][ga.attr],
            )
        else:
            raise ValueError(f"Unknown action type {ga.action}", ga.action)


def generate_forward_trajectory(g: Graph, max_nodes: int = None) -> List[Tuple[Graph, GraphAction]]:
    """Sample (uniformly) a trajectory that generates `g`"""
    # TODO: should this be a method of GraphBuildingEnv? handle set_node_attr flags and so on?
    gn = Graph()
    # Choose an arbitrary starting point, add to the stack
    stack: List[Tuple[int, ...]] = [(np.random.randint(0, len(g.nodes)),)]
    traj = []
    # This map keeps track of node labels in gn, since we have to start from 0
    relabeling_map: Dict[int, int] = {}
    while len(stack):
        # We pop from the stack until all nodes and edges have been
        # generated and their attributes have been set. Uninserted
        # nodes/edges will be added to the stack as the graph is
        # expanded from the starting point. Nodes/edges that have
        # attributes will be reinserted into the stack until those
        # attributes are "set".
        i = stack.pop(np.random.randint(len(stack)))

        gt = gn.copy()  # This is a shallow copy
        if len(i) > 1:  # i is an edge
            e = relabeling_map.get(i[0], None), relabeling_map.get(i[1], None)
            if e in gn.edges:
                # i exists in the new graph, that means some of its attributes need to be added.
                #
                # This remap is a special case for the fragment environment, due to the (poor) design
                # choice of treating directed edges as undirected edges. Until we have routines for
                # directed graphs, this may need to stay.
                def possibly_remap(attr):
                    if attr == f"{i[0]}_attach":
                        return f"{e[0]}_attach"
                    elif attr == f"{i[1]}_attach":
                        return f"{e[1]}_attach"
                    return attr

                attrs = [j for j in g.edges[i] if possibly_remap(j) not in gn.edges[e]]
                if len(attrs) == 0:
                    continue  # If nodes are in cycles edges leading to them get stack multiple times, disregard
                iattr = attrs[np.random.randint(len(attrs))]
                eattr = possibly_remap(iattr)
                gn.edges[e][eattr] = g.edges[i][iattr]
                act = GraphAction(
                    GraphActionType.SetEdgeAttr, source=e[0], target=e[1], attr=eattr, value=g.edges[i][iattr]
                )
            else:
                # i doesn't exist, add the edge
                if e[1] not in gn.nodes:
                    # The endpoint of the edge is not in the graph, this is a AddNode action
                    assert e[1] is None  # normally we shouldn't have relabeled i[1] yet
                    relabeling_map[i[1]] = len(relabeling_map)
                    e = e[0], relabeling_map[i[1]]
                    gn.add_node(e[1], v=g.nodes[i[1]]["v"])
                    gn.add_edge(*e)
                    for j in g[i[1]]:  # stack unadded edges/neighbours
                        jp = relabeling_map.get(j, None)
                        if jp not in gn or (e[1], jp) not in gn.edges:
                            stack.append((i[1], j))
                    act = GraphAction(GraphActionType.AddNode, source=e[0], value=g.nodes[i[1]]["v"])
                    if len(gn.nodes[e[1]]) < len(g.nodes[i[1]]):
                        stack.append((i[1],))  # we still have attributes to add to node i[1]
                else:
                    # The endpoint is in the graph, this is an AddEdge action
                    assert e[0] in gn.nodes
                    gn.add_edge(*e)
                    act = GraphAction(GraphActionType.AddEdge, source=e[0], target=e[1])

            if len(gn.edges[e]) < len(g.edges[i]):
                stack.append(i)  # we still have attributes to add to edge i
        else:  # i is a node, (u,)
            u = i[0]
            n = relabeling_map.get(u, None)
            if n not in gn.nodes:
                # u doesn't exist yet, this should only happen for the first node
                assert len(gn.nodes) == 0
                act = GraphAction(GraphActionType.AddNode, source=0, value=g.nodes[u]["v"])
                n = relabeling_map[u] = len(relabeling_map)
                gn.add_node(0, v=g.nodes[u]["v"])
                for j in g[u]:  # For every neighbour of node u
                    if relabeling_map.get(j, None) not in gn:
                        stack.append((u, j))  # push the (u,j) edge onto the stack
            else:
                # u exists, meaning we have attributes left to add
                attrs = [j for j in g.nodes[u] if j not in gn.nodes[n]]
                attr = attrs[np.random.randint(len(attrs))]
                gn.nodes[n][attr] = g.nodes[u][attr]
                act = GraphAction(GraphActionType.SetNodeAttr, source=n, attr=attr, value=g.nodes[u][attr])
            if len(gn.nodes[n]) < len(g.nodes[u]):
                stack.append((u,))  # we still have attributes to add to node u
        traj.append((gt, act))
    traj.append((gn, GraphAction(GraphActionType.Stop)))
    return traj


class GraphActionCategorical:
    def __init__(
        self,
        graphs: gd.Batch,
        raw_logits: List[torch.Tensor],
        keys: List[Union[str, None]],
        types: List[GraphActionType],
        deduplicate_edge_index=True,
        action_masks: List[torch.Tensor] = None,
        slice_dict: Optional[dict[str, torch.Tensor]] = None,
    ):
        """A multi-type Categorical compatible with generating structured actions.

        What is meant by type here is that there are multiple types of
        mutually exclusive actions, e.g. AddNode and AddEdge are
        mutually exclusive, but since their logits will be produced by
        different variable-sized tensors (corresponding to different
        elements of the graph, e.g. nodes or edges) it is inconvient
        to stack them all into one single Categorical. This class
        provides this convenient interaction between torch_geometric
        Batch objects and lists of logit tensors.

        Note on action-masking:
        Action masks depend on the environment logic (what are allowed v.s. prohibited actions).
        Thus, the action_masks should be created by the EnvContext (e.g. FragMolBuildingEnvContext)
        and passed to the GraphActionCategorical as a list of tensors. However, action masks
        should be applied to the logits within this class only to allow proper masking
        when computing log probabilities and sampling and avoid confusion about
        the state of the logits (masked or not) for external members.
        For this reason, the constructor takes as input the raw (unmasked) logits and the
        masks separately. The (masked) logits are cached in the _masked_logits attribute.
        Both the (masked) logits and the masks are private properties, and attempts to edit the masks or the logits will
        apply the masks to the raw_logits again.

        Parameters
        ----------
        graphs: Batch
            A Batch of graphs to which the logits correspond
        raw_logits: List[Tensor]
            A list of tensors of shape `(n, m)` representing raw (unmasked) logits
            over a variable number of graph elements (e.g. nodes) for
            which there are `m` possible actions. `n` should thus be
            equal to the sum of the number of such elements for each
            graph in the Batch object. The length of the `logits` list
            should thus be equal to the number of element types (in
            other words there should be one tensor per type).
        keys: List[Union[str, None]]
            The keys corresponding to the Graph elements for each
            tensor in the logits list. Used to extract the `_batch`
            and slice attributes. For example, if the first logit
            tensor is a per-node action logit, and the second is a
            per-edge, `keys` could be `['x', 'edge_index']`. If
            keys[i] is None, the corresponding logits are assumed to
            be graph-level (i.e. if there are `k` graphs in the Batch
            object, this logit tensor would have shape `(k, m)`)
        types: List[GraphActionType]
            The action type each logit corresponds to.
        deduplicate_edge_index: bool, default=True
            If true, this means that the 'edge_index' keys have been reduced
            by e_i[::2] (presumably because the graphs are undirected)
        masks: List[Tensor], default=None
            If not None, a list of broadcastable tensors that
            mask out logits of invalid actions
        slice_dist: Optional[dict[str, Tensor]], default=None
            If not None, a map of tensors that indicate the start (and end) the graph index
            of each object keyed. If None, uses the `_slice_dict` attribute of the graphs.
        """
        self.num_graphs = graphs.num_graphs
        assert all([i.ndim == 2 for i in raw_logits])
        assert len(raw_logits) == len(types) == len(keys)
        if action_masks is not None:
            assert len(raw_logits) == len(action_masks)
            assert all([i.ndim == 2 for i in action_masks])
        # The logits
        self.raw_logits = raw_logits
        self.types = types
        self.keys = keys
        self.dev = dev = graphs.x.device
        self._epsilon = 1e-38
        # TODO: mask is only used by graph_sampler, but maybe we should be more careful with it
        # (e.g. in a softmax and such)
        # Can be set to indicate which raw_logits are masked out (shape must match raw_logits or have
        # broadcast dimensions already set)
        self._action_masks: List[Any] = action_masks
        self._apply_action_masks()

        # I'm extracting batches and slices in a slightly hackish way,
        # but I'm not aware of a proper API to torch_geometric that
        # achieves this "neatly" without accessing private attributes

        # This is the minibatch index of each entry in the logits
        # i.e., if graph i in the Batch has N[i] nodes,
        #    g.batch == [0,0,0, ...,  1,1,1,1,1, ... ]
        #                 N[0] times    N[1] times
        # This generalizes to edges and non-edges.
        # Append '_batch' to keys except for 'x', since TG has a special case (done by default for 'x')
        self.batch = [
            (
                getattr(graphs, f"{k}_batch" if k != "x" else "batch")
                if k is not None
                # None signals a global logit rather than a per-instance logit
                else torch.arange(graphs.num_graphs, device=dev)
            )
            for k in keys
        ]
        # This is the cumulative sum (prefixed by 0) of N[i]s
        slice_dict = graphs._slice_dict if slice_dict is None else slice_dict
        self.slice = [
            slice_dict[k].to(dev) if k is not None else torch.arange(graphs.num_graphs + 1, device=dev) for k in keys
        ]
        self.logprobs = None
        self.log_n = None

        if deduplicate_edge_index and "edge_index" in keys:
            for idx, k in enumerate(keys):
                if k != "edge_index":
                    continue
                self.batch[idx] = self.batch[idx][::2]
                self.slice[idx] = self.slice[idx].div(2, rounding_mode="floor")

    @property
    def logits(self):
        return self._masked_logits

    @logits.setter
    def logits(self, new_raw_logits):
        self.raw_logits = new_raw_logits
        self._apply_action_masks()

    @property
    def action_masks(self):
        return self._action_masks

    @action_masks.setter
    def action_masks(self, new_action_masks):
        self._action_masks = new_action_masks
        self._apply_action_masks()

    def _apply_action_masks(self):
        self._masked_logits = (
            [self._mask(logits, mask) for logits, mask in zip(self.raw_logits, self._action_masks)]
            if self._action_masks is not None
            else self.raw_logits
        )

    def _mask(self, x, m):
        assert m.dtype == torch.float
        return x.masked_fill(m == 0.0, -torch.inf)

    def detach(self):
        new = copy.copy(self)
        new._masked_logits = [i.detach() for i in new._masked_logits]
        if new.logprobs is not None:
            new.logprobs = [i.detach() for i in new.logprobs]
        if new.log_n is not None:
            new.log_n = new.log_n.detach()
        return new

    def to(self, device):
        self.dev = device
        self.raw_logits = [i.to(device) for i in self.raw_logits]
        self._masked_logits = [i.to(device) for i in self._masked_logits]
        self.batch = [i.to(device) for i in self.batch]
        self.slice = [i.to(device) for i in self.slice]
        if self.logprobs is not None:
            self.logprobs = [i.to(device) for i in self.logprobs]
        if self.log_n is not None:
            self.log_n = self.log_n.to(device)
        if self._action_masks is not None:
            self._action_masks = [i.to(device) for i in self._action_masks]
        return self

    def log_n_actions(self):
        if self.log_n is None:
            self.log_n = (
                sum(
                    [
                        scatter(m.broadcast_to(i.shape).int().sum(1), b, dim=0, dim_size=self.num_graphs, reduce="sum")
                        for m, i, b in zip(self._action_masks, self._masked_logits, self.batch)
                    ]
                )
                .clamp(1)
                .float()
                .log()
                .clamp(1)
            )
        return self.log_n

    def _compute_batchwise_max(
        self,
        x: List[torch.Tensor],
        detach: bool = True,
        batch: Optional[List[torch.Tensor]] = None,
        reduce_columns: bool = True,
    ):
        """Compute the maximum value of each batch element in `x`

        Parameters
        ----------
        x: List[torch.Tensor]
            A list of tensors of shape `(n, m)` (e.g. representing _masked_logits)
        detach: bool, default=True
            If true, detach the tensors before computing the max
        batch: List[torch.Tensor], default=None
            The batch index of each element in `x`. If None, uses self.batch
        reduce_columns: bool, default=True
            If true computes the max over the columns, and returns a tensor of shape `(k,)`
            If false, only reduces over rows, returns a list of (values, indexes) tuples.

        Returns
        -------
        maxl: (values: torch.Tensor, indices: torch.Tensor)
            A named tuple of tensors of shape `(k,)` where `k` is the number of graphs in the batch, unless
            reduce_columns is False. In the latter case, returns a list of named tuples that don't have columns reduced.
        """
        if detach:
            x = [i.detach() for i in x]
        if batch is None:
            batch = self.batch
        # First we prefill `out` with the minimum values in case
        # there are no corresponding logits (this can happen if e.g. a
        # graph has no edges), we don't want to accidentally take the
        # max of that type, since we'd get 0.
        min_val = torch.finfo().min
        outs = [torch.zeros(self.num_graphs, i.shape[1], device=self.dev) + min_val for i in x]
        maxl = [scatter_max(i, b, dim=0, out=out) for i, b, out in zip(x, batch, outs)]
        if reduce_columns:
            return torch.cat([values for values, indices in maxl], dim=1).max(1)
        return maxl

    def logsoftmax(self):
        """Compute log-probabilities given logits"""
        if self.logprobs is not None:
            return self.logprobs
        # Use the `subtract by max` trick to avoid precision errors.
        maxl = self._compute_batchwise_max(self._masked_logits).values
        # substract by max then take exp
        # x[b, None] indexes by the batch to map back to each node/edge and adds a broadcast dim
        corr_logits = [(i - maxl[b, None]) for i, b in zip(self._masked_logits, self.batch)]
        exp_logits = [i.exp().clamp(self._epsilon) for i, b in zip(corr_logits, self.batch)]
        # sum corrected exponentiated logits, to get log(Z') = log(Z - max) = log(sum(exp(logits - max)))
        logZ = sum(
            [
                scatter(i, b, dim=0, dim_size=self.num_graphs, reduce="sum").sum(1)
                for i, b in zip(exp_logits, self.batch)
            ]
        ).log()
        # log probabilities is log(exp(logit) / Z) = (logit - max) - log(Z')
        self.logprobs = [i - logZ[b, None] for i, b in zip(corr_logits, self.batch)]
        return self.logprobs

    def logsumexp(self, x=None):
        """Reduces `x` (the _masked_logits by default) to one scalar per graph"""
        if x is None:
            x = self._masked_logits
        # Use the `subtract by max` trick to avoid precision errors.
        maxl = self._compute_batchwise_max(x).values
        # substract by max then take exp
        # x[b, None] indexes by the batch to map back to each node/edge and adds a broadcast dim
        exp_vals = [(i - maxl[b, None]).exp().clamp(self._epsilon) for i, b in zip(x, self.batch)]
        # sum corrected exponentiated _masked_logits, to get log(Z - max) = log(sum(exp(_masked_logits)) - max)
        reduction = sum(
            [scatter(i, b, dim=0, dim_size=self.num_graphs, reduce="sum").sum(1) for i, b in zip(exp_vals, self.batch)]
        ).log()
        # Add back max
        return reduction + maxl

    def sample(self) -> List[ActionIndex]:
        """Samples this categorical
        Returns
        -------
        actions: List[ActionIndex]
            A list of indices representing [action type, element index, action index]. See constructor.
        """
        # Use the Gumbel trick to sample categoricals
        # i.e. if X ~ argmax(logits - log(-log(uniform(logits.shape))))
        # then  p(X = i) = exp(logits[i]) / Z
        # Here we have to do the argmax first over the variable number
        # of rows of each element type for each graph in the
        # minibatch, then over the different types (since they are
        # mutually exclusive).

        # Uniform noise
        u = [torch.rand(i.shape, device=self.dev) for i in self._masked_logits]
        # Gumbel noise
        gumbel = [logit - (-noise.log()).log() for logit, noise in zip(self._masked_logits, u)]
        # Take the argmax
        return self.argmax(x=gumbel)

    def argmax(
        self,
        x: List[torch.Tensor],
        batch: List[torch.Tensor] = None,
        dim_size: int = None,
    ) -> List[ActionIndex]:
        """Takes the argmax, i.e. if x are the logits, returns the most likely action.

        Parameters
        ----------
        x: List[Tensor]
            Tensors in the same format as the logits (see constructor).
        batch: List[Tensor]
            Tensors in the same format as the batch indices of torch_geometric, default `self.batch`.
        dim_size: int
            The reduction dimension, default `self.num_graphs`.
        Returns
        -------
        actions: List[ActionIndex]
            A list of indices representing [action type, element index, action index]. See constructor.
        """
        # scatter_max and .max create a (values, indices) pair
        # These logits are 2d (num_obj_of_type, num_actions_of_type),
        # first reduce-max over the batch, which preserves the
        # columns, so we get (minibatch_size, num_actions_of_type).
        if batch is None:
            batch = self.batch
        if dim_size is None:
            dim_size = self.num_graphs
        # We don't want to reduce over the columns, since we want to keep the index within each column of the max
        mnb_max = self._compute_batchwise_max(x, batch=batch, reduce_columns=False)
        # Then over cols, this gets us which col holds the max value,
        # so we get (minibatch_size,)
        col_max = [values.max(1) for values, idx in mnb_max]
        # Now we look up which row in those argmax cols was the max:
        row_pos = [idx_mnb[torch.arange(len(idx_col)), idx_col] for (_, idx_mnb), (_, idx_col) in zip(mnb_max, col_max)]
        # The maxes themselves
        maxs = [values for values, idx in col_max]
        # Now we need to check which type of logit has the actual max
        type_max_val, type_max_idx = torch.stack(maxs).max(0)
        if torch.isfinite(type_max_val).logical_not_().any():
            raise ValueError("Non finite max value in sample", (type_max_val, x))

        # Now we can return the indices of where the actions occured
        # in the form List[(type, row, column)]
        assert dim_size == type_max_idx.shape[0]
        argmaxes = []
        for i in range(type_max_idx.shape[0]):
            t = type_max_idx[i]
            # Subtract from the slice of that type and index, since the computed
            # row position is batch-wise rather graph-wise
            argmaxes.append(
                ActionIndex(
                    action_type=int(t), row_idx=int(row_pos[t][i] - self.slice[t][i]), col_idx=int(col_max[t][1][i])
                )
            )
        # It's now up to the Context class to create GraphBuildingAction instances
        # if it wants to convert these indices to env-compatible actions
        return argmaxes

    def log_prob(self, actions: List[ActionIndex], logprobs: torch.Tensor = None, batch: torch.Tensor = None):
        """The log-probability of a list of action tuples, effectively indexes `logprobs` using internal
        slice indices.

        Parameters
        ----------
        actions: List[ActionIndex]
            A list of n action tuples denoting indices
        logprobs: List[Tensor]
            [Optional] The log-probablities to be indexed (self.logsoftmax() by default) in order (i.e. this
            assumes there are n graphs represented by this object).
        batch: Tensor
            [Optional] The batch of each action. If None (default) then this is arange(num_graphs), i.e. one
            action per graph is selected, in order.

        Returns
        -------
        log_prob: Tensor
            The log probability of each action.
        """
        N = self.num_graphs
        if logprobs is None:
            logprobs = self.logsoftmax()
        if batch is None:
            batch = torch.arange(N, device=self.dev)
        # We want to do the equivalent of this:
        #    [logprobs[t][row + self.slice[t][i], col] for i, (t, row, col) in zip(batch, actions)]
        # but faster.

        # each action is a 3-tuple ActionIndex (type, row, column), where type is the index of the action type group.
        actions = torch.as_tensor(actions, device=self.dev, dtype=torch.long)
        assert actions.shape[0] == batch.shape[0]  # Check there are as many actions as batch indices
        # To index the log probabilities efficiently, we will ravel the array, and compute the
        # indices of the raveled actions.
        # First, flatten and cat:
        all_logprobs = torch.cat([i.flatten() for i in logprobs])
        # The action type offset depends on how many elements each logit group has, and we retrieve by
        # the type index 0:
        t_offsets = torch.tensor([0] + [i.numel() for i in logprobs], device=self.dev).cumsum(0)[actions[:, 0]]
        # The row offset depends on which row the graph's corresponding logits start (since they are
        # all concatenated together). This is stored in self.slice; each logit group has its own
        # slice tensor of shape N+1 (since the 0th entry is always 0).
        # We want slice[t][i] for every graph i in the batch, since each slice has N+1 elements we
        # multiply t by N+1, batch is by default arange(N) so it just gets each graph's
        # corresponding row index.
        graph_row_offsets = torch.cat(self.slice)[actions[:, 0] * (N + 1) + batch]
        # Now we add the row value. To do that we need to know the number of elements of each row in
        # the flattened array, this is simply i.shape[1].
        row_lengths = torch.tensor([i.shape[1] for i in logprobs], device=self.dev)
        # Now we can multiply the length of the row for each type t by the actual row index,
        # offsetting by the row at which each graph's logits start.
        row_offsets = row_lengths[actions[:, 0]] * (actions[:, 1] + graph_row_offsets)
        # This is the last index in the raveled tensor, therefore the offset is just the column value
        col_offsets = actions[:, 2]
        # Index the flattened array
        return all_logprobs[t_offsets + row_offsets + col_offsets]

    def entropy(self, logprobs=None):
        """The entropy for each graph categorical in the batch

        Parameters
        ----------
        logprobs: List[Tensor]
            The log-probablities of the policy (self.logsoftmax() by default)

        Returns
        -------
        entropies: Tensor
            The entropy for each graph categorical in the batch
        """
        if logprobs is None:
            logprobs = self.logsoftmax()
        entropy = -sum(
            [
                scatter(i * i.exp(), b, dim=0, dim_size=self.num_graphs, reduce="sum").sum(1)
                for i, b in zip(logprobs, self.batch)
            ]
        )
        return entropy


class GraphBuildingEnvContext:
    """A context class defines what the graphs are, how they map to and from data"""

    device: torch.device
    action_type_order: List[GraphActionType]
    bck_action_type_order: List[GraphActionType]

    def ActionIndex_to_GraphAction(self, g: gd.Data, aidx: ActionIndex, fwd: bool = True) -> GraphAction:
        """Translate an action index (e.g. from a GraphActionCategorical) to a GraphAction
        Parameters
        ----------
        g: gd.Data
            The graph to which the action is being applied
        aidx: ActionIndex
            The tensor indices for the corresponding action
        fwd: bool
            If True (default) then this is a forward action

        Returns
        -------
        action: GraphAction
            A graph action that could be applied to the original graph coressponding to g.
        """
        raise NotImplementedError()

    def GraphAction_to_ActionIndex(self, g: gd.Data, action: GraphAction) -> ActionIndex:
        """Translate a GraphAction to an ActionIndex (e.g. from a GraphActionCategorical)
        Parameters
        ----------
        g: gd.Data
            The graph to which the action is being applied
        action: GraphAction
            A graph action that could be applied to the original graph coressponding to g.

        Returns
        -------
        aidx: ActionIndex
            The tensor indices for the corresponding action
        """
        raise NotImplementedError()

    def graph_to_Data(self, g: Graph) -> gd.Data:
        """Convert a networkx Graph to a torch geometric Data instance
        The logic to build masks for prohibited actions can be implemented here,
        packed in the data object and used in the GraphActionCategorical.
        Parameters
        ----------
        g: Graph
            A graph instance.

        Returns
        -------
        torch_g: gd.Data
            The corresponding torch_geometric graph.
        """
        raise NotImplementedError()

    def collate(self, graphs: List[gd.Data]) -> gd.Batch:
        """Convert a list of torch geometric Data instances to a Batch
        instance.  This exists so that environment contexts can set
        custom batching attributes, e.g. by using `follow_batch`.

        Parameters
        ----------
        graphs: List[gd.Data]
            Graph instances

        Returns
        -------
        batch: gd.Batch
            The corresponding batch.
        """
        return gd.Batch.from_data_list(graphs)

    def is_sane(self, g: Graph) -> bool:
        """Verifies whether a graph is sane according to the context. This can
        catch, e.g. impossible molecules.

        Parameters
        ----------
        g: Graph
            A graph.

        Returns
        -------
        is_sane: bool:
            True if the environment considers g to be sane.
        """
        raise NotImplementedError()

    def obj_to_graph(self, obj: Any) -> Graph:
        """Converts a native object into a generic Graph that the environment can handle

        Parameters
        ----------
        obj: Any
            An object

        Returns
        -------
        g: Graph
            The corresponding Graph representation of that object.
        """
        raise NotImplementedError()

    def object_to_log_repr(self, g: Graph) -> str:
        """Convert a Graph to a string representation for logging purposes"""
        return json.dumps(
            [[(i, g.nodes[i]) for i in g.nodes], [(e, g.edges[e]) for e in g.edges]], separators=(",", ":")
        )

    def graph_to_obj(self, g: Graph) -> Any:
        """Convert a graph back to an object"""
        raise NotImplementedError()

    def has_n(self) -> bool:
        return False

    def log_n(self, g) -> float:
        return 0.0

    def traj_log_n(self, traj):
        return [self.log_n(g) for g, _ in traj]


def action_type_to_mask(t: GraphActionType, gbatch: gd.Batch, assert_mask_exists: bool = False):
    if assert_mask_exists:
        assert hasattr(gbatch, t.mask_name), f"Mask {t.mask_name} not found in graph data"
    return getattr(gbatch, t.mask_name) if hasattr(gbatch, t.mask_name) else torch.ones((1, 1), device=gbatch.x.device)
