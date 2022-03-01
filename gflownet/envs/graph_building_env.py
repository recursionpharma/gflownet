from collections import defaultdict
import enum
from typing import List

import networkx as nx
from networkx.algorithms.isomorphism import is_isomorphic

import torch
import torch_geometric.data as gd
from torch_scatter import scatter


class Graph(nx.Graph):
    # Subclassing nx.Graph for debugging purposes
    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{list(self.nodes)}, {list(self.edges)}, {list(self.nodes[i]["v"] for i in self.nodes)}>'


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


class GraphAction:
    def __init__(self, action: GraphActionType, source=None, target=None, value=None, attr=None, relabel=None):
        """A single graph-building action

        Parameters
        ----------
        action: GraphActionType
            the action type
        source: int
            the source node this action is applied on
        target: int, optional
            the target node (i.e. if specified this is an edge action)
        attr: str, optional
            the set attribute of a node/edge
        value: Any, optional
            the value (e.g. new node type) applied
        relabel: int, optional
            for AddNode actions, relabels the new node with that id
        """
        self.action = action
        self.source = source
        self.target = target
        self.attr = attr
        self.value = value
        self.relabel = relabel


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
        allow_add_edge: bool
            if True, allows this action and computes AddEdge parents (i.e. if False, this env only allows for tree generation)
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
                assert action.source is None  # TODO: this may not be useful
                gp.add_node(action.relabel or 0, v=action.value)
            else:
                assert action.source in g.nodes
                e = [action.source, max(g.nodes) + 1]
                if kw and 'relabel' in kw:
                    e[1] = kw['relabel']  # for `parent` consistency, allow relabeling
                assert not g.has_edge(*e)
                gp.add_node(e[1], v=action.value)
                gp.add_edge(*e)

        elif action is GraphActionType.SetNodeAttr:
            assert self.allow_node_attr
            assert action.source in gp.nodes
            gp.nodes[action.source][action.attr] = action.value

        elif action is GraphActionType.SetEdgeAttr:
            assert self.allow_edge_attr
            assert g.has_edge(action.source, action.target)
            gp.edges[(action.source, action.target)][action.attr] = action.value
        else:
            # TODO: backward actions if we want to support MCMC-GFN style algorithms
            raise ValueError(f'Unknown action type {action.action}', action.action)

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
        parents = []
        # Count node degrees
        degree = defaultdict(int)
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
                    add_parent((self.add_edge, a, b), new_g)
            for k in g.edges[(a, b)]:
                add_parent((self.set_edge_attr, (a, b), k, g.edges[(a, b)][k]), graph_without_edge_attr(g, (a, b), k))

        for i in g.nodes:
            # Can only remove leaf nodes and without attrs (except 'v'),
            # and without edges with attrs.
            if (degree[i] == 1 and len(g.nodes[i]) == 1):
                edge = list(g.edges(i))[0]  # There should only be one since deg == 1
                if len(g.edges[edge]) == 0:
                    anchor = edge[0] if edge[1] == i else edge[1]
                    new_g = graph_without_node(g, i)
                    add_parent((self.add_node, anchor, g.nodes[i]['v'], {'relabel': i}), new_g)
            if len(g.nodes) == 1:
                # The final node is degree 0, need this special case to remove it
                # and end up with S0, the empty graph root
                add_parent((self.add_node, None, g.nodes[i]['v'], {'relabel': i}), graph_without_node(g, i))
            for k in g.nodes[i]:
                if k == 'v':
                    continue
                add_parent((self.set_node_attr, i, k, g.nodes[i][k]), graph_without_node_attr(g, i, k))
        return parents


def generate_forward_trajectory(g: Graph):
    """Sample (uniformly) a trajectory that generates `g`"""
    # TODO: should this be a method of GraphBuildingEnv? handle set_node_attr flags and so on?
    gn = Graph()
    # Choose an arbitrary starting point, add to the stack
    stack = [np.random.randint(0, len(g.nodes))]
    traj = []
    while len(stack):
        # We pop from the stack until all nodes and edges have been
        # generated and their attributes have been set. Uninserted
        # nodes/edges will be added to the stack as the graph is
        # expanded from the starting point. Nodes/edges that have
        # attributes will be reinserted into the stack until those
        # attributes are "set".
        i = stack.pop(np.random.randint(len(stack)))
        if type(i) is tuple:  # i is an edge
            if i in gn.edges:
                # i exists in the new graph, that means some of its attributes need to be added
                attrs = [j for j in g.edges[i] if j not in gn.edges[i]]
                attr = attrs[np.random.randint(len(attrs))]
                gn.edges[i][attr] = g.edges[i][attr]
                traj.append(
                    GraphAction(GraphActionType.SetEdgeAttr, source=i[0], target=i[1], attr=attr,
                                value=g.edges[i][attr]))
            else:
                # i doesn't exist, add the edge
                if i[1] not in gn.nodes:
                    # The endpoint of the edge is not in the graph, this is a AddNode action
                    gn.add_node(i[1], v=g.nodes[i[1]]['v'])
                    for j in g[i[1]]:
                        if j not in gn:
                            stack.append((i[1], j))
                    gn.add_edge(*i)
                    traj.append(
                        GraphAction(GraphActionType.AddNode, source=i[0], value=g.nodes[i[1]]['v'], relabel=i[1]))
                    if len(gn.nodes[i[1]]) < len(g.nodes[i[1]]):
                        stack.append(i[1])  # we still have attributes to add to node i[1]
                else:
                    # The endpoint is in the graph, this is an AddEdge action
                    gn.add_edge(*i)
                    traj.append(GraphAction(GraphActionType.AddEdge, source=i[0], target=i[1]))

            if len(gn.edges[i]) < len(g.edges[i]):
                stack.append(i)  # we still have attributes to add to edge i
        else:  # i is a node
            if i not in gn.nodes:
                # i doesn't exist yet, this should only happen for the first node
                assert len(gn.nodes) == 0
                traj.append(GraphAction(GraphActionType.AddNode, source=None, value=g.nodes[i]['v'], relabel=i))
                gn.add_node(i, v=g.nodes[i]['v'])
                for j in g[i]:  # For every neighbour of node i
                    if j not in gn:
                        stack.append((i, j))  # push the (i,j) edge onto the stack
            else:
                # i exists, meaning we have attributes left to add
                attrs = [j for j in g.nodes[i] if j not in gn.nodes[i]]
                attr = attrs[np.random.randint(len(attrs))]
                gn.nodes[i][attr] = g.nodes[i][attr]
                traj.append((env.set_node_attr, i, attr, g.nodes[i][attr]))
            if len(gn.nodes[i]) < len(g.nodes[i]):
                stack.append(i)  # we still have attributes to add to node i
    return traj


class GraphActionCategorical:
    def __init__(self, graphs: gd.Batch, logits: List[torch.Tensor], keys: List[str], types: List[GraphActionType]):
        """A multi-type Categorical compatible with generating structured actions.
        
        What is meant by type here is that there are multiple types of
        mutually exclusive actions, e.g. AddNode and AddEdge are
        mutually exclusive, but since their logits will be produced by
        different variable-sized tensors (corresponding to different
        elements of the graph, e.g. nodes or edges) it is inconvient
        to stack them all into one single Categorical. This class
        provides this convenient interaction between torch_geometric
        Batch objects and lists of logit tensors.

        Parameters
        ----------
        graphs: Batch
            A Batch of graphs to which the logits correspond
        logits: List[Tensor]
            A list of tensors of shape `(n, m)` representing logits
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

        """
        # TODO: handle legal action masks? (e.g. can't add a node attr to a node that already has an attr)
        # TODO: cuda-ize
        self.g = graphs
        # The logits
        self.logits = logits
        self.types = types

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
            getattr(graphs, f'{k}_batch' if k != 'x' else 'batch') if k is not None
            # None signals a global logit rather than a per-instance logit
            else torch.arange(graphs.num_graphs) for k in keys
        ]
        # This is the cumulative sum (prefixed by 0) of N[i]s
        self.slice = [graphs._slice_dict[k] if k is not None else torch.arange(graphs.num_graphs) for k in keys]
        self.logprobs = None

    def logsoftmax(self):
        """Compute log-probabilities given logits"""
        if self.logprobs is not None:
            return self.logprobs
        # Use the `subtract by max` trick to avoid precision errors:
        # compute max
        maxl = torch.cat(
            [scatter(i, b, dim=0, dim_size=self.g.num_graphs, reduce='max') for i, b in zip(self.logits, self.batch)],
            dim=1).max(1).values
        # substract by max then take exp
        # x[b, None] indexes by the batch to map back to each node/edge and adds a broadcast dim
        exp_logits = [(i - maxl[b, None]).exp() for i, b in zip(self.logits, self.batch)]
        # sum corrected exponentiated logits, to get log(Z - max) = log(sum(exp(logits)) - max)
        logZ = sum([
            scatter(i, b, dim=0, dim_size=self.g.num_graphs, reduce='sum').sum(1)
            for i, b in zip(exp_logits, self.batch)
        ]).log()
        # log probabilities is log(exp(logit) / Z)
        self.logprobs = [i.log() - logZ[b, None] for i, b in zip(exp_logits, self.batch)]
        return self.logprobs

    def sample(self) -> List[tuple[int, int, int]]:
        # Use the Gumbel trick to sample categoricals
        # i.e. if X ~ argmax(logits - log(-log(uniform(logits.shape))))
        # then  p(X = i) = exp(logits[i]) / Z
        # Uniform noise
        u = [torch.rand(i.shape) for i in self.logits]
        # Gumbel noise
        gumbel = [logit - (-noise.log()).log() for logit, noise in zip(self.logits, u)]
        # scatter_max and .max create a (values, indices) pair
        # These logits are 2d (num_obj_of_type, num_actions_of_type),
        # first reduce-max over the batch, which preserves the
        # columns, so we get (minibatch_size, num_actions_of_type)
        mnb_max = [scatter_max(i, b, dim=0) for i, b in zip(gumbel, self.batch)]
        # Then over cols, this gets us which col holds the max value,
        # so we get (minibatch_size,)
        col_max = [values.max(1) for values, idx in mnb_max]
        # Now we look up which row in those argmax cols was the max:
        row_pos = [idx_mnb[torch.arange(len(idx_col)), idx_col] for (_, idx_mnb), (_, idx_col) in zip(mnb_max, col_max)]
        # The maxes themselves
        maxs = [values for values, idx in col_max]
        # Now we need to check which type of logit has the actual max
        type_max_val, type_max_idx = torch.stack(maxs).max(0)

        # Now we can return the indices of where the actions occured
        # in the form List[(type, row, column)]
        actions = []
        for i in range(type_max_idx.shape[0]):
            t = type_max_idx[i]
            # Subtract from the slice of that type and index, since the computed
            # row position is batch-wise rather graph-wise
            actions.append((t, row_pos[t][i] - self.slice[t][i], col_max[t][1][i]))
        # It's now up to the Context class to create GraphBuildingAction instances
        # if it wants to convert these indices to env-compatible actions
        return actions

    def log_prob(self, actions: List[tuple[int, int, int]]):
        logprobs = self.logsoftmax()
        return torch.stack([logprobs[t][row + self.slice[t][i], col] for i, (t, row, col) in enumerate(actions)])
