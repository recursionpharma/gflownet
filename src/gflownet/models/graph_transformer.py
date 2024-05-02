from itertools import chain
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.utils import add_self_loops

from gflownet.config import Config
from gflownet.envs.graph_building_env import GraphActionCategorical, GraphActionType, action_type_to_mask


def mlp(n_in, n_hid, n_out, n_layer, act=nn.LeakyReLU):
    """Creates a fully-connected network with no activation after the last layer.
    If `n_layer` is 0 then this corresponds to `nn.Linear(n_in, n_out)`.
    """
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(*sum([[nn.Linear(n[i], n[i + 1]), act()] for i in range(n_layer + 1)], [])[:-1])


class GraphTransformer(nn.Module):
    """An agnostic GraphTransformer class, and the main model used by other model classes

    This graph model takes in node features, edge features, and graph features (referred to as
    conditional information, since they condition the output). The graph features are projected to
    virtual nodes (one per graph), which are fully connected.

    The per node outputs are the final (post graph-convolution) node embeddings.

    The per graph outputs are the concatenation of a global mean pooling operation, of the final
    node embeddings, and of the final virtual node embeddings.
    """

    def __init__(
        self, x_dim, e_dim, g_dim, num_emb=64, num_layers=3, num_heads=2, num_noise=0, ln_type="pre", concat=True
    ):
        """
        Parameters
        ----------
        x_dim: int
            The number of node features
        e_dim: int
            The number of edge features
        g_dim: int
            The number of graph-level features
        num_emb: int
            The number of hidden dimensions, i.e. embedding size. Default 64.
        num_layers: int
            The number of Transformer layers.
        num_heads: int
            The number of Transformer heads per layer.
        num_noise: int
            The number of noise features to add to the node features.
            This can be used as a simple positional encoding mechanism.
        ln_type: str
            The location of Layer Norm in the transformer, either 'pre' or 'post', default 'pre'.
            (apparently, before is better than after, see https://arxiv.org/pdf/2002.04745.pdf)
        concat: bool
            Whether each head uses num_emb units (True) or num_emb // num_heads (False) units. Defaults to True.
            If True this implies num_emb * num_heads output units within the attention mechanism (which are later
            reprojected to num_emb units).
        """
        super().__init__()
        self.num_layers = num_layers
        self.num_noise = num_noise
        assert ln_type in ["pre", "post"]
        self.ln_type = ln_type

        self.x2h = mlp(x_dim + num_noise, num_emb, num_emb, 2)
        self.e2h = mlp(e_dim, num_emb, num_emb, 2)
        self.c2h = mlp(max(1, g_dim), num_emb, num_emb, 2)
        n_att = num_emb * num_heads if concat else num_emb
        self.graph2emb = nn.ModuleList(
            sum(
                [
                    [
                        gnn.GENConv(num_emb, num_emb, num_layers=1, aggr="add", norm=None),
                        gnn.TransformerConv(num_emb * 2, n_att // num_heads, edge_dim=num_emb, heads=num_heads),
                        nn.Linear(n_att, num_emb),
                        gnn.LayerNorm(num_emb, affine=False),
                        mlp(num_emb, num_emb * 4, num_emb, 1),
                        gnn.LayerNorm(num_emb, affine=False),
                        nn.Linear(num_emb, num_emb * 2),
                    ]
                    for i in range(self.num_layers)
                ],
                [],
            )
        )

    def forward(self, g: gd.Batch, cond: Optional[torch.Tensor]):
        """Forward pass

        Parameters
        ----------
        g: gd.Batch
            A standard torch_geometric Batch object. Expects `edge_attr` to be set.
        cond: torch.Tensor
            The per-graph conditioning information. Shape: (g.num_graphs, self.g_dim).

        Returns
        node_embeddings: torch.Tensor
            Per node embeddings. Shape: (g.num_nodes, self.num_emb).
        graph_embeddings: torch.Tensor
            Per graph embeddings. Shape: (g.num_graphs, self.num_emb * 2).
        """
        if self.num_noise > 0:
            x = torch.cat([g.x, torch.rand(g.x.shape[0], self.num_noise, device=g.x.device)], 1)
        else:
            x = g.x
        o = self.x2h(x)
        e = self.e2h(g.edge_attr)
        c = self.c2h(cond if cond is not None else torch.ones((g.num_graphs, 1), device=g.x.device))
        num_total_nodes = g.x.shape[0]
        # Augment the edges with a new edge to the conditioning
        # information node. This new node is connected to every node
        # within its graph.
        u, v = torch.arange(num_total_nodes, device=o.device), g.batch + num_total_nodes
        aug_edge_index = torch.cat([g.edge_index, torch.stack([u, v]), torch.stack([v, u])], 1)
        e_p = torch.zeros((num_total_nodes * 2, e.shape[1]), device=g.x.device)
        e_p[:, 0] = 1  # Manually create a bias term
        aug_e = torch.cat([e, e_p], 0)
        aug_edge_index, aug_e = add_self_loops(aug_edge_index, aug_e, "mean")
        aug_batch = torch.cat([g.batch, torch.arange(c.shape[0], device=o.device)], 0)

        # Append the conditioning information node embedding to o
        o = torch.cat([o, c], 0)
        for i in range(self.num_layers):
            # Run the graph transformer forward
            gen, trans, linear, norm1, ff, norm2, cscale = self.graph2emb[i * 7 : (i + 1) * 7]
            cs = cscale(c[aug_batch])
            if self.ln_type == "post":
                agg = gen(o, aug_edge_index, aug_e)
                l_h = linear(trans(torch.cat([o, agg], 1), aug_edge_index, aug_e))
                scale, shift = cs[:, : l_h.shape[1]], cs[:, l_h.shape[1] :]
                o = norm1(o + l_h * scale + shift, aug_batch)
                o = norm2(o + ff(o), aug_batch)
            else:
                o_norm = norm1(o, aug_batch)
                agg = gen(o_norm, aug_edge_index, aug_e)
                l_h = linear(trans(torch.cat([o_norm, agg], 1), aug_edge_index, aug_e))
                scale, shift = cs[:, : l_h.shape[1]], cs[:, l_h.shape[1] :]
                o = o + l_h * scale + shift
                o = o + ff(norm2(o, aug_batch))

        o_final = o[: -c.shape[0]]
        glob = torch.cat([gnn.global_mean_pool(o_final, g.batch), o[-c.shape[0] :]], 1)
        return o_final, glob


class GraphTransformerGFN(nn.Module):
    """GraphTransformer class for a GFlowNet which outputs a GraphActionCategorical.

    Outputs logits corresponding to the action types used by the env_ctx argument.
    """

    # The GraphTransformer outputs per-node, per-edge, and per-graph embeddings, this routes the
    # embeddings to the right MLP
    _action_type_to_graph_part = {
        GraphActionType.Stop: "graph",
        GraphActionType.AddNode: "node",
        GraphActionType.SetNodeAttr: "node",
        GraphActionType.AddEdge: "non_edge",
        GraphActionType.SetEdgeAttr: "edge",
        GraphActionType.RemoveNode: "node",
        GraphActionType.RemoveNodeAttr: "node",
        GraphActionType.RemoveEdge: "edge",
        GraphActionType.RemoveEdgeAttr: "edge",
    }
    # The torch_geometric batch key each graph part corresponds to
    _graph_part_to_key = {
        "graph": None,
        "node": "x",
        "non_edge": "non_edge_index",
        "edge": "edge_index",
    }

    action_type_to_key = lambda action_type: GraphTransformerGFN._graph_part_to_key.get(  # noqa: E731
        GraphTransformerGFN._action_type_to_graph_part.get(action_type)
    )

    def __init__(
        self,
        env_ctx,
        cfg: Config,
        num_graph_out=1,
        do_bck=False,
    ):
        """See `GraphTransformer` for argument values"""
        super().__init__()
        self.transf = GraphTransformer(
            x_dim=env_ctx.num_node_dim,
            e_dim=env_ctx.num_edge_dim,
            g_dim=env_ctx.num_cond_dim,
            num_emb=cfg.model.num_emb,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.graph_transformer.num_heads,
            ln_type=cfg.model.graph_transformer.ln_type,
            concat=cfg.model.graph_transformer.concat_heads,
        )
        self.env_ctx = env_ctx
        num_emb = cfg.model.num_emb
        num_final = num_emb
        num_glob_final = num_emb * 2
        num_edge_feat = num_emb if env_ctx.edges_are_unordered else num_emb * 2
        self.edges_are_duplicated = env_ctx.edges_are_duplicated
        self.edges_are_unordered = env_ctx.edges_are_unordered
        self.action_type_order = env_ctx.action_type_order

        # Every action type gets its own MLP that is fed the output of the GraphTransformer.
        # Here we define the number of inputs and outputs of each of those (potential) MLPs.
        self._action_type_to_num_inputs_outputs = {
            GraphActionType.Stop: (num_glob_final, 1),
            GraphActionType.AddNode: (num_final, env_ctx.num_new_node_values),
            GraphActionType.SetNodeAttr: (num_final, env_ctx.num_node_attr_logits),
            GraphActionType.AddEdge: (num_edge_feat, 1),
            GraphActionType.SetEdgeAttr: (num_edge_feat, env_ctx.num_edge_attr_logits),
            GraphActionType.RemoveNode: (num_final, 1),
            GraphActionType.RemoveNodeAttr: (num_final, env_ctx.num_node_attrs - 1),
            GraphActionType.RemoveEdge: (num_edge_feat, 1),
            GraphActionType.RemoveEdgeAttr: (num_edge_feat, env_ctx.num_edge_attrs),
        }
        self._action_type_to_key = {
            at: self._graph_part_to_key[self._action_type_to_graph_part[at]] for at in self._action_type_to_graph_part
        }

        # Here we create only the embedding -> logit mapping MLPs that are required by the environment
        mlps = {}
        for atype in chain(env_ctx.action_type_order, env_ctx.bck_action_type_order if do_bck else []):
            num_in, num_out = self._action_type_to_num_inputs_outputs[atype]
            mlps[atype.cname] = mlp(num_in, num_emb, num_out, cfg.model.graph_transformer.num_mlp_layers)
        self.mlps = nn.ModuleDict(mlps)

        self.do_bck = do_bck
        if do_bck:
            self.bck_action_type_order = env_ctx.bck_action_type_order

        self.emb2graph_out = mlp(num_glob_final, num_emb, num_graph_out, cfg.model.graph_transformer.num_mlp_layers)
        # TODO: flag for this
        self._logZ = mlp(max(1, env_ctx.num_cond_dim), num_emb * 2, 1, 2)

    def logZ(self, cond_info: Optional[torch.Tensor]):
        if cond_info is None:
            return self._logZ(torch.ones((1, 1), device=self._logZ[0].weight.device))
        return self._logZ(cond_info)

    def _make_cat(self, g: gd.Batch, emb: Dict[str, Tensor], action_types: list[GraphActionType]):
        return GraphActionCategorical(
            g,
            raw_logits=[self.mlps[t.cname](emb[self._action_type_to_graph_part[t]]) for t in action_types],
            keys=[self._action_type_to_key[t] for t in action_types],
            action_masks=[action_type_to_mask(t, g) for t in action_types],
            types=action_types,
        )

    def forward(self, g: gd.Batch, cond: Optional[torch.Tensor]):
        node_embeddings, graph_embeddings = self.transf(g, cond)
        # "Non-edges" are edges not currently in the graph that we could add
        if hasattr(g, "non_edge_index"):
            ne_row, ne_col = g.non_edge_index
            if self.edges_are_unordered:
                non_edge_embeddings = node_embeddings[ne_row] + node_embeddings[ne_col]
            else:
                non_edge_embeddings = torch.cat([node_embeddings[ne_row], node_embeddings[ne_col]], 1)
        else:
            # If the environment context isn't setting non_edge_index, we can safely assume that
            # action is not in ctx.action_type_order.
            non_edge_embeddings = None
        if self.edges_are_duplicated:
            # On `::2`, edges are typically duplicated to make graphs undirected, only take the even ones
            e_row, e_col = g.edge_index[:, ::2]
        else:
            e_row, e_col = g.edge_index
        if self.edges_are_unordered:
            edge_embeddings = node_embeddings[e_row] + node_embeddings[e_col]
        else:
            edge_embeddings = torch.cat([node_embeddings[e_row], node_embeddings[e_col]], 1)

        emb = {
            "graph": graph_embeddings,
            "node": node_embeddings,
            "edge": edge_embeddings,
            "non_edge": non_edge_embeddings,
        }

        graph_out = self.emb2graph_out(graph_embeddings)
        fwd_cat = self._make_cat(g, emb, self.action_type_order)
        if self.do_bck:
            bck_cat = self._make_cat(g, emb, self.bck_action_type_order)
            return fwd_cat, bck_cat, graph_out
        return fwd_cat, graph_out
