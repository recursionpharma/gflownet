"""
This test shows an example of how to setup a model and environment.
It trains a model to overfit generating one single molecule.
"""

import torch
import torch.nn as nn
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from tqdm import tqdm

from gflownet.envs.graph_building_env import (
    GraphActionCategorical,
    GraphActionType,
    GraphBuildingEnv,
    generate_forward_trajectory,
)
from gflownet.envs.mol_building_env import MolBuildingEnvContext


class Model(nn.Module):
    def __init__(self, env_ctx, num_emb=64):
        super().__init__()
        self.x2h = nn.Linear(env_ctx.num_node_dim, num_emb)
        self.e2h = nn.Linear(env_ctx.num_edge_dim, num_emb)
        self.graph2emb = nn.ModuleList(
            sum(
                [
                    [
                        gnn.GENConv(num_emb, num_emb, num_layers=1, aggr="add"),
                        gnn.TransformerConv(num_emb, num_emb, edge_dim=num_emb),
                    ]
                    for i in range(6)
                ],
                [],
            )
        )

        def h2l(nl):
            return nn.Sequential(nn.Linear(num_emb, num_emb), nn.LeakyReLU(), nn.Linear(num_emb, nl))

        self.emb2add_edge = h2l(1)
        self.emb2add_node = h2l(env_ctx.num_new_node_values)
        self.emb2add_node_attr = h2l(env_ctx.num_node_attr_logits)
        self.emb2add_edge_attr = h2l(env_ctx.num_edge_attr_logits)
        self.emb2stop = h2l(1)
        self.action_type_order = [
            GraphActionType.Stop,
            GraphActionType.AddNode,
            GraphActionType.SetNodeAttr,
            GraphActionType.AddEdge,
            GraphActionType.SetEdgeAttr,
        ]

    def forward(self, g: gd.Batch):
        o = self.x2h(g.x)
        e = self.e2h(g.edge_attr)
        for layer in self.graph2emb:
            o = o + layer(o, g.edge_index, e)
        glob = gnn.global_mean_pool(o, g.batch)
        ne_row, ne_col = g.non_edge_index
        # On `::2`, edges are duplicated to make graphs undirected, only take the even ones
        e_row, e_col = g.edge_index[:, ::2]
        cat = GraphActionCategorical(
            g,
            raw_logits=[
                self.emb2stop(glob),
                self.emb2add_node(o),
                self.emb2add_node_attr(o),
                self.emb2add_edge(o[ne_row] + o[ne_col]),
                self.emb2add_edge_attr(o[e_row] + o[e_col]),
            ],
            keys=[None, "x", "x", "non_edge_index", "edge_index"],
            types=self.action_type_order,
        )
        return cat


def main(smi, n_steps):
    """This trains a model to overfit producing a molecule, runs a
    generative episode and tests whether the model has successfully
    generated that molecule

    """
    import networkx as nx
    import numpy as np
    from rdkit import Chem

    np.random.seed(123)
    env = GraphBuildingEnv()
    ctx = MolBuildingEnvContext()
    model = Model(ctx, num_emb=64)
    opt = torch.optim.Adam(model.parameters(), 5e-4)
    mol = Chem.MolFromSmiles(smi)
    molg = ctx.obj_to_graph(mol)
    traj = generate_forward_trajectory(molg)
    for g, a in traj:
        print(a.action, a.source, a.target, a.value)
    graphs = [ctx.graph_to_Data(i) for i, _ in traj]
    traj_batch = ctx.collate(graphs)
    actions = [ctx.GraphAction_to_ActionIndex(g, a) for g, a in zip(graphs, [i[1] for i in traj])]

    # Train to overfit
    for i in tqdm(range(n_steps)):
        fwd_cat = model(traj_batch)
        logprob = fwd_cat.log_prob(actions)
        loss = -logprob.mean()
        if not i % 100:
            print(fwd_cat.logits)
            print(logprob.exp())
            print(loss)
        loss.backward()
        opt.step()
        opt.zero_grad()
    print()
    # Generation episode
    model.eval()
    g = env.new()
    for t in range(100):
        tg = ctx.graph_to_Data(g)
        with torch.no_grad():
            fwd_cat = model(ctx.collate([tg]))
        fwd_cat.logsoftmax()
        print("stop:", fwd_cat.logprobs[0].exp())
        action = fwd_cat.sample()[0]
        print("action prob:", fwd_cat.log_prob([action]).exp())
        if fwd_cat.log_prob([action]).exp().item() < 0.2:
            # This test should work but obviously it's not perfect,
            # some probability is left on unlikely (wrong) steps
            print("oops, starting step over")
            continue
        graph_action = ctx.ActionIndex_to_GraphAction(tg, action)
        print(graph_action.action, graph_action.source, graph_action.target, graph_action.value)
        if graph_action.action is GraphActionType.Stop:
            break
        g = env.step(g, graph_action)
        # Make sure the subgraph is isomorphic to the target molecule
        issub = nx.algorithms.isomorphism.GraphMatcher(molg, g).subgraph_is_monomorphic()
        print(issub)
        if not issub:
            raise ValueError()
        print(g)
    new_mol = ctx.graph_to_obj(g)
    print(Chem.MolToSmiles(new_mol))
    # This should be True as well
    print(new_mol.HasSubstructMatch(mol) and mol.HasSubstructMatch(new_mol))


if __name__ == "__main__":
    # Simple mol
    main("C1N2C3C2C2C4OC12C34", 500)
    # More complicated mol
    # main("O=C(NC1=CC=2NC(=NC2C=C1)C=3C=CC=CC3)C4=NN(C=C4N(=O)=O)C", 2000)
