import tarfile
import pandas as pd
import numpy as np

import rdkit.Chem as Chem

import torch
import torch.nn as nn
import torch_geometric.data as gd
import torch_geometric.nn as gnn

from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphActionType, GraphActionCategorical
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.algo.trajectory_balance import TrajectoryBalance


def load_qm9_data():
    # TODO: point to bh, or config
    print('Loading QM9')
    f = tarfile.TarFile('/mnt/ps/home/CORP/emmanuel.bengio/data/qm9/qm9.xyz.tar', 'r')
    labels = ['rA', 'rB', 'rC', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    all_mols = []
    for pt in f:
        pt = f.extractfile(pt)
        data = pt.read().decode().splitlines()
        all_mols.append(data[-2].split()[:1] + list(map(float, data[1].split()[2:])))
        if len(all_mols) > 1000:
            break
    df = pd.DataFrame(all_mols, columns=['SMILES'] + labels)
    print('Done.')
    return df


class Model(nn.Module):
    def __init__(self, env_ctx, num_emb=64, initial_Z_guess=3):
        super().__init__()
        self.x2h = nn.Linear(env_ctx.num_node_dim, num_emb)
        self.e2h = nn.Linear(env_ctx.num_edge_dim, num_emb)
        self.graph2emb = nn.ModuleList(
            sum([[
                gnn.TransformerConv(num_emb, num_emb, edge_dim=num_emb),
                gnn.GENConv(num_emb, num_emb, num_layers=1, aggr='add'),
            ] for i in range(6)], []))

        def h2l(nl):
            return nn.Sequential(nn.Linear(num_emb, num_emb), nn.LeakyReLU(), nn.Linear(num_emb, nl))

        self.emb2add_edge = h2l(1)
        self.emb2add_node = h2l(env_ctx.num_new_node_values)
        #self.emb2add_node_attr = h2l(env_ctx.num_node_attr_logits)
        self.emb2add_edge_attr = h2l(env_ctx.num_edge_attr_logits)
        self.emb2stop = h2l(1)
        self.emb2reward = h2l(1)
        self.logZ = nn.Parameter(torch.tensor([initial_Z_guess], dtype=torch.float))
        self.action_type_order = [
            GraphActionType.Stop,
            GraphActionType.AddNode,
            # GraphActionType.SetNodeAttr,
            GraphActionType.AddEdge,
            GraphActionType.SetEdgeAttr
        ]

    def forward(self, g: gd.Batch):
        o = self.x2h(g.x)
        e = self.e2h(g.edge_attr)
        for layer in self.graph2emb:
            o = layer(o, g.edge_index, e)
        glob = gnn.global_mean_pool(o, g.batch)
        ne_row, ne_col = g.non_edge_index
        # On `::2`, edges are duplicated to make graphs undirected, only take the even ones
        e_row, e_col = g.edge_index[:, ::2]
        cat = GraphActionCategorical(
            g,
            logits=[
                self.emb2stop(glob),
                self.emb2add_node(o),
                # self.emb2add_node_attr(o),
                self.emb2add_edge(o[ne_row] + o[ne_col]),
                self.emb2add_edge_attr(o[e_row] + o[e_col]),
            ],
            keys=[None, 'x', 'non_edge_index', 'edge_index'],
            types=self.action_type_order,
        )
        return cat, self.emb2reward(glob)


def main():
    rng = np.random.default_rng(142857)
    env = GraphBuildingEnv()
    ctx = MolBuildingEnvContext(['H', 'C', 'N', 'F', 'O'])
    dev = torch.device('cpu')
    model = Model(ctx, num_emb=8, initial_Z_guess=8)
    model.to(dev)
    model.device = dev
    mb_size = 2
    # Was Adam failing? Why?
    opt = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
    #opt = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9, dampening=0.9)

    tb = TrajectoryBalance(rng, random_action_prob=0.01, max_nodes=9)
    df = load_qm9_data()
    train_idcs = rng.choice(len(df), size=int(len(df) * 0.9), replace=False)

    for i in range(1000):
        idcs = rng.choice(train_idcs, mb_size)
        graphs = [ctx.mol_to_graph(Chem.MolFromSmiles(df['SMILES'][i])) for i in idcs]
        rewards = torch.tensor([1 / df['gap'][i] for i in idcs], device=model.device)
        online_losses = tb.sample_model_losses(env, ctx, model, mb_size)
        offline_losses = tb.compute_data_losses(env, ctx, model, graphs, rewards)
        loss = (online_losses.mean() + offline_losses.mean()) / 2
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(loss.item())


if __name__ == '__main__':
    main()
