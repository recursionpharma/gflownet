import os
import signal
import tarfile
import time
import pandas as pd
import numpy as np
from typing import Tuple, List, Any, Dict

import rdkit.Chem as Chem
from rdkit import RDLogger

from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch_geometric.data as gd
import torch_geometric.nn as gnn
from torch.utils.data import Dataset, IterableDataset

from gflownet.envs.graph_building_env import GraphBuildingEnv, GraphActionType, GraphActionCategorical
from gflownet.envs.graph_building_env import generate_forward_trajectory
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.algo.trajectory_balance import TrajectoryBalance


class QM9Dataset(Dataset):
    def __init__(self, h5_file=None, xyz_file=None, train=True, split_seed=142857, ratio=0.9):
        if h5_file is not None:
            self.df = pd.HDFStore(h5_file, 'r')['df']
        elif xyz_file is not None:
            self.load_tar()
        rng = np.random.default_rng(split_seed)
        idcs = np.arange(len(self.df))
        rng.shuffle(idcs)
        self._min = self.df['gap'].min()
        self._max = self.df['gap'].max()
        self._gap = self._max - self._min
        #self._rtrans = 'exp'
        self._rtrans = 'unit'
        if train:
            self.idcs = idcs[:int(np.floor(ratio * len(self.df)))]
        else:
            self.idcs = idcs[int(np.floor(ratio * len(self.df))):]
            
    def load_tar(self, xyz_file):
        f = tarfile.TarFile(xyz_file, 'r')
        labels = ['rA', 'rB', 'rC', 'mu', 'alpha', 'homo', 'lumo',
                  'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        all_mols = []
        for pt in f:
            pt = f.extractfile(pt)
            data = pt.read().decode().splitlines()
            all_mols.append(data[-2].split()[:1] + list(map(float, data[1].split()[2:])))
        self.df = pd.DataFrame(all_mols, columns=['SMILES']+labels)

    def reward_transform(self, r):
        if self._rtrans == 'exp':
            return np.exp(-(r - self._min) / self._gap)
        elif self._rtrans == 'unit':
            return 1 - (r - self._min) / (self._gap + 1e-4)

    def __len__(self):
        return len(self.idcs)

    def __getitem__(self, idx):
        return (self.df['SMILES'][self.idcs[idx]], self.reward_transform(self.df['gap'][self.idcs[idx]]))

    
class QM9SamplingIterator(IterableDataset):
    def __init__(self, qm9_dataset, env, ctx, model, batch_size, algo, ratio=0.5, stream=True):
        self._data = qm9_dataset
        self.model = model
        self.model.device = next(model.parameters()).device
        self.offline_batch_size = int(np.ceil(batch_size * ratio))
        self.online_batch_size = int(np.floor(batch_size * (1 - ratio)))
        self.ratio = ratio
        self.env = env
        self.ctx = ctx
        self.algo = algo
        self.stream = stream

    def idx_iterator(self):
        RDLogger.DisableLog('rdApp.*')
        if self.stream:
            while True:
                yield self.rng.integers(0, len(self._data.idcs), self.offline_batch_size)
        else:
            worker_info = torch.utils.data.get_worker_info()
            n = len(self._data.idcs)
            if worker_info is None:
                start, end = 0, n
                wid = -1
            else:
                nw = worker_info.num_workers
                wid = worker_info.id
                start, end = int(np.floor(n / nw * wid)), int(np.ceil(n / nw * (wid+1)))
            bs = self.offline_batch_size
            if end - start < bs:
                yield np.arange(start, end)
                return
            for i in range(start, end - bs, bs):
                yield np.arange(i, i + bs)
            if i + bs < end:
                yield np.arange(i + bs, end)

    def __len__(self):
        if self.stream:
            return int(1e6)
        return len(self._data.idcs)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid = (worker_info.id if worker_info is not None else 0)
        self.rng = np.random.default_rng(142857 + wid)
        self.algo.rng = self.rng
        for idcs in self.idx_iterator():
            # Sample some dataset data
            smiles, rewards = zip(*[self._data[i] for i in idcs])
            _r = rewards
            graphs = [self.ctx.mol_to_graph(Chem.MolFromSmiles(s)) for s in smiles]
            trajs = [generate_forward_trajectory(i) for i in graphs]
            temps = self.algo.sample_temperatures(len(smiles))[:, None]
            cond_info = list(temps)
            rewards = [r ** t for r, t in zip(rewards, temps)]
            # Sample some on-policy data
            online_trajs = [[[], None, None] for i in range(self.online_batch_size)]
            if self.online_batch_size > 0:
                with torch.no_grad():
                    self.algo.sample_model_losses(self.env, self.ctx,
                                                  self.model,
                                                  self.online_batch_size,
                                                  cond_info='sample',
                                                  trajectories=online_trajs)
                    trajs += [i[0] for i in online_trajs]
                    rewards += [i[1].cpu() for i in online_trajs]
                    cond_info += [i[2].cpu() for i in online_trajs]
                    for i in []:# online_trajs:
                        print(i[1].item(), i[2].item())
                        for t in i[0]:
                            print(' ',t)

            # Construct batch
            # TODO: is this TB specific logic?
            torch_graphs = [self.ctx.graph_to_Data(i[0]) for tj in trajs for i in tj]
            actions = [self.ctx.GraphAction_to_aidx(g, a, self.model.action_type_order)
                       for g, a in zip(torch_graphs, [i[1] for tj in trajs for i in tj])]
            num_backward = torch.tensor([
                self.env.count_backward_transitions(tj[i + 1][0]) if i + 1 < len(tj) else 1
                #if tj[i][1].action is not GraphActionType.Stop# and len(tj[i][0].nodes) < self.algo.max_nodes
                #else 1
                for tj in trajs for i in range(len(tj))
            ])
            batch = self.ctx.collate(torch_graphs)
            batch.traj_lens = torch.tensor([len(i) for i in trajs])
            batch.num_backward = num_backward
            batch.actions = torch.tensor(actions)
            batch.smiles = smiles
            batch.flat_rewards = torch.tensor(_r).float()
            batch.pin_memory()
            yield batch, torch.tensor(rewards).float(), torch.tensor(cond_info).float() / 8

def mlp(n_in, n_hid, n_out, n_layer):
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(*sum([[nn.Linear(n[i], n[i+1]), nn.ReLU()] for i in range(n_layer + 1)], [])[:-1])
    
class Model(nn.Module):
    def __init__(self, env_ctx, num_emb=64):
        super().__init__()
        self.x2h = mlp(env_ctx.num_node_dim, num_emb, num_emb, 2)
        self.e2h = mlp(env_ctx.num_edge_dim, num_emb, num_emb, 2)
        self.c2h = mlp(env_ctx.num_cond_dim, num_emb, num_emb, 2)
        num_heads = 4
        self.num_layers = 6
        self.graph2emb = nn.ModuleList(
            sum([[
                gnn.GENConv(num_emb, num_emb, num_layers=3, aggr='add'),
                gnn.TransformerConv(num_emb, num_emb, edge_dim=num_emb, heads=num_heads),
                nn.Linear(num_heads * num_emb, num_emb),
                gnn.LayerNorm(num_emb),
                mlp(num_emb, num_emb * 4, num_emb, 1),
                gnn.LayerNorm(num_emb),
            ] for i in range(self.num_layers)], []))
        self.emb2add_edge = mlp(num_emb, num_emb, 1, 2)
        self.emb2add_node = mlp(num_emb, num_emb, env_ctx.num_new_node_values, 2)
        self.emb2set_node_attr = mlp(num_emb, num_emb, env_ctx.num_node_attr_logits, 2)
        self.emb2set_edge_attr = mlp(num_emb, num_emb, env_ctx.num_edge_attr_logits, 2)
        self.emb2stop = mlp(num_emb * 3, num_emb, 1, 2)
        self.emb2reward = mlp(num_emb * 3, num_emb, 1, 2)
        self.logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 3)
        self.action_type_order = [
            GraphActionType.Stop,
            GraphActionType.AddNode,
            GraphActionType.SetNodeAttr,
            GraphActionType.AddEdge,
            GraphActionType.SetEdgeAttr
        ]


    def forward(self, g: gd.Batch, cond: torch.tensor):
        o = self.x2h(g.x)
        e = self.e2h(g.edge_attr)
        c = self.c2h(cond)
        num_total_nodes = g.x.shape[0]
        # Augment the edges with a new edge to the conditioning
        # information node. This new node is connected to every node
        # within its graph.
        u, v = torch.arange(num_total_nodes, device=o.device), g.batch + num_total_nodes
        aug_edge_index = torch.cat(
            [g.edge_index,
             torch.stack([u, v]),
             torch.stack([v, u])],
            1)
        e_p = torch.zeros((num_total_nodes * 2, e.shape[1]), device=g.x.device)
        e_p[:, 0] = 1 # Manually create a bias term
        aug_e = torch.cat([e, e_p], 0)
        aug_batch = torch.cat([g.batch, torch.arange(c.shape[0], device=o.device)], 0)
            
        # Cat the node embedding to o
        o_0 = o = torch.cat([o, c], 0)
        for i in range(self.num_layers):
            gen, trans, linear, norm1, ff, norm2 = self.graph2emb[i * 6: (i+1) * 6]
            o = norm1(o + linear(trans(gen(o, aug_edge_index, aug_e), aug_edge_index, aug_e)))
            o = norm2(o + ff(o))
            
        glob = torch.cat([gnn.global_mean_pool(o[:-c.shape[0]], g.batch), o[-c.shape[0]:], c], 1)
        o = o[:-c.shape[0]]
        
        ne_row, ne_col = g.non_edge_index
        # On `::2`, edges are duplicated to make graphs undirected, only take the even ones
        e_row, e_col = g.edge_index[:, ::2]
        cat = GraphActionCategorical(
            g,
            logits=[
                self.emb2stop(glob),
                self.emb2add_node(o),
                self.emb2set_node_attr(o),
                self.emb2add_edge(o[ne_row] + o[ne_col]),
                self.emb2set_edge_attr(o[e_row] + o[e_col]),
            ],
            keys=[None, 'x', 'x', 'non_edge_index', 'edge_index'],
            types=self.action_type_order,
        )
        return cat, self.emb2reward(glob)


class QM9Trial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        
        self.num_workers = context.get_hparam('num_data_loader_workers')
        if self.num_workers > 0:
            mp.set_start_method('spawn')
        RDLogger.DisableLog('rdApp.*')
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        self.ctx = MolBuildingEnvContext(['H', 'C', 'N', 'F', 'O'], num_cond_dim=1)
        print(context.n_gpus, context.distributed.size)
        
        model = Model(self.ctx, num_emb=context.get_hparam('num_emb'))
        self.model = context.wrap_model(model)
        Z_params = list(model.logZ.parameters())
        non_Z_params = [i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)]
        print(len(non_Z_params), len(list(self.model.parameters())))
        self.opt = context.wrap_optimizer(
            torch.optim.Adam(non_Z_params, context.get_hparam('learning_rate'))#, (0.95, 0.999))
        )
        self.opt_Z = context.wrap_optimizer(
            torch.optim.SGD(Z_params, context.get_hparam('learning_rate') * 0.1))
        self.tb = TrajectoryBalance(self.env, self.ctx, self.rng, random_action_prob=0.01, max_nodes=9,
                                    epsilon=context.get_hparam('tb_epsilon'))
        self.tb.reward_loss_multiplier = context.get_hparam('reward_loss_multiplier')
        self.tb.temperature_sample_dist = context.get_hparam('temperature_sample_dist')
        self.tb.temperature_dist_params = eval(context.get_hparam('temperature_dist_params'))
        self.mb_size = self.context.get_per_slot_batch_size()
        # See https://docs.determined.ai/latest/training-apis/api-pytorch-advanced.html#customizing-a-reproducible-dataset
        if isinstance(context, PyTorchTrialContext):
            context.experimental.disable_dataset_reproducibility_checks()

    def _sample_temperatures(self, n):
        if self.temperature_sample_dist == 'gamma':
            return self.rng.gamma(*self.temperature_dist_params, n).astype(np.float32)
        elif self.temperature_sample_dist == 'uniform':
            return self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
        raise ValueError(self.temperature_sample_dist)

    def build_training_data_loader(self) -> DataLoader:
        data = QM9Dataset(self.context.get_data_config()['h5_path'], train=True)
        iterator = QM9SamplingIterator(data, self.env, self.ctx,
                                       self.model, self.mb_size * 2,
                                       self.tb)
        return torch.utils.data.DataLoader(iterator, batch_size=None, num_workers=self.num_workers, persistent_workers=self.num_workers > 0)
    
    def build_validation_data_loader(self) -> DataLoader:
        data = QM9Dataset(self.context.get_data_config()['h5_path'], train=False)
        iterator = QM9SamplingIterator(data, self.env, self.ctx,
                                       self.model, self.mb_size * 2,
                                       self.tb, ratio=1, stream=False)
        return torch.utils.data.DataLoader(iterator, batch_size=None, num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def train_batch(self, batch: Tuple[List[str], torch.Tensor], epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        if not hasattr(self.model, 'device'):
            self.model.device = self.context.to_device(torch.ones(1)).device
        if 1:
            t = [time.time()]
            batch, rewards, cond_info = batch
            losses, info = self.tb.compute_batch_losses(
                self.model, batch, rewards.squeeze(), cond_info, num_bootstrap=self.mb_size)
            t += [time.time()]
            avg_offline_loss = losses[:self.mb_size].mean()
            avg_online_loss = losses[self.mb_size:].mean()
            reward_losses = info['reward_losses'].mean()
            loss = losses.mean() + reward_losses * self.tb.reward_loss_multiplier
            self.context.backward(loss)
            t += [time.time()]
            self.context.step_optimizer(
                self.opt,
                clip_grads=lambda params: torch.nn.utils.clip_grad_value_(params, 10))
            self.context.step_optimizer(self.opt_Z)
            t += [time.time()]
            #print('train_batch', ' '.join(f"{t[i+1]-t[i]:.3f}" for i in range(len(t)-1)))
            return {'loss': loss.item(),
                    'avg_online_loss': avg_online_loss.item(),
                    'avg_offline_loss': avg_offline_loss.item(),
                    'reward_loss': reward_losses.item(),
                    'invalid_trajectories': info['invalid_trajectories'].item(),
                    'unnorm_traj_losses': info['unnorm_traj_losses'].mean().item()}
        else:
            batch, rewards, cond_info = batch
            smiles = batch.smiles
            flat_rewards = batch.rewards
            
        
        #smiles, flat_rewards = batch
        mb_size = len(smiles)
        t = [time.time()]
        graphs = [self.ctx.mol_to_graph(Chem.MolFromSmiles(s)) for s in smiles]
        temp = self.tb.sample_temperatures(mb_size * 2)
        cond_info = self.context.to_device(torch.tensor(temp).reshape((-1, 1)))
        rewards = flat_rewards ** cond_info[:mb_size, 0]
        t += [time.time()]
        offline_losses, off_info = self.tb.compute_data_losses(
            self.env, self.ctx, self.model, graphs, rewards, cond_info=cond_info[:mb_size])
        t += [time.time()]
        online_losses = self.tb.sample_model_losses(
            self.env, self.ctx, self.model, mb_size, cond_info=cond_info[mb_size:])
        t += [time.time()]
        avg_online_loss = online_losses.mean()
        avg_offline_loss = offline_losses.mean()
        loss = (avg_offline_loss + avg_online_loss) / 2
        
        self.context.backward(loss)
        self.context.step_optimizer(
            self.opt,
            clip_grads=lambda params: torch.nn.utils.clip_grad_value_(params, 1))
        t += [time.time()]
        #print('train_batch', ' '.join(f"{t[i+1]-t[i]:.3f}" for i in range(len(t)-1)))
        return {'loss': loss,
                'avg_online_loss': avg_online_loss,
                'avg_offline_loss': avg_offline_loss,
                'reward_loss': off_info['reward_losses'].mean().item(),
                'unnorm_traj_losses': off_info['unnorm_traj_losses'].mean().item()}

    def evaluate_batch(self, batch: Tuple[List[str], torch.Tensor]) -> Dict[str, Any]:
        if not hasattr(self.model, 'device'):
            self.model.device = self.context.to_device(torch.ones(1)).device
        if 1:
            batch, rewards, cond_info = batch
            losses, info = self.tb.compute_batch_losses(
                self.model, batch, rewards.squeeze(), cond_info, num_bootstrap=len(batch.smiles))#self.mb_size * 2)
            loss = losses.mean()
            reward_losses = info['reward_losses'].mean()
            return {'validation_loss': loss,
                    'reward_loss': reward_losses.item(),
                    'unnorm_traj_losses': info['unnorm_traj_losses'].mean().item()}
        smiles, flat_rewards = batch
        mb_size = len(smiles)
        graphs = [self.ctx.mol_to_graph(Chem.MolFromSmiles(s)) for s in smiles]
        temp = self.tb.sample_temperatures(mb_size)
        cond_info = self.context.to_device(torch.tensor(temp).reshape((-1, 1)))
        rewards = flat_rewards ** cond_info[:, 0]
        losses, info = self.tb.compute_data_losses(self.env, self.ctx,
                                             self.model, graphs, rewards,
                                             cond_info=cond_info)
        return {'validation_loss': losses.mean().item(),
                'reward_loss': info['reward_losses'].mean().item(),
                'unnorm_traj_losses': info['unnorm_traj_losses'].mean().item()}

class DummyContext:

    def __init__(self, hps, device):
        self.hps = hps
        self.dev = device
    
    def wrap_model(self, model):
        self.model = model
        return model.to(self.dev)

    def wrap_optimizer(self, opt):
        return opt

    def get_hparam(self, hp):
        return self.hps[hp]

    def get_data_config(self):
        return {'h5_path': '/data/chem/qm9/qm9.h5'}

    def get_per_slot_batch_size(self):
        return self.hps['global_batch_size']

    def to_device(self, x):
        return x.to(self.dev)

    def backward(self, loss):
        loss.backward()

    def step_optimizer(self, opt, clip_grads=None):
        if clip_grads is not None:
            [clip_grads(i) for i in self.model.parameters()]
        opt.step()
        opt.zero_grad()
    
def main():
    hps = {
        'learning_rate': 2e-4,
        'global_batch_size': 128,
        'num_emb': 64,
        'tb_epsilon': -60,
        'reward_loss_multiplier': 1,
        'temperature_sample_dist': 'gamma',
        'temperature_dist_params': '(1.5, 1.5)',
        'num_data_loader_workers': 6,
    }
    dummy_context = DummyContext(hps, torch.device('cuda'))
    trial = QM9Trial(dummy_context)

    train_dl = trial.build_training_data_loader()
    valid_dl = trial.build_validation_data_loader()

    for epoch in range(10):
        t0 = time.time()
        for it, batch in enumerate(train_dl):
            t1 = time.time()
            print('load', t1-t0)
            batch = [i.to(dummy_context.dev, non_blocking=True) if hasattr(i, 'to') else i for i in batch]
            t2 = time.time()
            print('transfer', t2-t1)
            r = trial.train_batch(batch, epoch, it)
            print(it, ' '.join(f"{k}: {v:.4f}" for k, v in r.items()))
            #trial.evaluate_batch(batch)
            t0 = t3 = time.time()
            print('train', t3-t2)
            if not it % 200:
                torch.save({'models_state_dict': [trial.model.state_dict()]}, open('temp_2.pt', 'wb'))
        break

if __name__ == '__main__':
    main()
