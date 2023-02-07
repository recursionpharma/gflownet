import ast
import gzip
import pickle
from typing import Any, Callable, Dict, List, Tuple

import networkx as nx
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch_geometric.data as gd
from networkx.algorithms.isomorphism import is_isomorphic
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter_add
from tqdm import tqdm

from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.cliques_env import CliquesEnvContext
from gflownet.envs.graph_building_env import GraphBuildingEnv, Graph
from gflownet.models.graph_transformer import GraphTransformerGFN, GraphTransformer
from gflownet.train import FlatRewards, GFNTask, GFNTrainer, RewardScalar
from gflownet.utils.transforms import thermometer


def n_clique_reward(g, n=4):
    cliques = list(nx.algorithms.clique.find_cliques(g))
    # The number of cliques each node belongs to
    num_cliques = np.bincount(sum(cliques, []))
    cliques_match = [len(i) == n for i in cliques]
    return np.mean(cliques_match) - np.mean(num_cliques)


def colored_n_clique_reward(g, n=4):
    cliques = list(nx.algorithms.clique.find_cliques(g))
    # The number of cliques each node belongs to
    num_cliques = np.bincount(sum(cliques, []))
    colors = {i: g.nodes[i]['v'] for i in g.nodes}
    color_match = lambda c: np.bincount([colors[i] for i in c]).max() >= n - 1
    cliques_match = [float(len(i) == n) * (1 if color_match(i) else 0.5) for i in cliques]
    return np.maximum(np.sum(cliques_match) - np.sum(num_cliques) + len(g) - 1, -10)


def load_clique_data():
    data = pickle.load(gzip.open('/scratch/emmanuel.bengio/data/cliques/two_col_7_graphs.pkl.gz', 'rb'))
    return data


class GraphTransformerRegressor(GraphTransformer):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.g2o = torch.nn.Linear(kw['num_emb'] * 2, 1)

    def forward(self, g: gd.Batch, cond: torch.Tensor):
        per_node_pred, per_graph_pred = super().forward(g, cond)
        return self.g2o(per_graph_pred)[:, 0]


class CliqueDataset(Dataset):
    def __init__(self, data, ctx, n_clique, train=True, output_graphs=False, split_seed=142857, ratio=0.9,
                 modal_seed=None):
        idcs = np.arange(len(data))
        if modal_seed is None:
            rng = np.random.default_rng(split_seed)
            rng.shuffle(idcs)
            if train:
                self.idcs = idcs[:int(np.floor(ratio * len(data)))]
            else:
                self.idcs = idcs[int(np.floor(ratio * len(data))):]
        elif 1:
            _, mode_labels = pickle.load(
                #open('/scratch/emmanuel.bengio/data/cliques/two_col_7_mode_label_assignments.pkl', 'rb'))
                open('/scratch/emmanuel.bengio/data/cliques/two_col_7_mode_label_assignments_xnoise2.pkl', 'rb'))
            rng = np.random.default_rng(split_seed + modal_seed)
            is_mode_present = rng.uniform(size=mode_labels.max() + 1) < ratio
            if train:
                self.idcs = idcs[is_mode_present[mode_labels]]
            else:
                self.idcs = idcs[~is_mode_present[mode_labels]]
        else:
            mode_labels, *_ = pickle.load(open('/scratch/emmanuel.bengio/data/cliques/two_col_7_mode_50.pkl', 'rb'))
            rng = np.random.default_rng(split_seed + modal_seed)
            is_mode_present = np.bool_(np.zeros(mode_labels.max() +
                                                1))  #rng.uniform(size=mode_labels.max() + 1) < ratio
            is_mode_present[0] = True  # Mode 0 is always included (it's the "idk" class of DBSCAN)
            while is_mode_present[mode_labels].mean() < ratio:
                is_mode_present[rng.integers(is_mode_present.shape[0])] = True
            if train:
                self.idcs = idcs[is_mode_present[mode_labels]]
            else:
                self.idcs = idcs[~is_mode_present[mode_labels]]

        print(train, self.idcs.shape)
        self.data = data
        self.ctx = ctx
        self.n_clique = n_clique
        self.output_graphs = output_graphs
        self._gc = nx.complete_graph(7)
        self._enum_edges = list(self._gc.edges)

    def __len__(self):
        return len(self.idcs)

    def reward(self, g):
        return colored_n_clique_reward(g, self.n_clique)

    def collate_fn(self, batch):
        graphs, rewards = zip(*batch)
        batch = self.ctx.collate(graphs)
        batch.y = torch.as_tensor(rewards)
        return batch

    def __getitem__(self, idx):
        idx = self.idcs[idx]
        g = self.data[idx]
        r = torch.tensor(self.reward(g).reshape((1,)))
        if self.output_graphs:
            return self.ctx.graph_to_Data(g), r
        else:
            return g, r


class CliqueTask(GFNTask):
    def __init__(self, dataset: CliqueDataset, temperature_distribution: str, temperature_parameters: Tuple[float],
                 wrap_model: Callable[[nn.Module], nn.Module] = None, hps: Dict = {}):
        self._wrap_model = wrap_model
        self.dataset = dataset
        self.hps = hps
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters
        self.seeded_preference = None
        self.experimental_dirichlet = hps.get('use_experimental_dirichlet', False)
        self.num_objectives = hps.get('num_objectives', 1)

    def flat_reward_transform(self, y: Tensor) -> FlatRewards:
        return FlatRewards(y.float())

    def sample_conditional_information(self, n):
        # TODO factorize this and other MOO boilerplate
        beta = None
        if self.temperature_sample_dist == 'gamma':
            loc, scale = self.temperature_dist_params
            beta = self.rng.gamma(loc, scale, n).astype(np.float32)
            upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
        elif self.temperature_sample_dist == 'uniform':
            beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
            upper_bound = self.temperature_dist_params[1]
        elif self.temperature_sample_dist == 'loguniform':
            beta = np.exp(self.rng.uniform(*np.log(self.temperature_dist_params), n).astype(np.float32))
            upper_bound = self.temperature_dist_params[1]
        elif self.temperature_sample_dist == 'beta':
            beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
            upper_bound = 1
        elif self.temperature_sample_dist == 'const':
            beta = np.ones(n) * self.temperature_dist_params
            upper_bound = 1
        beta_enc = thermometer(torch.tensor(beta), 32, 0, upper_bound)
        return {'beta': torch.tensor(beta), 'encoding': torch.zeros((n, 1))}  #beta_enc}

    def cond_info_to_reward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            if isinstance(flat_reward[0], Tensor):
                flat_reward = torch.stack(flat_reward)
            else:
                flat_reward = torch.tensor(flat_reward)
        scalar_reward = flat_reward[:, 0]
        return RewardScalar(scalar_reward * cond_info['beta'])

    def compute_flat_rewards(self, mols: List[Graph]) -> Tuple[FlatRewards, Tensor]:
        if not len(mols):
            return FlatRewards(torch.zeros((0, self.num_objectives))), torch.zeros((0,)).bool()
        is_valid = torch.ones(len(mols)).bool()
        flat_rewards = torch.tensor([self.dataset.reward(i) for i in mols]).float().reshape((-1, 1))
        return FlatRewards(flat_rewards), is_valid

    def encode_conditional_information(self, info):
        # TODO: redo with temperature once we get there
        encoding = torch.ones((len(info), 1))
        return {'beta': torch.ones(len(info)), 'encoding': encoding.float(), 'preferences': info.float()}


class CliquesTrainer(GFNTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            'bootstrap_own_reward': False,
            'learning_rate': 1e-4,
            'global_batch_size': 64,
            'num_emb': 96,
            'num_layers': 8,
            'num_mlp_layers': 0,
            'tb_epsilon': None,
            'tb_do_subtb': True,
            'illegal_action_logreward': -80,
            'temperature_sample_dist': 'const',
            'temperature_dist_params': '1',
            'weight_decay': 1e-8,
            'num_data_loader_workers': 8,
            'momentum': 0.9,
            'adam_eps': 1e-8,
            'lr_decay': 20000,
            'Z_lr_decay': 20000,
            'clip_grad_type': 'norm',
            'clip_grad_param': 10,
            'random_action_prob': .001,
            'use_experimental_dirichlet': True,
            'offline_ratio': 0.0,
            'valid_sample_cond_info': False,
        }

    def setup(self):
        hps = self.hps
        self.log_dir = hps['log_dir']
        print(self.log_dir)
        self.rng = np.random.default_rng(142857)
        self._data = load_clique_data()
        self.ctx = CliquesEnvContext(4, 2, num_cond_dim=1, graph_data=self._data)
        self.env = GraphBuildingEnv()
        self._do_supervised = hps.get('do_supervised', False)

        self.training_data = CliqueDataset(self._data, self.ctx, 4, train=True, ratio=hps.get('train_ratio', 0.9),
                                           modal_seed=hps.get('modal_seed', None))
        self.test_data = CliqueDataset(self._data, self.ctx, 4, train=False, ratio=hps.get('train_ratio', 0.9),
                                       modal_seed=hps.get('modal_seed', None))
        num_emb, num_layers, num_heads = hps['num_emb'], hps['num_layers'], hps.get('num_heads', 2)
        if self._do_supervised:
            model = GraphTransformerRegressor(x_dim=self.ctx.num_node_dim, e_dim=self.ctx.num_edge_dim, g_dim=1,
                                              num_emb=num_emb, num_layers=num_layers, num_heads=num_heads)
        else:
            model = GraphTransformerGFN(self.ctx, num_emb=hps['num_emb'], num_layers=hps['num_layers'],
                                        num_mlp_layers=hps['num_mlp_layers'])
            self.test_data = RepeatedPreferenceDataset(np.ones((32, 1)), 8)

        self.model = self.sampling_model = model
        params = [i for i in self.model.parameters()]
        self.opt = torch.optim.Adam(params, hps['learning_rate'], (hps['momentum'], 0.999),
                                    weight_decay=hps['weight_decay'], eps=hps['adam_eps'])
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2**(-steps / hps['lr_decay']))

        eps = hps['tb_epsilon']
        hps['tb_epsilon'] = ast.literal_eval(eps) if isinstance(eps, str) else eps
        self.algo = TrajectoryBalance(self.env, self.ctx, self.rng, hps, max_nodes=self.ctx.recommended_max_nodes,
                                      max_len=22)
        self.task = CliqueTask(self.training_data, hps['temperature_sample_dist'],
                               ast.literal_eval(hps['temperature_dist_params']), wrap_model=self._wrap_model_mp,
                               hps=hps)
        self.sampling_tau = hps.get('sampling_tau', 0)
        self.mb_size = hps['global_batch_size']
        self.clip_grad_param = hps['clip_grad_param']
        self.clip_grad_callback = {
            'value': (lambda params: torch.nn.utils.clip_grad_value_(params, self.clip_grad_param)),
            'norm': (lambda params: torch.nn.utils.clip_grad_norm_(params, self.clip_grad_param)),
            'none': (lambda x: None)
        }[hps['clip_grad_type']]
        self.valid_offline_ratio = 0

        self.algo.task = self.task
        if not self._do_supervised:
            self.exact_prob_cb = ExactProbCompCallback(self, [self.env.new()] + self.training_data.data, self.device)
            self._callbacks = {'true_px_error': self.exact_prob_cb}
        else:
            self._callbacks = {}

    def build_callbacks(self):
        return self._callbacks

    def step(self, loss: Tensor):
        loss.backward()
        for i in self.model.parameters():
            self.clip_grad_callback(i)
        self.opt.step()
        self.opt.zero_grad()
        self.lr_sched.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))


class RepeatedPreferenceDataset:
    def __init__(self, preferences, repeat):
        self.prefs = preferences
        self.repeat = repeat

    def __len__(self):
        return len(self.prefs) * self.repeat

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        return torch.tensor(self.prefs[int(idx // self.repeat)])


def hashg(g):
    return nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(g, node_attr='v')


class ExactProbCompCallback:
    def __init__(self, trial, states, dev, mbs=128,
                 cache_path='/scratch/emmanuel.bengio/data/cliques/two_col_7_precomp_px.pkl.gz'):
        self.trial = trial
        self.mbs = mbs
        self.dev = dev
        self.states = states
        self.cache_path = cache_path
        if self.cache_path is not None:
            bs, ids = pickle.load(gzip.open(cache_path, 'rb'))
            self.precomputed_batches, self.precomputed_indices = ([
                i.to(dev) for i in bs
            ], [[(j[0].to(dev), j[1].to(dev)) for j in i] for i in ids])
        else:
            self._compute_things()
        self.log_rewards = np.array([self.trial.training_data.reward(i) for i in tqdm(self.states, disable=None)])
        self.logZ = np.log(np.sum(np.exp(self.log_rewards)))
        self.true_log_probs = self.log_rewards - self.logZ

    def on_validation_end(self, metrics):
        # Compute exact sampling probabilities of the model, last probability is p(illegal), remove it.
        log_probs = self.compute_prob(self.trial.model).cpu().numpy()[:-1]
        lp, p = log_probs, np.exp(log_probs)
        lq, q = self.true_log_probs, np.exp(self.true_log_probs)
        metrics['L1_logpx_error'] = np.mean(abs(lp - lq))
        metrics['JS_divergence'] = (p * (lp - lq) + q * (lq - lp)).sum() / 2

    def _compute_things(self):
        states, mbs, dev = self.states, self.mbs, self.dev
        self.precomputed_batches = []
        self.precomputed_indices = []
        states_hash = [hashg(i) for i in tqdm(states, disable=None)]
        states_Data = [self.trial.ctx.graph_to_Data(i) for i in tqdm(states, disable=None)]
        ones = torch.ones((mbs, 1)).to(dev)
        hash_to_graphs = {}
        for i, h, g in zip(range(len(states)), states_hash, states):
            hash_to_graphs[h] = hash_to_graphs.get(h, list()) + [(g, i)]

        iso = lambda u, v: is_isomorphic(u, v, lambda a, b: a == b, lambda a, b: a == b)

        def get_graph_idx(g, default=None):
            h = hashg(g)
            if h not in hash_to_graphs and default is not None:
                return default
            bucket = hash_to_graphs[h]
            if len(bucket) == 1:
                return bucket[0][1]
            for i in bucket:
                if iso(i[0], g):
                    return i[1]
            if default is not None:
                return default
            raise ValueError(g)

        for bi in tqdm(range(0, len(states), mbs, disable=None)):
            bs = states[bi:bi + mbs]
            bD = states_Data[bi:bi + mbs]
            indices = list(range(bi, bi + len(bs)))
            non_terminals = [(i, j, k) for i, j, k in zip(bs, bD, indices) if not self.is_terminal(i)]
            if not len(non_terminals):
                self.precomputed_batches.append(None)
                self.precomputed_indices.append(None)
                continue
            bs, bD, indices = zip(*non_terminals)
            batch = self.trial.ctx.collate(bD).to(dev)
            self.precomputed_batches.append(batch)

            with torch.no_grad():
                cat, mo = self.trial.model(batch, ones[:len(bs)])
            actions = [list() for i in range(len(bs))]
            offset = 0
            for u, i in enumerate(cat.logsoftmax()):
                for k, j in enumerate(map(list, (i * 0 + 1).nonzero().cpu().numpy())):
                    jb = cat.batch[u][j[0]].item()
                    actions[jb].append((u, j[0] - cat.slice[u][jb].item(), j[1], k + offset))
                offset += i.numel()
            all_indices = []
            for jb, j_acts in enumerate(actions):
                end_indices = []
                being_indices = []
                for *a, srcidx in j_acts:
                    idx = indices[jb]
                    sp = (self.trial.env.step(bs[jb], self.trial.ctx.aidx_to_GraphAction(bD[jb], a[:3]))
                          if a[0] != 0 else bs[jb])
                    spidx = get_graph_idx(sp, len(states))
                    if a[0] == 0 or spidx >= len(states) or self.is_terminal(sp):
                        end_indices.append((idx, spidx, srcidx))
                    else:
                        being_indices.append((idx, spidx, srcidx))
                all_indices.append((torch.tensor(end_indices).T.to(dev), torch.tensor(being_indices).T.to(dev)))
            self.precomputed_indices.append(all_indices)

    def is_terminal(self, g):
        return len(g.nodes) > 7 or len(g.edges) >= 21

    def compute_prob(self, model):
        prob_of_being_t = torch.zeros(len(self.states) + 1).to(self.dev) - 100
        prob_of_being_t[0] = 0
        prob_of_ending_t = torch.zeros(len(self.states) + 1).to(self.dev) - 100
        ones = torch.ones((self.mbs, 1)).to(self.dev)
        # Note: visiting the states in order works because the ordering here is a natural topological sort.
        # Wrong results otherwise.
        for bi, batch, pre_indices in zip(tqdm(range(0, len(self.states), self.mbs), disable=None),
                                          self.precomputed_batches, self.precomputed_indices):
            bs = self.states[bi:bi + self.mbs]
            indices = list(range(bi, bi + len(bs)))
            non_terminals = [(i, j) for i, j in zip(bs, indices) if not self.is_terminal(i)]
            if not len(non_terminals):
                continue
            bs, indices = zip(*non_terminals)
            with torch.no_grad():
                cat, mo = model(batch, ones[:len(bs)])
            logprobs = torch.cat([i.flatten() for i in cat.logsoftmax()])
            for end_indices, being_indices in pre_indices:
                if end_indices.shape[0] > 0:
                    s_idces, sp_idces, a_idces = end_indices
                    prob_of_ending_t = scatter_add((prob_of_being_t[s_idces] + logprobs[a_idces]).exp(), sp_idces,
                                                   out=prob_of_ending_t.exp()).log()
                if being_indices.shape[0] > 0:
                    s_idces, sp_idces, a_idces = being_indices
                    prob_of_being_t = scatter_add((prob_of_being_t[s_idces] + logprobs[a_idces]).exp(), sp_idces,
                                                  out=prob_of_being_t.exp()).log()
        return prob_of_ending_t


class RewardMSE:
    def compute_batch_losses(self, model, batch, **kw):
        pred = model(batch, torch.ones((batch.y.shape[0], 1), device=batch.x.device))
        loss = (pred - batch.y).pow(2).mean()
        return loss, {'loss': loss}


class CliquesSupervisedTrainer(CliquesTrainer):
    def setup(self):
        super().setup()
        self.algo = RewardMSE()
        self.training_data.output_graphs = True
        self.test_data.output_graphs = True

    def build_training_data_loader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.training_data, batch_size=self.mb_size, num_workers=self.num_workers,
                                           persistent_workers=self.num_workers > 0,
                                           collate_fn=self.training_data.collate_fn)

    def build_validation_data_loader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.mb_size * 4, num_workers=self.num_workers,
                                           persistent_workers=self.num_workers > 0,
                                           collate_fn=self.test_data.collate_fn)


def main():
    _hps = {
        'num_training_steps': 500,
        'validate_every': 10_000,
        'num_data_loader_workers': 8,
        'learning_rate': 3e-4,
        'num_layers': 3,
        'num_mlp_layers': 0,
        'num_emb': 128,
        'global_batch_size': 128,
        'do_supervised': False,  # Change this to launch a supervised job
    }

    hps = [
        {
            **_hps,
            'log_dir': '/scratch/emmanuel.bengio/logs/cliques_gfn/run_0/',
        },
        {
            **_hps,
            'log_dir': '/scratch/emmanuel.bengio/logs/cliques_gfn/run_1/',
            'temperature_dist_params': '(1, 1.001)',
        },
    ]

    if 1:
        import sys
        trial = CliquesTrainer(hps[int(sys.argv[1])], torch.device('cuda'))
        trial.verbose = True
        trial.run()


if __name__ == '__main__':
    main()
