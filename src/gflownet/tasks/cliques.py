import ast
import bz2
import os
import pickle
from typing import Any, Callable, Dict, List, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import is_isomorphic
import numpy as np
import scipy.stats as stats
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch_geometric.data as gd
from torch_scatter import scatter_add
from tqdm import tqdm

from gflownet.algo.flow_matching import FlowMatching
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.envs.cliques_env import CliquesEnvContext
from gflownet.envs.graph_building_env import Graph
from gflownet.envs.graph_building_env import GraphActionType
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.models.graph_transformer import GraphTransformer
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.train import FlatRewards
from gflownet.train import GFNTask
from gflownet.train import GFNTrainer
from gflownet.train import RewardScalar
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


def even_neighbors_reward(g):
    total_correct = 0
    for n in g:
        num_diff_colr = 0
        c = g.nodes[n]['v']
        for i in g.neighbors(n):
            num_diff_colr += int(g.nodes[i]['v'] != c)
        total_correct += int(num_diff_colr % 2 == 0) - (1 if num_diff_colr == 0 else 0)
    return np.float32((total_correct - len(g.nodes) if len(g.nodes) > 3 else -5) * 10 / 7)


def count_reward(g):
    ncols = np.bincount([g.nodes[i]['v'] for i in g], minlength=2)
    return np.float32(-abs(ncols[0] + ncols[1] / 2 - 3) / 4 * 10)


def generate_two_col_data(data_root):
    atl = nx.generators.atlas.graph_atlas_g()
    # Filter out disconnected graphs
    conn = [i for i in atl if 1 <= len(i.nodes) <= 7 and nx.is_connected(i)]
    # Create all possible two-colored graphs
    two_col_graphs = [nx.Graph()]
    pb = tqdm(range(117142), disable=None)
    hashes = {}
    rejected = 0

    def node_eq(a, b):
        return a == b

    for g in conn:
        for i in range(2**len(g.nodes)):
            g = g.copy()
            for j in range(len(g.nodes)):
                bit = i % 2
                i //= 2
                g.nodes[j]['v'] = bit
            h = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(g, node_attr='v')
            if h not in hashes:
                hashes[h] = [g]
                two_col_graphs.append(g)
            else:
                if not any(nx.algorithms.isomorphism.is_isomorphic(g, gp, node_eq) for gp in hashes[h]):
                    hashes[h].append(g)
                    two_col_graphs.append(g)
                else:
                    pb.set_description(f'{rejected}', refresh=False)
                    rejected += 1
            pb.update(1)
    with bz2.open(data_root + '/two_col_7_graphs.pkl.bz', 'wb') as f:
        pickle.dump(two_col_graphs, f)
    return two_col_graphs


def load_two_col_data(data_root):
    with bz2.open(data_root + '/two_col_7_graphs.pkl.bz', 'rb') as f:
        data = pickle.load(f)
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
                 modal_seed=None, max_nodes=7):
        self.data = data
        self.ctx = ctx
        self.n_clique = n_clique
        self.output_graphs = output_graphs
        self.reward_func = 'cliques'
        self.idcs = [0]
        self.max_nodes = max_nodes
        if data is None:
            return

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
        self._gc = nx.complete_graph(7)
        self._enum_edges = list(self._gc.edges)
        self.compute_Fsa = False
        self.compute_normalized_Fsa = False

    def __len__(self):
        return len(self.idcs)

    def reward(self, g):
        if len(g.nodes) > self.max_nodes:
            return -100
        if self.reward_func == 'cliques':
            return colored_n_clique_reward(g, self.n_clique)
        elif self.reward_func == 'even_neighbors':
            return even_neighbors_reward(g)
        elif self.reward_func == 'count':
            return count_reward(g)
        elif self.reward_func == 'const':
            return np.float32(0)

    def collate_fn(self, batch):
        graphs, rewards, idcs = zip(*batch)
        batch = self.ctx.collate(graphs)
        if self.regress_to_F:
            batch.y = torch.as_tensor([self.epc.mdp_graph.nodes[i]['F'] for i in idcs])
        else:
            batch.y = torch.as_tensor(rewards)
        if self.compute_Fsa:
            all_targets = []
            for data_idx in idcs:
                targets = [
                    torch.zeros_like(getattr(self.epc._Data[data_idx], i.mask_name)) - 100
                    for i in self.ctx.action_type_order
                ]
                for neighbor in list(self.epc.mdp_graph.neighbors(data_idx)):
                    for _, edge in self.epc.mdp_graph.get_edge_data(data_idx, neighbor).items():
                        a, F = edge['a'], edge['F']
                        targets[a[0]][a[1], a[2]] = F
                if self.compute_normalized_Fsa:
                    logZ = torch.log(sum([i.exp().sum() for i in targets]))
                    targets = [i - logZ for i in targets]
                all_targets.append(targets)
            batch.y = torch.cat([torch.cat(i).flatten() for i in zip(*all_targets)])
        return batch

    def __getitem__(self, idx):
        idx = self.idcs[idx]
        g = self.data[idx]
        r = torch.tensor(self.reward(g).reshape((1,)))
        if self.output_graphs:
            return self.ctx.graph_to_Data(g), r, idx
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

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
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
        encoding = torch.zeros((len(info), 1))
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
            'tb_p_b_is_parameterized': False,
            'illegal_action_logreward': -30,
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
            'do_save_generated': False,
            'data_root': '/scratch/emmanuel.bengio/data/cliques',
        }

    def setup(self):
        hps = self.hps
        self.log_dir = hps['log_dir']
        max_nodes = hps.get('max_nodes', 7)
        print(self.log_dir)
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        self._data = load_two_col_data(hps['data_root'])
        self.ctx = CliquesEnvContext(max_nodes, 4, 2, num_cond_dim=1, graph_data=self._data)
        self._do_supervised = hps.get('do_supervised', False)

        self.training_data = CliqueDataset(self._data, self.ctx, 4, train=True, ratio=hps.get('train_ratio', 0.9),
                                           modal_seed=hps.get('modal_seed', None), max_nodes=max_nodes)
        self.test_data = CliqueDataset(self._data, self.ctx, 4, train=False, ratio=hps.get('train_ratio', 0.9),
                                       modal_seed=hps.get('modal_seed', None), max_nodes=max_nodes)
        self.training_data.reward_func = self.test_data.reward_func = self.hps.get('reward_func', 'cliques')
        num_emb, num_layers, num_heads = hps['num_emb'], hps['num_layers'], hps.get('num_heads', 2)
        i2h_width = hps.get('i2h_width', 1)
        if self._do_supervised and not hps.get('regress_to_Fsa', True):
            model = GraphTransformerRegressor(x_dim=self.ctx.num_node_dim, e_dim=self.ctx.num_edge_dim, g_dim=1,
                                              num_emb=num_emb, num_layers=num_layers, num_heads=num_heads,
                                              i2h_width=i2h_width, ln_type=hps.get('ln_type', 'pre'))
        else:
            model = GraphTransformerGFN(self.ctx, num_emb=hps['num_emb'], num_layers=hps['num_layers'],
                                        num_mlp_layers=hps['num_mlp_layers'], num_heads=num_heads, i2h_width=i2h_width,
                                        ln_type=hps.get('ln_type', 'pre'), do_bck=hps['tb_p_b_is_parameterized'])
            if not self._do_supervised:
                self.test_data = RepeatedPreferenceDataset(np.ones((32, 1)), 8)

        self.model = self.sampling_model = model
        params = [i for i in self.model.parameters()]
        if hps.get('opt', 'adam') == 'adam':
            self.opt = torch.optim.Adam(params, hps['learning_rate'], (hps['momentum'], 0.999),
                                        weight_decay=hps['weight_decay'], eps=hps['adam_eps'])
        elif hps.get('opt', 'adam') == 'SGD':
            self.opt = torch.optim.SGD(params, hps['learning_rate'], hps['momentum'], weight_decay=hps['weight_decay'])
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2**(-steps / hps['lr_decay']))

        eps = hps['tb_epsilon']
        hps['tb_epsilon'] = ast.literal_eval(eps) if isinstance(eps, str) else eps
        algo = hps.get('algo', 'TB')
        if algo == 'TB' or algo == 'subTB':
            if algo == 'subTB':
                hps['tb_do_subtb'] = True
            self.algo = TrajectoryBalance(self.env, self.ctx, self.rng, hps, max_nodes=max_nodes, max_len=25)
        elif algo == 'FM':
            self.algo = FlowMatching(self.env, self.ctx, self.rng, hps, max_nodes=max_nodes, max_len=25)
        self.task = CliqueTask(self.training_data, hps['temperature_sample_dist'],
                               ast.literal_eval(str(hps['temperature_dist_params'])), wrap_model=self._wrap_model_mp,
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
        self.exact_prob_cb = ExactProbCompCallback(self, self.training_data.data, self.device,
                                                   cache_root=hps['data_root'],
                                                   cache_path=hps['data_root'] + '/two_col_epc_cache.pkl')
        split_type = hps.get('test_split_type', 'random')
        if split_type == 'random':
            pass
        elif split_type == 'bck_traj':
            train_idcs, test_idcs = self.exact_prob_cb.get_bck_trajectory_test_split(hps.get('train_ratio', 0.9))
            self.training_data.idcs = train_idcs
            self.test_data.idcs = test_idcs
        elif split_type == 'subtrees':
            train_idcs, test_idcs = self.exact_prob_cb.get_subtree_test_split(hps.get('train_ratio', 0.9),
                                                                              hps.get('test_split_seed', 142857))
            self.training_data.idcs = train_idcs
            self.test_data.idcs = test_idcs
        if not self._do_supervised or hps.get('regress_to_Fsa', False):
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


class RepeatedPreferenceDataset(CliqueDataset):
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
    def __init__(self, trial, states, dev, mbs=128, cache_root=None, cache_path=None, do_save_px=True, log_rewards=None,
                 tqdm_disable=None, ctx=None, env=None):
        self.trial = trial
        self.ctx = trial.ctx if trial is not None else ctx
        self.env = trial.env if trial is not None else env
        self.mbs = mbs
        self.dev = dev
        self.states = states
        self.cache_root = cache_root
        self.cache_path = cache_path
        self.mdp_graph = None
        if self.cache_path is not None:
            self.load_cache(self.cache_path)
        if log_rewards is None:
            self.log_rewards = np.array(
                [self.trial.training_data.reward(i) for i in tqdm(self.states, disable=tqdm_disable)])
        else:
            self.log_rewards = log_rewards
        self.logZ = np.log(np.sum(np.exp(self.log_rewards)))
        self.true_log_probs = self.log_rewards - self.logZ
        # This is reward-dependent
        if self.mdp_graph is not None:
            self.recompute_flow()
        self.do_save_px = do_save_px
        if do_save_px:
            os.makedirs(self.trial.hps['log_dir'], exist_ok=True)
        self._save_increment = 0

    def load_cache(self, cache_path):
        print("Loading cache @", cache_path)
        cache = torch.load(open(cache_path, 'rb'))
        self.mdp_graph = cache['mdp']
        self._Data = cache['Data']
        self._hash_to_graphs = cache['hashmap']
        bs, ids = cache['batches'], cache['idces']
        print("Done")
        self.precomputed_batches, self.precomputed_indices = ([
            i.to(self.dev) for i in bs
        ], [[(j[0].to(self.dev), j[1].to(self.dev)) for j in i] for i in ids])

    def on_validation_end(self, metrics):
        # Compute exact sampling probabilities of the model, last probability is p(illegal), remove it.
        log_probs = self.compute_prob(self.trial.model).cpu().numpy()[:-1]
        lp, p = log_probs, np.exp(log_probs)
        lq, q = self.true_log_probs, np.exp(self.true_log_probs)
        metrics['L1_logpx_error'] = np.mean(abs(lp - lq))
        metrics['JS_divergence'] = (p * (lp - lq) + q * (lq - lp)).sum() / 2
        if self.do_save_px:
            torch.save(log_probs, open(self.trial.hps['log_dir'] + f'/log_px_{self._save_increment}.pt', 'wb'))
            self._save_increment += 1

    def get_graph_idx(self, g, default=None):
        iso = lambda u, v: is_isomorphic(u, v, lambda a, b: a == b, lambda a, b: a == b)
        h = hashg(g)
        if h not in self._hash_to_graphs and default is not None:
            return default
        bucket = self._hash_to_graphs[h]
        if len(bucket) == 1:
            return bucket[0]
        for i in bucket:
            if iso(self.states[i], g):
                return i
        if default is not None:
            return default
        raise ValueError(g)

    def compute_cache(self, tqdm_disable=None):
        states, mbs, dev = self.states, self.mbs, self.dev
        mdp_graph = nx.MultiDiGraph()
        self.precomputed_batches = []
        self.precomputed_indices = []
        self._hash_to_graphs = {}
        states_hash = [hashg(i) for i in tqdm(states, disable=tqdm_disable)]
        self._Data = states_Data = gd.Batch.from_data_list(
            [self.ctx.graph_to_Data(i) for i in tqdm(states, disable=tqdm_disable)])
        ones = torch.ones((mbs, self.ctx.num_cond_dim)).to(dev)
        for i, h in enumerate(states_hash):
            self._hash_to_graphs[h] = self._hash_to_graphs.get(h, list()) + [i]

        for bi in tqdm(range(0, len(states), mbs), disable=tqdm_disable):
            bs = states[bi:bi + mbs]
            bD = states_Data[bi:bi + mbs]
            indices = list(range(bi, bi + len(bs)))
            # TODO: if the environment's masks are well designed, this non_terminal business shouldn't be necessary
            #non_terminals = [(i, j, k) for i, j, k in zip(bs, bD, indices) if not self.is_terminal(i)]
            #if not len(non_terminals):
            #    self.precomputed_batches.append(None)
            #    self.precomputed_indices.append(None)
            #    continue
            # bs, bD, indices = zip(*non_terminals)
            batch = self.ctx.collate(bD).to(dev)
            self.precomputed_batches.append(batch)

            #with torch.no_grad():
            #    cat, *_, mo = self.trial.model(batch, ones[:len(bs)])
            actions = [list() for i in range(len(bs))]
            offset = 0
            for u, i in enumerate(ctx.action_type_order):
                # /!\ This assumes mask.shape == cat.logit[i].shape
                mask = getattr(batch, i.mask_name)
                batch_key = GraphTransformerGFN._graph_part_to_key[GraphTransformerGFN._action_type_to_graph_part[i]]
                batch_idx = (getattr(batch, f'{batch_key}_batch' if batch_key != 'x' else 'batch')
                             if batch_key is not None else torch.arange(batch.num_graphs, device=dev))
                mslice = batch._slice_dict[batch_key] if batch_key is not None else torch.arange(
                    batch.num_graphs + 1, device=dev)
                for j in mask.nonzero().cpu().numpy():
                    # We're using nonzero above to enumerate all positions, but we still need to check
                    # if the mask is nonzero since we only want the legal actions.
                    # We *don't* wan't mask.nonzero() because then `k` would be wrong
                    k = j[0] * mask.shape[1] + j[1] + offset
                    jb = batch_idx[j[0]].item()
                    actions[jb].append((u, j[0] - mslice[jb].item(), j[1], k))
                offset += mask.numel()
            all_indices = []
            for jb, j_acts in enumerate(actions):
                end_indices = []
                being_indices = []
                for *a, srcidx in j_acts:
                    idx = indices[jb]
                    sp = (self.env.step(bs[jb], self.ctx.aidx_to_GraphAction(bD[jb], a[:3])) if a[0] != 0 else bs[jb])
                    spidx = self.get_graph_idx(sp, len(states))
                    if a[0] == 0 or spidx >= len(states):
                        end_indices.append((idx, spidx, srcidx))
                        mdp_graph.add_edge(idx, spidx, srci=srcidx, term=True, a=a)
                    else:
                        being_indices.append((idx, spidx, srcidx))
                        mdp_graph.add_edge(idx, spidx, srci=srcidx, term=False, a=a)
                all_indices.append((torch.tensor(end_indices).T.to(dev), torch.tensor(being_indices).T.to(dev)))
            self.precomputed_indices.append(all_indices)
        self.mdp_graph = mdp_graph

    def save_cache(self, path):
        with open(path, 'wb') as f:
            torch.save(
                {
                    'batches': [i.cpu() for i in self.precomputed_batches],
                    'idces': [[(j[0].cpu(), j[1].cpu()) for j in i] for i in self.precomputed_indices],
                    'Data': self._Data,
                    'mdp': self.mdp_graph,
                    'hashmap': self._hash_to_graphs,
                }, f)

    def compute_prob(self, model, cond_info=None, tqdm_disable=None):
        # +1 to count illegal actions prob (may not be applicable to well-masked envs)
        prob_of_being_t = torch.zeros(len(self.states) + 1).to(self.dev) - 100
        prob_of_being_t[0] = 0
        prob_of_ending_t = torch.zeros(len(self.states) + 1).to(self.dev) - 100
        if cond_info is None:
            cond_info = torch.zeros((self.mbs, self.ctx.num_cond_dim)).to(self.dev)
        if cond_info.ndim == 1:
            cond_info = cond_info[None, :] * torch.ones((self.mbs, 1)).to(self.dev)
        if cond_info.ndim == 2 and cond_info.shape[0] == 1:
            cond_info = cond_info * torch.ones((self.mbs, 1)).to(self.dev)
        # Note: visiting the states in order works because the ordering here is a natural topological sort.
        # Wrong results otherwise.
        for bi, batch, pre_indices in zip(tqdm(range(0, len(self.states), self.mbs), disable=tqdm_disable),
                                          self.precomputed_batches, self.precomputed_indices):
            bs = self.states[bi:bi + self.mbs]
            # This isn't even right:
            # indices = list(range(bi, bi + len(bs)))
            # non_terminals = [(i, j) for i, j in zip(bs, indices) if not self.is_terminal(i)]
            # if not len(non_terminals):
            #     continue
            # bs, indices = zip(*non_terminals)
            with torch.no_grad():
                cat, *_, mo = model(batch, cond_info[:len(bs)])
            logprobs = torch.cat([i.flatten() for i in cat.logsoftmax()])
            for end_indices, being_indices in pre_indices:
                if being_indices.shape[0] > 0:
                    s_idces, sp_idces, a_idces = being_indices
                    prob_of_being_t = scatter_add((prob_of_being_t[s_idces] + logprobs[a_idces]).exp(), sp_idces,
                                                  out=prob_of_being_t.exp()).log()
                if end_indices.shape[0] > 0:
                    s_idces, sp_idces, a_idces = end_indices
                    prob_of_ending_t = scatter_add((prob_of_being_t[s_idces] + logprobs[a_idces]).exp(), sp_idces,
                                                   out=prob_of_ending_t.exp()).log()
        return prob_of_ending_t

    def recompute_flow(self, tqdm_disable=None):
        g = self.mdp_graph
        for i in g:
            g.nodes[i]['F'] = -100
        for i in tqdm(list(range(len(g)))[::-1], disable=tqdm_disable):
            p = sorted(list(g.predecessors(i)), reverse=True)
            num_back = len([n for n in p if n != i])
            for j in p:
                if j == i:
                    g.nodes[j]['F'] = np.logaddexp(g.nodes[j]['F'], self.log_rewards[j])
                    g.edges[(i, i, 0)]['F'] = self.log_rewards[j].item()
                else:
                    backflow = np.log(np.exp(g.nodes[i]['F']) / num_back)
                    g.nodes[j]['F'] = np.logaddexp(g.nodes[j]['F'], backflow)
                    # Here we're making a decision to split flow backwards equally for idempotent actions
                    # from the same state. I think it's ok?
                    ed = g.get_edge_data(j, i)
                    for k, vs in ed.items():
                        g.edges[(j, i, k)]['F'] = np.log(np.exp(backflow) / len(ed))

    def get_bck_trajectory_test_split(self, r, seed=142857):
        test_set = set()
        n = int((1 - r) * len(self.states))
        np.random.seed(seed)
        while len(test_set) < n:
            i0 = np.random.randint(len(self.states))
            s0 = self.states[i0]
            if len(s0.nodes) < 7:  # TODO: unhardcode this
                continue
            s = s0
            idx = i0
            while len(s.nodes) > 5:  # TODO: unhardcode this
                test_set.add(idx)
                actions = [(u, a.item(), b.item())
                           for u, ra in enumerate(self.ctx.bck_action_type_order)
                           for a, b in getattr(self._Data[idx], ra.mask_name).nonzero()]
                action = actions[np.random.randint(len(actions))]
                gaction = self.ctx.aidx_to_GraphAction(self._Data[idx], action, fwd=False)
                s = self.env.step(s, gaction)
                idx = self.get_graph_idx(s)  # This finds the graph index taking into account isomorphism
                s = self.states[idx]  # We still have to get the original graph so that the Data instance is correct
        train_set = list(set(range(len(self.states))).difference(test_set))
        test_set = list(test_set)
        np.random.shuffle(train_set)
        return train_set, test_set

    def get_subtree_test_split(self, r, seed=142857):
        cache_path = f'{self.cache_root}/subtree_split_{r}_{seed}.pkl'
        if self.cache_root is not None:
            if os.path.exists(cache_path):
                return pickle.load(open(cache_path, 'rb'))
        test_set = set()
        n = int((1 - r) * len(self.states))
        np.random.seed(seed)
        start_states_idx, available_start_states, start_states = [], [], []
        edge_limit = 11
        while len(test_set) < n:
            num_ss = len([i for i in start_states_idx if i not in test_set])
            if num_ss == 0 or len(available_start_states) == 0:
                start_states, start_states_idx = zip(*[(s0, i)
                                                       for i, s0 in enumerate(self.states)
                                                       if len(s0.nodes) == 6 and len(s0.edges) >= edge_limit])
                available_start_states = list(range(len(start_states)))
                edge_limit -= 1
            assi = np.random.randint(len(available_start_states))
            ssi = available_start_states.pop(assi)
            s0 = start_states[ssi]
            i0 = self.get_graph_idx(s0)
            if i0 in test_set:
                continue
            stack = [(s0, i0)]
            while len(stack):
                s, i = stack.pop()
                if i in test_set:
                    continue
                test_set.add(i)
                actions = [(u, a.item(), b.item()) for u, ra in enumerate(self.ctx.action_type_order)
                           if ra != GraphActionType.Stop for a, b in getattr(self._Data[i], ra.mask_name).nonzero()]
                for action in actions:
                    gaction = self.ctx.aidx_to_GraphAction(self._Data[i], action, fwd=True)
                    sp = self.env.step(s, gaction)
                    ip = self.get_graph_idx(sp)  # This finds the graph index taking into account isomorphism
                    if ip in test_set:
                        continue
                    sp = self.states[ip]  # We still have to get the original graph so that the Data instance is correct
                    stack.append((sp, ip))
        train_set = list(set(range(len(self.states))).difference(test_set))
        test_set = list(test_set)
        np.random.shuffle(train_set)
        if self.cache_root is not None:
            pickle.dump((np.array(train_set), np.array(test_set)), open(cache_path, 'wb'))
        return train_set, test_set


class Regression:
    def compute_batch_losses(self, model, batch, **kw):
        if self.regress_to_Fsa:
            fwd_cat, *other = model(batch, torch.zeros((batch.num_graphs, 1), device=batch.x.device))
            mask = torch.cat([i.flatten() for i in fwd_cat.masks])
            pred = torch.cat([i.flatten() for i in fwd_cat.logits]) * mask
            batch.y = batch.y * mask
        else:
            pred = model(batch, torch.zeros((batch.num_graphs, 1), device=batch.x.device))
        if self.loss_type == 'MSE':
            loss = (pred - batch.y).pow(2).mean()
        elif self.loss_type == 'MAE':
            loss = abs(pred - batch.y).mean()
        return loss, {'loss': loss}


class CliquesSupervisedTrainer(CliquesTrainer):
    def setup(self):
        super().setup()
        self.algo = Regression()
        self.algo.loss_type = self.hps.get('loss_type', 'MSE')
        self.algo.regress_to_Fsa = self.hps.get('regress_to_Fsa', False)
        self.training_data.output_graphs = True
        self.test_data.output_graphs = True
        for i in [self.training_data, self.test_data]:
            i.compute_Fsa = self.hps.get('regress_to_Fsa', False)
            i.regress_to_F = self.hps.get('regress_to_F', False)
            i.compute_normalized_Fsa = self.hps.get('regress_to_P_F', False)
            i.epc = self.exact_prob_cb

    def build_training_data_loader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.training_data, batch_size=self.mb_size, num_workers=self.num_workers,
                                           persistent_workers=self.num_workers > 0, shuffle=True,
                                           collate_fn=self.training_data.collate_fn)

    def build_validation_data_loader(self) -> DataLoader:
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.mb_size, num_workers=self.num_workers,
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

    _hps = {
        'num_training_steps': 5000,
        'validate_every': 500,
        'num_data_loader_workers': 8,
        'learning_rate': 3e-4,
        'num_layers': 3,
        'num_mlp_layers': 1,
        'num_emb': 128,
        'global_batch_size': 128,
        'do_supervised': True,  # Change this to launch a supervised job
    }

    hps = [
        {
            **_hps,
            'log_dir': './tmp/run_csup_0',
            'regress_to_F': True,
            'regress_to_Fsa': False,
        },
        {
            **_hps,
            'log_dir': './tmp/run_csup_1',
            'regress_to_F': True,
            'regress_to_Fsa': False,
            'num_layers': 6,
            'num_heads': 4,
            'reward_func': 'count',
        },
        {
            **_hps,  # Adds shuffle
            'log_dir': './tmp/run_csup_2',
            'regress_to_F': True,
            'regress_to_Fsa': False,
            'num_layers': 6,
            'num_heads': 4,
            'reward_func': 'count',
        },
        {
            **_hps,  # What does sup learning get?
            'log_dir': './tmp/run_csup_3',
            'regress_to_F': False,
            'regress_to_Fsa': False,
            'num_layers': 6,
            'num_heads': 4,
            'reward_func': 'count',
        },
        {
            **_hps,  # minor bug fix
            'log_dir': './tmp/run_csup_4',
            'regress_to_F': True,
            'regress_to_Fsa': False,
            'num_layers': 6,
            'num_heads': 4,
            'reward_func': 'count',
        },
        {
            **_hps,
            'log_dir': './tmp/run_csup_5',
            'regress_to_F': True,
            'regress_to_Fsa': False,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'i2h_width': 4,
            'reward_func': 'count',
        },
        {
            **_hps,
            'num_training_steps': 5000,
            'log_dir': './tmp/run_csup_6',
            'global_batch_size': 512,
            'regress_to_F': True,
            'regress_to_Fsa': False,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'i2h_width': 4,
            'reward_func': 'count',
        },
        {
            **_hps,
            'num_training_steps': 20000,
            'log_dir': './tmp/run_csup_7',
            'global_batch_size': 256,
            'learning_rate': 1e-3,
            'regress_to_F': True,
            'regress_to_Fsa': False,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'i2h_width': 4,
            'reward_func': 'count',
        },
        {
            **_hps,
            'num_training_steps': 50000,
            'log_dir': './tmp/run_csup_8',
            'global_batch_size': 256,
            'learning_rate': 1e-4,
            'regress_to_F': True,
            'regress_to_Fsa': False,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'i2h_width': 4,
            'reward_func': 'count',
        },
        {
            **_hps,
            'num_training_steps': 50000,
            'log_dir': './tmp/run_csup_9',
            'global_batch_size': 256,
            'learning_rate': 1e-4,
            'regress_to_F': False,
            'regress_to_Fsa': False,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'i2h_width': 4,
            'reward_func': 'count',
        },
        {
            **_hps,
            'num_training_steps': 50000,
            'log_dir': './tmp/run_csup_10',
            'global_batch_size': 256,
            'learning_rate': 1e-4,
            'regress_to_F': True,
            'regress_to_Fsa': False,
            'num_layers': 6,
            'num_heads': 3,
            'num_mlp_layers': 1,
            'i2h_width': 1,
            'reward_func': 'count',
            'data_root': '/mnt/bh1/scratch/emmanuel.bengio/data/cliques',
        },
        {
            **_hps,
            'num_training_steps': 50000,
            'log_dir': './tmp/run_csup_11',
            'global_batch_size': 256,
            'learning_rate': 1e-4,
            'regress_to_F': False,
            'regress_to_Fsa': True,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'i2h_width': 4,
            'reward_func': 'count',
            'data_root': '/mnt/bh1/scratch/emmanuel.bengio/data/cliques',
        },
        {
            **_hps,
            'num_training_steps': 50000,
            'log_dir': './tmp/run_csup_12',
            'global_batch_size': 256,
            'learning_rate': 1e-4,
            'regress_to_F': False,
            'regress_to_Fsa': True,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'i2h_width': 4,
            'weight_decay': 1e-3,
            'reward_func': 'count',
            'data_root': '/scratch/emmanuel.bengio/data/cliques',
        },
        {
            **_hps,
            'num_training_steps': 50000,
            'log_dir': './tmp/run_csup_13',
            'global_batch_size': 256,
            'learning_rate': 1e-4,
            'regress_to_F': False,
            'regress_to_Fsa': True,
            'regress_to_P_F': True,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'i2h_width': 4,
            'reward_func': 'count',
            'data_root': '/scratch/emmanuel.bengio/data/cliques',
        },
        # With p(x) measure
        {
            **_hps,
            'num_training_steps': 50000,
            'log_dir': './tmp/run_csup_14',
            'global_batch_size': 256,
            'learning_rate': 1e-4,
            'regress_to_F': False,
            'regress_to_Fsa': True,
            'regress_to_P_F': True,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'i2h_width': 4,
            'reward_func': 'count',
            'num_data_loader_workers': 0,
            'data_root': '/scratch/emmanuel.bengio/data/cliques',
        },
        {
            **_hps,
            'num_training_steps': 50000,
            'log_dir': './tmp/run_csup_15',
            'global_batch_size': 256,
            'learning_rate': 1e-4,
            'regress_to_F': False,
            'regress_to_Fsa': True,
            'regress_to_P_F': False,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'i2h_width': 4,
            'reward_func': 'count',
            'data_root': '/scratch/emmanuel.bengio/data/cliques',
        },
        {
            **_hps,
            'num_training_steps': 50000,
            'log_dir': './tmp/run_csup_16',
            'global_batch_size': 256,
            'learning_rate': 1e-4,
            'regress_to_F': False,
            'regress_to_Fsa': True,
            'regress_to_P_F': False,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'num_data_loader_workers': 20,
            'i2h_width': 4,
            'reward_func': 'count',
            'data_root': '/mnt/bh1/scratch/emmanuel.bengio/data/cliques',
            'train_ratio': 0.99,  ####
        },
        {
            **_hps,
            'num_training_steps': 4000,
            'log_dir': './tmp/run_gfn_offline_17p',
            'global_batch_size': 256 // 20,
            'learning_rate': 1e-4,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'num_data_loader_workers': 8,
            'i2h_width': 4,
            'reward_func': 'count',
            'test_split_type': 'subtrees',
            'data_root': '/rxrx/data/user/emmanuel.bengio/data/cliques',
            'train_ratio': 0.9,
            'offline_ratio': 1.0,
            'do_supervised': False,
            'tb_correct_idempotent': True,
            'validate_every': 100,
        },
        {
            **_hps,
            'num_training_steps': 4000,
            'log_dir': './tmp/run_gfn_offline_18p',
            'global_batch_size': 256 // 20,
            'learning_rate': 1e-4,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'num_data_loader_workers': 8,
            'i2h_width': 4,
            'reward_func': 'count',
            'test_split_type': 'subtrees',
            'data_root': '/rxrx/data/user/emmanuel.bengio/data/cliques',
            'train_ratio': 0.9,
            'offline_ratio': 1.0,
            'do_supervised': False,
            'tb_do_subtb': False,
            'tb_correct_idempotent': True,
            'validate_every': 100,
        },
        {
            **_hps,
            'num_training_steps': 4000,
            'log_dir': './tmp/run_gfn_offline_19p',
            'global_batch_size': 256 // 20,
            'learning_rate': 1e-4,
            'num_layers': 8,
            'num_heads': 4,
            'num_mlp_layers': 2,
            'num_data_loader_workers': 8,
            'i2h_width': 4,
            'reward_func': 'count',
            'test_split_type': 'subtrees',
            'data_root': '/rxrx/data/user/emmanuel.bengio/data/cliques',
            'train_ratio': 0.9,
            'offline_ratio': 1.0,
            'do_supervised': False,
            'tb_do_subtb': False,
            'validate_every': 100,
            'algo': 'FM',
            'fm_correct_idempotent': True,
        },
    ]

    if 1:
        import sys
        mhps = hps[int(sys.argv[1])]
        if mhps['do_supervised']:
            trial = CliquesSupervisedTrainer(mhps, torch.device('cuda'))
        else:
            trial = CliquesTrainer(mhps, torch.device('cuda'))
        trial.verbose = True
        trial.run()


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        if sys.argv[1] == '--recompute-all':
            #states = generate_two_col_data(sys.argv[2])
            states = load_two_col_data(sys.argv[2])
            env = GraphBuildingEnv()
            ctx = CliquesEnvContext(7, 4, 2, num_cond_dim=1, graph_data=states)
            epc = ExactProbCompCallback(None, states, torch.device('cpu'), ctx=ctx, env=env, do_save_px=False,
                                        log_rewards=1)
            epc.compute_cache()
            epc.save_cache(sys.argv[2] + '/two_col_epc_cache.pkl')
        else:
            raise ValueError(sys.argv)
    else:
        main()
