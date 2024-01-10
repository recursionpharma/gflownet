import bz2
import os
import shutil
import socket
import pickle
import pathlib
import distance
from typing import Dict, List, Tuple
from omegaconf import OmegaConf
from itertools import product
from collections import namedtuple

import networkx as nx
import numpy as np
import torch
import torch_geometric.data as gd
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_scatter import scatter, scatter_logsumexp, scatter_sum
from tqdm import tqdm


from gflownet.algo.config import TBVariant
from gflownet.algo.flow_matching import FlowMatching
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.config import Config
from gflownet.envs.seq_building_env import Seq, AutoregressiveSeqBuildingContext, SeqBuildingEnv
from gflownet.models.seq_transformer import SeqTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer, GFNTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional, LogZConditional
from gflownet.envs.graph_building_env import (
    Graph,
    GraphAction,
    GraphActionCategorical,
    GraphActionType,
    GraphBuildingEnv,
)


def hamming(s1, s2):
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def edit_distance(s1, s2):
    return distance.levenshtein(s1, s2)


def old_edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def get_seq_modes(states, num_modes=60, seed=142857):
    rng = np.random.default_rng(seed)
    modes_idcs = rng.choice(np.arange(0, len(states), 1), size=num_modes, replace=False)
    return [states[m] for m in modes_idcs]


def seq_reward(s, max_len=9):
    seqs = ["aa", "bb", "cc"]
    norm = max_len / min(map(len, seqs))
    return np.float32(sum([s.count(p) for p in seqs])) / norm


def edit_reward(s, modes, max_len=7):
    ds = [edit_distance(s, m) for m in modes]
    # return (1.0 - np.float32(min(ds))) / max_len # log rewards of exp((1 - d) / n)
    return (-np.float32(min(ds))) / max_len * 10 * 4  # log rewards of exp((-d) / n) * 10


def generate_seq_data(data_root, max_len=9, syms=["0", "1"]):
    all_seqs = sum([list(product(syms, repeat=n)) for n in range(max_len + 1)], [])
    return all_seqs


def old_generate_seq_data(data_root, max_len=9, syms=["0", "1"]):
    seq_objs = []
    for l in range(max_len):
        obj = list(product(syms, repeat=l + 1))
        [seq_objs.append("".join(s)) for s in obj]
    if data_root is None:
        return seq_objs
    else:
        with bz2.open(data_root + f"/toy_seq_{max_len}_objs.pkl.bz", "wb") as f:
            pickle.dump(seq_objs, f)
        return seq_objs


def load_seq_data(data_root, max_len=7, generate_if_missing=True):
    if data_root is None:
        return generate_seq_data(data_root, max_len=max_len)
    else:
        p = data_root + f"/toy_seq_{max_len}_objs.pkl.bz"
        print("Loading", p)
        if not os.path.exists(p) and generate_if_missing:
            return generate_seq_data(data_root, max_len=max_len)
        with bz2.open(p, "rb") as f:
            data = pickle.load(f)
        return data


class SeqTrajectoryBalance(TrajectoryBalance):
    # The graphs here are actually Seq objects
    def create_training_data_from_graphs(self, graphs):
        trajs = []
        for g in graphs:
            trajs.append(
                {
                    "traj": [
                        (Seq(g.seq[:i]), GraphAction(GraphActionType.AddNode, value=g.seq[i]))
                        for i in range(len(g.seq))
                    ]
                    + [(g, GraphAction(GraphActionType.Stop))],
                    "is_valid": True,
                    "is_sink": [0] * len(g.seq) + [1],
                    "bck_logprobs": torch.tensor([0] * len(g.seq) + [0]).to(self.ctx.device),
                    "result": g,
                }
            )
        return trajs


class SeqDataset(Dataset):
    def __init__(
        self,
        data,
        ctx,
        train=True,
        output_graphs=False,
        split_seed=142857,
        ratio=0.9,
        max_len=7,
        reward_func="edit",
        reward_reshape: bool = False,
        reward_corrupt: bool = False,
        reward_shuffle: bool = False,
        reward_temper: bool = False,
        reward_skewed_random: bool = False,
        reward_param: float = 0.0,
        regress_to_F: bool = False,
        compute_Fsa: bool = False,
        compute_normalized_Fsa: bool = False,
    ):
        self.data = data
        self.ctx = ctx
        self.output_graphs = output_graphs
        self.reward_func = reward_func
        self.reward_reshape = reward_reshape
        self.reward_corrupt = reward_corrupt
        self.reward_shuffle = reward_shuffle
        self.reward_temper = reward_temper
        self.reward_skewed_random = reward_skewed_random
        self.reward_param = reward_param
        self.idcs = [0]
        self.max_len = max_len
        if data is None:
            return

        # self.compute_Fsa = False
        # self.compute_normalized_Fsa = False
        # self.regress_to_F = False
        self.regress_to_F = regress_to_F
        self.compute_Fsa = compute_Fsa
        self.compute_normalized_Fsa = compute_normalized_Fsa  # === compute_P_F

        idcs_for_split = np.arange(len(data))  # o.g.
        rng = np.random.default_rng(split_seed)
        rng.shuffle(idcs_for_split)
        if train:
            self.idcs_for_split = idcs_for_split[: int(np.floor(ratio * len(data)))]
        else:
            self.idcs_for_split = idcs_for_split[int(np.floor(ratio * len(data))) :]

        print(train, self.idcs_for_split.shape)

        # ~We only want the intermediate states~
        if self.regress_to_F or self.compute_Fsa or self.compute_normalized_Fsa:
            # self.idcs = np.int32([i for i in range(len(self.data)) if len(self.data[i]) < max_len])
            # Actually we want all the states
            self.idcs = np.arange(len(self.data))
        else:
            self.idcs = self.idcs_for_split

        self.modes = get_seq_modes(self.data, num_modes=60, seed=split_seed)

        # pre-compute log_rewards and apply selected reward trasnformation(s)
        log_rewards = self.pre_compute_rewards()
        self.pre_computed_log_rewards = log_rewards

        self.adjusted_log_rewards, adjusted_log_rewards = None, log_rewards
        if self.reward_reshape:
            adjusted_log_rewards = self.monotonic_skew_reward_values(adjusted_log_rewards, lam=self.reward_param)
        if self.reward_corrupt:
            adjusted_log_rewards = self.corrupt_reward_values(adjusted_log_rewards, std=self.reward_param)
        if self.reward_shuffle:
            adjusted_log_rewards = self.shuffle_reward_values(adjusted_log_rewards)
        if self.reward_temper:
            adjusted_log_rewards = self.temper_reward_values(adjusted_log_rewards, beta=self.reward_param)
        if self.reward_skewed_random:
            adjusted_log_rewards = self.skewed_random_values(
                size=len(adjusted_log_rewards), sparse_reward=self.reward_param
            )

        if self.reward_reshape or self.reward_corrupt or self.reward_shuffle:
            self.adjusted_log_rewards = adjusted_log_rewards

        # compute MDP
        if train:
            self.mdp = nx.MultiDiGraph()
            self.s2id = {}  # can use this to lookup F(s)
            self.s2id[tuple()] = 0
            self.mdp.add_node(0, s="", r=0)
            print("\n Computing MDP ... ")
            Z = self.compute_flows([], 0)
            print("logZ:", np.log(Z))
            print("... MDP done \n")
            self.epc = namedtuple("epc", ["mdp_graph"])(self.mdp)
            self.is_doing_seq = True

        self._gc = nx.complete_graph(7)
        self._enum_edges = list(self._gc.edges)

    def __len__(self):
        return len(self.idcs)

    def reward(self, g):
        if self.adjusted_log_rewards is not None:
            g_idx = self.get_graph_idx(g, self.data)
            return self.adjusted_log_rewards[g_idx]
        # else:
        #    return self.reward_type(g)
        else:
            g_idx = self.get_graph_idx(g, self.data)
            return self.pre_computed_log_rewards[g_idx]

    def reward_type(self, g):
        if self.reward_func == "edit":
            return edit_reward(g, self.modes, self.max_len)
        else:
            return -100

    def monotonic_skew_reward_values(self, log_rewards, lam=0.1):
        """
        Apply monotonic trasnformation on reward values
        """
        return self.adjust_reward_skew(log_rewards, lam)

    def corrupt_reward_values(self, log_rewards, std=1.0):
        """
        Corrupt reward values with noised. Used to
        emulate "Rethinking Generalization" experiments, but for
        GFlowNets
            TODO:
                - Currently only for Guassian noise.
                  Could add implementation for Laplace and others.
                - Currently noise is just over one seed
        """
        if std <= 0.0:
            return log_rewards
        rng = np.random.default_rng(12345)
        noise = rng.normal(loc=0.0, scale=std, size=np.array(log_rewards).shape)
        return list(log_rewards + noise)

    def shuffle_reward_values(self, log_rewards):
        """
        Shuffles reward value pairing for given graphs. Used to
        emulate "Rethinking Generalization" experiments, but for
        GFlowNets
        """
        rng = np.random.default_rng(12345)
        aranged_ids = np.arange(start=0, stop=len(log_rewards))
        rand_ids = rng.choice(aranged_ids, size=aranged_ids.shape, replace=False)
        shuffled_log_rewards = np.array(log_rewards)[rand_ids]
        return list(shuffled_log_rewards)

    def temper_reward_values(self, log_rewards, beta=1.0):
        """
        Temper rewards for pre-computed log_rewards.
        """
        return list(np.array(log_rewards) * (1.0 / beta))

    def skewed_random_values(self, size_log_rewards, sparse_reward=0.0):
        """
        Defines random log-rewards sampled from Rayleigh dsitribution.
        Emulates log-reward skew to high and low rewards. 'Sparser' rewards
        skew log-reward distribution to higher mass around lower rewards.
        """
        rng = np.random.default_rng(12345)
        if sparse_reward > 0.0:
            x = rng.rayleigh(2.6, size=size_log_rewards) - 10
            idcs = x > 0
            x[idcs] = 0
            idcs = x < -10
            x[idcs] = -10
        else:
            x = -rng.rayleigh(2.6, size=size_log_rewards)
            idcs = x > 0
            x[idcs] = 0
            idcs = x < -10
            x[idcs] = -10
        return x

    def adjust_reward_skew(self, log_rewards, lam=0.1):
        """
        Skew the reward function towards favouring higher reward
        values.
        """
        r_bins = list(set(log_rewards))
        mono_weights = np.exp(-lam * np.array(r_bins))
        log_rewards_skew = []

        for r in log_rewards:
            i = np.where(r_bins == r)[0][0]
            log_rewards_skew.append(mono_weights[i] * r)

        log_rewards_skew = np.array(log_rewards_skew) / np.min(log_rewards_skew) * np.min(r_bins)
        return list(log_rewards_skew)

    def get_graph_idx(self, g, states, default=None):
        h = hash(g)
        if h not in self._hash_to_objs:
            if default is not None:
                return default
            else:
                print("Object not found in cache", h, g)
        bucket = self._hash_to_objs[h]
        if len(bucket) == 1:
            return bucket[0]
        for i in bucket:
            return i
        if default is not None:
            return default
        raise ValueError(g)

    def hash_for_objs(self):
        states = self.data
        _hash_to_objs = {}
        states_hash = [hash(i) for i in tqdm(states, disable=True)]
        for i, h in enumerate(states_hash):
            _hash_to_objs[h] = _hash_to_objs.get(h, list()) + [i]
        return _hash_to_objs

    def pre_compute_rewards(self):
        self._hash_to_objs = self.hash_for_objs()
        rewards = [self.reward_type(self.data[self.get_graph_idx(g, self.data)]) for g in tqdm(self.data)]
        return rewards

    def compute_flows(self, seq, parent):
        flow = r = self.mdp.nodes[parent]["r"]
        for i, token in enumerate(self.ctx.alphabet):
            n = len(self.mdp)
            new_seq = seq + [token]
            self.s2id[tuple(new_seq)] = n
            child_r = np.exp(self.reward(tuple(new_seq)))
            self.mdp.add_node(n, s="".join(new_seq), r=child_r)
            self.mdp.add_edge(parent, n, a=(1, 0, i))
            if len(new_seq) < self.max_len:
                edge_flow = self.compute_flows(new_seq, n)
            else:
                edge_flow = child_r
                self.mdp.add_edge(n, n, a=(0, 0, 0), F=np.log(child_r))
                self.mdp.nodes[n]["F"] = np.log(child_r)
            self.mdp.edges[(parent, n, 0)]["F"] = np.log(edge_flow)
            flow += edge_flow
        self.mdp.add_edge(parent, parent, a=(0, 0, 0), F=r)
        self.mdp.nodes[parent]["F"] = np.log(flow)
        return flow

    def collate_fn(self, batch):
        graphs, rewards, idcs = zip(*batch)
        batch = self.ctx.collate(graphs)
        if self.regress_to_F:
            batch.y = torch.as_tensor([self.epc.mdp_graph.nodes[i]["F"] for i in idcs])
        elif self.compute_Fsa:
            all_targets = []
            for data_idx in idcs:
                if self.is_doing_seq:
                    targets = [torch.zeros((1, n)) - 100 for n in [1, len(self.ctx.alphabet)]]  # Stop, AddNode
                else:
                    targets = [
                        torch.zeros_like(getattr(self.epc._Data[data_idx], i.mask_name)) - 100
                        for i in self.ctx.action_type_order
                    ]
                for neighbor in list(self.epc.mdp_graph.neighbors(data_idx)):
                    for _, edge in self.epc.mdp_graph.get_edge_data(data_idx, neighbor).items():
                        a, F = edge["a"], edge["F"]
                        targets[a[0]][a[1], a[2]] = F
                if self.compute_normalized_Fsa:
                    logZ = torch.log(sum([i.exp().sum() for i in targets]))
                    targets = [i - logZ for i in targets]
                all_targets.append(targets)
            batch.y = torch.cat([torch.cat(i).flatten() for i in zip(*all_targets)])
        else:
            batch.y = torch.as_tensor(rewards)
        return batch

    def __getitem__(self, idx):
        idx = self.idcs[idx]
        g = self.data[idx]
        r = torch.tensor(self.reward(g)).reshape((1,))
        if self.is_doing_seq:
            idx = self.s2id[tuple(g)]
            seq = Seq()
            seq.seq = [self.ctx.alphabet.index(i) for i in g]
            g = seq
        if self.output_graphs:
            return self.ctx.graph_to_Data(g), r, idx
        else:
            return g, r

    def old_collate_fn(self, batch):
        graphs, rewards, idcs = zip(*batch)
        batch = self.ctx.collate(graphs)
        if self.regress_to_F:
            batch.y = torch.as_tensor([self.epc.mdp_graph.nodes[i]["F"] for i in idcs])
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
                        a, F = edge["a"], edge["F"]
                        targets[a[0]][a[1], a[2]] = F
                if self.compute_normalized_Fsa:
                    logZ = torch.log(sum([i.exp().sum() for i in targets]))
                    targets = [i - logZ for i in targets]
                all_targets.append(targets)
            batch.y = torch.cat([torch.cat(i).flatten() for i in zip(*all_targets)])
        return batch

    def old__getitem__(self, idx):
        idx = self.idcs[idx]
        g = self.data[idx]
        r = torch.tensor(self.reward(g).reshape((1,)))
        if self.output_graphs:
            return self.ctx.graph_to_Data(g), r, idx
        else:
            return g, r


class ToySeqTask(GFNTask):
    def __init__(
        self,
        cfg: Config,
        dataset: SeqDataset,
        rng: np.random.Generator = None,
    ):
        self.dataset = dataset
        self.cfg = cfg
        self.rng = rng
        self.logZ_conditional = LogZConditional(cfg, rng)

    def flat_reward_transform(self, y: Tensor) -> FlatRewards:
        return FlatRewards(y.float())

    def sample_conditional_information(self, n: int, train_it: int = 0):
        if self.cfg.cond.logZ.sample_dist is not None:
            return self.logZ_conditional.sample(n)
        else:
            return {"encoding": torch.zeros((n, 1))}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(flat_reward[:, 0].float())

    def compute_flat_rewards(self, mols: List[Graph]) -> Tuple[FlatRewards, Tensor]:
        if not len(mols):
            return FlatRewards(torch.zeros((0, 1))), torch.zeros((0,)).bool()
        is_valid = torch.ones(len(mols)).bool()
        flat_rewards = torch.tensor([self.dataset.reward(tuple([*i])) for i in mols]).float().reshape((-1, 1))
        return FlatRewards(flat_rewards), is_valid

    def encode_conditional_information(self, info):
        if self.cfg.cond.logZ.sample_dist is not None:
            encoding = self.logZ_conditional.encode(info)
            return {
                "beta": torch.ones(len(info)),
                "encoding": encoding.float(),
                "preferences": torch.tensor(info).float(),
            }
        else:
            encoding = torch.zeros((len(info), 1))
            return {"beta": torch.ones(len(info)), "encoding": encoding.float(), "preferences": info.float()}


class OldToySeqTask(GFNTask):
    """Sets up a task where the reward is the number of times some sequences appear in the input. Normalized to be
    in [0,1]"""

    def __init__(
        self,
        cfg: Config,
        dataset: SeqDataset,
        rng: np.random.Generator,
    ):
        self.dataset = dataset
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def flat_reward_transform(self, y: Tensor) -> FlatRewards:
        return FlatRewards(y.float())

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, flat_reward))

    def compute_flat_rewards(self, objs: List[str]) -> Tuple[FlatRewards, Tensor]:
        if not len(objs):
            return FlatRewards(torch.zeros((0, 1))), torch.zeros((0,)).bool()
        is_valid = torch.ones(len(objs)).bool()
        flat_rewards = torch.tensor([self.dataset.reward(tuple([*i])) for i in objs]).float().reshape((-1, 1))
        return FlatRewards(flat_rewards), is_valid

    def encode_conditional_information(self, info):
        encoding = torch.zeros((len(info), 1))
        return {"beta": torch.ones(len(info)), "encoding": encoding.float(), "preferences": info.float()}


class ToySeqTrainer(GFNTrainer):  # o.g. inheritence from StandardOnlineTrainer
    task: ToySeqTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.opt.learning_rate = 3e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "none"
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 32
        cfg.algo.offline_ratio = 0.  # This works now, incl 0.0-1.0
        cfg.model.num_emb = 64
        cfg.model.num_layers = 4

        # This seems to work: self.cfg.task.toy_seq.train_ratio = 0.75

        cfg.algo.method = "TB"
        # cfg.algo.max_nodes = cfg.algo.max_nodes
        # cfg.algo.max_len = cfg.algo.max_len
        cfg.algo.sampling_tau = 0.0
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-2  # This is not being respected
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False
        cfg.algo.tb.do_correct_idempotent = False  # No idepotent correction for append oly MDPs

    def setup(self):
        assert self.cfg.algo.max_nodes + 1 == self.cfg.algo.max_len, "For implementation reasons this needs to be true"
        mcfg = self.cfg.task.toy_seq
        self.log_sampling_g_distribution = None
        self.rng = np.random.default_rng(self.cfg.seed)

        self.env = SeqBuildingEnv("")
        if mcfg.reward_func == "edit":
            self.ctx = AutoregressiveSeqBuildingContext(["0", "1"], num_cond_dim=1, max_len=self.cfg.algo.max_nodes)
        else:
            raise ValueError("Invalid reward function")
        self._do_supervised = self.cfg.task.toy_seq.do_supervised

        self._data = load_seq_data(None, max_len=self.cfg.algo.max_nodes)

        self.training_data = SeqDataset(
            self._data,
            self.ctx,
            train=True,
            ratio=mcfg.train_ratio,
            split_seed=mcfg.test_split_seed,
            max_len=self.cfg.algo.max_nodes,
            reward_func=mcfg.reward_func,
            reward_reshape=mcfg.reward_reshape,
            reward_corrupt=mcfg.reward_corrupt,
            reward_shuffle=mcfg.reward_shuffle,
            reward_temper=mcfg.reward_temper,
            reward_param=mcfg.reward_param,
            regress_to_F=mcfg.regress_to_F,
            compute_Fsa=mcfg.regress_to_Fsa,
            compute_normalized_Fsa=mcfg.regress_to_Fsa,
        )
        self.test_data = SeqDataset(
            self._data,
            self.ctx,
            train=False,
            ratio=mcfg.train_ratio,
            split_seed=mcfg.test_split_seed,
            max_len=self.cfg.algo.max_nodes,
            reward_func=mcfg.reward_func,
            reward_reshape=mcfg.reward_reshape,
            reward_corrupt=mcfg.reward_corrupt,
            reward_shuffle=mcfg.reward_shuffle,
            reward_temper=mcfg.reward_temper,
            reward_param=mcfg.reward_param,
            regress_to_F=mcfg.regress_to_F,
            compute_Fsa=mcfg.regress_to_Fsa,
            compute_normalized_Fsa=mcfg.regress_to_Fsa,
        )

        self.exact_prob_cb = ExactSeqProbCompCallback(
            self,
            self.training_data.data,
            self.training_data.mdp,
            self.cfg.algo.max_nodes,
            self.device,
            log_rewards=self.training_data.adjusted_log_rewards
            if mcfg.reward_reshape or mcfg.reward_corrupt or mcfg.reward_shuffle
            else None,
            logits_shuffle=mcfg.logits_shuffle,
        )

        model = SeqTransformerGFN(
            self.ctx,
            self.cfg,
        )

        self.task = ToySeqTask(
            cfg=self.cfg,
            dataset=self.training_data,
            rng=self.rng,
        )

        self.model = self.sampling_model = model

        params = [i for i in self.model.parameters()]
        if self.cfg.opt.opt == "adam":
            self.opt = torch.optim.Adam(
                params,
                self.cfg.opt.learning_rate,
                (self.cfg.opt.momentum, 0.999),
                weight_decay=self.cfg.opt.weight_decay,
                eps=self.cfg.opt.adam_eps,
            )
        elif self.cfg.opt.opt == "SGD":
            self.opt = torch.optim.SGD(
                params, self.cfg.opt.learning_rate, self.cfg.opt.momentum, weight_decay=self.cfg.opt.weight_decay
            )
        elif self.cfg.opt.opt == "RMSProp":
            self.opt = torch.optim.RMSprop(params, self.cfg.opt.learning_rate, weight_decay=self.cfg.opt.weight_decay)
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2 ** (-steps / self.cfg.opt.lr_decay))

        algo = self.cfg.algo.method
        if algo == "TB" or algo == "subTB":
            self.algo = SeqTrajectoryBalance(self.env, self.ctx, self.rng, self.cfg)
        elif algo == "FM":
            self.algo = FlowMatching(self.env, self.ctx, self.rng, self.cfg)

        self.algo.model_is_autoregressive = True

        # For offline training -- set p(x) to be used for sampling x ~ p(x)
        if isinstance(model, SeqTransformerGFN):
            # select use of true log_Z
            if self.cfg.algo.use_true_log_Z:
                self.cfg.algo.true_log_Z = float(self.exact_prob_cb.logZ)
            # select x ~ p(x) sampling
            if self.cfg.algo.offline_sampling_g_distribution == "log_rewards":  # x ~ R(x)/Z
                self.log_sampling_g_distribution = self.exact_prob_cb.true_log_probs
            elif self.cfg.algo.offline_sampling_g_distribution == "log_p":  # x ~ p(x; \theta)
                self.log_sampling_g_distribution = (
                    self.exact_prob_cb.compute_prob(model.to(self.cfg.device))[0].cpu().numpy()[:-1]
                )
            elif (
                self.cfg.algo.offline_sampling_g_distribution == "l2_log_error_gfn"
                or self.cfg.algo.offline_sampling_g_distribution == "l1_error_gfn"
            ):  # x ~ ||p(x; \theta) - p(x)||
                model_log_probs = self.exact_prob_cb.compute_prob(model.to(self.cfg.device))[0].cpu().numpy()[:-1]
                true_log_probs = self.exact_prob_cb.true_log_probs
                err = []
                for lq, lp in zip(model_log_probs, true_log_probs):
                    if self.cfg.algo.offline_sampling_g_distribution == "l2_log_error_gfn":
                        err.append((lq - lp) ** 2)
                    else:
                        err.append(np.abs(np.exp(lq) - np.exp(lp)))
                err = np.array(err)
                err = err / np.sum(err)
                self.log_sampling_g_distribution = np.log(err)
            elif self.cfg.algo.offline_sampling_g_distribution == "uniform":  # x ~ Unif(x)
                self.log_sampling_g_distribution = -1 * np.ones_like(
                    self.exact_prob_cb.true_log_probs
                )  # uniform distribution
            elif self.cfg.algo.offline_sampling_g_distribution == "random":
                rng = np.random.default_rng(self.cfg.seed)
                self.log_sampling_g_distribution = rng.uniform(0, 10, len(self.exact_prob_cb.true_log_probs))
            else:
                self.log_sampling_g_distribution = None
        self.sampling_tau = self.cfg.algo.sampling_tau
        self.mb_size = self.cfg.algo.global_batch_size
        self.clip_grad_param = self.cfg.opt.clip_grad_param
        self.clip_grad_callback = {
            "value": (lambda params: torch.nn.utils.clip_grad_value_(params, self.clip_grad_param)),
            "norm": (lambda params: torch.nn.utils.clip_grad_norm_(params, self.clip_grad_param)),
            "none": (lambda x: None),
        }[self.cfg.opt.clip_grad_type]

        self.algo.task = self.task
        if self.cfg.task.toy_seq.test_split_type == "random":
            pass
        elif self.cfg.task.toy_seq.test_split_type == "bck_traj":
            train_idcs, test_idcs = self.exact_prob_cb.get_bck_trajectory_test_split(self.cfg.task.toy_seq.train_ratio)
            self.training_data.idcs = train_idcs
            self.test_data.idcs = test_idcs
        elif self.cfg.task.toy_seq.test_split_type == "subtrees":
            train_idcs, test_idcs = self.exact_prob_cb.get_subtree_test_split(
                self.cfg.task.toy_seq.train_ratio, self.cfg.task.toy_seq.test_split_seed
            )
            self.training_data.idcs = train_idcs
            self.test_data.idcs = test_idcs
        if not self._do_supervised or self.cfg.task.toy_seq.regress_to_Fsa:
            self._callbacks = {"true_px_error": self.exact_prob_cb}
        else:
            self._callbacks = {}

        os.makedirs(self.cfg.log_dir, exist_ok=True)
        print("\n\nHyperparameters:\n")
        yaml = OmegaConf.to_yaml(self.cfg)
        print(yaml)
        with open(pathlib.Path(self.cfg.log_dir) / "hps.yaml", "w") as f:
            f.write(yaml)

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


class ExactSeqProbCompCallback:
    ctx: AutoregressiveSeqBuildingContext
    trial: ToySeqTrainer
    mdp_graph: nx.DiGraph

    def __init__(
        self,
        trial,
        states,
        mdp,
        max_len,
        dev,
        mbs=128,
        do_save_px=True,
        log_rewards=None,
        logits_shuffle=False,
        tqdm_disable=None,
        ctx=None,
        env=None,
    ):
        self.trial = trial
        self.ctx = trial.ctx if trial is not None else ctx
        self.env = trial.env if trial is not None else env
        self.mbs = mbs
        self.dev = dev
        self.states = states
        self.mdp_graph = mdp
        self.max_len = max_len
        # if self.cache_path is not None:
        #    self.load_cache(self.cache_path)
        if log_rewards is None:
            self.log_rewards = np.array(
                [self.trial.training_data.reward(i) for i in tqdm(self.states, disable=tqdm_disable)]
            )
        else:
            self.log_rewards = log_rewards
        self.logZ = np.log(np.sum(np.exp(self.log_rewards)))
        self.true_log_probs = self.log_rewards - self.logZ
        self.logits_shuffle = logits_shuffle
        # This is reward-dependent
        if self.mdp_graph is not None:
            self.recompute_flow()
        self.do_save_px = do_save_px
        if do_save_px:
            os.makedirs(self.trial.cfg.log_dir, exist_ok=True)
        self._save_increment = 0

    def compute_metrics(self, log_probs, state_flows, log_rewards_estimate, valid_batch_ids=None):
        log_probs = log_probs.cpu().numpy()[:-1]
        state_flows = state_flows.cpu().numpy().flatten()
        log_rewards_estimate = log_rewards_estimate.cpu().numpy().flatten()
        log_rewards = self.log_rewards
        lp, p = log_probs, np.exp(log_probs)
        lq, q = self.true_log_probs, np.exp(self.true_log_probs)
        self.trial.model_log_probs, self.trial.true_log_probs = log_probs, self.true_log_probs
        mae_log_probs = np.mean(abs(lp - lq))
        js_log_probs = (p * (lp - lq) + q * (lq - lp)).sum() / 2
        mae_log_rewards = np.mean(abs(log_rewards_estimate - log_rewards))
        print("L1 logpx error", mae_log_probs, "JS divergence", js_log_probs)

        if self.do_save_px and self.trial.cfg.cond.logZ.sample_dist is None:
            torch.save(log_probs, open(self.trial.cfg.log_dir + f"/log_px_{self._save_increment}.pt", "wb"))
            self._save_increment += 1

        metrics_dict = {}
        if valid_batch_ids is not None:
            lp_valid, p_valid = log_probs[valid_batch_ids], np.exp(log_probs[valid_batch_ids])
            lq_valid, q_valid = self.true_log_probs[valid_batch_ids], np.exp(self.true_log_probs[valid_batch_ids])
            test_mae_log_probs = np.mean(abs(lp_valid - lq_valid))
            metrics_dict["test_graphs-L1_logpx_error"] = test_mae_log_probs
            if self.trial.cfg.algo.dir_model_pretrain_for_sampling is None:
                test_mae_log_rewards = np.mean(
                    abs(log_rewards_estimate[valid_batch_ids] - log_rewards[valid_batch_ids])
                )
                metrics_dict["test_graphs-L1_log_R_error"] = test_mae_log_rewards

        metrics_dict["L1_logpx_error"] = mae_log_probs
        metrics_dict["JS_divergence"] = js_log_probs
        metrics_dict["L1_log_R_error"] = mae_log_rewards

        return metrics_dict

    def on_validation_end(self, metrics, valid_batch_ids=None):
        # Compute exact sampling probabilities of the model, last probability is p(illegal), remove it.
        if self.trial.cfg.cond.logZ.sample_dist is not None:
            logZ_true = self.logZ * torch.ones(
                (1, 1)
            )  # * torch.ones((1, self.trial.cfg.cond.logZ.num_thermometer_dim + 1)).to(self.dev)
            logZ_true_enc = self.trial.task.encode_conditional_information(logZ_true)
            cond_info = logZ_true_enc["encoding"].squeeze(0).to(self.dev)
            log_probs, state_flows, log_rewards_estimate = self.compute_prob(
                self.trial.model, cond_info=cond_info
            )  # compute once using correct logZ
            metrics_true_logZ = self.compute_metrics(log_probs, state_flows, log_rewards_estimate, valid_batch_ids)

            if self.do_save_px:
                torch.save(
                    log_probs,
                    open(
                        self.trial.cfg.log_dir + f"/log_px_val_iter_{self._save_increment}_logZ_{logZ_true.mean()}.pt",
                        "wb",
                    ),
                )

            dist_params = self.trial.cfg.cond.logZ.dist_params
            num_logZ = self.trial.cfg.cond.logZ.num_valid_logZ_samples
            metrics_range_logZ = {k: [v] for k, v in metrics_true_logZ.items()}

            for logz in np.linspace(
                dist_params[0], dist_params[1], num_logZ
            ).tolist():  # select size of range for logZ's
                logZ_sampled = logz * torch.ones(
                    (1, 1)
                )  # * torch.ones((1, self.trial.cfg.cond.logZ.num_thermometer_dim + 1)).to(self.dev)
                logZ_sampled_enc = self.trial.task.encode_conditional_information(logZ_sampled)
                cond_info = logZ_sampled_enc["encoding"].squeeze(0).to(self.dev)
                log_probs, state_flows, log_rewards_estimate = self.compute_prob(self.trial.model, cond_info=cond_info)
                metrics_tmp = self.compute_metrics(log_probs, state_flows, log_rewards_estimate, valid_batch_ids)

                if self.do_save_px:
                    torch.save(
                        log_probs,
                        open(self.trial.cfg.log_dir + f"/log_px_val_iter_{self._save_increment}_logZ_{logz}.pt", "wb"),
                    )

                for k in metrics_range_logZ.keys():
                    metrics_range_logZ[k].append(metrics_tmp[k])

            for k, v in metrics_range_logZ.items():
                metrics[k] = np.array(v)

            if self.do_save_px:
                self._save_increment += 1

        else:
            log_probs, state_flows, log_rewards_estimate = self.compute_prob(self.trial.model)
            metrics_pre = self.compute_metrics(log_probs, state_flows, log_rewards_estimate, valid_batch_ids)
            for k, v in metrics_pre.items():
                metrics[k] = np.array(v)

    def get_graph_idx(self, g, default=None):
        h = hash(g)
        if h not in self._hash_to_graphs:
            if default is not None:
                return default
            else:
                print("Object not found in cache", h, g)
        bucket = self._hash_to_graphs[h]
        if len(bucket) == 1:
            return bucket[0]
        for i in bucket:
            return i
        if default is not None:
            return default
        raise ValueError(g)

    def get_data_batch_actions(self, s):
        # If the string is max_len then there's no stop action
        is_max_len = len(s) == self.max_len

        seq = Seq()
        seq.seq = [self.ctx.alphabet.index(i) for i in s]
        true_seq = seq.seq
        if is_max_len:
            seq.seq = seq.seq[:-1]  # the last action is adding the last token, so no need for that state
            # because the mask is going to make p(stop) = 1 anyways
        data = self.ctx.graph_to_Data(seq)
        actions = torch.zeros(len(seq.seq) + 1, 3)
        tokens = torch.tensor(true_seq)
        actions[: len(tokens), 0] = 1
        actions[: len(tokens), 2] = tokens
        if not is_max_len:
            actions[len(seq.seq), 0] = 0
        return data, actions

    def compute_prob(self, model, cond_info=None, tqdm_disable=None):
        # +1 to count illegal actions prob (may not be applicable to well-masked envs)
        prob_of_being_t = torch.zeros(len(self.states) + 1).to(self.dev) - 100
        prob_of_being_t[0] = 0
        prob_of_ending_t = torch.zeros(len(self.states) + 1).to(self.dev) - 100
        state_log_flows = torch.zeros((len(self.states), 1)).to(self.dev)
        log_rewards_estimate = torch.zeros((len(self.states), 1)).to(self.dev)
        if cond_info is None:
            cond_info = torch.zeros((self.mbs, self.ctx.num_cond_dim)).to(self.dev)
        if cond_info.ndim == 1:
            cond_info = cond_info[None, :] * torch.ones((self.mbs, 1)).to(self.dev)
        if cond_info.ndim == 2 and cond_info.shape[0] == 1:
            cond_info = cond_info * torch.ones((self.mbs, 1)).to(self.dev)
        # Note: visiting the states in order works because the ordering here is a natural topological sort.
        # Wrong results otherwise.

        # all_seqs = [torch.tensor(list(map(int, [*s]))) for s in self.states]
        for bi in tqdm(range(0, len(self.states), self.mbs), disable=tqdm_disable):
            bs = self.states[bi : bi + self.mbs]
            all_term_states, all_actions = zip(*[self.get_data_batch_actions(i) for i in bs])
            batch = self.ctx.collate(all_term_states).to(self.dev)
            actions = torch.cat(all_actions, 0)

            with torch.no_grad():
                cat, o = model(batch, torch.zeros((len(batch.lens)), 1).to(self.dev), batched=True)
                # cat, o = model(batch, torch.zeros((batch.lens.sum(), 1)).to(self.dev), batched=True) # leads to [1, 128, 64] x [1, 777, 64]
            lp_of_steps = cat.log_prob(actions)
            lp_of_ending = scatter_sum(
                lp_of_steps, torch.arange(len(batch.seqs)).to(self.dev).repeat_interleave(batch.lens)
            )

            prob_of_ending_t[bi : bi + len(bs)] = lp_of_ending
            # print(o)
            # print(o.shape)
            # print(cat.logsoftmax()[0].shape)
            # print(batch.lens)
            # print(batch.lens.shape)
            # print(o[batch.lens])
            # print(o[batch.lens].shape)
            # final_graph_idx = torch.cumsum(batch.lens, 0) - 1 # maybe need this?
            state_log_flows[bi : bi + len(bs)] = o[batch.lens]
            log_rewards_estimate[bi : bi + len(bs)] = o[batch.lens] + (cat.logsoftmax()[0])[batch.lens]

        # print("\n Full probs")
        # print(prob_of_ending_t.exp())
        # print(prob_of_ending_t.exp().sum())

        return prob_of_ending_t, state_log_flows, log_rewards_estimate

    def recompute_flow(self, tqdm_disable=None):
        g = self.mdp_graph
        if self.logits_shuffle:
            rng = np.random.default_rng(seed=142857)
            for i in g:
                g.nodes[i]["F"] = -100
            for i in tqdm(list(range(len(g)))[::-1], disable=tqdm_disable):
                p = sorted(list(g.predecessors(i)), reverse=True)
                num_back = len([n for n in p if n != i])
                for j in p:
                    if j == i:
                        g.nodes[j]["F"] = rng.uniform(-10, 0)
                        g.edges[(i, i, 0)]["F"] = rng.uniform(-10, 0)
                    else:
                        # backflow = np.log(np.exp(g.nodes[i]["F"]) / num_back)
                        g.nodes[j]["F"] = rng.uniform(-10, 0)
                        # Here we're making a decision to split flow backwards equally for idempotent actions
                        # from the same state. I think it's ok?
                        ed = g.get_edge_data(j, i)
                        for k, vs in ed.items():
                            g.edges[(j, i, k)]["F"] = rng.uniform(-10, 0)
        else:
            for i in g:
                g.nodes[i]["F"] = -100
            for i in tqdm(list(range(len(g)))[::-1], disable=tqdm_disable):
                p = sorted(list(g.predecessors(i)), reverse=True)
                num_back = len([n for n in p if n != i])
                for j in p:
                    if j == i:
                        g.nodes[j]["F"] = np.logaddexp(g.nodes[j]["F"], self.log_rewards[j])
                        g.edges[(i, i, 0)]["F"] = self.log_rewards[j].item()
                    else:
                        backflow = np.log(np.exp(g.nodes[i]["F"]) / num_back)
                        g.nodes[j]["F"] = np.logaddexp(g.nodes[j]["F"], backflow)
                        # Here we're making a decision to split flow backwards equally for idempotent actions
                        # from the same state. I think it's ok?
                        ed = g.get_edge_data(j, i)
                        for k, vs in ed.items():
                            g.edges[(j, i, k)]["F"] = np.log(np.exp(backflow) / len(ed))

    def get_subtree_test_split(self, r, seed=142857):
        cache_path = f"{self.cache_root}/subtree_split_{r}_{seed}.pkl"
        if self.cache_root is not None:
            if os.path.exists(cache_path):
                return pickle.load(open(cache_path, "rb"))
        test_set = set()
        n = int((1 - r) * len(self.states))
        np.random.seed(seed)
        start_states_idx, available_start_states, start_states = [], [], []
        edge_limit = 11
        while len(test_set) < n:
            num_ss = len([i for i in start_states_idx if i not in test_set])
            if num_ss == 0 or len(available_start_states) == 0:
                start_states, start_states_idx = zip(
                    *[(s0, i) for i, s0 in enumerate(self.states) if len(s0.nodes) == 6 and len(s0.edges) >= edge_limit]
                )
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
                actions = [
                    (u, a.item(), b.item())
                    for u, ra in enumerate(self.ctx.action_type_order)
                    if ra != GraphActionType.Stop
                    for a, b in getattr(self._Data[i], ra.mask_name).nonzero()
                ]
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
            pickle.dump((np.array(train_set), np.array(test_set)), open(cache_path, "wb"))
        return train_set, test_set


def main():
    """Example of how this model can be run outside of Determined"""
    import sys

    if len(sys.argv) >= 3:
        # Example call:
        # python toy_seq.py --recompute-all ./data/toy_01_seq 13
        # specifically generates cache for "01"-symbols environment with max_nodes=13
        if sys.argv[1] == "--recompute-all":
            max_nodes = 13 if len(sys.argv) == 3 else int(sys.argv[3])
            states = load_seq_data(sys.argv[2], max_nodes, generate_if_missing=True)
            env = GraphBuildingEnv()
            ctx = AutoregressiveSeqBuildingContext("01", 1)
            epc = ExactSeqProbCompCallback(
                None, states, torch.device("cpu"), ctx=ctx, env=env, do_save_px=False, log_rewards=1
            )
            epc.compute_cache()
            epc.save_cache(sys.argv[2] + f"/toy_seq_epc_cache_{max_nodes}.pkl")
        else:
            raise ValueError(sys.argv)

    else:
        hps = {
            "log_dir": "./logs/debug_run_toy_seq",
            "device": "cuda",
            "overwrite_existing_exp": True,
            "num_training_steps": 5_000,
            "validate_every": 200,
            "checkpoint_every": 200,
            "num_workers": 0,
            "log_tags": [],
            "cond": {
                "temperature": {
                    "sample_dist": "constant",
                    "dist_params": [2.0],
                    "num_thermometer_dim": 1,
                }
            },
            "algo": {"train_random_action_prob": 0.05, "max_nodes": 5, "max_len": 6},
        }
        if os.path.exists(hps["log_dir"]):
            if hps["overwrite_existing_exp"]:
                shutil.rmtree(hps["log_dir"])
            else:
                raise ValueError(
                    f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it."
                )
        os.makedirs(hps["log_dir"])

        trial = ToySeqTrainer(hps)
        trial.print_every = 1
        trial.run()


if __name__ == "__main__":
    main()
