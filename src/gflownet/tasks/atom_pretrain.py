import os
import pathlib
import shutil
import socket
import copy
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol, ChiralType
from rdkit.Chem import Descriptors, Crippen, AllChem
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds, CalcFractionCSP3

from torch import Tensor

from gflownet.config import Config
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.tasks.seh_frag import SEHTask
from gflownet.utils.conditioning import Conditional
from gflownet.utils import sascore


class LogZConditional(Conditional):
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        tmp_cfg = self.cfg.cond.logZ
        self.upper_bound = 1024
        if tmp_cfg.sample_dist == "gamma":
            loc, scale = tmp_cfg.dist_params
            self.upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
        elif tmp_cfg.sample_dist == "uniform":
            self.upper_bound = tmp_cfg.dist_params[1]
        elif tmp_cfg.sample_dist == "loguniform":
            self.upper_bound = tmp_cfg.dist_params[1]
        elif tmp_cfg.sample_dist == "beta":
            self.upper_bound = 1

    def encoding_size(self):
        return self.cfg.cond.logZ.num_thermometer_dim

    def sample(self, n):
        cfg = self.cfg.cond.logZ
        logZ = None
        if cfg.sample_dist == "constant":
            assert type(cfg.dist_params[0]) is float
            logZ = np.array(cfg.dist_params[0]).repeat(n).astype(np.float32)
            logZ_enc = torch.zeros((n, cfg.num_thermometer_dim))
        else:
            if cfg.sample_dist == "gamma":
                loc, scale = cfg.dist_params
            elif cfg.sample_dist == "uniform":
                a, b = float(cfg.dist_params[0]), float(cfg.dist_params[1])
                logZ = self.rng.uniform(a, b, n).astype(np.float32)
            elif cfg.sample_dist == "loguniform":
                low, high = np.log(cfg.dist_params)
                logZ = np.exp(self.rng.uniform(low, high, n).astype(np.float32))
            elif cfg.sample_dist == "beta":
                a, b = float(cfg.dist_params[0]), float(cfg.dist_params[1])
                logZ = self.rng.beta(a, b, n).astype(np.float32)
            # logZ_enc = thermometer(torch.tensor(logZ), cfg.num_thermometer_dim, 0, self.upper_bound)
            logZ_enc = self.encode(logZ)

        return {"encoding": logZ_enc}

    def transform(self, cond_info: Dict[str, Tensor], linear_reward: Tensor) -> Tensor:
        return linear_reward

    def encode(self, conditional: Tensor) -> Tensor:
        cfg = self.cfg.cond.logZ
        if cfg.sample_dist == "constant":
            return torch.zeros((conditional.shape[0], cfg.num_thermometer_dim))
        enc = thermometer(torch.tensor(conditional), cfg.num_thermometer_dim - 1, 0, self.upper_bound)
        return torch.cat([torch.tensor(conditional).unsqueeze(-1), enc], dim=1)


class AtomPropConditional(Conditional):
    def __init__(self, cfg: Config, rng: np.random.Generator, props: List[str]):
        self.cfg = cfg
        self.rng = rng
        self.props = props
        self.cbs = {
            "wt": lambda mol: Descriptors.MolWt(mol),
            "logp": lambda mol: Crippen.MolLogP(mol),
            "tpsa": lambda mol: CalcTPSA(mol),
            "fsp3": lambda mol: CalcFractionCSP3(mol),
            "nrb": lambda mol: CalcNumRotatableBonds(mol),
            "rings": lambda mol: mol.GetRingInfo().NumRings(),
            "sa": lambda mol: sascore.calculateScore(mol),
        }
        self.bounds = {p: getattr(cfg.cond.atom_prop, p + "_bounds") for p in self.props}

    def transform(self, cond_info: Dict[str, Tensor], properties: Tensor) -> Tensor:
        return torch.ones((properties.shape[0], 1), device=properties.device)

    def sample(self, n: int):
        return {}


class AtomPretrainTask(GFNTask):
    """ """

    def __init__(
        self,
        cfg: Config,
        rng: np.random.Generator,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self.rng = rng
        self.props = ["wt", "logp", "tpsa", "fsp3", "nrb", "rings", "sa"]
        self.task = cfg.task.atom_pt.task
        if self.task == "props":
            self.conditional = AtomPropConditional(cfg, rng, self.props)
        else:
            self.conditional = LogZConditional(cfg, rng)
        if self.task == "seh":
            self.seh = SEHTask([], cfg, rng, wrap_model)
            assert cfg.cond.temperature.sample_dist == "constant", "Chained conditionals not implemented yet"
        self.num_cond_dim = self.conditional.encoding_size()

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if self.task == "props":
            return RewardScalar(self.conditional.transform(cond_info, flat_reward))
        elif self.task == "const":
            return flat_reward.flatten()
        elif self.task == "seh":
            return self.seh.cond_info_to_logreward(cond_info, flat_reward)

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        if self.task == "props":
            frs = torch.tensor(
                [[self.conditional.cbs[p](mol) for p in self.props] for mol in mols], dtype=torch.float32
            )
            return FlatRewards(frs), torch.ones((frs.shape[0]), dtype=torch.bool)
        elif self.task == "seh":
            return self.seh.compute_flat_rewards(mols)
        else:
            return torch.zeros((len(mols), 1)), torch.ones((len(mols),), dtype=torch.bool)


class ChemblDataset:
    def __init__(self, ctx, train: bool = True, n=10_000, split_seed=142857):
        self.smis, self.seh_score = pickle.load(
            open("/mnt/ps/home/CORP/emmanuel.bengio/project/data/chembl_sorted_seh.pkl", "rb")
        )
        rng = np.random.RandomState(split_seed)
        self.idcs = np.arange(len(smis))
        rng.shuffle(self.idcs)
        if train:
            self.idcs = self.idcs[:n]
        else:
            self.idcs = self.idcs[-n:]
        self.ctx = ctx

    def __len__(self):
        return len(self.idcs)

    def __getitem__(self, idx):
        return (
            self.ctx.mol_to_graph(Chem.MolFromSmiles(self.smis[self.idcs[idx]])),
            self.seh_score[self.idcs[idx]] * 8,
        )


class AtomPretrainTrainer(StandardOnlineTrainer):
    task: AtomPretrainTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = True
        cfg.num_workers = 8
        cfg.checkpoint_every = 1000
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 64
        cfg.algo.offline_ratio = 0
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 75
        cfg.algo.max_edges = 90
        cfg.algo.max_len = 100
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -256
        cfg.algo.train_random_action_prob = 0.01
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.valid_offline_ratio = 0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

    def setup_task(self):
        self.task = AtomPretrainTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_env_context(self):
        self.ctx = MolBuildingEnvContext(
            ["C", "N", "O", "S", "F", "Cl", "Br"],
            charges=[0],
            chiral_types=[ChiralType.CHI_UNSPECIFIED],
            num_rw_feat=0,
            max_nodes=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            # allow_5_valence_nitrogen=True,  # We need to fix backward trajectories to use masks!
            # And make sure the Nitrogen-related backward masks make sense
        )
        if hasattr(self.ctx, "graph_def"):
            self.env.graph_cls = self.ctx.graph_cls


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        "log_dir": f"./logs/atom_pt/run_debug/",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 2000,
        "validate_every": 0,
        "num_workers": 4,
        "opt": {
            "lr_decay": 20000,
        },
        "algo": {
            "illegal_action_logreward": -512,
            "sampling_tau": 0.99,
            "global_batch_size": 128,
            "tb": {"variant": "TB"},
        },
        "cond": {
            "temperature": {
                "sample_dist": "constant",
                "dist_params": [64.0],
            }
        },
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = AtomPretrainTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
