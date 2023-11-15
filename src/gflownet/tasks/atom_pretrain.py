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
from gflownet.utils.conditioning import Conditional
from gflownet.utils import sascore

class AtomPropConditional(Conditional):
    def __init__(self, cfg: Config, rng: np.random.Generator, props: List[str]):
        self.cfg = cfg
        self.rng = rng
        self.props = props
        self.cbs = {
            'wt': lambda mol: Descriptors.MolWt(mol),
            'logp': lambda mol: Crippen.MolLogP(mol),
            'tpsa': lambda mol: CalcTPSA(mol),
            'fsp3': lambda mol: CalcFractionCSP3(mol),
            'nrb': lambda mol: CalcNumRotatableBonds(mol),
            'rings': lambda mol: mol.GetRingInfo().NumRings(),
            'sa': lambda mol: sascore.calculateScore(mol),
        }

    def transform(self, cond_info: Dict[str, Tensor], properties: Tensor) -> Tensor:
        return torch.ones((properties.shape[0], 1), device=properties.device)

    def sample(self, n: int):
        return {}

class AtomPretrainTask(GFNTask):
    """
    """

    def __init__(
        self,
        cfg: Config,
        rng: np.random.Generator,
    ):
        self.rng = rng
        self.props = ['wt', 'logp', 'tpsa', 'fsp3', 'nrb', 'rings', 'sa']
        self.conditional = AtomPropConditional(cfg, rng, self.props)
        self.num_cond_dim = self.conditional.encoding_size()

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.conditional.transform(cond_info, flat_reward))

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        frs = torch.tensor([[self.conditional.cbs[p](mol) for p in self.props] for mol in mols], dtype=torch.float32)
        return FlatRewards(frs), torch.ones((frs.shape[0]), dtype=torch.bool)


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
        cfg.algo.max_nodes = 50
        cfg.algo.max_edges = 70
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
        self.greedy_task = copy.copy(self.task)
        # Ignore temperature for greedy task
        self.greedy_task.cond_info_to_logreward = lambda cond_info, flat_reward: RewardScalar(
            flat_reward.reshape((-1,))
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
