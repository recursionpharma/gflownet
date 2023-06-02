import ast
import copy
import json
import os
import pathlib
import shutil
import socket
from typing import Any, Callable, Dict, List, Tuple, Union

import git
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit import RDLogger
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset

from gflownet.algo.advantage_actor_critic import A2C
from gflownet.algo.soft_q_learning import SoftQLearning
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.algo.flow_matching import FlowMatching
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.models import bengio2021flow
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.train import FlatRewards, GFNTask, GFNTrainer, RewardScalar
from gflownet.utils.transforms import thermometer
from gflownet.config import config_class, Config, config_to_dict


@config_class("task.seh")
class SEHTaskConfig:
    # TODO: a proper class for temperature-conditional sampling
    temperature_sample_dist: str = "uniform"
    temperature_dist_params: List[Any] = [0.5, 32]
    num_thermometer_dim: int = 32


class SEHTask(GFNTask):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching.
    """

    def __init__(
        self,
        dataset: Dataset,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = cfg.task.seh.temperature_sample_dist
        self.temperature_dist_params = cfg.task.seh.temperature_dist_params
        self.num_thermometer_dim = cfg.task.seh.num_thermometer_dim

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y) / 8)

    def inverse_flat_reward_transform(self, rp):
        return rp * 8

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model, send_to_device=True)
        return {"seh": model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        beta = None
        if self.temperature_sample_dist == "constant":
            assert type(self.temperature_dist_params) is float
            beta = np.array(self.temperature_dist_params).repeat(n).astype(np.float32)
            beta_enc = torch.zeros((n, self.num_thermometer_dim))
        else:
            if self.temperature_sample_dist == "gamma":
                loc, scale = self.temperature_dist_params
                beta = self.rng.gamma(loc, scale, n).astype(np.float32)
                upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
            elif self.temperature_sample_dist == "uniform":
                beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == "loguniform":
                low, high = np.log(self.temperature_dist_params)
                beta = np.exp(self.rng.uniform(low, high, n).astype(np.float32))
                upper_bound = self.temperature_dist_params[1]
            elif self.temperature_sample_dist == "beta":
                beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
                upper_bound = 1
            beta_enc = thermometer(torch.tensor(beta), self.num_thermometer_dim, 0, upper_bound)

        assert len(beta.shape) == 1, f"beta should be a 1D array, got {beta.shape}"
        return {"beta": torch.tensor(beta), "encoding": beta_enc}

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            flat_reward = torch.tensor(flat_reward)
        scalar_logreward = flat_reward.squeeze().clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == len(
            cond_info["beta"].shape
        ), f"dangerous shape mismatch: {scalar_logreward.shape} vs {cond_info['beta'].shape}"
        return RewardScalar(scalar_logreward * cond_info["beta"])

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)
        preds = self.models["seh"](batch).reshape((-1,)).data.cpu()
        preds[preds.isnan()] = 0
        preds = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds), is_valid


class SEHFragTrainer(GFNTrainer):
    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
        cfg.num_workers = 8
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20_000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.global_batch_size = 64
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4

        cfg.algo.method = "TB"
        cfg.algo.max_nodes = 9
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.train_random_action_prob = 0.0
        cfg.algo.valid_random_action_prob = 0.0
        cfg.algo.tb.epsilon = None
        cfg.algo.tb.bootstrap_own_reward = False
        cfg.algo.tb.Z_learning_rate = 1e-3
        cfg.algo.tb.Z_lr_decay = 50_000
        cfg.algo.tb.do_parameterize_p_b = False

        cfg.replay.use = False
        cfg.replay.capacity = 10_000
        cfg.replay.warmup = 1_000

    def setup_algo(self):
        algo = self.cfg.algo.method
        if algo == "TB":
            algo = TrajectoryBalance
        elif algo == "FM":
            algo = FlowMatching
        elif algo == "A2C":
            algo = A2C
        else:
            raise ValueError(algo)
        self.algo = algo(self.env, self.ctx, self.rng, self.cfg)

    def setup_task(self):
        self.task = SEHTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )

    def setup_model(self):
        model = GraphTransformerGFN(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
        )
        self.model = model

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes, num_cond_dim=self.cfg.task.seh.num_thermometer_dim
        )

    def setup(self):
        RDLogger.DisableLog("rdApp.*")
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        self.training_data = []
        self.test_data = []
        self.offline_ratio = 0
        self.valid_offline_ratio = 0
        self.replay_buffer = ReplayBuffer(self.cfg, self.rng) if self.cfg.replay.use else None
        self.setup_env_context()
        self.setup_algo()
        self.setup_task()
        self.setup_model()

        # Separate Z parameters from non-Z to allow for LR decay on the former
        Z_params = list(self.model.logZ.parameters())
        non_Z_params = [i for i in self.model.parameters() if all(id(i) != id(j) for j in Z_params)]
        self.opt = torch.optim.Adam(
            non_Z_params,
            self.cfg.opt.learning_rate,
            (self.cfg.opt.momentum, 0.999),
            weight_decay=self.cfg.opt.weight_decay,
            eps=self.cfg.opt.adam_eps,
        )
        self.opt_Z = torch.optim.Adam(Z_params, self.cfg.algo.tb.Z_learning_rate, (0.9, 0.999))
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2 ** (-steps / self.cfg.opt.lr_decay))
        self.lr_sched_Z = torch.optim.lr_scheduler.LambdaLR(
            self.opt_Z, lambda steps: 2 ** (-steps / self.cfg.algo.tb.Z_lr_decay)
        )

        self.sampling_tau = self.cfg.algo.sampling_tau
        if self.sampling_tau > 0:
            self.sampling_model = copy.deepcopy(self.model)
        else:
            self.sampling_model = self.model

        self.mb_size = self.cfg.algo.global_batch_size
        self.clip_grad_callback = {
            "value": (lambda params: torch.nn.utils.clip_grad_value_(params, self.cfg.opt.clip_grad_param)),
            "norm": (lambda params: torch.nn.utils.clip_grad_norm_(params, self.cfg.opt.clip_grad_param)),
            "none": (lambda x: None),
        }[self.cfg.opt.clip_grad_type]

        # saving hyperparameters
        git_hash = git.Repo(__file__, search_parent_directories=True).head.object.hexsha[:7]
        self.cfg.git_hash = git_hash

        os.makedirs(self.cfg.log_dir, exist_ok=True)
        fmt_hps = "\n".join([f"{f'{k}':40}:\t{f'({type(v).__name__})':10}\t{v}" for k, v in sorted(self.hps.items())])
        print(f"\n\nHyperparameters:\n{'-'*50}\n{fmt_hps}\n{'-'*50}\n\n")
        with open(pathlib.Path(self.cfg.log_dir) / "hps.json", "w") as f:
            json.dump(config_to_dict(self.cfg), f)

    def step(self, loss: Tensor):
        loss.backward()
        for i in self.model.parameters():
            self.clip_grad_callback(i)
        self.opt.step()
        self.opt.zero_grad()
        self.opt_Z.step()
        self.opt_Z.zero_grad()
        self.lr_sched.step()
        self.lr_sched_Z.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.model.parameters(), self.sampling_model.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        "log_dir": "./logs/debug_run",
        "overwrite_existing_exp": True,
        "num_training_steps": 10_000,
        "num_workers": 8,
        "opt.lr_decay": 20000,
        "algo.sampling_tau": 0.99,
        "task.seh.temperature_dist_params": (0.0, 64.0),
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = SEHFragTrainer(hps, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
