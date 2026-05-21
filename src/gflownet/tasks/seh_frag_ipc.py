import logging
import socket
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch_geometric.data import Data

from gflownet import LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.tasks.seh_frag import SOME_MOLS, LittleSEHDataset
from gflownet.utils.communication.reward import RewardModule
from gflownet.utils.communication.task import IPCTask
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.transforms import to_logreward


class TemperatureConditional_IPCTask(IPCTask):
    """IPCTask for temperature-conditioned GFlowNet"""

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def setup_communication(self):
        """Set the communication settings"""
        self._ipc_timeout = 60  # wait up to 1 min for reward function
        self._ipc_tick = 0.1  # 0.1 sec

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], obj_props: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(obj_props)))


class SEHReward(RewardModule):
    """Reward Module which communicates with trainer running on the other process."""

    def __init__(self, gfn_log_dir: str | Path, device: str = "cuda"):
        super().__init__(gfn_log_dir, verbose_level=logging.INFO)
        self.model = bengio2021flow.load_original_model().to(device)
        self.device = device

    def setup_communication(self):
        """Set the communication settings"""
        self._ipc_timeout = 60  # wait up to 1 min for gflownet
        self._ipc_tick = 0.1  # 0.1 sec

    @property
    def num_objectives(self) -> int:
        return 1

    def convert_object(self, obj: RDMol) -> Data:
        return bengio2021flow.mol2graph(obj)

    def filter_object(self, obj: Data) -> bool:
        return obj is not None

    def compute_obj_prop_batch(self, objs: list[Data]) -> list[list[float]]:
        """Modify here if parallel computation is required

        Parameters
        ----------
        objs : list[Graph]
            A list of valid graphs

        Returns
        -------
        rewards_list: list[list[float]]
            Each item of list should be list of reward for each objective
            assert len(rewards_list) == len(objs)
        """
        batch = gd.Batch.from_data_list(objs)
        batch.to(self.device)
        preds = self.model(batch).reshape((-1,)).data.cpu() / 8
        preds[preds.isnan()] = 0
        preds = preds.clip(1e-4, 100).reshape(-1, 1)
        return preds.tolist()

    def log(self, objs: list[Data], rewards: list[list[float]], is_valid: list[bool]):
        info: OrderedDict[str, float | int] = OrderedDict()
        info["num_objects"] = len(is_valid)
        info["invalid_trajectories"] = 1.0 - float(np.mean(is_valid))
        info["sampled_rewards_avg"] = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
        self.logger.info(f"iteration {self.oracle_idx} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))


class SEHFragTrainer_IPC(StandardOnlineTrainer):
    task: TemperatureConditional_IPCTask

    def setup_task(self):
        self.task = TemperatureConditional_IPCTask(cfg=self.cfg)

    # Equal to SEHFragTrainer
    def set_default_hps(self, base: Config):
        base.hostname = socket.gethostname()
        base.pickle_mp_messages = False
        base.num_workers = 8
        base.opt.learning_rate = 1e-4
        base.opt.weight_decay = 1e-8
        base.opt.momentum = 0.9
        base.opt.adam_eps = 1e-8
        base.opt.lr_decay = 20_000
        base.opt.clip_grad_type = "norm"
        base.opt.clip_grad_param = 10
        base.algo.num_from_policy = 64
        base.model.num_emb = 128
        base.model.num_layers = 4

        base.algo.method = "TB"
        base.algo.max_nodes = 9
        base.algo.sampling_tau = 0.9
        base.algo.illegal_action_logreward = -75
        base.algo.train_random_action_prob = 0.0
        base.algo.valid_random_action_prob = 0.0
        base.algo.valid_num_from_policy = 64
        base.num_validation_gen_steps = 10
        base.algo.tb.epsilon = None
        base.algo.tb.bootstrap_own_reward = False
        base.algo.tb.Z_learning_rate = 1e-3
        base.algo.tb.Z_lr_decay = 50_000
        base.algo.tb.do_parameterize_p_b = False
        base.algo.tb.do_sample_p_b = True

        base.replay.use = False
        base.replay.capacity = 10_000
        base.replay.warmup = 1_000

    def setup_data(self):
        super().setup_data()
        if self.cfg.task.seh.reduced_frag:
            # The examples don't work with the 18 frags
            self.training_data = LittleSEHDataset([])
        else:
            self.training_data = LittleSEHDataset(SOME_MOLS)

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(
            max_frags=self.cfg.algo.max_nodes,
            num_cond_dim=self.task.num_cond_dim,
            fragments=bengio2021flow.FRAGMENTS_18 if self.cfg.task.seh.reduced_frag else bengio2021flow.FRAGMENTS,
        )

    def setup(self):
        super().setup()
        self.training_data.setup(self.task, self.ctx)


def main_gfn(log_dir: str):
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = log_dir
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True
    config.num_training_steps = 1_00
    config.validate_every = 20
    config.num_final_gen_steps = 10
    config.num_workers = 1
    config.opt.lr_decay = 20_000
    config.algo.sampling_tau = 0.99
    config.cond.temperature.sample_dist = "uniform"
    config.cond.temperature.dist_params = [0, 64.0]

    # set ipc module using TCP/IP protocol (recommended)
    config.communication.method = "network"
    config.communication.network.host = "localhost"
    config.communication.network.port = 14285

    trial = SEHFragTrainer_IPC(config)
    trial.run()
    trial.task.terminate_reward()


def main_reward(gfn_log_dir: str):
    """Example of how this reward function can be run."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reward_module = SEHReward(gfn_log_dir, device=device)
    reward_module.run()


if __name__ == "__main__":
    import sys

    process = sys.argv[1]
    assert process in ("gfn", "reward")

    log_dir = "./logs/debug_run_seh_frag"

    if process == "gfn":
        main_gfn(log_dir)
    else:
        main_reward(log_dir)
