import copy
import os
import pathlib
import shutil
import socket
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset

from gflownet.algo.q_learning import QLearning
from gflownet.config import Config
from gflownet.data.double_iterator import BatchTuple, DoubleIterator
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.models import bengio2021flow
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.tasks.seh_frag import SEHTask


class SEHDoubleModelTrainer(StandardOnlineTrainer):
    task: SEHTask

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
        cfg.algo.max_nodes = 9
        cfg.algo.max_edges = 70
        cfg.algo.sampling_tau = 0.9
        cfg.algo.illegal_action_logreward = -256
        cfg.algo.train_random_action_prob = 0.05
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

    def setup_algo(self):
        super().setup_algo()

        cfgp = copy.deepcopy(self.cfg)
        cfgp.algo.input_timestep = True  # Hmmm?
        cfgp.algo.illegal_action_logreward = -10
        ctxp = copy.deepcopy(self.ctx)
        ctxp.num_cond_dim += 32  # Add an extra dimension for the timestep input [do we still need that?]
        if self.cfg.second_model_allow_back_and_forth:
            # Merge fwd and bck action types
            ctxp.action_type_order = ctxp.action_type_order + ctxp.bck_action_type_order
            ctxp.bck_action_type_order = ctxp.action_type_order  # Make sure the backward action types are the same
            self.second_algo.graph_sampler.compute_uniform_bck = False  # I think this might break things, to be checked
        self.second_algo = QLearning(self.env, ctxp, self.rng, cfgp)
        # True is already the default, just leaving this as a reminder the we need to turn this off
        self.second_ctx = ctxp

    def setup_task(self):
        self.task = SEHTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )
        self.second_task = copy.copy(self.task)
        # Ignore temperature for RL task
        self.second_task.cond_info_to_logreward = lambda cond_info, flat_reward: RewardScalar(
            flat_reward.reshape((-1,))
        )

    def setup_env_context(self):
        self.ctx = FragMolBuildingEnvContext(num_cond_dim=self.task.num_cond_dim)

    def setup_model(self):
        super().setup_model()
        self.second_model = GraphTransformerGFN(
            self.second_ctx,
            self.cfg,
        )
        self._get_additional_parameters = lambda: list(self.second_model.parameters())
        # Maybe only do this if we are using DDQN?
        self.second_model_lagged = copy.deepcopy(self.second_model)
        self.second_model_lagged.to(self.device)
        self.dqn_tau = self.cfg.dqn_tau

    def build_training_data_loader(self):
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        gmodel, dev = self._wrap_for_mp(self.second_model, send_to_device=True)
        iterator = DoubleIterator(
            model,
            gmodel,
            self.ctx,
            self.algo,
            self.second_algo,
            self.task,
            self.second_task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "train"),
            random_action_prob=self.cfg.algo.train_random_action_prob,
            hindsight_ratio=self.cfg.replay.hindsight_ratio,  # remove?
            illegal_action_logrewards=(
                self.cfg.algo.illegal_action_logreward,
                self.second_algo.illegal_action_logreward,
            ),
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            # The 2 here is an odd quirk of torch 1.10, it is fixed and
            # replaced by None in torch 2.
            prefetch_factor=1 if self.cfg.num_workers else 2,
        )

    def train_batch(self, batch: BatchTuple, epoch_idx: int, batch_idx: int, train_it: int) -> Dict[str, Any]:
        gfn_batch, second_batch = batch
        loss, info = self.algo.compute_batch_losses(self.model, gfn_batch)
        sloss, sinfo = self.second_algo.compute_batch_losses(self.second_model, second_batch, self.second_model_lagged)
        self.step(loss + sloss)  # TODO: clip second model gradients?
        info.update({f"sec_{k}": v for k, v in sinfo.items()})
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def step(self, loss):
        super().step(loss)
        if self.dqn_tau > 0:
            for a, b in zip(self.second_model.parameters(), self.second_model_lagged.parameters()):
                b.data.mul_(self.dqn_tau).add_(a.data * (1 - self.dqn_tau))

    def _save_state(self, it):
        torch.save(
            {
                "models_state_dict": [self.model.state_dict(), self.second_model.state_dict()],
                "cfg": self.cfg,
                "step": it,
            },
            open(pathlib.Path(self.cfg.log_dir) / "model_state.pt", "wb"),
        )


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        "log_dir": "./logs/twomod/run_debug/",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 2000,
        "validate_every": 0,
        "num_workers": 8,
        "opt": {
            "lr_decay": 20000,
        },
        "algo": {
            "sampling_tau": 0.95,
            "global_batch_size": 64,  # This is a lie, since we're sampling for each model, so it's really 2N
            "tb": {"variant": "SubTB1"},
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

    trial = SEHDoubleModelTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
