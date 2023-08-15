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
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset

from gflownet.config import Config
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.envs.frag_mol_env import FragMolBuildingEnvContext
from gflownet.data.greedyfier_iterator import GreedyfierIterator, BatchTuple
from gflownet.models import bengio2021flow
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.algo.q_learning import QLearning


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
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        self.num_cond_dim = self.temperature_conditional.encoding_size()

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y) / 8)

    def inverse_flat_reward_transform(self, rp):
        return rp * 8

    def _load_task_models(self):
        model = bengio2021flow.load_original_model()
        model, self.device = self._wrap_model(model, send_to_device=True)
        return {"seh": model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, flat_reward))

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


class SEHAtomTrainer(StandardOnlineTrainer):
    task: SEHTask

    def set_default_hps(self, cfg: Config):
        cfg.hostname = socket.gethostname()
        cfg.pickle_mp_messages = False
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

    def setup_algo(self):
        super().setup_algo()
        cfgp = copy.deepcopy(self.cfg)
        cfgp.algo.max_len = cfgp.greedy_max_steps
        cfgp.algo.input_timestep = True
        cfgp.algo.illegal_action_logreward = -10
        ctxp = copy.deepcopy(self.ctx)
        ctxp.num_cond_dim += 1  # Add an extra dimension for the timestep input
        ctxp.action_type_order = ctxp.action_type_order + ctxp.bck_action_type_order  # Merge fwd and bck action types
        ctxp.bck_action_type_order = ctxp.action_type_order  # Make sure the backward action types are the same
        self.greedy_algo = QLearning(self.env, ctxp, self.rng, cfgp)
        self.greedy_algo.graph_sampler.compute_uniform_bck = False
        self.greedy_ctx = ctxp

    def setup_task(self):
        self.task = SEHTask(
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
        if 1:
            self.ctx = FragMolBuildingEnvContext(num_cond_dim=self.task.num_cond_dim)
            # Why do we need this? The greedy algorithm might remove edge attributes which make the fragment graph
            # invalid, we want to know that we've landed in an invalid state in such a case.
            self.ctx.fail_on_missing_attr = True
        else:
            self.ctx = MolBuildingEnvContext(
                ["C", "N", "O", "S", "F", "Cl", "Br"],
                charges=[0],
                chiral_types=None,
                num_rw_feat=0,
                max_nodes=self.cfg.algo.max_nodes,
                num_cond_dim=self.task.num_cond_dim,
                allow_5_valence_nitrogen=True,  # We need to fix backward trajectories to use masks!
                # And make sure the Nitrogen-related backward masks make sense
            )

    def setup_model(self):
        super().setup_model()
        self.greedy_model = GraphTransformerGFN(
            self.greedy_ctx,
            self.cfg,
        )
        self._get_additional_parameters = lambda: list(self.greedy_model.parameters())
        self.greedy_model_lagged = copy.deepcopy(self.greedy_model)
        self.greedy_model_lagged.to(self.device)
        self.dqn_tau = self.cfg.dqn_tau

    def build_training_data_loader(self):
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        gmodel, dev = self._wrap_for_mp(self.greedy_model, send_to_device=True)
        iterator = GreedyfierIterator(
            model,
            gmodel,
            self.ctx,
            self.algo,
            self.greedy_algo,
            self.task,
            self.greedy_task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "train"),
            random_action_prob=self.cfg.algo.train_random_action_prob,
            hindsight_ratio=self.cfg.replay.hindsight_ratio,  # remove?
            illegal_action_logrewards=(
                self.cfg.algo.illegal_action_logreward,
                self.greedy_algo.illegal_action_logreward,
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
        gfn_batch, greedy_batch = batch
        loss, info = self.algo.compute_batch_losses(self.model, gfn_batch)
        gloss, ginfo = self.greedy_algo.compute_batch_losses(self.greedy_model, greedy_batch, self.greedy_model_lagged)
        self.step(loss + gloss)  # TODO: clip greedy model gradients?
        info.update({f"greedy_{k}": v for k, v in ginfo.items()})
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def step(self, loss):
        super().step(loss)
        if self.dqn_tau > 0:
            for a, b in zip(self.greedy_model.parameters(), self.greedy_model_lagged.parameters()):
                b.data.mul_(self.dqn_tau).add_(a.data * (1 - self.dqn_tau))

    def _save_state(self, it):
        torch.save(
            {
                "models_state_dict": [self.model.state_dict(), self.greedy_model.state_dict()],
                "cfg": self.cfg,
                "step": it,
            },
            open(pathlib.Path(self.cfg.log_dir) / "model_state.pt", "wb"),
        )


def main():
    """Example of how this model can be run outside of Determined"""
    hps = {
        "log_dir": f"./logs/greedy/run_debug/",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overwrite_existing_exp": True,
        "num_training_steps": 2000,
        "validate_every": 0,
        "num_workers": 0,
        "opt": {
            "lr_decay": 20000,
        },
        "algo": {"sampling_tau": 0.95, "global_batch_size": 4, "tb": {"do_subtb": True}},
        "cond": {
            "temperature": {
                "sample_dist": "uniform",
                "dist_params": [8.0, 64.0],
            }
        },
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = SEHAtomTrainer(hps)
    trial.print_every = 1
    trial.run()


if __name__ == "__main__":
    main()
