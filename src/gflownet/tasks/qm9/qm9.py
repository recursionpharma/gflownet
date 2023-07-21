import os
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from ruamel.yaml import YAML
from torch import Tensor
from torch.utils.data import Dataset

import gflownet.models.mxmnet as mxmnet
from gflownet.config import Config
from gflownet.data.qm9 import QM9Dataset
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.trainer import FlatRewards, GFNTask, RewardScalar
from gflownet.utils.conditioning import TemperatureConditional


class QM9GapTask(GFNTask):
    """This class captures conditional information generation and reward transforms"""

    def __init__(
        self,
        dataset: Dataset,
        cfg: Config,
        rng: np.random.Generator = None,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.rng = rng
        self.models = self.load_task_models(cfg.task.qm9.model_path)
        self.dataset = dataset
        self.temperature_conditional = TemperatureConditional(cfg, rng)
        # TODO: fix interface
        self._min, self._max, self._percentile_95 = self.dataset.get_stats(percentile=0.05)  # type: ignore
        self._width = self._max - self._min
        self._rtrans = "unit+95p"  # TODO: hyperparameter

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        """Transforms a target quantity y (e.g. the LUMO energy in QM9) to a positive reward scalar"""
        if self._rtrans == "exp":
            flat_r = np.exp(-(y - self._min) / self._width)
        elif self._rtrans == "unit":
            flat_r = 1 - (y - self._min) / self._width
        elif self._rtrans == "unit+95p":
            # Add constant such that 5% of rewards are > 1
            flat_r = 1 - (y - self._percentile_95) / self._width
        else:
            raise ValueError(self._rtrans)
        return FlatRewards(flat_r)

    def inverse_flat_reward_transform(self, rp):
        if self._rtrans == "exp":
            return -np.log(rp) * self._width + self._min
        elif self._rtrans == "unit":
            return (1 - rp) * self._width + self._min
        elif self._rtrans == "unit+95p":
            return (1 - rp + (1 - self._percentile_95)) * self._width + self._min

    def load_task_models(self, path):
        gap_model = mxmnet.MXMNet(mxmnet.Config(128, 6, 5.0))
        # TODO: this path should be part of the config?
        state_dict = torch.load(path)
        gap_model.load_state_dict(state_dict)
        gap_model.cuda()
        gap_model, self.device = self._wrap_model(gap_model, send_to_device=True)
        return {"mxmnet_gap": gap_model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        return RewardScalar(self.temperature_conditional.transform(cond_info, flat_reward))

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        graphs = [mxmnet.mol2graph(i) for i in mols]  # type: ignore[attr-defined]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)
        preds = self.models["mxmnet_gap"](batch).reshape((-1,)).data.cpu() / mxmnet.HAR2EV  # type: ignore[attr-defined]
        preds[preds.isnan()] = 1
        preds = self.flat_reward_transform(preds).clip(1e-4, 2).reshape((-1, 1))
        return FlatRewards(preds), is_valid


class QM9GapTrainer(StandardOnlineTrainer):
    def set_default_hps(self, cfg: Config):
        cfg.num_workers = 8
        cfg.num_training_steps = 100000
        cfg.opt.learning_rate = 1e-4
        cfg.opt.weight_decay = 1e-8
        cfg.opt.momentum = 0.9
        cfg.opt.adam_eps = 1e-8
        cfg.opt.lr_decay = 20000
        cfg.opt.clip_grad_type = "norm"
        cfg.opt.clip_grad_param = 10
        cfg.algo.max_nodes = 9
        cfg.algo.global_batch_size = 64
        cfg.algo.train_random_action_prob = 0.001
        cfg.algo.illegal_action_logreward = -75
        cfg.algo.sampling_tau = 0.0
        cfg.model.num_emb = 128
        cfg.model.num_layers = 4
        cfg.cond.temperature.sample_dist = "uniform"
        cfg.cond.temperature.dist_params = [0.5, 32.0]
        cfg.cond.temperature.num_thermometer_dim = 32

    def setup_env_context(self):
        self.ctx = MolBuildingEnvContext(
            ["C", "N", "F", "O"], expl_H_range=[0, 1, 2, 3], num_cond_dim=32, allow_5_valence_nitrogen=True
        )
        # Note: we only need the allow_5_valence_nitrogen flag because of how we generate trajectories
        # from the dataset. For example, consider tue Nitrogen atom in this: C[NH+](C)C, when s=CN(C)C, if the action
        # for setting the explicit hydrogen is used before the positive charge is set, it will be considered
        # an invalid action. However, generate_forward_trajectory does not consider this implementation detail,
        # it assumes that attribute-setting will always be valid. For the molecular environment, as of writing
        # (PR #98) this edge case is the only case where the ordering in which attributes are set can matter.

    def setup_data(self):
        self.training_data = QM9Dataset(self.cfg.task.qm9.h5_path, train=True, target="gap")
        self.test_data = QM9Dataset(self.cfg.task.qm9.h5_path, train=False, target="gap")

    def setup_task(self):
        self.task = QM9GapTask(
            dataset=self.training_data,
            cfg=self.cfg,
            rng=self.rng,
            wrap_model=self._wrap_for_mp,
        )


def main():
    """Example of how this model can be run."""
    yaml = YAML(typ="safe", pure=True)
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qm9.yaml")
    with open(config_file, "r") as f:
        hps = yaml.load(f)
    trial = QM9GapTrainer(hps, torch.device("cuda"))
    trial.run()


if __name__ == "__main__":
    main()
