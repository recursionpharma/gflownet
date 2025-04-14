from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import Dataset

import gflownet.models.mxmnet as mxmnet
from gflownet import GFNTask, LogScalar, ObjectProperties
from gflownet.config import Config, init_empty
from gflownet.data.qm9 import QM9Dataset
from gflownet.envs.mol_building_env import MolBuildingEnvContext
from gflownet.online_trainer import StandardOnlineTrainer
from gflownet.utils.conditioning import TemperatureConditional
from gflownet.utils.misc import get_worker_device
from gflownet.utils.transforms import to_logreward


class QM9GapTask(GFNTask):
    """This class captures conditional information generation and reward transforms"""

    def __init__(
        self,
        dataset: Dataset,
        cfg: Config,
        wrap_model: Callable[[nn.Module], nn.Module] = None,
    ):
        self._wrap_model = wrap_model
        self.device = get_worker_device()
        self.models = self.load_task_models(cfg.task.qm9.model_path)
        self.dataset = dataset
        self.temperature_conditional = TemperatureConditional(cfg)
        self.num_cond_dim = self.temperature_conditional.encoding_size()
        # TODO: fix interface
        self._min, self._max, self._percentile_95 = self.dataset.get_stats("gap", percentile=0.05)  # type: ignore
        self._width = self._max - self._min
        self._rtrans = "unit+95p"  # TODO: hyperparameter

    def reward_transform(self, y: Union[float, Tensor]) -> ObjectProperties:
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
        return ObjectProperties(flat_r)

    def inverse_reward_transform(self, rp):
        if self._rtrans == "exp":
            return -np.log(rp) * self._width + self._min
        elif self._rtrans == "unit":
            return (1 - rp) * self._width + self._min
        elif self._rtrans == "unit+95p":
            return (1 - rp + (1 - self._percentile_95)) * self._width + self._min

    def load_task_models(self, path):
        gap_model = mxmnet.MXMNet(mxmnet.Config(128, 6, 5.0))
        # TODO: this path should be part of the config?
        try:
            state_dict = torch.load(path, map_location=self.device)
        except Exception as e:
            print(
                "Could not load model.",
                e,
                "\nModel weights can be found at",
                "https://storage.valencelabs.com/gflownet/models/mxmnet_gap_model.pt",
            )
        gap_model.load_state_dict(state_dict)
        gap_model.to(self.device)
        gap_model = self._wrap_model(gap_model)
        return {"mxmnet_gap": gap_model}

    def sample_conditional_information(self, n: int, train_it: int) -> Dict[str, Tensor]:
        return self.temperature_conditional.sample(n)

    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LogScalar:
        return LogScalar(self.temperature_conditional.transform(cond_info, to_logreward(flat_reward)))

    def compute_reward_from_graph(self, graphs: List[gd.Data]) -> Tensor:
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(
            self.models["mxmnet_gap"].device if hasattr(self.models["mxmnet_gap"], "device") else get_worker_device()
        )
        preds = self.models["mxmnet_gap"](batch).reshape((-1,)).data.cpu() / mxmnet.HAR2EV  # type: ignore[attr-defined]
        preds[preds.isnan()] = 1
        preds = (
            self.reward_transform(preds)
            .clip(1e-4, 2)
            .reshape(
                -1,
            )
        )
        return preds

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[ObjectProperties, Tensor]:
        graphs = [mxmnet.mol2graph(i) for i in mols]  # type: ignore[attr-defined]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return ObjectProperties(torch.zeros((0, 1))), is_valid

        preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))
        assert len(preds) == is_valid.sum()
        return ObjectProperties(preds), is_valid


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
        cfg.algo.num_from_policy = 32
        cfg.algo.num_from_dataset = 32
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
            ["C", "N", "F", "O"],
            expl_H_range=[0, 1, 2, 3],
            num_cond_dim=self.task.num_cond_dim,
            allow_5_valence_nitrogen=True,
        )
        # Note: we only need the allow_5_valence_nitrogen flag because of how we generate trajectories
        # from the dataset. For example, consider tue Nitrogen atom in this: C[NH+](C)C, when s=CN(C)C, if the action
        # for setting the explicit hydrogen is used before the positive charge is set, it will be considered
        # an invalid action. However, generate_forward_trajectory does not consider this implementation detail,
        # it assumes that attribute-setting will always be valid. For the molecular environment, as of writing
        # (PR #98) this edge case is the only case where the ordering in which attributes are set can matter.

    def setup_data(self):
        self.training_data = QM9Dataset(self.cfg.task.qm9.h5_path, train=True, targets=["gap"])
        self.test_data = QM9Dataset(self.cfg.task.qm9.h5_path, train=False, targets=["gap"])
        self.to_terminate.append(self.training_data.terminate)
        self.to_terminate.append(self.test_data.terminate)

    def setup_task(self):
        self.task = QM9GapTask(
            dataset=self.training_data,
            cfg=self.cfg,
            wrap_model=self._wrap_for_mp,
        )

    def setup(self):
        super().setup()
        self.training_data.setup(self.task, self.ctx)
        self.test_data.setup(self.task, self.ctx)


def main():
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.num_workers = 0
    config.num_training_steps = 100000
    config.validate_every = 100
    config.log_dir = "./logs/debug_qm9"
    config.opt.lr_decay = 10000
    config.task.qm9.h5_path = "/rxrx/data/chem/qm9/qm9.h5"
    config.task.qm9.model_path = "/rxrx/data/chem/qm9/mxmnet_gap_model.pt"

    trial = QM9GapTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()
