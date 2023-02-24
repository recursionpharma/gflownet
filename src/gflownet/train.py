import os
import pathlib
import os
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple

from rdkit.Chem.rdchem import Mol as RDMol
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.utils.tensorboard
import torch_geometric.data as gd

from gflownet.data.sampling_iterator import SamplingIterator
from gflownet.envs.graph_building_env import GraphActionCategorical
from gflownet.envs.graph_building_env import GraphBuildingEnv
from gflownet.envs.graph_building_env import GraphBuildingEnvContext
from gflownet.utils.misc import create_logger
from gflownet.utils.multiprocessing_proxy import wrap_model_mp

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType('FlatRewards', Tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType('RewardScalar', Tensor)  # type: ignore


class GFNAlgorithm:
    def compute_batch_losses(self, model: nn.Module, batch: gd.Batch,
                             num_bootstrap: Optional[int] = 0) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Computes the loss for a batch of data, and proves logging informations
        Parameters
        ----------
        model: nn.Module
            The model being trained or evaluated
        batch: gd.Batch
            A batch of graphs
        num_bootstrap: Optional[int]
            The number of trajectories with reward targets in the batch (if applicable).
        Returns
        -------
        loss: Tensor
            The loss for that batch
        info: Dict[str, Tensor]
            Logged information about model predictions.
        """
        raise NotImplementedError()


class GFNTask:
    def cond_info_to_reward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        """Combines a minibatch of reward signal vectors and conditional information into a scalar reward.

        Parameters
        ----------
        cond_info: Dict[str, Tensor]
            A dictionary with various conditional informations (e.g. temperature)
        flat_reward: FlatRewards
            A 2d tensor where each row represents a series of flat rewards.

        Returns
        -------
        reward: RewardScalar
            A 1d tensor, a scalar reward for each minibatch entry.
        """
        raise NotImplementedError()

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        """Compute the flat rewards of mols according the the tasks' proxies

        Parameters
        ----------
        mols: List[RDMol]
            A list of RDKit molecules.
        Returns
        -------
        reward: FlatRewards
            A 2d tensor, a vector of scalar reward for valid each molecule.
        is_valid: Tensor
            A 1d tensor, a boolean indicating whether the molecule is valid.
        """
        raise NotImplementedError()


class GFNTrainer:
    def __init__(self, hps: Dict[str, Any], device: torch.device):
        """A GFlowNet trainer. Contains the main training loop in `run` and should be subclassed.

        Parameters
        ----------
        hps: Dict[str, Any]
            A dictionary of hyperparameters. These override default values obtained by the `default_hps` method.
        device: torch.device
            The torch device of the main worker.
        """
        # self.setup should at least set these up:
        self.training_data: Dataset
        self.test_data: Dataset
        self.model: nn.Module
        # `sampling_model` is used by the data workers to sample new objects from the model. Can be
        # the same as `model`.
        self.sampling_model: nn.Module
        self.mb_size: int
        self.env: GraphBuildingEnv
        self.ctx: GraphBuildingEnvContext
        self.task: GFNTask
        self.algo: GFNAlgorithm

        # Override default hyperparameters with the constructor arguments
        self.hps = {**self.default_hps(), **hps}
        self.device = device
        # The number of processes spawned to sample object and do CPU work
        self.num_workers: int = self.hps.get('num_data_loader_workers', 0)
        # The ratio of samples drawn from `self.training_data` during training. The rest is drawn from
        # `self.sampling_model`.
        self.offline_ratio = self.hps.get('offline_ratio', 0.5)
        # idem, but from `self.test_data` during validation.
        self.valid_offline_ratio = 1
        # If True, print messages during training
        self.verbose = False
        # These hooks allow us to compute extra quantities when sampling data
        self.sampling_hooks: List[Callable] = []
        self.valid_sampling_hooks: List[Callable] = []
        # Will check if parameters are finite at every iteration (can be costly)
        self._validate_parameters = False

        self.setup()

    def default_hps(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def setup(self):
        raise NotImplementedError()

    def step(self, loss: Tensor):
        raise NotImplementedError()

    def _wrap_model_mp(self, model):
        """Wraps a nn.Module instance so that it can be shared to `DataLoader` workers.  """
        model.to(self.device)
        if self.num_workers > 0:
            placeholder = wrap_model_mp(model, self.num_workers, cast_types=(gd.Batch, GraphActionCategorical))
            return placeholder, torch.device('cpu')
        return model, self.device

    def build_callbacks(self):
        return {}

    def build_training_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.sampling_model)
        iterator = SamplingIterator(self.training_data, model, self.mb_size * 2, self.ctx, self.algo, self.task, dev,
                                    ratio=self.offline_ratio, log_dir=self.hps['log_dir'])
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(iterator, batch_size=None, num_workers=self.num_workers,
                                           persistent_workers=self.num_workers > 0)

    def build_validation_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.model)
        iterator = SamplingIterator(self.test_data, model, self.mb_size, self.ctx, self.algo, self.task, dev,
                                    ratio=self.valid_offline_ratio, stream=False,
                                    sample_cond_info=self.hps.get('valid_sample_cond_info', True))
        for hook in self.valid_sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(iterator, batch_size=None, num_workers=self.num_workers,
                                           persistent_workers=self.num_workers > 0)

    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int) -> Dict[str, Any]:
        try:
            loss, info = self.algo.compute_batch_losses(self.model, batch)
            if not torch.isfinite(loss):
                raise ValueError('loss is not finite')
            step_info = self.step(loss)
            if self._validate_parameters and not all([torch.isfinite(i).all() for i in self.model.parameters()]):
                raise ValueError('parameters are not finite')
        except ValueError as e:
            os.makedirs(self.hps['log_dir'], exist_ok=True)
            torch.save([self.model.state_dict(), batch, loss, info], open(self.hps['log_dir'] + '/dump.pkl', 'wb'))
            raise e

        if step_info is not None:
            info.update(step_info)
        if hasattr(batch, 'extra_info'):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, 'item') else v for k, v in info.items()}

    def evaluate_batch(self, batch: gd.Batch, epoch_idx: int = 0, batch_idx: int = 0) -> Dict[str, Any]:
        loss, info = self.algo.compute_batch_losses(self.model, batch)
        if hasattr(batch, 'extra_info'):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, 'item') else v for k, v in info.items()}

    def run(self, logger=None):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        os.makedirs(self.hps['log_dir'], exist_ok=True)
        torch.save({
            'hps': self.hps,
        }, open(pathlib.Path(self.hps['log_dir']) / 'hps.pt', 'wb'))
        if logger is None:
            logger = create_logger(logfile=self.hps['log_dir'] + '/train.log')
        self.model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 1)
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()
        callbacks = self.build_callbacks()
        start = self.hps.get('start_at_step', 0) + 1
        logger.info("Starting training")
        for it, batch in zip(range(start, 1 + self.hps['num_training_steps']), cycle(train_dl)):
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx)
            self.log(info, it, 'train')
            if self.verbose:
                logger.info(f"iteration {it} : " + ' '.join(f'{k}:{v:.2f}' for k, v in info.items()))

            if it % self.hps['validate_every'] == 0:
                for batch in valid_dl:
                    info = self.evaluate_batch(batch.to(self.device), epoch_idx, batch_idx)
                    self.log(info, it, 'valid')
                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, 'on_validation_end'):
                        c.on_validation_end(end_metrics)
                self.log(end_metrics, it, 'valid_end')
                self._save_state(it)
        self._save_state(self.hps['num_training_steps'])

    def _save_state(self, it):
        torch.save({
            'models_state_dict': [self.model.state_dict()],
            'hps': self.hps,
            'step': it,
        }, open(pathlib.Path(self.hps['log_dir']) / 'model_state.pt', 'wb'))

    def log(self, info, index, key):
        if not hasattr(self, '_summary_writer'):
            self._summary_writer = torch.utils.tensorboard.SummaryWriter(self.hps['log_dir'])
        for k, v in info.items():
            self._summary_writer.add_scalar(f'{key}_{k}', v, index)


def cycle(it):
    while True:
        for i in it:
            yield i
