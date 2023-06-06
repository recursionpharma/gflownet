import os
import pathlib
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch_geometric.data as gd
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from gflownet.config import Config, config_class, make_config, update_config
from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.data.sampling_iterator import SamplingIterator
from gflownet.envs.graph_building_env import GraphActionCategorical, GraphBuildingEnv, GraphBuildingEnvContext
from gflownet.utils.misc import create_logger
from gflownet.utils.multiprocessing_proxy import mp_object_wrapper

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType("FlatRewards", Tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType("RewardScalar", Tensor)  # type: ignore


@config_class("@base")
class BaseConfig:
    """Base configuration for training

    Attributes
    ----------
    log_dir : str
        The directory where to store logs, checkpoints, and samples.
    seed : int
        The random seed
    validate_every : int
        The number of training steps after which to validate the model
    checkpoint_every : Optional[int]
        The number of training steps after which to checkpoint the model
    start_at_step : int
        The training step to start at (default: 0)
    num_final_gen_steps : Optional[int]
        After training, the number of steps to generate graphs for
    num_training_steps : int
        The number of training steps
    num_workers : int
        The number of workers to use for creating minibatches (0 = no multiprocessing)
    hostname : Optional[str]
        The hostname of the machine on which the experiment is run
    pickle_mp_messages : bool
        Whether to pickle messages sent between processes (only relevant if num_workers > 0)
    git_hash : Optional[str]
        The git hash of the current commit
    overwrite_existing_exp : bool
        Whether to overwrite the contents of the log_dir if it already exists
    """

    log_dir: str
    seed: int = 0
    validate_every: int = 1000
    checkpoint_every: Optional[int] = None
    start_at_step: int = 0
    num_final_gen_steps: Optional[int] = None
    num_training_steps: int = 10_000
    num_workers: int = 0
    hostname: Optional[str] = None
    pickle_mp_messages: bool = False
    git_hash: Optional[str] = None
    overwrite_existing_exp: bool = True


@config_class("opt")
class OptimizerConfig:
    """Generic configuration for optimizers

    Attributes
    ----------
    opt : str
        The optimizer to use (either "adam" or "sgd")
    learning_rate : float
        The learning rate
    lr_decay : float
        The learning rate decay (in steps, f = 2 ** (-steps / self.cfg.opt.lr_decay))
    weight_decay : float
        The L2 weight decay
    momentum : float
        The momentum parameter value
    clip_grad_type : str
        The type of gradient clipping to use (either "norm" or "value")
    clip_grad_param : float
        The parameter for gradient clipping
    adam_eps : float
        The epsilon parameter for Adam
    """

    opt: str = "adam"
    learning_rate: float = 1e-4
    lr_decay: float = 20_000
    weight_decay: float = 1e-8
    momentum: float = 0.9
    clip_grad_type: str = "norm"
    clip_grad_param: float = 10.0
    adam_eps: float = 1e-8


@config_class("model")
class ModelConfig:
    """Generic configuration for models

    Attributes
    ----------
    num_layers : int
        The number of layers in the model
    num_emb : int
        The number of dimensions of the embedding
    """

    num_layers: int = 3
    num_emb: int = 128


@config_class("algo")
class AlgoConfig:
    """Generic configuration for algorithms

    Attributes
    ----------
    method : str
        The name of the algorithm to use (e.g. "TB")
    global_batch_size : int
        The batch size for training
    max_len : int
        The maximum length of a trajectory
    max_nodes : int
        The maximum number of nodes in a generated graph
    max_edges : int
        The maximum number of edges in a generated graph
    illegal_action_logreward : float
        The log reward an agent gets for illegal actions
    offline_ratio: float
        The ratio of samples drawn from `self.training_data` during training. The rest is drawn from
        `self.sampling_model`
    train_random_action_prob : float
        The probability of taking a random action during training
    valid_random_action_prob : float
        The probability of taking a random action during validation
    valid_sample_cond_info : bool
        Whether to sample conditioning information during validation (if False, expects a validation set of cond_info)
    sampling_tau : float
        The EMA factor for the sampling model (theta_sampler = tau * theta_sampler + (1-tau) * theta)
    """

    method: str = "TB"
    global_batch_size: int = 64
    max_len: int = 128
    max_nodes: int = 128
    max_edges: int = 128
    illegal_action_logreward: float = -100
    offline_ratio: float = 0.5
    train_random_action_prob: float = 0.0
    valid_random_action_prob: float = 0.0
    valid_sample_cond_info: bool = True
    sampling_tau: float = 0.0


class GFNAlgorithm:
    def compute_batch_losses(
        self, model: nn.Module, batch: gd.Batch, num_bootstrap: Optional[int] = 0
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
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
    def cond_info_to_logreward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
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
            A 1d tensor, a scalar log-reward for each minibatch entry.
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
        self.replay_buffer: Optional[ReplayBuffer]
        self.mb_size: int
        self.env: GraphBuildingEnv
        self.ctx: GraphBuildingEnvContext
        self.task: GFNTask
        self.algo: GFNAlgorithm

        # There are three sources of config values
        #   - The default values specified in individual config classes
        #   - The default values specified in the `default_hps` method, typically what is defined by a task
        #   - The values passed in the constructor, typically what is called by the used
        # The final config is obtained by merging the three sources
        self.cfg = make_config()
        self.set_default_hps(self.cfg)
        update_config(self.cfg, hps)
        self.device = device
        # idem, but from `self.test_data` during validation.
        self.valid_offline_ratio = 1
        # Print the loss every `self.print_every` iterations
        self.print_every = 1000
        # These hooks allow us to compute extra quantities when sampling data
        self.sampling_hooks: List[Callable] = []
        self.valid_sampling_hooks: List[Callable] = []
        # Will check if parameters are finite at every iteration (can be costly)
        self._validate_parameters = False

        self.setup()

    def set_default_hps(self, base: Config):
        raise NotImplementedError()

    def setup(self):
        raise NotImplementedError()

    def step(self, loss: Tensor):
        raise NotImplementedError()

    def _wrap_for_mp(self, obj, send_to_device=False):
        """Wraps an object in a placeholder whose reference can be sent to a
        data worker process (only if the number of workers is non-zero)."""
        if send_to_device:
            obj.to(self.device)
        if self.cfg.num_workers > 0 and obj is not None:
            placeholder = mp_object_wrapper(
                obj,
                self.cfg.num_workers,
                cast_types=(gd.Batch, GraphActionCategorical),
                pickle_messages=self.cfg.pickle_mp_messages,
            )
            return placeholder, torch.device("cpu")
        else:
            return obj, self.device

    def build_callbacks(self):
        return {}

    def build_training_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        replay_buffer, _ = self._wrap_for_mp(self.replay_buffer, send_to_device=False)
        iterator = SamplingIterator(
            self.training_data,
            model,
            self.cfg,
            self.ctx,
            self.algo,
            self.task,
            dev,
            replay_buffer=replay_buffer,
            ratio=self.cfg.algo.offline_ratio,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "train"),
            random_action_prob=self.cfg.algo.train_random_action_prob,
            hindsight_ratio=self.cfg.replay.hindsight_ratio,
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

    def build_validation_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.model, send_to_device=True)
        iterator = SamplingIterator(
            self.test_data,
            model,
            self.cfg,
            self.ctx,
            self.algo,
            self.task,
            dev,
            ratio=self.valid_offline_ratio,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "valid"),
            sample_cond_info=self.cfg.algo.valid_sample_cond_info,
            stream=False,
            random_action_prob=self.cfg.algo.valid_random_action_prob,
        )
        for hook in self.valid_sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else 2,
        )

    def build_final_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        iterator = SamplingIterator(
            self.training_data,
            model,
            self.cfg,
            self.ctx,
            self.algo,
            self.task,
            dev,
            replay_buffer=None,
            ratio=0.0,
            log_dir=os.path.join(self.cfg.log_dir, "final"),
            random_action_prob=0.0,
            hindsight_ratio=0.0,
            init_train_iter=self.cfg.num_training_steps,
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else 2,
        )

    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int, train_it: int) -> Dict[str, Any]:
        try:
            loss, info = self.algo.compute_batch_losses(self.model, batch)
            if not torch.isfinite(loss):
                raise ValueError("loss is not finite")
            step_info = self.step(loss)
            if self._validate_parameters and not all([torch.isfinite(i).all() for i in self.model.parameters()]):
                raise ValueError("parameters are not finite")
        except ValueError as e:
            os.makedirs(self.cfg.log_dir, exist_ok=True)
            torch.save([self.model.state_dict(), batch, loss, info], open(self.cfg.log_dir + "/dump.pkl", "wb"))
            raise e

        if step_info is not None:
            info.update(step_info)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def evaluate_batch(self, batch: gd.Batch, epoch_idx: int = 0, batch_idx: int = 0) -> Dict[str, Any]:
        loss, info = self.algo.compute_batch_losses(self.model, batch)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def run(self, logger=None):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        if logger is None:
            logger = create_logger(logfile=self.cfg.log_dir + "/train.log")
        self.model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 1)
        valid_freq = self.cfg.validate_every
        # If checkpoint_every is not specified, checkpoint at every validation epoch
        ckpt_freq = self.cfg.checkpoint_every if self.cfg.checkpoint_every is not None else valid_freq
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()
        if self.cfg.num_final_gen_steps:
            final_dl = self.build_final_data_loader()
        callbacks = self.build_callbacks()
        start = self.cfg.start_at_step + 1
        num_training_steps = self.cfg.num_training_steps
        logger.info("Starting training")
        for it, batch in zip(range(start, 1 + num_training_steps), cycle(train_dl)):
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            if self.replay_buffer is not None and len(self.replay_buffer) < self.replay_buffer.warmup:
                logger.info(
                    f"iteration {it} : warming up replay buffer {len(self.replay_buffer)}/{self.replay_buffer.warmup}"
                )
                continue
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx, it)
            self.log(info, it, "train")
            if it % self.print_every == 0:
                logger.info(f"iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))

            if valid_freq > 0 and it % valid_freq == 0:
                for batch in valid_dl:
                    info = self.evaluate_batch(batch.to(self.device), epoch_idx, batch_idx)
                    self.log(info, it, "valid")
                    logger.info(f"validation - iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))
                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, "on_validation_end"):
                        c.on_validation_end(end_metrics)
                self.log(end_metrics, it, "valid_end")
            if ckpt_freq > 0 and it % ckpt_freq == 0:
                self._save_state(it)
        self._save_state(num_training_steps)

        num_final_gen_steps = self.cfg.num_final_gen_steps
        if num_final_gen_steps:
            logger.info(f"Generating final {num_final_gen_steps} batches ...")
            for it, batch in zip(
                range(num_training_steps, num_training_steps + num_final_gen_steps + 1),
                cycle(final_dl),
            ):
                pass
            logger.info("Final generation steps completed.")

    def _save_state(self, it):
        torch.save(
            {
                "models_state_dict": [self.model.state_dict()],
                "cfg": self.cfg,
                "step": it,
            },
            open(pathlib.Path(self.cfg.log_dir) / "model_state.pt", "wb"),
        )

    def log(self, info, index, key):
        if not hasattr(self, "_summary_writer"):
            self._summary_writer = torch.utils.tensorboard.SummaryWriter(self.cfg.log_dir)
        for k, v in info.items():
            self._summary_writer.add_scalar(f"{key}_{k}", v, index)


def cycle(it):
    while True:
        for i in it:
            yield i
