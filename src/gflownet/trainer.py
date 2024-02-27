import gc
import os
import pathlib
import shutil
import time
from typing import Any, Callable, Dict, List, NewType, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch_geometric.data as gd
import wandb
from omegaconf import OmegaConf
from rdkit import RDLogger
from rdkit.Chem.rdchem import Mol as RDMol
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.data.sampling_iterator import SamplingIterator
from gflownet.envs.graph_building_env import GraphActionCategorical, GraphBuildingEnv, GraphBuildingEnvContext
from gflownet.envs.seq_building_env import SeqBatch
from gflownet.utils.misc import create_logger
from gflownet.utils.multiprocessing_proxy import mp_object_wrapper

from .config import Config

# This type represents an unprocessed list of reward signals/conditioning information
FlatRewards = NewType("FlatRewards", Tensor)  # type: ignore

# This type represents the outcome for a multi-objective task of
# converting FlatRewards to a scalar, e.g. (sum R_i omega_i) ** beta
RewardScalar = NewType("RewardScalar", Tensor)  # type: ignore


class GFNAlgorithm:
    updates: int = 0

    def step(self):
        self.updates += 1

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


class Closable(Protocol):
    def close(self):
        pass


class GFNTrainer:
    def __init__(self, config: Config, print_config=True):
        """A GFlowNet trainer. Contains the main training loop in `run` and should be subclassed.

        Parameters
        ----------
        config: Config
            The hyperparameters for the trainer.
        """
        self.print_config = print_config
        self.to_terminate: List[Closable] = []
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
        #   - The values passed in the constructor, typically what is called by the user
        # The final config is obtained by merging the three sources with the following precedence:
        #   config classes < default_hps < constructor (i.e. the constructor overrides the default_hps, and so on)
        self.default_cfg: Config = Config()
        self.set_default_hps(self.default_cfg)
        assert isinstance(self.default_cfg, Config) and isinstance(
            config, Config
        )  # make sure the config is a Config object, and not the Config class itself
        self.cfg = OmegaConf.merge(self.default_cfg, config)

        self.device = torch.device(self.cfg.device)
        # Print the loss every `self.print_every` iterations
        self.print_every = self.cfg.print_every
        # These hooks allow us to compute extra quantities when sampling data
        self.sampling_hooks: List[Callable] = []
        self.valid_sampling_hooks: List[Callable] = []
        # Will check if parameters are finite at every iteration (can be costly)
        self._validate_parameters = False

        self.setup()

    def set_default_hps(self, base: Config):
        raise NotImplementedError()

    def setup_env_context(self):
        raise NotImplementedError()

    def setup_task(self):
        raise NotImplementedError()

    def setup_model(self):
        raise NotImplementedError()

    def setup_algo(self):
        raise NotImplementedError()

    def setup_data(self):
        pass

    def step(self, loss: Tensor):
        raise NotImplementedError()

    def setup(self):
        if os.path.exists(self.cfg.log_dir):
            if self.cfg.overwrite_existing_exp:
                shutil.rmtree(self.cfg.log_dir)
            else:
                raise ValueError(
                    f"Log dir {self.cfg.log_dir} already exists. Set overwrite_existing_exp=True to delete it."
                )
        os.makedirs(self.cfg.log_dir)

        RDLogger.DisableLog("rdApp.*")
        self.rng = np.random.default_rng(142857)
        self.env = GraphBuildingEnv()
        self.setup_data()
        self.setup_task()
        self.setup_env_context()
        self.setup_algo()
        self.setup_model()

    def _wrap_for_mp(self, obj, send_to_device=False):
        """Wraps an object in a placeholder whose reference can be sent to a
        data worker process (only if the number of workers is non-zero)."""
        if send_to_device:
            obj.to(self.device)
        if self.cfg.num_workers > 0 and obj is not None:
            wapper = mp_object_wrapper(
                obj,
                self.cfg.num_workers,
                cast_types=(gd.Batch, GraphActionCategorical, SeqBatch),
                pickle_messages=self.cfg.pickle_mp_messages,
            )
            self.to_terminate.append(wapper.terminate)
            return wapper.placeholder, torch.device("cpu")
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
            self.ctx,
            self.algo,
            self.task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            replay_buffer=replay_buffer,
            ratio=self.cfg.algo.offline_ratio,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "train"),
            random_action_prob=self.cfg.algo.train_random_action_prob,
            det_after=self.cfg.algo.train_det_after,
            hindsight_ratio=self.cfg.replay.hindsight_ratio,
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else None,
        )

    def build_validation_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.model, send_to_device=True)
        iterator = SamplingIterator(
            self.test_data,
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
            ratio=self.cfg.algo.valid_offline_ratio,
            log_dir=str(pathlib.Path(self.cfg.log_dir) / "valid"),
            sample_cond_info=self.cfg.cond.valid_sample_cond_info,
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
            prefetch_factor=1 if self.cfg.num_workers else None,
        )

    def build_final_data_loader(self) -> DataLoader:
        model, dev = self._wrap_for_mp(self.sampling_model, send_to_device=True)
        iterator = SamplingIterator(
            self.training_data,
            model,
            self.ctx,
            self.algo,
            self.task,
            dev,
            batch_size=self.cfg.algo.global_batch_size,
            illegal_action_logreward=self.cfg.algo.illegal_action_logreward,
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
            prefetch_factor=1 if self.cfg.num_workers else None,
        )

    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int, train_it: int) -> Dict[str, Any]:
        tick = time.time()
        self.model.train()
        try:
            loss, info = self.algo.compute_batch_losses(self.model, batch)
            if not torch.isfinite(loss):
                raise ValueError("loss is not finite")
            step_info = self.step(loss)
            self.algo.step()
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
        info["train_time"] = time.time() - tick
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def evaluate_batch(self, batch: gd.Batch, epoch_idx: int = 0, batch_idx: int = 0) -> Dict[str, Any]:
        tick = time.time()
        self.model.eval()
        loss, info = self.algo.compute_batch_losses(self.model, batch)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        info["eval_time"] = time.time() - tick
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
        start_time = time.time()
        for it, batch in zip(range(start, 1 + num_training_steps), cycle(train_dl)):
            # the memory fragmentation or allocation keeps growing, how often should we clean up?
            # is changing the allocation strategy helpful?

            if it % 1024 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            if self.replay_buffer is not None and len(self.replay_buffer) < self.replay_buffer.warmup:
                logger.info(
                    f"iteration {it} : warming up replay buffer {len(self.replay_buffer)}/{self.replay_buffer.warmup}"
                )
                continue
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx, it)
            info["time_spent"] = time.time() - start_time
            start_time = time.time()
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
        final_info = {}
        if num_final_gen_steps:
            logger.info(f"Generating final {num_final_gen_steps} batches ...")
            for it, batch in zip(
                range(num_training_steps + 1, num_training_steps + num_final_gen_steps + 1),
                cycle(final_dl),
            ):
                if hasattr(batch, "extra_info"):
                    for k, v in batch.extra_info.items():
                        if k not in final_info:
                            final_info[k] = []
                        if hasattr(v, "item"):
                            v = v.item()
                        final_info[k].append(v)
                if it % self.print_every == 0:
                    logger.info(f"Generating mols {it - num_training_steps}/{num_final_gen_steps}")
            final_info = {k: np.mean(v) for k, v in final_info.items()}

            logger.info("Final generation steps completed - " + " ".join(f"{k}:{v:.2f}" for k, v in final_info.items()))
            self.log(final_info, num_training_steps, "final")

        # for pypy and other GC having implementations, we need to manually clean up
        del train_dl
        del valid_dl
        if self.cfg.num_final_gen_steps:
            del final_dl

    def terminate(self):
        for hook in self.sampling_hooks:
            if hasattr(hook, "terminate") and hook.terminate not in self.to_terminate:
                hook.terminate()

        for terminate in self.to_terminate:
            terminate()

    def _save_state(self, it):
        state = {
            "models_state_dict": [self.model.state_dict()],
            "cfg": self.cfg,
            "step": it,
        }
        if self.sampling_model is not self.model:
            state["sampling_model_state_dict"] = [self.sampling_model.state_dict()]
        fn = pathlib.Path(self.cfg.log_dir) / "model_state.pt"
        with open(fn, "wb") as fd:
            torch.save(
                state,
                fd,
            )
        if self.cfg.store_all_checkpoints:
            shutil.copy(fn, pathlib.Path(self.cfg.log_dir) / f"model_state_{it}.pt")

    def log(self, info, index, key):
        if not hasattr(self, "_summary_writer"):
            self._summary_writer = torch.utils.tensorboard.SummaryWriter(self.cfg.log_dir)
        for k, v in info.items():
            self._summary_writer.add_scalar(f"{key}_{k}", v, index)
        if wandb.run is not None:
            wandb.log({f"{key}_{k}": v for k, v in info.items()}, step=index)

    def __del__(self):
        self.terminate()


def cycle(it):
    while True:
        for i in it:
            yield i
