import os
import sqlite3
from collections.abc import Iterable
from copy import deepcopy
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
from rdkit import RDLogger
from torch.utils.data import Dataset, IterableDataset

from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.envs.graph_building_env import GraphActionCategorical


class MixedIterator(IterableDataset):
    """An iterator that mixes offline and online data. """

    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,
        ctx,
        algo,
        task,
        device,
        batch_size: int = 2,
        illegal_action_logreward: float = -50,
        ratio: float = 0.5,
        stream: bool = True,
        replay_buffer: ReplayBuffer = None,
        log_dir: str = None,
        sample_cond_info: bool = True,
        random_action_prob: float = 0.0,
        hindsight_ratio: float = 0.0,
        init_train_iter: int = 0,
    ):
        """Parameters
        ----------
        dataset: Dataset
            A dataset instance
        model: nn.Module
            The model we sample from (must be on CUDA already or share_memory() must be called so that
            parameters are synchronized between each worker)
        ctx:
            The context for the environment, e.g. a MolBuildingEnvContext instance
        algo:
            The training algorithm, e.g. a TrajectoryBalance instance
        task: GFNTask
            A Task instance, e.g. a MakeRingsTask instance
        device: torch.device
            The device the model is on
        replay_buffer: ReplayBuffer
            The replay buffer for training on past data
        batch_size: int
            The number of trajectories, each trajectory will be comprised of many graphs, so this is
            _not_ the batch size in terms of the number of graphs (that will depend on the task)
        illegal_action_logreward: float
            The logreward for invalid trajectories
        ratio: float
            The ratio of offline trajectories in the batch.
        stream: bool
            If True, data is sampled iid for every batch. Otherwise, this is a normal in-order
            dataset iterator.
        log_dir: str
            If not None, logs each SamplingIterator worker's generated molecules to that file.
        sample_cond_info: bool
            If True (default), then the dataset is a dataset of points used in offline training.
            If False, then the dataset is a dataset of preferences (e.g. used to validate the model)
        random_action_prob: float
            The probability of taking a random action, passed to the graph sampler
        init_train_iter: int
            The initial training iteration, incremented and passed to task.sample_conditional_information
        """
        self.data = dataset
        self.model = model
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.illegal_action_logreward = illegal_action_logreward
        self.offline_batch_size = int(np.ceil(self.batch_size * ratio))
        self.online_batch_size = int(np.floor(self.batch_size * (1 - ratio)))
        self.ratio = ratio
        self.ctx = ctx
        self.algo = algo
        self.task = task
        self.device = device
        self.sample_cond_info = sample_cond_info
        self.random_action_prob = random_action_prob

        self.log_hooks: List[Callable] = []

    def add_log_hook(self, hook: Callable):
        self.log_hooks.append(hook)

    def __len__(self):
        return int(1e12)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # Now that we know we are in a worker instance, we can initialize per-worker things
        self.rng = self.algo.rng = self.task.rng = np.random.default_rng(142857 + self._wid)
        self.ctx.device = self.device

        while True:
            objs = self.data.sample(self.offline_batch_size)

            if self.sample_cond_info:
                num_online = self.online_batch_size
                cond_info = self.task.sample_conditional_information(
                    num_offline + self.online_batch_size, self.train_it
                )

                # Sample some dataset data
                graphs, flat_rewards = map(list, zip(*[self.data[i] for i in idcs])) if len(idcs) else ([], [])
                flat_rewards = (
                    list(self.task.flat_reward_transform(torch.stack(flat_rewards))) if len(flat_rewards) else []
                )
                trajs = self.algo.create_training_data_from_graphs(
                    graphs, self.model, cond_info["encoding"][:num_offline], 0
                )

            else:  # If we're not sampling the conditionals, then the idcs refer to listed preferences
                num_online = num_offline
                num_offline = 0
                cond_info = self.task.encode_conditional_information(
                    steer_info=torch.stack([self.data[i] for i in idcs])
                )
                trajs, flat_rewards = [], []

            # Sample some on-policy data
            is_valid = torch.ones(num_offline + num_online).bool()
            if num_online > 0:
                with torch.no_grad():
                    trajs += self.algo.create_training_data_from_own_samples(
                        self.model,
                        num_online,
                        cond_info["encoding"][num_offline:],
                        random_action_prob=self.random_action_prob,
                    )
                if self.algo.bootstrap_own_reward:
                    # The model can be trained to predict its own reward,
                    # i.e. predict the output of cond_info_to_logreward
                    pred_reward = [i["reward_pred"].cpu().item() for i in trajs[num_offline:]]
                    flat_rewards += pred_reward
                else:
                    # Otherwise, query the task for flat rewards
                    valid_idcs = torch.tensor(
                        [i + num_offline for i in range(num_online) if trajs[i + num_offline]["is_valid"]]
                    ).long()
                    # fetch the valid trajectories endpoints
                    mols = [self.ctx.graph_to_mol(trajs[i]["result"]) for i in valid_idcs]
                    # ask the task to compute their reward
                    online_flat_rew, m_is_valid = self.task.compute_flat_rewards(mols)
                    assert (
                        online_flat_rew.ndim == 2
                    ), "FlatRewards should be (mbsize, n_objectives), even if n_objectives is 1"
                    # The task may decide some of the mols are invalid, we have to again filter those
                    valid_idcs = valid_idcs[m_is_valid]
                    pred_reward = torch.zeros((num_online, online_flat_rew.shape[1]))
                    pred_reward[valid_idcs - num_offline] = online_flat_rew
                    is_valid[num_offline:] = False
                    is_valid[valid_idcs] = True
                    flat_rewards += list(pred_reward)
                    # Override the is_valid key in case the task made some mols invalid
                    for i in range(num_online):
                        trajs[num_offline + i]["is_valid"] = is_valid[num_offline + i].item()

            # Compute scalar rewards from conditional information & flat rewards
            flat_rewards = torch.stack(flat_rewards)
            log_rewards = self.task.cond_info_to_logreward(cond_info, flat_rewards)
            log_rewards[torch.logical_not(is_valid)] = self.illegal_action_logreward
            
            #  note: we convert back into natural rewards for logging purposes
            #  (allows to take averages and plot in objective space)
            #  TODO: implement that per-task (in case they don't apply the same beta and log transformations)
            rewards = torch.exp(log_rewards / cond_info["beta"])
            if num_online > 0:
                for hook in self.log_hooks:
                    extra_info.update(
                        hook(
                            deepcopy(trajs[num_offline:]),
                            deepcopy(rewards[num_offline:]),
                            deepcopy(flat_rewards[num_offline:]),
                            {k: v[num_offline:] for k, v in deepcopy(cond_info).items()},
                        )
                    )

            # Construct batch
            batch = self.algo.construct_batch(trajs, cond_info["encoding"], log_rewards)
            batch.num_offline = num_offline
            batch.num_online = num_online
            batch.flat_rewards = flat_rewards
            batch.preferences = cond_info.get("preferences", None)
            batch.focus_dir = cond_info.get("focus_dir", None)
            batch.extra_info = {}
            # TODO: we could very well just pass the cond_info dict to construct_batch above,
            # and the algo can decide what it wants to put in the batch object

            self.train_it += worker_info.num_workers if worker_info is not None else 1
            yield batch
