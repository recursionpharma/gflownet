import os
import sqlite3
from collections.abc import Iterable
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from rdkit import RDLogger
from torch.utils.data import Dataset, IterableDataset

from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.envs.graph_building_env import GraphActionCategorical


class SamplingIterator(IterableDataset):
    """This class allows us to parallelise and train faster.

    By separating sampling data/the model and building torch geometric
    graphs from training the model, we can do the former in different
    processes, which is much faster since much of graph construction
    is CPU-bound.

    """

    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,
        ctx,
        algo,
        task,
        device,
        batch_size: int = 1,
        illegal_action_logreward: float = -50,
        ratio: float = 0.5,
        stream: bool = True,
        replay_buffer: ReplayBuffer = None,
        log_dir: str = None,
        sample_cond_info: bool = True,
        random_action_prob: float = 0.0,
        det_after: Optional[int] = None,
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
        self.stream = stream
        self.sample_online_once = True  # TODO: deprecate this, disallow len(data) == 0 entirely
        self.sample_cond_info = sample_cond_info
        self.random_action_prob = random_action_prob
        self.hindsight_ratio = hindsight_ratio
        self.train_it = init_train_iter
        self.do_validate_batch = False  # Turn this on for debugging
        self.iter = 0
        self.det_after = det_after
        # Slightly weird semantics, but if we're sampling x given some fixed cond info (data)
        # then "offline" now refers to cond info and online to x, so no duplication and we don't end
        # up with 2*batch_size accidentally
        if not sample_cond_info:
            self.offline_batch_size = self.online_batch_size = self.batch_size

        # This SamplingIterator instance will be copied by torch DataLoaders for each worker, so we
        # don't want to initialize per-worker things just yet, such as where the log the worker writes
        # to. This must be done in __iter__, which is called by the DataLoader once this instance
        # has been copied into a new python process.
        self.log_dir = log_dir
        self.log = SQLiteLog()
        self.log_hooks: List[Callable] = []

    def add_log_hook(self, hook: Callable):
        self.log_hooks.append(hook)

    def _idx_iterator(self):
        RDLogger.DisableLog("rdApp.*")
        if self.stream:
            # If we're streaming data, just sample `offline_batch_size` indices
            while True:
                if self.offline_batch_size == 0 or len(self.data) == 0:
                    yield np.arange(0, 0)
                else:
                    yield self.rng.integers(0, len(self.data), self.offline_batch_size)
        else:
            # Otherwise, figure out which indices correspond to this worker
            worker_info = torch.utils.data.get_worker_info()
            n = len(self.data)
            if n == 0:
                yield np.arange(0, 0)
                return
            assert (
                self.offline_batch_size > 0
            ), "offline_batch_size must be > 0 if not streaming and len(data) > 0 (have you set ratio=0?)"
            if worker_info is None:  # no multi-processing
                start, end, wid = 0, n, -1
            else:  # split the data into chunks (per-worker)
                nw = worker_info.num_workers
                wid = worker_info.id
                start, end = int(np.round(n / nw * wid)), int(np.round(n / nw * (wid + 1)))
            bs = self.offline_batch_size
            if end - start <= bs:
                yield np.arange(start, end)
                return
            for i in range(start, end - bs, bs):
                yield np.arange(i, i + bs)
            if i + bs < end:
                yield np.arange(i + bs, end)

    def __len__(self):
        if self.stream:
            return int(1e6)
        if len(self.data) == 0 and self.sample_online_once:
            return 1
        return len(self.data)

    def __iter__(self):
        self.iter += 1
        if self.det_after is not None and self.iter > self.det_after:
            self.random_action_prob = 0
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # Now that we know we are in a worker instance, we can initialize per-worker things
        self.rng = self.algo.rng = self.task.rng = np.random.default_rng(142857 + self._wid)
        self.ctx.device = self.device
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f"{self.log_dir}/generated_mols_{self._wid}.db"
            self.log.connect(self.log_path)

        for idcs in self._idx_iterator():
            num_offline = idcs.shape[0]  # This is in [0, self.offline_batch_size]
            # Sample conditional info such as temperature, trade-off weights, etc.

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

            assert len(trajs) == num_online + num_offline
            # Computes some metrics
            extra_info = {"random_action_prob": self.random_action_prob}
            if num_online > 0:
                H = sum(i["fwd_logprob"] for i in trajs[num_offline:])
                extra_info["entropy"] = -H / num_online
                extra_info["length"] = np.mean([len(i["traj"]) for i in trajs[num_offline:]])
            if not self.sample_cond_info:
                # If we're using a dataset of preferences, the user may want to know the id of the preference
                for i, j in zip(trajs, idcs):
                    i["data_idx"] = j
            #  note: we convert back into natural rewards for logging purposes
            #  (allows to take averages and plot in objective space)
            #  TODO: implement that per-task (in case they don't apply the same beta and log transformations)
            rewards = torch.exp(log_rewards / cond_info["beta"])
            if num_online > 0 and self.log_dir is not None:
                self.log_generated(
                    trajs[num_offline:],
                    rewards[num_offline:],
                    flat_rewards[num_offline:],
                    {k: v[num_offline:] for k, v in cond_info.items()},
                )
            if num_online > 0:
                extra_info["sampled_reward_avg"] = rewards[num_offline:].mean().item()
                for hook in self.log_hooks:
                    extra_info.update(
                        hook(
                            trajs[num_offline:],
                            rewards[num_offline:],
                            flat_rewards[num_offline:],
                            {k: v[num_offline:] for k, v in cond_info.items()},
                        )
                    )

            if self.replay_buffer is not None:
                # If we have a replay buffer, we push the online trajectories in it
                # and resample immediately such that the "online" data in the batch
                # comes from a more stable distribution (try to avoid forgetting)

                # cond_info is a dict, so we need to convert it to a list of dicts
                cond_info = [{k: v[i] for k, v in cond_info.items()} for i in range(num_offline + num_online)]

                # push the online trajectories in the replay buffer and sample a new 'online' batch
                for i in range(num_offline, len(trajs)):
                    self.replay_buffer.push(
                        trajs[i],
                        log_rewards[i],
                        flat_rewards[i],
                        cond_info[i],
                        is_valid[i],
                    )
                replay_trajs, replay_logr, replay_fr, replay_condinfo, replay_valid = self.replay_buffer.sample(
                    num_online
                )

                # append the online trajectories to the offline ones
                trajs = trajs[:num_offline] + replay_trajs
                log_rewards = torch.cat([log_rewards[:num_offline], replay_logr], dim=0)
                flat_rewards = torch.cat([flat_rewards[:num_offline], replay_fr], dim=0)
                cond_info = cond_info[:num_offline] + replay_condinfo  # list of dicts
                is_valid = torch.cat([is_valid[:num_offline], replay_valid], dim=0)

                # convert cond_info back to a dict
                cond_info = {k: torch.stack([d[k] for d in cond_info]) for k in cond_info[0]}

            if self.hindsight_ratio > 0.0:
                # Relabels some of the online trajectories with hindsight
                assert hasattr(
                    self.task, "relabel_condinfo_and_logrewards"
                ), "Hindsight requires the task to implement relabel_condinfo_and_logrewards"
                # samples indexes of trajectories without repeats
                hindsight_idxs = torch.randperm(num_online)[: int(num_online * self.hindsight_ratio)] + num_offline
                cond_info, log_rewards = self.task.relabel_condinfo_and_logrewards(
                    cond_info, log_rewards, flat_rewards, hindsight_idxs
                )
                log_rewards[torch.logical_not(is_valid)] = self.illegal_action_logreward

            # Construct batch
            batch = self.algo.construct_batch(trajs, cond_info["encoding"], log_rewards)
            batch.num_offline = num_offline
            batch.num_online = num_online
            batch.flat_rewards = flat_rewards
            batch.preferences = cond_info.get("preferences", None)
            batch.focus_dir = cond_info.get("focus_dir", None)
            batch.extra_info = extra_info
            if self.ctx.has_n():
                log_ns = [self.ctx.traj_log_n(i["traj"]) for i in trajs]
                batch.log_n = torch.tensor([i[-1] for i in log_ns], dtype=torch.float32)
                batch.log_ns = torch.tensor(sum(log_ns, start=[]), dtype=torch.float32)
            # TODO: we could very well just pass the cond_info dict to construct_batch above,
            # and the algo can decide what it wants to put in the batch object

            # Only activate for debugging your environment or dataset (e.g. the dataset could be
            # generating trajectories with illegal actions)
            if self.do_validate_batch:
                self.validate_batch(batch, trajs)

            self.train_it += worker_info.num_workers if worker_info is not None else 1
            yield batch

    def validate_batch(self, batch, trajs):
        for actions, atypes in [(batch.actions, self.ctx.action_type_order)] + (
            [(batch.bck_actions, self.ctx.bck_action_type_order)]
            if hasattr(batch, "bck_actions") and hasattr(self.ctx, "bck_action_type_order")
            else []
        ):
            mask_cat = GraphActionCategorical(
                batch,
                [self.model._action_type_to_mask(t, batch) for t in atypes],
                [self.model._action_type_to_key[t] for t in atypes],
                [None for _ in atypes],
            )
            masked_action_is_used = 1 - mask_cat.log_prob(actions, logprobs=mask_cat.logits)
            num_trajs = len(trajs)
            batch_idx = torch.arange(num_trajs, device=batch.x.device).repeat_interleave(batch.traj_lens)
            first_graph_idx = torch.zeros_like(batch.traj_lens)
            torch.cumsum(batch.traj_lens[:-1], 0, out=first_graph_idx[1:])
            if masked_action_is_used.sum() != 0:
                invalid_idx = masked_action_is_used.argmax().item()
                traj_idx = batch_idx[invalid_idx].item()
                timestep = invalid_idx - first_graph_idx[traj_idx].item()
                raise ValueError("Found an action that was masked out", trajs[traj_idx]["traj"][timestep])

    def log_generated(self, trajs, rewards, flat_rewards, cond_info):
        if hasattr(self.ctx, "object_to_log_repr"):
            mols = [self.ctx.object_to_log_repr(t["result"]) if t["is_valid"] else "" for t in trajs]
        else:
            mols = [""] * len(trajs)

        flat_rewards = flat_rewards.reshape((len(flat_rewards), -1)).data.numpy().tolist()
        rewards = rewards.data.numpy().tolist()
        preferences = cond_info.get("preferences", torch.zeros((len(mols), 0))).data.numpy().tolist()
        focus_dir = cond_info.get("focus_dir", torch.zeros((len(mols), 0))).data.numpy().tolist()
        logged_keys = [k for k in sorted(cond_info.keys()) if k not in ["encoding", "preferences", "focus_dir"]]

        data = [
            [mols[i], rewards[i]]
            + flat_rewards[i]
            + preferences[i]
            + focus_dir[i]
            + [cond_info[k][i].item() for k in logged_keys]
            for i in range(len(trajs))
        ]

        data_labels = (
            ["smi", "r"]
            + [f"fr_{i}" for i in range(len(flat_rewards[0]))]
            + [f"pref_{i}" for i in range(len(preferences[0]))]
            + [f"focus_{i}" for i in range(len(focus_dir[0]))]
            + [f"ci_{k}" for k in logged_keys]
        )

        self.log.insert_many(data, data_labels)


class SQLiteLog:
    def __init__(self, timeout=300):
        """Creates a log instance, but does not connect it to any db."""
        self.is_connected = False
        self.db = None
        self.timeout = timeout

    def connect(self, db_path: str):
        """Connects to db_path

        Parameters
        ----------
        db_path: str
            The sqlite3 database path. If it does not exist, it will be created.
        """
        self.db = sqlite3.connect(db_path, timeout=self.timeout)
        cur = self.db.cursor()
        self._has_results_table = len(
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'").fetchall()
        )
        cur.close()

    def _make_results_table(self, types, names):
        type_map = {str: "text", float: "real", int: "real"}
        col_str = ", ".join(f"{name} {type_map[t]}" for t, name in zip(types, names))
        cur = self.db.cursor()
        cur.execute(f"create table results ({col_str})")
        self._has_results_table = True
        cur.close()

    def insert_many(self, rows, column_names):
        assert all(
            [isinstance(x, str) or not isinstance(x, Iterable) for x in rows[0]]
        ), "rows must only contain scalars"
        if not self._has_results_table:
            self._make_results_table([type(i) for i in rows[0]], column_names)
        cur = self.db.cursor()
        cur.executemany(f'insert into results values ({",".join("?"*len(rows[0]))})', rows)  # nosec
        cur.close()
        self.db.commit()
