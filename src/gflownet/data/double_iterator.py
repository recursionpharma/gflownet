import os
import sqlite3
from collections.abc import Iterable
from copy import deepcopy
from typing import Callable, List
import warnings

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset, IterableDataset

from gflownet.trainer import GFNTask, GFNAlgorithm
from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.envs.graph_building_env import (
    GraphActionCategorical,
    GraphActionType,
    GraphBuildingEnv,
    GraphBuildingEnvContext,
)
from gflownet.algo.graph_sampling import GraphSampler


class BatchTuple:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def to(self, device):
        return BatchTuple(self.a.to(device), self.b.to(device))

    def __getitem__(self, idx: int):
        if idx == 0:
            return self.a
        elif idx == 1:
            return self.b
        else:
            raise IndexError("Index must be 0 or 1")

    def __iter__(self):
        yield self.a
        yield self.b


class DoubleIterator(IterableDataset):
    """This iterator runs two models in sequence, and constructs batches for each model from each other's data"""

    def __init__(
        self,
        first_model: nn.Module,
        second_model: nn.Module,
        ctx: GraphBuildingEnvContext,
        first_algo: GFNAlgorithm,
        second_algo: GFNAlgorithm,
        first_task: GFNTask,
        second_task: GFNTask,
        device,
        batch_size: int,
        log_dir: str,
        random_action_prob: float = 0.0,
        hindsight_ratio: float = 0.0,
        init_train_iter: int = 0,
        illegal_action_logrewards: tuple[float, float] = (-100.0, -10.0),
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
        self.first_model = first_model
        self.second_model = second_model
        self.batch_size = batch_size
        self.ctx = ctx
        self.first_algo = first_algo
        self.second_algo = second_algo
        self.first_task = first_task
        self.second_task = second_task
        self.device = device
        self.random_action_prob = random_action_prob
        self.hindsight_ratio = hindsight_ratio
        self.train_it = init_train_iter
        self.illegal_action_logrewards = illegal_action_logrewards
        self.seed_second_trajs_with_firsts = False  # Disabled for now

        # This SamplingIterator instance will be copied by torch DataLoaders for each worker, so we
        # don't want to initialize per-worker things just yet, such as where the log the worker writes
        # to. This must be done in __iter__, which is called by the DataLoader once this instance
        # has been copied into a new python process.
        self.log_dir = log_dir
        self.log = SQLiteLog()
        self.log_hooks: List[Callable] = []
        self.log_molecule_smis = True

    def add_log_hook(self, hook: Callable):
        self.log_hooks.append(hook)

    def __len__(self):
        return int(1e6)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # Now that we know we are in a worker instance, we can initialize per-worker things
        self.rng = self.first_algo.rng = self.first_task.rng = np.random.default_rng(142857 + self._wid)
        self.ctx.device = self.device
        self.second_algo.ctx.device = self.device  # TODO: fix
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f"{self.log_dir}/generated_mols_{self._wid}.db"
            self.log.connect(self.log_path)

        while True:
            cond_info = self.first_task.sample_conditional_information(self.batch_size, self.train_it)
            with torch.no_grad():
                first_trajs = self.first_algo.create_training_data_from_own_samples(
                    self.first_model,
                    self.batch_size,
                    cond_info["encoding"],
                    random_action_prob=self.random_action_prob,
                )
                if self.seed_second_trajs_with_firsts:
                    _optional_starts = {"starts": [i["result"] for i in first_trajs[: self.batch_size - 1]]}
                else:
                    _optional_starts = {}

                # Note to self: if using a deterministic policy this makes no sense, make sure that epsilon-greedy
                # is turned on!
                if self.random_action_prob == 0:
                    warnings.warn("If second_algo is a deterministic policy, this is probably not what you want!")
                second_trajs = self.second_algo.create_training_data_from_own_samples(
                    self.second_model,
                    self.batch_size - 1,
                    cond_info["encoding"],
                    random_action_prob=self.random_action_prob,
                    **_optional_starts,
                )

            all_trajs = first_trajs + second_trajs

            def safe(f, a, default):
                try:
                    return f(a)
                except Exception as e:
                    return default

            results = [safe(self.ctx.graph_to_mol, i["result"], None) for i in trajs_for_first]
            pred_reward, is_valid = self.first_task.compute_flat_rewards(results)
            assert pred_reward.ndim == 2, "FlatRewards should be (mbsize, n_objectives), even if n_objectives is 1"
            flat_rewards = list(pred_reward)
            # Compute scalar rewards from conditional information & flat rewards
            flat_rewards = torch.stack(flat_rewards)
            # This is a bit ugly but we've sampled from the same cond_info twice, so we need to repeat
            # cond_info_to_logreward twice
            first_log_rewards = torch.cat(
                [
                    self.first_task.cond_info_to_logreward(cond_info, flat_rewards[: self.batch_size]),
                    self.first_task.cond_info_to_logreward(cond_info, flat_rewards[self.batch_size :]),
                ],
            )
            first_log_rewards[torch.logical_not(is_valid)] = self.illegal_action_logrewards[0]

            # Second task may choose to transform rewards differently
            second_log_rewards = torch.cat(
                [
                    self.second_task.cond_info_to_logreward(cond_info, flat_rewards[: self.batch_size]),
                    self.second_task.cond_info_to_logreward(cond_info, flat_rewards[self.batch_size :]),
                ],
            )
            second_log_rewards[torch.logical_not(is_valid)] = self.illegal_action_logrewards[1]

            # Computes some metrics
            if self.log_dir is not None:
                self.log_generated(
                    deepcopy(first_trajs),
                    deepcopy(first_log_rewards[: self.batch_size]),
                    deepcopy(flat_rewards[: self.batch_size]),
                    {k: v for k, v in deepcopy(cond_info).items()},
                )
                self.log_generated(
                    deepcopy(second_trajs),
                    deepcopy(second_log_rewards[self.batch_size :]),
                    deepcopy(flat_rewards[self.batch_size :]),
                    {k: v for k, v in deepcopy(cond_info).items()},
                )
            for hook in self.log_hooks:
                raise NotImplementedError()

            # Construct batch
            batch = self.first_algo.construct_batch(all_trajs, cond_info["encoding"].repeat(2, 1), first_log_rewards)
            batch.num_online = len(all_trajs)
            batch.num_offline = 0
            batch.flat_rewards = flat_rewards

            # self.validate_batch(self.first_model, batch, trajs_for_first, self.ctx)

            second_batch = self.second_algo.construct_batch(
                all_trajs, cond_info["encoding"].repeat(2, 1), second_log_rewards
            )
            second_batch.num_online = len(all_trajs)
            second_batch.num_offline = 0
            # self.validate_batch(self.second_model, second_batch, trajs_for_second, self.second_algo.ctx)

            self.train_it += worker_info.num_workers if worker_info is not None else 1
            yield BatchTuple(batch, second_batch)

    def log_generated(self, trajs, rewards, flat_rewards, cond_info):
        if self.log_molecule_smis:
            mols = [
                Chem.MolToSmiles(self.ctx.graph_to_mol(trajs[i]["result"])) if trajs[i]["is_valid"] else ""
                for i in range(len(trajs))
            ]
        else:
            mols = [nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(t["result"], None, "v") for t in trajs]

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

    def validate_batch(self, model, batch, trajs, ctx):
        env = GraphBuildingEnv()
        for traj in trajs:
            tp = traj["traj"] + [(traj["result"], None)]
            for t in range(len(tp) - 1):
                if tp[t][1].action == GraphActionType.Stop:
                    continue
                gp = env.step(tp[t][0], tp[t][1])
                assert nx.is_isomorphic(gp, tp[t + 1][0], lambda a, b: a == b, lambda a, b: a == b)

        for actions, atypes in [(batch.actions, ctx.action_type_order)] + (
            [(batch.bck_actions, ctx.bck_action_type_order)]
            if hasattr(batch, "bck_actions") and hasattr(ctx, "bck_action_type_order")
            else []
        ):
            mask_cat = GraphActionCategorical(
                batch,
                [model._action_type_to_mask(t, batch) for t in atypes],
                [model._action_type_to_key[t] for t in atypes],
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
        assert all([type(x) is str or not isinstance(x, Iterable) for x in rows[0]]), "rows must only contain scalars"
        if not self._has_results_table:
            self._make_results_table([type(i) for i in rows[0]], column_names)
        cur = self.db.cursor()
        cur.executemany(f'insert into results values ({",".join("?"*len(rows[0]))})', rows)  # nosec
        cur.close()
        self.db.commit()
