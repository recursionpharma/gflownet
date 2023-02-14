import os
import sqlite3
from typing import Callable, List

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset


class SamplingIterator(IterableDataset):
    """This class allows us to parallelise and train faster.

    By separating sampling data/the model and building torch geometric
    graphs from training the model, we can do the former in different
    processes, which is much faster since much of graph construction
    is CPU-bound.

    """
    def __init__(self, dataset: Dataset, model: nn.Module, batch_size: int, ctx, algo, task, device, ratio=0.5,
                 stream=True, log_dir: str = None, sample_cond_info=True):
        """Parameters
        ----------
        dataset: Dataset
            A dataset instance
        model: nn.Module
            The model we sample from (must be on CUDA already or share_memory() must be called so that
            parameters are synchronized between each worker)
        batch_size: int
            The number of trajectories, each trajectory will be comprised of many graphs, so this is
            _not_ the batch size in terms of the number of graphs (that will depend on the task)
        algo:
            The training algorithm, e.g. a TrajectoryBalance instance
        task: ConditionalTask
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

        """
        self.data = dataset
        self.model = model
        self.batch_size = batch_size
        self.offline_batch_size = int(np.ceil(batch_size * ratio))
        self.online_batch_size = int(np.floor(batch_size * (1 - ratio)))
        self.ratio = ratio
        self.ctx = ctx
        self.algo = algo
        self.task = task
        self.device = device
        self.stream = stream
        self.sample_online_once = True  # TODO: deprecate this, disallow len(data) == 0 entirely
        self.sample_cond_info = sample_cond_info
        self.log_molecule_smis = not hasattr(self.ctx, 'not_a_molecule_env')  # TODO: make this a proper flag
        if not sample_cond_info:
            # Slightly weird semantics, but if we're sampling x given some fixed (data) cond info
            # then "offline" refers to cond info and online to x, so no duplication and we don't end
            # up with 2*batch_size accidentally
            self.offline_batch_size = self.online_batch_size = batch_size
        self.log_dir = log_dir if self.ratio < 1 and self.stream else None
        # This SamplingIterator instance will be copied by torch DataLoaders for each worker, so we
        # don't want to initialize per-worker things just yet, such as where the log the worker writes
        # to. This must be done in __iter__, which is called by the DataLoader once this instance
        # has been copied into a new python process.
        self.log = SQLiteLog()
        self.log_hooks: List[Callable] = []

    def add_log_hook(self, hook: Callable):
        self.log_hooks.append(hook)

    def _idx_iterator(self):
        RDLogger.DisableLog('rdApp.*')
        if self.stream:
            # If we're streaming data, just sample `offline_batch_size` indices
            while True:
                yield self.rng.integers(0, len(self.data), self.offline_batch_size)
        else:
            # Otherwise, figure out which indices correspond to this worker
            worker_info = torch.utils.data.get_worker_info()
            n = len(self.data)
            if n == 0:
                yield np.arange(0, 0)
                return
            if worker_info is None:
                start, end, wid = 0, n, -1
            else:
                nw = worker_info.num_workers
                wid = worker_info.id
                start, end = int(np.round(n / nw * wid)), int(np.round(n / nw * (wid + 1)))
            bs = self.offline_batch_size
            if end - start < bs:
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
        worker_info = torch.utils.data.get_worker_info()
        self._wid = (worker_info.id if worker_info is not None else 0)
        # Now that we know we are in a worker instance, we can initialize per-worker things
        self.rng = self.algo.rng = self.task.rng = np.random.default_rng(142857 + self._wid)
        self.ctx.device = self.device
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f'{self.log_dir}/generated_mols_{self._wid}.db'
            self.log.connect(self.log_path)

        for idcs in self._idx_iterator():
            num_offline = idcs.shape[0]  # This is in [0, self.offline_batch_size]
            # Sample conditional info such as temperature, trade-off weights, etc.

            if self.sample_cond_info:
                cond_info = self.task.sample_conditional_information(num_offline + self.online_batch_size)
                # Sample some dataset data
                mols, flat_rewards = map(list, zip(*[self.data[i] for i in idcs])) if len(idcs) else ([], [])
                flat_rewards = list(self.task.flat_reward_transform(
                    torch.stack(flat_rewards))) if len(flat_rewards) else []
                graphs = [self.ctx.mol_to_graph(m) for m in mols]
                trajs = self.algo.create_training_data_from_graphs(graphs)
                num_online = self.online_batch_size
            else:  # If we're not sampling the conditionals, then the idcs refer to listed preferences
                num_online = num_offline
                num_offline = 0
                cond_info = self.task.encode_conditional_information(torch.stack([self.data[i] for i in idcs]))
                trajs, flat_rewards = [], []

            is_valid = torch.ones(num_offline + num_online).bool()
            # Sample some on-policy data
            if num_online > 0:
                with torch.no_grad():
                    trajs += self.algo.create_training_data_from_own_samples(self.model, num_online,
                                                                             cond_info['encoding'][num_offline:])
                if self.algo.bootstrap_own_reward:
                    # The model can be trained to predict its own reward,
                    # i.e. predict the output of cond_info_to_reward
                    pred_reward = [i['reward_pred'].cpu().item() for i in trajs[num_offline:]]
                    flat_rewards += pred_reward
                else:
                    # Otherwise, query the task for flat rewards
                    valid_idcs = torch.tensor(
                        [i + num_offline for i in range(num_online) if trajs[i + num_offline]['is_valid']]).long()
                    # fetch the valid trajectories endpoints
                    mols = [self.ctx.graph_to_mol(trajs[i]['result']) for i in valid_idcs]
                    # ask the task to compute their reward
                    preds, m_is_valid = self.task.compute_flat_rewards(mols)
                    assert preds.ndim == 2, "FlatRewards should be (mbsize, n_objectives), even if n_objectives is 1"
                    # The task may decide some of the mols are invalid, we have to again filter those
                    valid_idcs = valid_idcs[m_is_valid]
                    valid_mols = [m for m, v in zip(mols, m_is_valid) if v]
                    pred_reward = torch.zeros((num_online, preds.shape[1]))
                    pred_reward[valid_idcs - num_offline] = preds
                    # TODO: reintegrate bootstrapped reward predictions
                    # if preds.shape[0] > 0:
                    #     for i in range(self.number_of_objectives):
                    #         pred_reward[valid_idcs - num_offline, i] = preds[range(preds.shape[0]), i]
                    is_valid[num_offline:] = False
                    is_valid[valid_idcs] = True
                    flat_rewards += list(pred_reward)
                    # Override the is_valid key in case the task made some mols invalid
                    for i in range(num_online):
                        trajs[num_offline + i]['is_valid'] = is_valid[num_offline + i].item()
                    if self.log_molecule_smis:
                        for i, m in zip(valid_idcs, valid_mols):
                            trajs[i]['smi'] = Chem.MolToSmiles(m)
            flat_rewards = torch.stack(flat_rewards)
            # Compute scalar rewards from conditional information & flat rewards
            log_rewards = self.task.cond_info_to_reward(cond_info, flat_rewards)
            log_rewards[torch.logical_not(is_valid)] = self.algo.illegal_action_logreward
            # Construct batch
            batch = self.algo.construct_batch(trajs, cond_info['encoding'], log_rewards)
            batch.num_offline = num_offline
            batch.num_online = num_online
            batch.flat_rewards = flat_rewards
            batch.mols = mols
            # TODO: we could very well just pass the cond_info dict to construct_batch above,
            # and the algo can decide what it wants to put in the batch object
            batch.preferences = cond_info.get('preferences', None)
            if not self.sample_cond_info:
                # If we're using a dataset of preferences, the user may want to know the id of the preference
                for i, j in zip(trajs, idcs):
                    i['data_idx'] = j

            if num_online > 0 and self.log_dir is not None:
                self.log_generated(trajs[num_offline:], log_rewards[num_offline:], flat_rewards[num_offline:],
                                   {k: v[num_offline:] for k, v in cond_info.items()})
            if num_online > 0:
                extra_info = {}
                for hook in self.log_hooks:
                    extra_info.update(
                        hook(trajs[num_offline:], log_rewards[num_offline:], flat_rewards[num_offline:],
                             {k: v[num_offline:] for k, v in cond_info.items()}))
                batch.extra_info = extra_info
            yield batch

    def log_generated(self, trajs, log_rewards, flat_rewards, cond_info):
        if self.log_molecule_smis:
            mols = [
                Chem.MolToSmiles(self.ctx.graph_to_mol(trajs[i]['result'])) if trajs[i]['is_valid'] else ''
                for i in range(len(trajs))
            ]
        else:
            mols = [nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(t['result'], None, 'v') for t in trajs]

        flat_rewards = flat_rewards.reshape((len(flat_rewards), -1)).data.numpy().tolist()
        log_rewards = log_rewards.data.numpy().tolist()
        preferences = cond_info.get('preferences', torch.zeros((len(mols), 0))).data.numpy().tolist()
        logged_keys = [k for k in sorted(cond_info.keys()) if k not in ['encoding', 'preferences']]

        data = ([[mols[i], log_rewards[i]] + flat_rewards[i] + preferences[i] +
                 [cond_info[k][i].item() for k in logged_keys] for i in range(len(trajs))])
        data_labels = (['smi', 'r'] + [f'fr_{i}' for i in range(len(flat_rewards[0]))] +
                       [f'pref_{i}' for i in range(len(preferences[0]))] + [f'ci_{k}' for k in logged_keys])
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
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'").fetchall())
        cur.close()

    def _make_results_table(self, types, names):
        type_map = {str: 'text', float: 'real', int: 'real'}
        col_str = ', '.join(f'{name} {type_map[t]}' for t, name in zip(types, names))
        cur = self.db.cursor()
        cur.execute(f'create table results ({col_str})')
        self._has_results_table = True
        cur.close()

    def insert_many(self, rows, column_names):
        if not self._has_results_table:
            self._make_results_table([type(i) for i in rows[0]], column_names)
        cur = self.db.cursor()
        cur.executemany(f'insert into results values ({",".join("?"*len(rows[0]))})', rows)  # nosec
        cur.close()
        self.db.commit()
