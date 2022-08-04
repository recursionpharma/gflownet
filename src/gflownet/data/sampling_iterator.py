import json
import os

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
                 stream=True, log_dir: str = None):
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
        self.log_dir = log_dir if self.ratio < 1 and self.stream else None

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
                start, end = int(np.floor(n / nw * wid)), int(np.ceil(n / nw * (wid + 1)))
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
        return len(self.data)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = (worker_info.id if worker_info is not None else 0)
        self.rng = self.algo.rng = self.task.rng = np.random.default_rng(142857 + self._wid)
        self.ctx.device = self.device
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f'{self.log_dir}/generated_mols_{self._wid}.csv'
        for idcs in self._idx_iterator():
            num_offline = idcs.shape[0]  # This is in [1, self.offline_batch_size]
            # Sample conditional info such as temperature, trade-off weights, etc.
            cond_info = self.task.sample_conditional_information(num_offline + self.online_batch_size)
            is_valid = torch.ones(cond_info['beta'].shape[0]).bool()

            # Sample some dataset data
            mols, flat_rewards = map(list, zip(*[self.data[i] for i in idcs])) if len(idcs) else ([], [])
            flat_rewards = list(self.task.flat_reward_transform(torch.tensor(flat_rewards)))
            graphs = [self.ctx.mol_to_graph(m) for m in mols]
            trajs = self.algo.create_training_data_from_graphs(graphs)
            # Sample some on-policy data
            if self.online_batch_size > 0:
                with torch.no_grad():
                    trajs += self.algo.create_training_data_from_own_samples(self.model, self.online_batch_size,
                                                                             cond_info['encoding'][num_offline:])
                if self.algo.bootstrap_own_reward:
                    # The model can be trained to predict its own reward,
                    # i.e. predict the output of cond_info_to_reward
                    pred_reward = [i['reward_pred'].cpu().item() for i in trajs[num_offline:]]
                    flat_rewards += pred_reward
                else:
                    # Otherwise, query the task for flat rewards
                    valid_idcs = torch.tensor([
                        i + num_offline for i in range(self.online_batch_size) if trajs[i + num_offline]['is_valid']
                    ]).long()
                    pred_reward = torch.zeros((self.online_batch_size))
                    # fetch the valid trajectories endpoints
                    mols = [self.ctx.graph_to_mol(trajs[i]['traj'][-1][0]) for i in valid_idcs]
                    # ask the task to compute their reward
                    preds, m_is_valid = self.task.compute_flat_rewards(mols)
                    # The task may decide some of the mols are invalid, we have to again filter those
                    valid_idcs = valid_idcs[m_is_valid]
                    pred_reward[valid_idcs - num_offline] = preds
                    is_valid[num_offline:] = False
                    is_valid[valid_idcs] = True
                    flat_rewards += list(pred_reward)
                    # Override the is_valid key in case the task made some mols invalid
                    for i in range(self.online_batch_size):
                        trajs[num_offline + i]['is_valid'] = is_valid[num_offline + i].item()
            # Compute scalar rewards from conditional information & flat rewards
            rewards = self.task.cond_info_to_reward(cond_info, flat_rewards)
            rewards[torch.logical_not(is_valid)] = np.exp(self.algo.illegal_action_logreward)
            # Construct batch
            batch = self.algo.construct_batch(trajs, cond_info['encoding'], rewards)
            batch.num_offline = num_offline
            batch.num_online = self.online_batch_size
            # TODO: There is a smarter way to do this
            # batch.pin_memory()
            if self.online_batch_size > 0 and self.log_dir is not None:
                self.log_generated(trajs[num_offline:], rewards[num_offline:], flat_rewards[num_offline:],
                                   {k: v[num_offline:] for k, v in cond_info.items()})
            yield batch

    def log_generated(self, trajs, rewards, flat_rewards, cond_info):
        mols = [
            Chem.MolToSmiles(self.ctx.graph_to_mol(trajs[i]['traj'][-1][0])) if trajs[i]['is_valid'] else ''
            for i in range(len(trajs))
        ]

        def un_tensor(v):
            if isinstance(v, torch.Tensor):
                return v.data.numpy().tolist()

        with open(self.log_path, 'a') as logfile:
            flat_rewards = un_tensor(torch.as_tensor(flat_rewards))
            for i in range(len(trajs)):
                serializable_ci = {k: un_tensor(v[i]) for k, v in cond_info.items() if k != 'encoding'}
                logfile.write(f'{mols[i]},{rewards[i]},{json.dumps(flat_rewards[i])},{json.dumps(serializable_ci)}\n')
