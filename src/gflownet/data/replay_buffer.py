import heapq
from threading import Lock
from typing import Any, List

import numpy as np
import torch

from gflownet.config import Config
from gflownet.utils.misc import get_worker_rng


class ReplayBuffer(object):
    def __init__(self, cfg: Config):
        """
        Replay buffer for storing and sampling arbitrary data (e.g. transitions or trajectories)
        In self.push(), the buffer detaches any torch tensor and sends it to the CPU.
        """
        self.capacity = cfg.replay.capacity or int(1e6)
        self.warmup = cfg.replay.warmup or 0
        assert self.warmup <= self.capacity, "ReplayBuffer warmup must be smaller than capacity"

        self.buffer: List[tuple] = []
        self.position = 0

        self.treat_as_heap = cfg.replay.keep_highest_rewards
        self.filter_uniques = cfg.replay.keep_only_uniques
        self._uniques: set[Any] = set()

        self._lock = Lock()

    def push(self, *args, unique_obj=None, priority=None):
        """unique_obj must be hashable and comparable"""
        if len(self.buffer) == 0:
            self._input_size = len(args)
        else:
            assert self._input_size == len(args), "ReplayBuffer input size must be constant"
        if self.filter_uniques and unique_obj in self._uniques:
            return
        args = detach_and_cpu(args)
        self._lock.acquire()
        if self.treat_as_heap:
            if len(self.buffer) >= self.capacity:
                if priority is None or priority > self.buffer[0][0]:
                    # We will use self.position for tie-breaking
                    *_, pop_unique = heapq.heappushpop(self.buffer, (priority, self.position, args, unique_obj))
                    self.position += 1
                    if self.filter_uniques:
                        self._uniques.remove(pop_unique)
                        self._uniques.add(unique_obj)
                else:
                    pass  # If the priority is lower than the lowest in the heap, we don't add it
            else:
                heapq.heappush(self.buffer, (priority, self.position, args, unique_obj))
                self.position += 1
                if self.filter_uniques:
                    self._uniques.add(unique_obj)
        else:
            if len(self.buffer) < self.capacity:
                self.buffer.append(())
            if self.filter_uniques:
                if self.position == 0 and len(self.buffer) == self.capacity:
                    # We're about to wrap around, so remove the oldest element
                    self._uniques.remove(self.buffer[0][2])
            self.buffer[self.position] = (priority, args, unique_obj)
            if self.filter_uniques:
                self._uniques.add(unique_obj)
            self.position = (self.position + 1) % self.capacity
        self._lock.release()

    def sample(self, batch_size):
        idxs = get_worker_rng().choice(len(self.buffer), batch_size)
        out = list(zip(*[self.buffer[idx][2] for idx in idxs]))
        for i in range(len(out)):
            # stack if all elements are numpy arrays or torch tensors
            # (this is much more efficient to send arrays through multiprocessing queues)
            if all([isinstance(x, np.ndarray) for x in out[i]]):
                out[i] = np.stack(out[i], axis=0)
            elif all([isinstance(x, torch.Tensor) for x in out[i]]):
                out[i] = torch.stack(out[i], dim=0)
            else:
                out[i] = list(out[i])
        return out

    def __len__(self):
        return len(self.buffer)


def detach_and_cpu(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    elif isinstance(x, dict):
        x = {k: detach_and_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        x = [detach_and_cpu(v) for v in x]
    elif isinstance(x, tuple):
        x = tuple(detach_and_cpu(v) for v in x)
    return x
