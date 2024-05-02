from typing import List

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
        self.capacity = cfg.replay.capacity
        self.warmup = cfg.replay.warmup
        assert self.warmup <= self.capacity, "ReplayBuffer warmup must be smaller than capacity"

        self.buffer: List[tuple] = []
        self.position = 0

    def push(self, *args):
        if len(self.buffer) == 0:
            self._input_size = len(args)
        else:
            assert self._input_size == len(args), "ReplayBuffer input size must be constant"
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        args = detach_and_cpu(args)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idxs = get_worker_rng().choice(len(self.buffer), batch_size)
        out = list(zip(*[self.buffer[idx] for idx in idxs]))
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
