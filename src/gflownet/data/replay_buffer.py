from typing import List

import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity: int = 100000, warmup: int = 0, rng: np.random.Generator = None):
        self.capacity = capacity
        self.warmup = warmup
        assert self.warmup <= self.capacity, "ReplayBuffer warmup must be smaller than capacity"

        self.buffer: List[tuple] = []
        self.position = 0
        self.rng = rng

    def push(self, traj, flat_rewards, cond_info):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (traj, flat_rewards, cond_info)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idxs = self.rng.choice(len(self.buffer), batch_size)
        trajs, flat_rewards, cond_info = zip(*[self.buffer[idx] for idx in idxs])
        return trajs, flat_rewards, cond_info

    def __len__(self):
        return len(self.buffer)
