import time
from typing import Any

import torch
from torch import Tensor

from gflownet import GFNTask, ObjectProperties
from gflownet.config import Config
from gflownet.utils.communication.method import FileSystemIPC, FileSystemIPC_CSV, IPCModule, NetworkIPC


class IPCTask(GFNTask):
    """The rewards of objects are calculated by different processs"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.setup_ipc_module()
        self.setup_communication()

    def setup_ipc_module(self):
        ipc_cfg = self.cfg.communication
        if ipc_cfg.method == "network":
            ipc_module = NetworkIPC("sampler", ipc_cfg.network.host, ipc_cfg.network.port)
        elif ipc_cfg.method == "file":
            fs_cfg = ipc_cfg.filesystem
            ipc_module = FileSystemIPC("sampler", fs_cfg.workdir, fs_cfg.overwrite_existing_exp)
        elif ipc_cfg.method == "file-csv":
            fs_cfg = ipc_cfg.filesystem
            ipc_module = FileSystemIPC_CSV(
                "sampler", fs_cfg.workdir, fs_cfg.num_objectives, fs_cfg.overwrite_existing_exp
            )
        else:
            raise NotImplementedError
        self.ipc_module: IPCModule = ipc_module

    def setup_communication(self):
        """Set the communication settings"""
        self._ipc_timeout = 600  # wait up to 10 min for reward function
        self._ipc_tick = 0.1  # 0.1 sec

    def compute_obj_properties(self, objs: list[Any]) -> tuple[ObjectProperties, Tensor]:
        assert self.to_reward(objs)
        self.wait_reward()
        rewards, is_valid = self.from_reward()
        assert len(objs) == is_valid.shape[0]
        assert rewards.shape[0] == is_valid.sum()
        return ObjectProperties(rewards), is_valid

    def to_reward(self, objs: list[Any]) -> bool:
        """Send objects to Reward Process

        Parameters
        ----------
        objs : list[Any]
            A list of n sampled objects

        Returns
        -------
        status: bool
        """
        flag = self.ipc_module.sampler_to_reward(objs)
        self.ipc_module.sampler_unlock_reward()
        return flag

    def wait_reward(self) -> bool:
        """Wait Reward Process"""
        tick_st = time.time()
        tick = lambda: time.time() - tick_st  # noqa: E731
        while self.ipc_module.sampler_wait_reward():
            assert tick() <= self._ipc_timeout, f"Timeout! ({tick()} sec)"
            time.sleep(self._ipc_tick)
        return True

    def from_reward(self) -> tuple[Tensor, Tensor]:
        """Receive rewards from Reward Process
        Returns
        -------
        tuple[Tensor, Tensor]
            - rewards: FloatTensor [m, n_objectives]
                rewards of n<=m valid objects
            - is_valid: BoolTensor [n,]
                flags whether the objects are valid or not
        """
        fr, is_valid = self.ipc_module.sampler_from_reward()
        return torch.tensor(fr, dtype=torch.float32), torch.tensor(is_valid, dtype=torch.bool)

    def terminate_reward(self):
        """Terminate Reward Process"""
        self.ipc_module.sampler_terminate_reward()
