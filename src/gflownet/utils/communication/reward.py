import logging
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from gflownet.utils.communication.config import CommunicationConfig
from gflownet.utils.communication.method import FileSystemIPC, FileSystemIPC_CSV, IPCModule, NetworkIPC


class RewardModule:
    """Reward Module which communicates with trainer running on the other process.
    To avoid the dependency issues, gflownet sources are not imported.
    Therefore, it is available to run without torch library.
    If you want to run reward function on your own dependencies or environments, copy this class.
    """

    def __init__(self, gfn_log_dir, verbose_level: int = logging.INFO):
        # Wait up to 10 mins while the gflownet starts and load gfn config
        self.gfn_log_dir = Path(gfn_log_dir)
        self.gfn_config_path = self.gfn_log_dir / "config.yaml"
        st = time.time()
        while (not self.gfn_config_path.exists()) and (time.time() - st < 600):
            time.sleep(0.1)
        assert self.gfn_config_path.exists(), "Timeout (10 mins) - GFN log direcotry is not created"

        self.gfn_cfg = OmegaConf.load(self.gfn_config_path)
        self.cfg: CommunicationConfig = self.gfn_cfg.communication

        # Setup IPC Module
        self.setup_ipc_module()
        self.setup_communication()
        self.logger: logging.Logger = create_logger("reward", verbose_level)
        self.oracle_idx = 0

    def setup_ipc_module(self):
        ipc_cfg = self.cfg
        if ipc_cfg.method == "network":
            ipc_module = NetworkIPC("reward", ipc_cfg.network.host, ipc_cfg.network.port)
        elif ipc_cfg.method == "file":
            fs_cfg = ipc_cfg.filesystem
            ipc_module = FileSystemIPC("reward", fs_cfg.workdir, fs_cfg.overwrite_existing_exp)
        elif ipc_cfg.method == "file-csv":
            fs_cfg = ipc_cfg.filesystem
            ipc_module = FileSystemIPC_CSV(
                "reward", fs_cfg.workdir, fs_cfg.num_objectives, fs_cfg.overwrite_existing_exp
            )
        else:
            raise NotImplementedError
        self.ipc_module: IPCModule = ipc_module

    def setup_communication(self):
        """Write here the communication settings"""
        self._ipc_timeout = 60  # wait up to 1 min for gflownet
        self._ipc_tick = 0.1  # 0.1 sec

    """Running API"""

    def run(self):
        while self.wait_sampler():
            # communication: receive objects
            objs = self.from_sampler()
            # compute obj_props
            obj_props, is_valid = self.compute_obj_properties(objs)
            # communication: send obj_props
            self.to_sampler(obj_props, is_valid)
            # logging (async)
            self.log(objs, obj_props, is_valid)
            self.oracle_idx += 1

    def compute_obj_properties(self, objs: list[Any]) -> tuple[list[list[float]], list[bool]]:
        self.logger.debug(f"receive {len(objs)} objects")
        # convert and filter the objects before reward calculation
        st = time.time()
        converted_objs = [self.convert_object(obj) for obj in objs]
        is_valid = [self.filter_object(obj) for obj in converted_objs]
        valid_objs = [obj for flag, obj in zip(is_valid, converted_objs, strict=True) if flag]
        tick = time.time() - st
        self.logger.debug(f"get the {len(valid_objs)} valid objects (filter: {tick:.3f} sec)")

        # reward calculation
        st = time.time()
        obj_props = self._compute_obj_props(valid_objs)
        tick = time.time() - st
        self.logger.debug(f"finish obj_prop calculation! (obj_prop: {tick:.3f} sec)")
        return obj_props, is_valid

    """Implement following methods!"""

    @property
    def num_objectives(self) -> int:
        raise NotImplementedError

    def convert_object(self, obj: Any) -> Any | None:
        """Implement the conversion function if required

        Parameters
        ----------
        obj : Any
            A valid object(graph, seq, mol, ...)

        Returns
        -------
        obj: Any
            return a converted object
        """
        return obj

    def filter_object(self, obj: Any) -> bool:
        """Implement the constraint here if required

        Parameters
        ----------
        obj : Any
            A valid object(graph, seq, mol, ...)

        Returns
        -------
        is_valid: bool
            return whether the object is valid or not
        """
        return obj is not None

    def compute_obj_prop_batch(self, objs: list[Any]) -> list[list[float]]:
        """Modify here if parallel computation is required

        Parameters
        ----------
        objs : list[Any]
            A list of valid objects(graphs, seqs, mols, ...)

        Returns
        -------
        obj_props: list[list[float]]
            Each item of list should be list of reward for each objective
            assert len(obj_props) == len(objs)
        """
        return [self.compute_obj_prop_single(obj) for obj in objs]

    def compute_obj_prop_single(self, obj: Any) -> list[float]:
        """Implement the reward function which calculates the reward of each object individually

        Parameters
        ----------
        obj : Any
            A valid object(graph, seq, mol, ...)

        Returns
        -------
        obj_props: list[float]
            It shoule be list of the property for each objective
            The negative value would be clipped to 0 because GFlowNets require a non-negative reward.
            assert len(obj_prop) == self.num_objectives
        """
        raise NotImplementedError

    def log(self, objs: list[Any], obj_props: list[list[float]], is_valid: list[bool]):
        """Log Hook

        Parameters
        ----------
        objs : list[Any]
            A list of objects
        obj_props : list[list[float]]
            A list of reward for each object
        is_valid : list[bool]
            A list of valid flag of each objects
        """
        pass

    """Inner function"""

    def wait_sampler(self) -> bool:
        tick_st = time.time()
        tick = lambda: time.time() - tick_st  # noqa: E731
        while self.ipc_module.reward_wait_sampler():
            if self.ipc_module.reward_is_terminated():
                self.logger.info("Sampler is terminated")
                return False
            if tick() > self._ipc_timeout:
                self.logger.warning(f"Timeout! ({tick()} sec)")
                return False
            time.sleep(self._ipc_tick)
        return True

    def from_sampler(self) -> list[Any]:
        """Receive objects from GFlowNet process
        if the gflownet process is terminated, it will return None"""
        return self.ipc_module.reward_from_sampler()

    def to_sampler(self, obj_props: list[list[float]], is_valid: list[bool]):
        """Send reward to GFlowNet process"""
        self.ipc_module.reward_to_sampler(obj_props, is_valid)
        self.ipc_module.reward_unlock_sampler()

    def _compute_obj_props(self, objs: list[Any]) -> list[list[float]]:
        """To prevent unsafe actions"""
        if len(objs) == 0:
            return []
        else:
            rs = self.compute_obj_prop_batch(objs)
            for r in rs[1:]:
                assert (
                    len(r) == self.num_objectives
                ), f"The number of rewards ({len(r)}) is different to the number of objectives ({self.num_objectives})"
            assert len(rs) == len(
                objs
            ), f"The number of outputs {len(rs)} should be same to the number of samples ({len(objs)})"
            return rs


def create_logger(name="logger", loglevel=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    while len([logger.removeHandler(i) for i in logger.handlers]):
        pass  # Remove all handlers (only useful when debugging)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - {} - %(message)s".format(name),
        datefmt="%d/%m/%Y %H:%M:%S",
    )

    handlers = []
    handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
