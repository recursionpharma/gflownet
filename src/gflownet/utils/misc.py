import logging
import sys

import numpy as np
import torch


def create_logger(name="logger", loglevel=logging.INFO, logfile=None, streamHandle=True):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    while len([logger.removeHandler(i) for i in logger.handlers]):
        pass  # Remove all handlers (only useful when debugging)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - {} - %(message)s".format(name),
        datefmt="%d/%m/%Y %H:%M:%S",
    )

    handlers = []
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="a"))
    if streamHandle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


_worker_rngs = {}
_worker_rng_seed = [142857]
_main_process_device = [torch.device("cpu")]


def get_worker_rng():
    worker_info = torch.utils.data.get_worker_info()
    wid = worker_info.id if worker_info is not None else 0
    if wid not in _worker_rngs:
        _worker_rngs[wid] = np.random.RandomState(_worker_rng_seed[0] + wid)
    return _worker_rngs[wid]


def set_worker_rng_seed(seed):
    _worker_rng_seed[0] = seed
    for wid in _worker_rngs:
        _worker_rngs[wid].seed(seed + wid)


def set_main_process_device(device):
    _main_process_device[0] = device


def get_worker_device():
    worker_info = torch.utils.data.get_worker_info()
    return _main_process_device[0] if worker_info is None else torch.device("cpu")


class StrictDataClass:
    """
    A dataclass that raises an error if any field is created outside of the __init__ method.
    """

    def __setattr__(self, name, value):
        if hasattr(self, name) or name in self.__annotations__:
            super().__setattr__(name, value)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'."
                f" '{type(self).__name__}' is a StrictDataClass object."
                f" Attributes can only be defined in the class definition."
            )
