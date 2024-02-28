import logging
import sys

import numpy as np
import torch


def create_logger(name="logger", loglevel=logging.INFO, logfile=None, streamHandle=True):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
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
