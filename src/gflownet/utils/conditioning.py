from typing import Any, Dict, List

import numpy as np
import torch
from scipy import stats
from torch import Tensor

from gflownet.config import Config, config_class
from gflownet.utils.transforms import thermometer


class Conditional:
    def sample(self, n):
        raise NotImplementedError

    def compute_reward(self, cond_info: Dict[str, Tensor], flat_reward: Tensor) -> Tensor:
        raise NotImplementedError


@config_class("cond.temperature")
class TempCondConfig:
    """Config for the temperature conditional.

    Attributes
    ----------

    sample_dist : str
        The distribution to sample the inverse temperature from. Can be one of:
        - "uniform": uniform distribution
        - "loguniform": log-uniform distribution
        - "gamma": gamma distribution
        - "constant": constant temperature
        - "beta": beta distribution
    dist_params : List[Any]
        The parameters of the temperature distribution. E.g. for the "uniform" distribution, this is the range.
    num_thermometer_dim : int
        The number of thermometer encoding dimensions to use.
    """

    sample_dist: str
    dist_params: List[Any]
    num_thermometer_dim: int


class TemperatureConditional(Conditional):
    def __init__(self, cfg: Config):
        self.cfg = cfg.cond.temperature

    def sample(self, n):
        beta = None
        if self.cfg.sample_dist == "constant":
            assert type(self.cfg.dist_params) is float
            beta = np.array(self.cfg.dist_params).repeat(n).astype(np.float32)
            beta_enc = torch.zeros((n, self.cfg.num_thermometer_dim))
        else:
            if self.cfg.sample_dist == "gamma":
                loc, scale = self.cfg.dist_params
                beta = self.rng.gamma(loc, scale, n).astype(np.float32)
                upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
            elif self.cfg.sample_dist == "uniform":
                a, b = float(self.cfg.dist_params[0]), float(self.cfg.dist_params[1])
                beta = self.rng.uniform(a, b, n).astype(np.float32)
                upper_bound = self.cfg.dist_params[1]
            elif self.cfg.sample_dist == "loguniform":
                low, high = np.log(self.cfg.dist_params)
                beta = np.exp(self.rng.uniform(low, high, n).astype(np.float32))
                upper_bound = self.cfg.dist_params[1]
            elif self.cfg.sample_dist == "beta":
                a, b = float(self.cfg.dist_params[0]), float(self.cfg.dist_params[1])
                beta = self.rng.beta(a, b, n).astype(np.float32)
                upper_bound = 1
            beta_enc = thermometer(torch.tensor(beta), self.cfg.num_thermometer_dim, 0, upper_bound)

        assert len(beta.shape) == 1, f"beta should be a 1D array, got {beta.shape}"
        return {"beta": torch.tensor(beta), "encoding": beta_enc}

    def compute_reward(self, cond_info: Dict[str, Tensor], flat_reward: Tensor) -> Tensor:
        if isinstance(flat_reward, list):
            flat_reward = torch.tensor(flat_reward)
        scalar_logreward = flat_reward.squeeze().clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == len(
            cond_info["beta"].shape
        ), f"dangerous shape mismatch: {scalar_logreward.shape} vs {cond_info['beta'].shape}"
        return scalar_logreward * cond_info["beta"]
