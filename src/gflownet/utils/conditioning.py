from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy import stats
from torch import Tensor
from torch.distributions.dirichlet import Dirichlet

from gflownet.config import Config, config_class
from gflownet.utils.transforms import thermometer


class Conditional:
    def sample(self, n):
        raise NotImplementedError()

    def compute_logreward(self, cond_info: Dict[str, Tensor], flat_reward: Tensor) -> Tensor:
        raise NotImplementedError()

    def embedding_size(self):
        raise NotImplementedError()


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

    sample_dist: str = "uniform"
    dist_params: List[Any] = [0, 32]
    num_thermometer_dim: int = 32


class TemperatureConditional(Conditional):
    def __init__(self, cfg: Config):
        self.cfg = cfg.cond.temperature

    def embedding_size(self):
        return self.cfg.num_thermometer_dim

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

    def compute_logreward(self, cond_info: Dict[str, Tensor], flat_reward: Tensor) -> Tensor:
        scalar_logreward = flat_reward.squeeze().clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == len(
            cond_info["beta"].shape
        ), f"dangerous shape mismatch: {scalar_logreward.shape} vs {cond_info['beta'].shape}"
        return scalar_logreward * cond_info["beta"]


@config_class("cond.mowp")
class MOWPConfig:
    preference_type: Optional[str] = "dirichlet"
    num_objectives: int = 2
    num_thermometer_dim: int = 16


class MultiObjectiveWeightedPreferences(Conditional):
    def __init__(self, cfg: Config):
        self.cfg = cfg.cond.mowp
        if self.cfg.preference_type == "seeded":
            self.seeded_prefs = np.random.default_rng(142857 + int(cfg.seed)).dirichlet([1] * self.cfg.num_objectives)

    def sample(self, n):
        if self.cfg.preference_type is None:
            preferences = torch.ones((n, self.cfg.num_objectives))
        elif self.cfg.preference_type == "seeded":
            preferences = torch.tensor(self.seeded_prefs).float().repeat(n, 1)
        elif self.cfg.preference_type == "dirichlet_exponential":
            a = np.random.dirichlet([1] * self.cfg.num_objectives, n)
            b = np.random.exponential(1, n)[:, None]
            preferences = Dirichlet(torch.tensor(a * b)).sample([1])[0].float()
        elif self.cfg.preference_type == "dirichlet":
            m = Dirichlet(torch.FloatTensor([1.0] * self.cfg.num_objectives))
            preferences = m.sample([n])
        else:
            raise ValueError(f"Unknown preference type {self.cfg.preference_type}")
        preferences = torch.as_tensor(preferences).float()
        if self.cfg.num_thermometer_dim > 0:
            enc = thermometer(preferences, self.cfg.num_thermometer_dim, 0, 1)
        else:
            enc = preferences.unsqueeze(1)
        return {"preferences": preferences, "encoding": enc}

    def compute_logreward(self, cond_info: Dict[str, Tensor], flat_reward: Tensor) -> Tensor:
        scalar_logreward = (flat_reward * cond_info["preferences"]).sum(1).clamp(min=1e-30).log()
        assert len(scalar_logreward.shape) == 1, f"scalar_logreward should be a 1D array, got {scalar_logreward.shape}"
        return scalar_logreward

    def embedding_size(self):
        return max(1, self.cfg.num_thermometer_dim)
