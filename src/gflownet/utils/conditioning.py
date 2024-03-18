import abc
from copy import deepcopy
from typing import Dict, Generic, Optional, TypeVar

import numpy as np
import torch
from scipy import stats
from torch import Tensor
from torch.distributions.dirichlet import Dirichlet
from torch_geometric import data as gd

from gflownet import LinScalar, LogScalar, ObjectProperties
from gflownet.config import Config
from gflownet.utils import metrics
from gflownet.utils.focus_model import TabularFocusModel
from gflownet.utils.misc import get_worker_device, get_worker_rng
from gflownet.utils.transforms import thermometer

Tin = TypeVar("Tin")
Tout = TypeVar("Tout")


class Conditional(abc.ABC, Generic[Tin, Tout]):
    def sample(self, n):
        raise NotImplementedError()

    @abc.abstractmethod
    def transform(self, cond_info: Dict[str, Tensor], data: Tin) -> Tout:
        raise NotImplementedError()

    def encoding_size(self):
        raise NotImplementedError()

    def encode(self, conditional: Tensor) -> Tensor:
        raise NotImplementedError()


class TemperatureConditional(Conditional[LogScalar, LogScalar]):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        tmp_cfg = self.cfg.cond.temperature
        self.upper_bound = 1024
        if tmp_cfg.sample_dist == "gamma":
            loc, scale = tmp_cfg.dist_params
            self.upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
        elif tmp_cfg.sample_dist == "uniform":
            self.upper_bound = tmp_cfg.dist_params[1]
        elif tmp_cfg.sample_dist == "loguniform":
            self.upper_bound = tmp_cfg.dist_params[1]
        elif tmp_cfg.sample_dist == "beta":
            self.upper_bound = 1

    def encoding_size(self):
        return self.cfg.cond.temperature.num_thermometer_dim

    def sample(self, n):
        cfg = self.cfg.cond.temperature
        beta = None
        rng = get_worker_rng()
        if cfg.sample_dist == "constant":
            if isinstance(cfg.dist_params[0], (float, int, np.int64, np.int32)):
                beta = np.array(cfg.dist_params[0]).repeat(n).astype(np.float32)
                beta_enc = torch.zeros((n, cfg.num_thermometer_dim))
            else:
                raise ValueError(f"{cfg.dist_params[0]} is not a float)")
        else:
            if cfg.sample_dist == "gamma":
                loc, scale = cfg.dist_params
                beta = rng.gamma(loc, scale, n).astype(np.float32)
            elif cfg.sample_dist == "uniform":
                a, b = float(cfg.dist_params[0]), float(cfg.dist_params[1])
                beta = rng.uniform(a, b, n).astype(np.float32)
            elif cfg.sample_dist == "loguniform":
                low, high = np.log(cfg.dist_params)
                beta = np.exp(rng.uniform(low, high, n).astype(np.float32))
            elif cfg.sample_dist == "beta":
                a, b = float(cfg.dist_params[0]), float(cfg.dist_params[1])
                beta = rng.beta(a, b, n).astype(np.float32)
            beta_enc = thermometer(torch.tensor(beta), cfg.num_thermometer_dim, 0, self.upper_bound)

        assert len(beta.shape) == 1, f"beta should be a 1D array, got {beta.shape}"
        return {"beta": torch.tensor(beta), "encoding": beta_enc}

    def transform(self, cond_info: Dict[str, Tensor], logreward: LogScalar) -> LogScalar:
        assert len(logreward.shape) == len(
            cond_info["beta"].shape
        ), f"dangerous shape mismatch: {logreward.shape} vs {cond_info['beta'].shape}"
        return LogScalar(logreward * cond_info["beta"])

    def encode(self, conditional: Tensor) -> Tensor:
        cfg = self.cfg.cond.temperature
        if cfg.sample_dist == "constant":
            return torch.zeros((conditional.shape[0], cfg.num_thermometer_dim))
        return thermometer(torch.tensor(conditional), cfg.num_thermometer_dim, 0, self.upper_bound)


class MultiObjectiveWeightedPreferences(Conditional[ObjectProperties, LinScalar]):
    def __init__(self, cfg: Config):
        self.cfg = cfg.cond.weighted_prefs
        self.num_objectives = cfg.cond.moo.num_objectives
        self.num_thermometer_dim = cfg.cond.moo.num_thermometer_dim
        if self.cfg.preference_type == "seeded":
            self.seeded_prefs = np.random.default_rng(142857 + int(cfg.seed)).dirichlet([1] * self.num_objectives)

    def sample(self, n):
        if self.cfg.preference_type is None:
            preferences = torch.ones((n, self.num_objectives))
        elif self.cfg.preference_type == "seeded":
            preferences = torch.tensor(self.seeded_prefs).float().repeat(n, 1)
        elif self.cfg.preference_type == "dirichlet_exponential":
            a = np.random.dirichlet([self.cfg.preference_param] * self.num_objectives, n)
            b = np.random.exponential(1, n)[:, None]
            preferences = Dirichlet(torch.tensor(a * b)).sample([1])[0].float()
        elif self.cfg.preference_type == "dirichlet":
            m = Dirichlet(torch.FloatTensor([self.cfg.preference_param] * self.num_objectives))
            preferences = m.sample([n])
        else:
            raise ValueError(f"Unknown preference type {self.cfg.preference_type}")
        preferences = torch.as_tensor(preferences).float()
        return {"preferences": preferences, "encoding": self.encode(preferences)}

    def transform(self, cond_info: Dict[str, Tensor], flat_reward: ObjectProperties) -> LinScalar:
        scalar_reward = (flat_reward * cond_info["preferences"]).sum(1)
        assert len(scalar_reward.shape) == 1, f"scalar_reward should be a 1D array, got {scalar_reward.shape}"
        return LinScalar(scalar_reward)

    def encoding_size(self):
        return max(1, self.num_thermometer_dim * self.num_objectives)

    def encode(self, conditional: Tensor) -> Tensor:
        if self.num_thermometer_dim > 0:
            return thermometer(conditional, self.num_thermometer_dim, 0, 1).reshape(conditional.shape[0], -1)
        else:
            return conditional.unsqueeze(1)


class FocusRegionConditional(Conditional[tuple[ObjectProperties, LogScalar], LogScalar]):
    def __init__(self, cfg: Config, n_valid: int):
        self.cfg = cfg.cond.focus_region
        self.n_valid = n_valid
        self.n_objectives = cfg.cond.moo.num_objectives
        self.ocfg = cfg
        self.num_thermometer_dim = cfg.cond.moo.num_thermometer_dim if self.cfg.use_steer_thermomether else 0

        focus_type = self.cfg.focus_type
        if focus_type is not None and "learned" in focus_type:
            if focus_type == "learned-tabular":
                self.focus_model = TabularFocusModel(
                    device=get_worker_device(),
                    n_objectives=cfg.cond.moo.num_objectives,
                    state_space_res=self.cfg.focus_model_state_space_res,
                )
            else:
                raise NotImplementedError("Unknown focus model type {self.focus_type}")
        else:
            self.focus_model = None
        self.setup_focus_regions()

    def encoding_size(self):
        if self.num_thermometer_dim > 0:
            return self.num_thermometer_dim * self.n_objectives
        return self.n_objectives

    def setup_focus_regions(self):
        # focus regions
        if self.cfg.focus_type is None:
            valid_focus_dirs = np.zeros((self.n_valid, self.n_objectives))
            self.fixed_focus_dirs = valid_focus_dirs
        elif self.cfg.focus_type == "centered":
            valid_focus_dirs = np.ones((self.n_valid, self.n_objectives))
            self.fixed_focus_dirs = valid_focus_dirs
        elif self.cfg.focus_type == "partitioned":
            valid_focus_dirs = metrics.partition_hypersphere(d=self.n_objectives, k=self.n_valid, normalisation="l2")
            self.fixed_focus_dirs = valid_focus_dirs
        elif self.cfg.focus_type in ["dirichlet", "learned-gfn"]:
            valid_focus_dirs = metrics.partition_hypersphere(d=self.n_objectives, k=self.n_valid, normalisation="l1")
            self.fixed_focus_dirs = None
        elif self.cfg.focus_type in ["hyperspherical", "learned-tabular"]:
            valid_focus_dirs = metrics.partition_hypersphere(d=self.n_objectives, k=self.n_valid, normalisation="l2")
            self.fixed_focus_dirs = None
        elif isinstance(self.cfg.focus_type, list):
            if len(self.cfg.focus_type) == 1:
                valid_focus_dirs = np.array([self.cfg.focus_type[0]] * self.n_valid)
                self.fixed_focus_dirs = valid_focus_dirs
            else:
                valid_focus_dirs = np.array(self.cfg.focus_type)
                self.fixed_focus_dirs = valid_focus_dirs
        else:
            raise NotImplementedError(
                f"focus_type should be None, a list of fixed_focus_dirs, or a string describing one of the supported "
                f"focus_type, but here: {self.cfg.focus_type}"
            )
        self.valid_focus_dirs = valid_focus_dirs

    def sample(self, n: int, train_it: Optional[int] = None):
        train_it = train_it or 0
        rng = get_worker_rng()
        if self.fixed_focus_dirs is not None:
            focus_dir = torch.tensor(
                np.array(self.fixed_focus_dirs)[rng.choice(len(self.fixed_focus_dirs), n)].astype(np.float32)
            )
        elif self.cfg.focus_type == "dirichlet":
            m = Dirichlet(torch.FloatTensor([1.0] * self.n_objectives))
            focus_dir = m.sample(torch.Size((n,)))
        elif self.cfg.focus_type == "hyperspherical":
            focus_dir = torch.tensor(
                metrics.sample_positiveQuadrant_ndim_sphere(n, self.n_objectives, normalisation="l2")
            ).float()
        elif self.cfg.focus_type is not None and "learned" in self.cfg.focus_type:
            if (
                self.focus_model is not None
                and train_it >= self.cfg.focus_model_training_limits[0] * self.cfg.max_train_it
            ):
                focus_dir = self.focus_model.sample_focus_directions(n)
            else:
                focus_dir = torch.tensor(
                    metrics.sample_positiveQuadrant_ndim_sphere(n, self.n_objectives, normalisation="l2")
                ).float()
        else:
            raise NotImplementedError(f"Unsupported focus_type={type(self.cfg.focus_type)}")

        return {"focus_dir": focus_dir, "encoding": self.encode(focus_dir)}

    def encode(self, conditional: Tensor) -> Tensor:
        return (
            thermometer(conditional, self.ocfg.cond.moo.num_thermometer_dim, 0, 1).reshape(conditional.shape[0], -1)
            if self.cfg.use_steer_thermomether
            else conditional
        )

    def transform(self, cond_info: Dict[str, Tensor], data: tuple[ObjectProperties, LogScalar]) -> LogScalar:
        flat_rewards, scalar_logreward = data
        focus_coef, in_focus_mask = metrics.compute_focus_coef(
            flat_rewards, cond_info["focus_dir"], self.cfg.focus_cosim, self.cfg.focus_limit_coef
        )
        scalar_logreward = LogScalar(scalar_logreward.clone())  # Avoid modifying the original tensor
        scalar_logreward[in_focus_mask] += torch.log(focus_coef[in_focus_mask])
        scalar_logreward[~in_focus_mask] = self.ocfg.algo.illegal_action_logreward

        return scalar_logreward

    def step_focus_model(self, batch: gd.Batch, train_it: int):
        focus_model_training_limits = self.cfg.focus_model_training_limits
        max_train_it = self.ocfg.num_training_steps
        if (
            self.focus_model is not None
            and train_it >= focus_model_training_limits[0] * max_train_it
            and train_it <= focus_model_training_limits[1] * max_train_it
        ):
            self.focus_model.update_belief(deepcopy(batch.focus_dir), deepcopy(batch.flat_rewards))
