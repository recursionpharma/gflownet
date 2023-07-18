from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
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
    dist_params: List[Any] = field(default_factory=lambda: [0.5, 32])
    num_thermometer_dim: int = 32


@dataclass
class MultiObjectiveConfig:
    num_objectives: int = 2
    num_thermometer_dim: int = 16


@dataclass
class WeightedPreferencesConfig:
    """Config for the weighted preferences conditional.

    Attributes
    ----------
    preference_type : str
        The preference sampling distribution, defaults to "dirichlet". Can be one of:
        - "dirichlet": Dirichlet distribution
        - "dirichlet_exponential": Dirichlet distribution with exponential temperature
        - "seeded": Enumerated preferences
        - None: All rewards equally weighted"""

    preference_type: Optional[str] = "dirichlet"


@dataclass
class FocusRegionConfig:
    """Config for the focus region conditional.

    Attributes
    ----------
    focus_type : str
        The type of focus distribtuion used, see FocusRegionConditon.setup_focus_regions. Can be one of:
        [None, "centered", "partitioned", "dirichlet", "hyperspherical", "learned-gfn", "learned-tabular"]
    """

    focus_type: Optional[str] = "learned-tabular"
    use_steer_thermomether: bool = False
    focus_cosim: float = 0.98
    focus_limit_coef: float = 0.1
    focus_model_training_limits: tuple[float, float] = (0.25, 0.75)
    focus_model_state_space_res: int = 30
    max_train_it: int = 20_000


@dataclass
class ConditionalsConfig:
    temperature: TempCondConfig = TempCondConfig()
    moo: MultiObjectiveConfig = MultiObjectiveConfig()
    weighted_prefs: WeightedPreferencesConfig = WeightedPreferencesConfig()
    focus_region: FocusRegionConfig = FocusRegionConfig()
