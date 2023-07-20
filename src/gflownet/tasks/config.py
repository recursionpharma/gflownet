from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SEHTaskConfig:
    pass  # SEH just uses a temperature conditional


@dataclass
class SEHMOOTaskConfig:
    """Config for the SEHMOOTask

    Attributes
    ----------
    use_steer_thermometer : bool
        Whether to use a thermometer encoding for the steering.
    preference_type : Optional[str]
        The preference sampling distribution, defaults to "dirichlet".
    focus_type : Union[list, str, None]
        The type of focus distribtuion used, see SEHMOOTask.setup_focus_regions.
    focus_cosim : float
        The cosine similarity threshold for the focus distribution.
    focus_limit_coef : float
        The smoothing coefficient for the focus reward.
    focus_model_training_limits : Optional[Tuple[int, int]]
        The training limits for the focus sampling model (if used).
    focus_model_state_space_res : Optional[int]
        The state space resolution for the focus sampling model (if used).
    max_train_it : Optional[int]
        The maximum number of training iterations for the focus sampling model (if used).
    n_valid : int
        The number of valid cond_info tensors to sample
    n_valid_repeats : int
        The number of times to repeat the valid cond_info tensors
    objectives : List[str]
        The objectives to use for the multi-objective optimization. Should be a subset of ["seh", "qed", "sa", "wt"].
    """

    use_steer_thermometer: bool = False
    preference_type: Optional[str] = "dirichlet"
    focus_type: Optional[str] = None
    focus_dirs_listed: Optional[List[List[float]]] = None
    focus_cosim: float = 0.0
    focus_limit_coef: float = 1.0
    focus_model_training_limits: Optional[Tuple[int, int]] = None
    focus_model_state_space_res: Optional[int] = None
    max_train_it: Optional[int] = None
    n_valid: int = 15
    n_valid_repeats: int = 128
    objectives: List[str] = field(default_factory=lambda: ["seh", "qed", "sa", "mw"])


@dataclass
class QM9TaskConfig:
    h5_path: str = "./data/qm9/qm9.h5"  # see src/gflownet/data/qm9.py
    model_path: str = "./data/qm9/qm9_model.pt"


@dataclass
class TasksConfig:
    qm9: QM9TaskConfig = QM9TaskConfig()
    seh: SEHTaskConfig = SEHTaskConfig()
    seh_moo: SEHMOOTaskConfig = SEHMOOTaskConfig()
