from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SEHTaskConfig:
    reduced_frag: bool = False


@dataclass
class SEHMOOTaskConfig:
    """Config for the SEHMOOTask

    Attributes
    ----------
    n_valid : int
        The number of valid cond_info tensors to sample
    n_valid_repeats : int
        The number of times to repeat the valid cond_info tensors
    objectives : List[str]
        The objectives to use for the multi-objective optimization. Should be a subset of ["seh", "qed", "sa", "mw"].
    """

    n_valid: int = 15
    n_valid_repeats: int = 128
    objectives: List[str] = field(default_factory=lambda: ["seh", "qed", "sa", "mw"])


@dataclass
class QM9TaskConfig:
    h5_path: str = "./data/qm9/qm9.h5"  # see src/gflownet/data/qm9.py
    model_path: str = "./data/qm9/qm9_model.pt"


@dataclass
class QM9MOOTaskConfig:
    """Config for the QM9MOOTask

    Attributes
    ----------
    n_valid : int
        The number of valid cond_info tensors to sample
    n_valid_repeats : int
        The number of times to repeat the valid cond_info tensors
    objectives : List[str]
        The objectives to use for the multi-objective optimization. Should be a subset of ["gap", "qed", "sa", "mw"].
        While "mw" can be used, it is not recommended as the molecules are already small.
    """

    n_valid: int = 15
    n_valid_repeats: int = 128
    objectives: List[str] = field(default_factory=lambda: ["gap", "qed", "sa"])


@dataclass
class TasksConfig:
    qm9: QM9TaskConfig = QM9TaskConfig()
    qm9_moo: QM9MOOTaskConfig = QM9MOOTaskConfig()
    seh: SEHTaskConfig = SEHTaskConfig()
    seh_moo: SEHMOOTaskConfig = SEHMOOTaskConfig()
