from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from gflownet.utils.misc import StrictDataClass


class Backward(IntEnum):
    """
    See algo.trajectory_balance.TrajectoryBalance for details.
    The A variant of `Maxent` and `GSQL` equire the environment to provide $n$.
    This is true for sEH but not QM9.
    """

    Uniform = 1
    Free = 2
    Maxent = 3
    MaxentA = 4
    GSQL = 5
    GSQLA = 6


class NLoss(IntEnum):
    """See algo.trajectory_balance.TrajectoryBalance for details."""

    none = 0
    Transition = 1
    SubTB1 = 2
    TermTB1 = 3
    StartTB1 = 4
    TB = 5


class TBVariant(IntEnum):
    """See algo.trajectory_balance.TrajectoryBalance for details."""

    TB = 0
    SubTB1 = 1
    DB = 2


class LossFN(IntEnum):
    """
    The loss function to use.

    - GHL:  Kaan Gokcesu, Hakan Gokcesu
    https://arxiv.org/pdf/2108.12627.pdf,
    Note: This can be used as a differentiable version of HUB.
    """

    MSE = 0
    MAE = 1
    HUB = 2
    GHL = 3


@dataclass
class TBConfig(StrictDataClass):
    """Trajectory Balance config.

    Attributes
    ----------
    bootstrap_own_reward : bool
        Whether to bootstrap the reward with the own reward. (deprecated)
    epsilon : Optional[float]
        The epsilon parameter in log-flow smoothing (see paper)
    reward_loss_multiplier : float
        The multiplier for the reward loss when bootstrapping the reward. (deprecated)
    variant : TBVariant
        The loss variant. See algo.trajectory_balance.TrajectoryBalance for details.
    do_correct_idempotent : bool
        Whether to correct for idempotent actions
    do_parameterize_p_b : bool
        Whether to parameterize the P_B distribution (otherwise it is uniform)
    do_predict_n : bool
        Whether to predict the number of paths in the graph
    do_length_normalize : bool
        Whether to normalize the loss by the length of the trajectory
    subtb_max_len : int
        The maximum length trajectories, used to cache subTB computation indices
    Z_learning_rate : float
        The learning rate for the logZ parameter (only relevant when do_subtb is False)
    Z_lr_decay : float
        The learning rate decay for the logZ parameter (only relevant when do_subtb is False)
    loss_fn: LossFN
        The loss function to use
    loss_fn_par: float
        The loss function parameter in case of Huber loss, it is the delta
    n_loss: NLoss
        The $n$ loss to use (defaults to NLoss.none i.e., do not learn $n$)
    n_loss_multiplier: float
        The multiplier for the $n$ loss
    backward_policy: Backward
        The backward policy to use
    """

    bootstrap_own_reward: bool = False
    epsilon: Optional[float] = None
    reward_loss_multiplier: float = 1.0
    variant: TBVariant = TBVariant.TB
    do_correct_idempotent: bool = False
    do_parameterize_p_b: bool = False
    do_predict_n: bool = False
    do_sample_p_b: bool = False
    do_length_normalize: bool = False
    subtb_max_len: int = 128
    Z_learning_rate: float = 1e-4
    Z_lr_decay: float = 50_000
    cum_subtb: bool = True
    loss_fn: LossFN = LossFN.MSE
    loss_fn_par: float = 1.0
    n_loss: NLoss = NLoss.none
    n_loss_multiplier: float = 1.0
    backward_policy: Backward = Backward.Uniform


@dataclass
class MOQLConfig(StrictDataClass):
    gamma: float = 1
    num_omega_samples: int = 32
    num_objectives: int = 2
    lambda_decay: int = 10_000
    penalty: float = -10


@dataclass
class A2CConfig(StrictDataClass):
    entropy: float = 0.01
    gamma: float = 1
    penalty: float = -10


@dataclass
class FMConfig(StrictDataClass):
    epsilon: float = 1e-38
    balanced_loss: bool = False
    leaf_coef: float = 10
    correct_idempotent: bool = False


@dataclass
class SQLConfig(StrictDataClass):
    alpha: float = 0.01
    gamma: float = 1
    penalty: float = -10


@dataclass
class AlgoConfig(StrictDataClass):
    """Generic configuration for algorithms

    Attributes
    ----------
    method : str
        The name of the algorithm to use (e.g. "TB")
    num_from_policy : int
        The number of on-policy samples for a training batch.
        If using a replay buffer, see `replay.num_from_replay` for the number of samples from the replay buffer, and
        `replay.num_new_samples` for the number of new samples to add to the replay buffer (e.g. `num_from_policy=0`,
        and `num_new_samples=N` inserts `N` new samples in the replay buffer at each step, but does not make that data
        part of the training batch).
    num_from_dataset : int
        The number of samples from the dataset for a training batch
    valid_num_from_policy : int
        The number of on-policy samples for a validation batch
    valid_num_from_dataset : int
        The number of samples from the dataset for a validation batch
    max_len : int
        The maximum length of a trajectory
    max_nodes : int
        The maximum number of nodes in a generated graph
    max_edges : int
        The maximum number of edges in a generated graph
    illegal_action_logreward : float
        The log reward an agent gets for illegal actions
    train_random_action_prob : float
        The probability of taking a random action during training
    train_det_after: Optional[int]
        Do not take random actions after this number of steps
    valid_random_action_prob : float
        The probability of taking a random action during validation
    sampling_tau : float
        The EMA factor for the sampling model (theta_sampler = tau * theta_sampler + (1-tau) * theta)
    """

    method: str = "TB"
    num_from_policy: int = 64
    num_from_dataset: int = 0
    valid_num_from_policy: int = 64
    valid_num_from_dataset: int = 0
    max_len: int = 128
    max_nodes: int = 128
    max_edges: int = 128
    illegal_action_logreward: float = -100
    train_random_action_prob: float = 0.0
    train_det_after: Optional[int] = None
    valid_random_action_prob: float = 0.0
    sampling_tau: float = 0.0
    tb: TBConfig = field(default_factory=TBConfig)
    moql: MOQLConfig = field(default_factory=MOQLConfig)
    a2c: A2CConfig = field(default_factory=A2CConfig)
    fm: FMConfig = field(default_factory=FMConfig)
    sql: SQLConfig = field(default_factory=SQLConfig)
