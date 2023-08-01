from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from gflownet.algo.config import AlgoConfig
from gflownet.data.config import ReplayConfig
from gflownet.models.config import ModelConfig
from gflownet.tasks.config import TasksConfig
from gflownet.utils.config import ConditionalsConfig


@dataclass
class OptimizerConfig:
    """Generic configuration for optimizers

    Attributes
    ----------
    opt : str
        The optimizer to use (either "adam" or "sgd")
    learning_rate : float
        The learning rate
    lr_decay : float
        The learning rate decay (in steps, f = 2 ** (-steps / self.cfg.opt.lr_decay))
    weight_decay : float
        The L2 weight decay
    momentum : float
        The momentum parameter value
    clip_grad_type : str
        The type of gradient clipping to use (either "norm" or "value")
    clip_grad_param : float
        The parameter for gradient clipping
    adam_eps : float
        The epsilon parameter for Adam
    """

    opt: str = "adam"
    learning_rate: float = 1e-4
    lr_decay: float = 20_000
    weight_decay: float = 1e-8
    momentum: float = 0.9
    clip_grad_type: str = "norm"
    clip_grad_param: float = 10.0
    adam_eps: float = 1e-8


@dataclass
class Config:
    """Base configuration for training

    Attributes
    ----------
    log_dir : str
        The directory where to store logs, checkpoints, and samples.
    device : str
        The device to use for training (either "cpu" or "cuda[:<device_id>]")
    seed : int
        The random seed
    validate_every : int
        The number of training steps after which to validate the model
    checkpoint_every : Optional[int]
        The number of training steps after which to checkpoint the model
    print_every : int
        The number of training steps after which to print the training loss
    start_at_step : int
        The training step to start at (default: 0)
    num_final_gen_steps : Optional[int]
        After training, the number of steps to generate graphs for
    num_training_steps : int
        The number of training steps
    num_workers : int
        The number of workers to use for creating minibatches (0 = no multiprocessing)
    hostname : Optional[str]
        The hostname of the machine on which the experiment is run
    pickle_mp_messages : bool
        Whether to pickle messages sent between processes (only relevant if num_workers > 0)
    git_hash : Optional[str]
        The git hash of the current commit
    overwrite_existing_exp : bool
        Whether to overwrite the contents of the log_dir if it already exists
    """

    log_dir: str = MISSING
    device: str = "cuda"
    seed: int = 0
    validate_every: int = 1000
    checkpoint_every: Optional[int] = None
    print_every: int = 100
    start_at_step: int = 0
    num_final_gen_steps: Optional[int] = None
    num_training_steps: int = 10_000
    num_workers: int = 0
    hostname: Optional[str] = None
    pickle_mp_messages: bool = False
    git_hash: Optional[str] = None
    overwrite_existing_exp: bool = True
    algo: AlgoConfig = AlgoConfig()
    model: ModelConfig = ModelConfig()
    opt: OptimizerConfig = OptimizerConfig()
    replay: ReplayConfig = ReplayConfig()
    task: TasksConfig = TasksConfig()
    cond: ConditionalsConfig = ConditionalsConfig()
