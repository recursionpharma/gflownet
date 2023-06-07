# This file was generated automatically
# Do not edit by hand, your changes will be lost
# Regenerate by running `python -m foliconf src/gflownet/config.py`
from typing import Any, Optional, Union

class Config:
    checkpoint_every: Optional[int]
    """The number of training steps after which to checkpoint the model"""
    git_hash: Optional[str]
    """The git hash of the current commit"""
    hostname: Optional[str]
    """The hostname of the machine on which the experiment is run"""
    log_dir: str
    """The directory where to store logs, checkpoints, and samples."""
    num_final_gen_steps: Optional[int]
    """After training, the number of steps to generate graphs for"""
    num_training_steps: int
    """The number of training steps"""
    num_workers: int
    """The number of workers to use for creating minibatches (0 = no multiprocessing)"""
    overwrite_existing_exp: bool
    """Whether to overwrite the contents of the log_dir if it already exists"""
    pickle_mp_messages: bool
    """Whether to pickle messages sent between processes (only relevant if num_workers > 0)"""
    seed: int
    """The random seed"""
    start_at_step: int
    """The training step to start at (default: 0)"""
    validate_every: int
    """The number of training steps after which to validate the model"""

    class algo:
        global_batch_size: int
        """The batch size for training"""
        illegal_action_logreward: float
        """The log reward an agent gets for illegal actions"""
        max_edges: int
        """The maximum number of edges in a generated graph"""
        max_len: int
        """The maximum length of a trajectory"""
        max_nodes: int
        """The maximum number of nodes in a generated graph"""
        method: str
        """The name of the algorithm to use (e.g. "TB")"""
        offline_ratio: float
        sampling_tau: float
        """The EMA factor for the sampling model (theta_sampler = tau * theta_sampler + (1-tau) * theta)"""
        train_random_action_prob: float
        """The probability of taking a random action during training"""
        valid_random_action_prob: float
        """The probability of taking a random action during validation"""
        valid_sample_cond_info: bool
        """Whether to sample conditioning information during validation (if False, expects a validation set of cond_info)"""

        class a2c:
            entropy: float
            gamma: float
            penalty: float

        class fm:
            balanced_loss: bool
            correct_idempotent: bool
            epsilon: float
            leaf_coef: float

        class moql:
            gamma: float
            lambda_decay: int
            num_objectives: int
            num_omega_samples: int
            penalty: float

        class sql:
            alpha: float
            gamma: float
            penalty: float

        class tb:
            Z_learning_rate: float
            """The learning rate for the logZ parameter (only relevant when do_subtb is False)"""
            Z_lr_decay: float
            """The learning rate decay for the logZ parameter (only relevant when do_subtb is False)"""
            bootstrap_own_reward: bool
            """Whether to bootstrap the reward with the own reward. (deprecated)"""
            do_correct_idempotent: bool
            """Whether to correct for idempotent actions"""
            do_parameterize_p_b: bool
            """Whether to parameterize the P_B distribution (otherwise it is uniform)"""
            do_subtb: bool
            """Whether to use the full N^2 subTB loss"""
            epsilon: Optional[float]
            """The epsilon parameter in log-flow smoothing (see paper)"""
            reward_loss_multiplier: float
            """The multiplier for the reward loss when bootstrapping the reward. (deprecated)"""
            subtb_max_len: int
            """The maximum length trajectories, used to cache subTB computation indices"""

    class model:
        num_emb: int
        """The number of dimensions of the embedding"""
        num_layers: int
        """The number of layers in the model"""

        class graph_transformer:
            ln_type: str
            num_heads: int
            num_mlp_layers: int

    class opt:
        adam_eps: float
        """The epsilon parameter for Adam"""
        clip_grad_param: float
        """The parameter for gradient clipping"""
        clip_grad_type: str
        """The type of gradient clipping to use (either "norm" or "value")"""
        learning_rate: float
        """The learning rate"""
        lr_decay: float
        """The learning rate decay (in steps, f = 2 ** (-steps / self.cfg.opt.lr_decay))"""
        momentum: float
        """The momentum parameter value"""
        opt: str
        """The optimizer to use (either "adam" or "sgd")"""
        weight_decay: float
        """The L2 weight decay"""

    class replay:
        capacity: int
        """The capacity of the replay buffer"""
        hindsight_ratio: float
        """The ratio of hindsight samples within a batch"""
        use: bool
        """Whether to use a replay buffer"""
        warmup: int
        """The number of samples to collect before starting to sample from the replay buffer"""

    class task:
        class qm9:
            h5_path: str
            num_thermometer_dim: int
            temperature_dist_params: list[Any]
            temperature_sample_dist: str

        class seh:
            num_thermometer_dim: int
            """The number of thermometer encoding dimensions to use."""
            temperature_dist_params: list[Any]
            """The parameters of the temperature distribution. E.g. for the "uniform" distribution, this is the range."""
            temperature_sample_dist: str
            """The distribution to sample the inverse temperature from. Can be one of:
- "uniform": uniform distribution
- "loguniform": log-uniform distribution
- "gamma": gamma distribution
- "constant": constant temperature"""

        class seh_moo:
            focus_cosim: float
            """The cosine similarity threshold for the focus distribution."""
            focus_limit_coef: float
            """The smoothing coefficient for the focus reward."""
            focus_model_state_space_res: Optional[int]
            """The state space resolution for the focus sampling model (if used)."""
            focus_model_training_limits: Optional[tuple[int, int]]
            """The training limits for the focus sampling model (if used)."""
            focus_type: Union[list, str, None]
            """The type of focus distribtuion used, see SEHMOOTask.setup_focus_regions."""
            max_train_it: Optional[int]
            """The maximum number of training iterations for the focus sampling model (if used)."""
            n_valid: int
            """The number of valid cond_info tensors to sample"""
            n_valid_repeats: int
            """The number of times to repeat the valid cond_info tensors"""
            num_thermometer_dim: int
            """The number of thermometer encoding dimensions to use."""
            objectives: list[str]
            """The objectives to use for the multi-objective optimization. Should be a subset of ["seh", "qed", "sa", "wt"]."""
            preference_type: Optional[str]
            """The preference sampling distribution, defaults to "dirichlet"."""
            temperature_dist_params: list[Any]
            temperature_sample_dist: str
            """The distribution to sample the inverse temperature from. Can be one of:
- "uniform": uniform distribution
- "loguniform": log-uniform distribution
- "gamma": gamma distribution
- "constant": constant temperature"""
            use_steer_thermometer: bool
            """Whether to use a thermometer encoding for the steering."""

def config_class(name): ...
def config_from_dict(config_dict: dict[str, Any]) -> Config: ...
def make_config() -> Config: ...
def update_config(config: Config, config_dict: dict[str, Any]) -> Config: ...
def config_to_dict(config: Config) -> dict[str, Any]: ...
