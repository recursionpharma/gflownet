from typing import *

class Config:
    checkpoint_every: Optional[int]
    git_hash: Optional[str]
    hostname: Optional[str]
    log_dir: str
    """The directory where to store logs, checkpoints, and samples."""
    num_final_gen_steps: Optional[int]
    num_training_steps: int
    num_workers: int
    overwrite_existing_exp: bool
    pickle_mp_messages: bool
    seed: int
    start_at_step: int
    validate_every: int
    class algo:
        global_batch_size: int
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
        train_random_action_prob: float
        valid_random_action_prob: float
        valid_sample_cond_info: bool
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
            Z_lr_decay: float
            bootstrap_own_reward: bool
            do_correct_idempotent: bool
            do_parameterize_p_b: bool
            do_subtb: bool
            epsilon: Optional[float]
            reward_loss_multiplier: float
            subtb_max_len: int
    class model:
        num_emb: int
        num_layers: int
        """The number of layers in the model"""
        class graph_transformer:
            ln_type: str
            num_heads: int
            num_mlp_layers: int
    class opt:
        adam_eps: float
        clip_grad_param: float
        clip_grad_type: str
        learning_rate: float
        """The learning rate"""
        lr_decay: float
        momentum: float
        opt: str
        weight_decay: float
    class replay:
        capacity: int
        hindsight_ratio: float
        use: bool
        warmup: int
    class task:
        class qm9:
            h5_path: str
            num_thermometer_dim: int
            temperature_dist_params: List[typing.Any]
            temperature_sample_dist: str
        class seh:
            num_thermometer_dim: int
            temperature_dist_params: List[typing.Any]
            temperature_sample_dist: str
        class seh_moo:
            focus_cosim: float
            focus_limit_coef: float
            focus_model_state_space_res: Optional[int]
            focus_model_training_limits: Optional[typing.Tuple[int, int]]
            focus_type: Union[list, str, NoneType]
            max_train_it: Optional[int]
            n_valid: int
            n_valid_repeats: int
            num_thermometer_dim: int
            objectives: List[str]
            preference_type: Optional[str]
            temperature_parameters: List[typing.Any]
            temperature_sample_dist: str
            use_steer_thermometer: bool
def config_class(name): ...
def config_from_dict(config_dict: dict[str, Any]) -> Config: ...
def make_config() -> Config: ...
def update_config(config: Config, config_dict: dict[str, Any]) -> Config: ...
def config_to_dict(config: Config) -> dict[str, Any]: ...
