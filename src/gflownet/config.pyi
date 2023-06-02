from typing import *

class Config:
    class algo:
        class a2c:
            entropy: float
            gamma: float
            penalty: float
        class fm:
            balanced_loss: bool
            correct_idempotent: bool
            espilon: float
            leaf_coef: float
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
        class moql:
            gamma: float
            lambda_decay: int
            num_objectives: int
            num_omega_samples: int
            penalty: float
        offline_ratio: float
        class sql:
            alpha: float
            gamma: float
            penalty: float
        class tb:
            bootstrap_own_reward: bool
            do_correct_idempotent: bool
            do_parameterize_p_b: bool
            do_subtb: bool
            epsilon: Optional[float]
            reward_loss_multiplier: float
            subtb_max_len: int
        train_random_action_prob: float
        valid_random_action_prob: float
        valid_sample_cond_info: bool
    checkpoint_every: Optional[int]
    hostname: Optional[str]
    log_dir: str
    """The directory where to store logs, checkpoints, and samples."""
    num_final_gen_steps: Optional[int]
    num_training_steps: int
    num_workers: int
    pickle_mp_messages: bool
    class replay:
        capacity: int
        hindsight_ratio: float
        warmup: int
    class seh:
        ...
    start_at_step: int
    validate_every: int
def config_class(name): ...
def config_from_dict(config_dict: dict[str, Any]) -> Config: ...
