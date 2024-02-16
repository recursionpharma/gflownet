from pathlib import Path

import wandb

import gflownet
from gflownet.config import Config, init_empty

sweep_config = {
    "name": "sehFragMoo-Zlr-Zlrdecay",
    "program": "wandb_agent_main.py",
    "controller": {
        "type": "cloud",
    },
    "method": "grid",
    "parameters": {
        "config.algo.tb.Z_learning_rate": {"values": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]},
        "config.algo.tb.Z_lr_decay": {"values": [2_000, 10_000, 50_000, 250_000]},
    },
}


def wandb_config_merger():
    config = init_empty(Config())
    wandb_config = wandb.config

    # Set desired config values
    config.log_dir = str(Path(gflownet.__file__).parent / "sweeps" / sweep_config["name"] / "run_logs" / wandb.run.name)
    config.print_every = 100
    config.validate_every = 1000
    config.num_final_gen_steps = 1000
    config.num_training_steps = 40_000
    config.pickle_mp_messages = True
    config.overwrite_existing_exp = False
    config.algo.sampling_tau = 0.95
    config.algo.train_random_action_prob = 0.01
    config.algo.tb.Z_learning_rate = 1e-3
    config.task.seh_moo.objectives = ["seh", "qed"]
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [60.0]
    config.cond.weighted_prefs.preference_type = "dirichlet"
    config.cond.focus_region.focus_type = None
    config.replay.use = False

    # Merge the wandb sweep config with the nested config from gflownet
    config.algo.tb.Z_learning_rate = wandb_config["config.algo.tb.Z_learning_rate"]
    config.algo.tb.Z_lr_decay = wandb_config["config.algo.tb.Z_lr_decay"]

    return config


if __name__ == "__main__":
    wandb.sweep(sweep_config, entity="valencelabs", project="gflownet")
