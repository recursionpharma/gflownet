import os
import time

import wandb

import gflownet.tasks.seh_frag_moo as seh_frag_moo
from gflownet.config import Config, init_empty

TIME = time.strftime("%m-%d-%H-%M")
ENTITY = "valencelabs"
PROJECT = "syngfn"
SWEEP_NAME = f"{TIME}-sehFragMoo-Zlr-Zlrdecay"
STORAGE_DIR = f"~/storage/wandb_sweeps/{SWEEP_NAME}"


# Define the search space of the sweep
sweep_config = {
    "name": SWEEP_NAME,
    "program": "init_wandb_sweep.py",
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
    config.log_dir = (f"{STORAGE_DIR}/{wandb.run.name}-id-{wandb.run.id}",)
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
    # if there are arguments, this is a wandb agent
    if len(sweep_config["parameters"]) > 0:
        wandb.init(entity="valencelabs", project="gflownet")
        config = wandb_config_merger()
        trial = seh_frag_moo.SEHMOOFragTrainer(config)
        trial.run()

    # otherwise, initialize the sweep
    else:
        if os.path.exists(STORAGE_DIR):
            raise ValueError(f"Sweep storage directory {STORAGE_DIR} already exists.")

        wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)
