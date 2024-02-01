import torch
import os
import argparse
import shutil
from pathlib import Path
from gflownet.tasks.seh_frag_moo import SEHMOOFragTrainer

def parse_bool(x):
    return x.lower() in ["true", "t", "1"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", type=str, default="no_description")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--just_count", type=parse_bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    parent_dir = Path("./logs/count_state_space")
    parent_dir.mkdir(exist_ok=True, parents=True)
    existing_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]
    log_dir = parent_dir / f"{len(existing_dirs) + 1}_{args.desc}"

    hps = {
        "log_dir": log_dir,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pickle_mp_messages": True,
        "overwrite_existing_exp": False,
        "seed": args.seed,
        "num_training_steps": 10_000,
        "num_final_gen_steps": 1000,
        "validate_every": 1000,
        "num_workers": 0,
        "algo": {
            "global_batch_size": 64,
            "method": "TB",
            "sampling_tau": 0.95,
            "train_random_action_prob": 0.01,
            "tb": {
                "Z_learning_rate": 1e-3,
                "Z_lr_decay": 50000,
            },
        },
        "model": {
            "num_layers": 2,
            "num_emb": 256,
        },
        "task": {
            "seh_moo": {
                "objectives": ["seh", "qed"],
                "n_valid": 15,
                "n_valid_repeats": 128,
                "just_count": args.just_count,
            },
        },
        "opt": {
            "learning_rate": 1e-4,
            "lr_decay": 20000,
        },
        "cond": {
            "temperature": {
                "sample_dist": "constant",
                "dist_params": [60.0],
                "num_thermometer_dim": 32,
            },
            "weighted_prefs": {
                "preference_type": "dirichlet",
            },
            "focus_region": {
                "focus_type": None,  # "learned-tabular",
                "focus_cosim": 0.98,
                "focus_limit_coef": 1e-1,
                "focus_model_training_limits": (0.25, 0.75),
                "focus_model_state_space_res": 30,
                "max_train_it": 5_000,
            },
        },
        "replay": {
            "use": False,
            "warmup": 1000,
            "hindsight_ratio": 0.0,
        },
    }
    if os.path.exists(hps["log_dir"]):
        if hps["overwrite_existing_exp"]:
            shutil.rmtree(hps["log_dir"])
        else:
            raise ValueError(f"Log dir {hps['log_dir']} already exists. Set overwrite_existing_exp=True to delete it.")
    os.makedirs(hps["log_dir"])

    trial = SEHMOOFragTrainer(hps)
    trial.print_every = 100
    trial.run()
