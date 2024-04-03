import torch
from gflownet.config import Config, init_empty
from gflownet.tasks.seh_frag_moo import SEHMOOFragTrainer


def main():
    """Example of how this model can be run."""
    config = init_empty(Config())
    config.algo.method = "MOQL"
    config.desc = "debug_seh_frag_moo"
    config.log_dir = "./logs/debug_run_sfm"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.num_workers = 0
    config.print_every = 1
    config.algo.num_from_policy = 2
    config.validate_every = 1
    config.num_final_gen_steps = 5
    config.num_training_steps = 3
    config.pickle_mp_messages = True
    config.overwrite_existing_exp = True
    config.algo.sampling_tau = 0.95
    config.algo.train_random_action_prob = 0.01
    config.task.seh_moo.objectives = ["seh", "qed"]
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [60.0]
    config.cond.weighted_prefs.preference_type = "dirichlet"
    config.cond.focus_region.focus_type = None
    config.replay.use = False
    config.task.seh_moo.n_valid = 15
    config.task.seh_moo.n_valid_repeats = 2

    trial = SEHMOOFragTrainer(config)
    trial.run()


if __name__ == "__main__":
    main()