import datetime
import torch

from gflownet.config import Config, init_empty
from gflownet.tasks.seh_frag import SEHFragTrainer


if __name__ == "__main__":

    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = f"./logs/debug_run_seh_frag_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True
    
    config.num_training_steps = 100
    config.checkpoint_every = 100
    config.validate_every = 0
    config.num_final_gen_steps = 0
    config.num_workers = 4
    config.opt.lr_decay = 20_000
    config.opt.clip_grad_type = "total_norm"
    config.algo.sampling_tau = 0.95
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [80.0]
    config.replay.use = False
    config.mp_buffer_size = 32 * 1024**2

    if config.replay.use:
        config.algo.num_from_policy = 0
        config.replay.num_new_samples = 64
        config.replay.num_from_replay = 64
    else:
        config.algo.num_from_policy = 64

    trial = SEHFragTrainer(config, print_config=False)
    trial.run()
    trial.terminate()
