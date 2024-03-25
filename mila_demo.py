import datetime
import torch

from gflownet.config import Config, init_empty
from gflownet.tasks.seh_frag import SEHFragTrainer


if __name__ == "__main__":
    # This script runs on an A100 with 8 cpus and 32Gb memory, but the A100 is probably overkill here.
    # VRAM peaks at 6Gb and GPU usage peaks at 25%.
    config = init_empty(Config())
    config.print_every = 1
    config.log_dir = f"./logs/debug_run_seh_frag_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.overwrite_existing_exp = True
    
    config.num_training_steps = 1000 # Change this to train for longer
    config.checkpoint_every = 500
    config.validate_every = 0
    config.num_final_gen_steps = 0
    config.opt.lr_decay = 20_000
    config.opt.clip_grad_type = "total_norm"
    config.algo.sampling_tau = 0.9
    config.cond.temperature.sample_dist = "constant"
    config.cond.temperature.dist_params = [64.0]
    config.replay.use = False

    ###
    # Things it may be fun to play with
    config.num_workers = 8
    config.model.num_emb = 128
    config.model.num_layers = 4
    batch_size = 64
    ###

    if config.replay.use:
        config.algo.num_from_policy = 0
        config.replay.num_new_samples = batch_size
        config.replay.num_from_replay = batch_size
    else:
        config.algo.num_from_policy = batch_size

    # This may need to be adjusted if the batch_size is made bigger
    config.mp_buffer_size = 32 * 1024**2  # 32Mb

    trial = SEHFragTrainer(config, print_config=False)
    trial.run()
    trial.terminate()
