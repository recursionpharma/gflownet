import wandb
from init_wandb_sweep import wandb_config_merger

import gflownet.tasks.seh_frag_moo as seh_frag_moo

if __name__ == "__main__":
    wandb.init(entity="valencelabs", project="gflownet")
    config = wandb_config_merger()
    trial = seh_frag_moo.SEHMOOFragTrainer(config)
    trial.run()
