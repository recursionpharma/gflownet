These are the two files used to execute wandb searches:
1. `init_wandb_sweep.py` defines the base-configuration and the hyperparameters to sweep over.
2. `wandb_agent_main.py` is executed by wandb agents that are managed by the wandb sweep.

To launch the search
1. `python init_wandb_sweep.py` to intialize the sweep
2. `sbatch launch_wandb_agents.sh <SWEEP_ID>` to schedule a jobarray in slurm which will launch wandb agents with `wandb_agent_main.py` as entrypoint. The number of jobs in the sbatch file should reflect the size of the hyperparameter space that is being sweeped.
