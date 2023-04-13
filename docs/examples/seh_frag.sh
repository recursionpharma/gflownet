#!/bin/bash
#SBATCH -o /rxrx/scratch/sandbox/emmanuel.bengio/slurm_outs/job-%A_%a.out
#SBATCH --job-name=seh_frag
#SBATCH --partition=batch
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1


# To run e.g. the apr_5 array of hyperparameters use:
#     sbatch --array=0-7 seh_frag.sh apr_5
# This will run the 8 hyperparameter combinations in parallel through slurm.
# Make sure to adapt the above SBATCH parameters to your cluster.

python seh_frag.py $1 $SLURM_ARRAY_TASK_ID
