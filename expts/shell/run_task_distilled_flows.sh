#!/bin/bash
#SBATCH -o /h/lazar/gflownet/expts/slurm_logs_distilled_flows/log-%A-%a.out
#SBATCH --job-name=gfn
#SBATCH --partition=t4v2,rtx6000,a40
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=normal

# activate conda environment
conda init
conda activate gfn

# Launch these jobs with sbatch --array=0-N%M job.sh   (N is inclusive, M limits number of tasks run at once)
srun python expts/task_distilled_flows.py $SLURM_ARRAY_TASK_ID