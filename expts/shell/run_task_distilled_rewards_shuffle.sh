#!/bin/bash
#SBATCH -o /mnt/ps/home/CORP/lazar.atanackovic/gflownet/expts/slurm_logs_distilled_rewards_shuffle/log-%A-%a.out
#SBATCH --job-name=gfn-shuffle
#SBATCH --partition=long
#SBATCH --gres=gpu:1080ti:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --qos=normal

# activate conda environment
source venv-gfn/bin/activate

# Launch these jobs with sbatch --array=0-N%M job.sh   (N is inclusive, M limits number of tasks run at once)
python expts/task_distilled_rewards_shuffle.py $SLURM_ARRAY_TASK_ID