#!/bin/bash
#SBATCH --job-name=4M_training
#SBATCH --time=10:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:2
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=interactive_job.out
#SBATCH --error=interactive_job.err
#SBATCH --partition=l40s

CONFIG_FILE=$1
NUM_GPUS=$2

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS run_training.py --config $CONFIG_FILE