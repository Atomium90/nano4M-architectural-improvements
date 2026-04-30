#!/bin/bash
# scripts/submit_job.sh — Submit a named experiment to SLURM.
#
# Usage:
#   bash scripts/submit_job.sh <CONFIG> <EXP_NAME> [NUM_GPUS] [WANDB_KEY]
#
# Examples:
#   bash scripts/submit_job.sh cfgs/nano4M/baseline/multiclevr_d6-6w512.yaml baseline 2 $WANDB_API_KEY
#   bash scripts/submit_job.sh cfgs/nano4M/variants/rope.yaml rope_v1 2 $WANDB_API_KEY
#
# What it does:
#   - Passes --run_name to run_training.py, which auto-sets output_dir and wandb_run_name
#   - Each experiment gets its own output folder and its own wandb run (no overrides)
#   - SLURM logs go to slurm_logs/<EXP_NAME>_<jobid>.out/.err

set -euo pipefail

CONFIG="${1:?Error: missing config file. Usage: submit_job.sh <config> <exp_name> [num_gpus] [wandb_key]}"
EXP_NAME="${2:?Error: missing experiment name. Usage: submit_job.sh <config> <exp_name> [num_gpus] [wandb_key]}"
NUM_GPUS="${3:-2}"
WANDB="${4:-}"

mkdir -p slurm_logs

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}
#SBATCH --time=10:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/${EXP_NAME}_%j.out
#SBATCH --error=slurm_logs/${EXP_NAME}_%j.err
#SBATCH --partition=l40s

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

${WANDB:+export WANDB_API_KEY=${WANDB}}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=${NUM_GPUS} run_training.py \\
    --config ${CONFIG} \\
    --run_name ${EXP_NAME}
EOF

echo "✓ Submitted: ${EXP_NAME}"
echo "  config  → ${CONFIG}"
echo "  output  → ./outputs/${EXP_NAME}/"
echo "  logs    → ./slurm_logs/${EXP_NAME}_<jobid>.out"
echo "  wandb   → epfl-com304-group16 / COM304_nano4M / ${EXP_NAME}"