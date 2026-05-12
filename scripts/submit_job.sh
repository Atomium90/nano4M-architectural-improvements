#!/bin/bash
# scripts/submit_job.sh — Submit a named experiment to SLURM.
#
# Usage:
#   bash scripts/submit_job.sh <CONFIG> [EXP_NAME] [PARTITION] [NUM_GPUS] [WANDB_KEY]
#
# Examples:
#   bash scripts/submit_job.sh cfgs/nano4M/variants/init_he.yaml
#       → auto-names: init_he_v1 (or _v2 if v1 already exists in scratch)
#
#   bash scripts/submit_job.sh cfgs/nano4M/variants/init_he.yaml init_he_custom
#       → uses the given name as-is (default partition=l40s, num_gpus=2, no wandb)
#
#   bash scripts/submit_job.sh cfgs/nano4M/variants/rope.yaml rope_v1 h100 2 $WANDB_API_KEY
#       → uses name 'rope_v1', runs on h100 with 2 GPUs and wandb enabled
#
#   bash scripts/submit_job.sh cfgs/nano4M/variants/rope.yaml "" h100 "" ""
#       → auto-name, default GPUs, no wandb, but runs on the h100 partition
#
# What it does:
#   - Derives EXP_NAME automatically if not given (init_he_v1, _v2, ...)
#   - Writes checkpoints to /scratch/$USER/nano4M/<EXP_NAME>/ (auto-created)
#   - Each experiment gets its own output folder and its own wandb run (no overrides)
#   - SLURM logs go to slurm_logs/<EXP_NAME>_<jobid>.out/.err

set -euo pipefail

CONFIG="${1:?Error: missing config file. Usage: submit_job.sh <config> [exp_name] [partition] [num_gpus] [wandb_key]}"
PARTITION="${3:-l40s}"
NUM_GPUS="${4:-2}"
WANDB="${5:-}"

# -- Scratch storage ------------------------------------------------------------
# Checkpoints go to scratch to avoid filling the home quota (100 GB limit).
# /scratch/$USER exists on the cluster; we just create the nano4M subfolder.
SCRATCH_BASE="/scratch/${USER}/nano4M"
mkdir -p "${SCRATCH_BASE}"

# -- Auto-versioning ------------------------------------------------------------
# If EXP_NAME is not provided (or is empty), derive it from the config filename
# and pick the next available _vN suffix by scanning the scratch output dir.

if [[ -z "${2:-}" ]]; then
    BASE="$(basename "${CONFIG}" .yaml)"
    N=1
    while [[ -d "${SCRATCH_BASE}/${BASE}_v${N}" ]]; do
        N=$((N + 1))
    done
    EXP_NAME="${BASE}_v${N}"
    echo "→ auto-naming: ${EXP_NAME}"
else
    EXP_NAME="${2}"
fi

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
#SBATCH --partition=${PARTITION}

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

${WANDB:+export WANDB_API_KEY=${WANDB}}

mkdir -p ${SCRATCH_BASE}/${EXP_NAME}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=${NUM_GPUS} run_training.py \\
    --config ${CONFIG} \\
    --run_name ${EXP_NAME} \\
    --output_dir ${SCRATCH_BASE}/${EXP_NAME}
EOF

echo "✓ Submitted: ${EXP_NAME}"
echo "  config    → ${CONFIG}"
echo "  output    → ${SCRATCH_BASE}/${EXP_NAME}/"
echo "  logs      → ./slurm_logs/${EXP_NAME}_<jobid>.out"
echo "  wandb     → epfl-com304-group16 / COM304_nano4M / ${EXP_NAME}"
echo "  partition → ${PARTITION}"