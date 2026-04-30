#!/bin/bash
# scripts/run_local.sh — Quick local run for debugging (no SLURM).
#
# Usage:
#   bash scripts/run_local.sh <CONFIG> <EXP_NAME> [NUM_GPUS]
#
# Example:
#   bash scripts/run_local.sh cfgs/nano4M/variants/rope.yaml rope_debug 1

set -euo pipefail

CONFIG="${1:?Error: missing config file.}"
EXP_NAME="${2:?Error: missing experiment name.}"
NUM_GPUS="${3:-1}"

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate nanofm 2>/dev/null || true

echo "Running locally: ${EXP_NAME} (${NUM_GPUS} GPU(s))"

OMP_NUM_THREADS=1 torchrun --nproc_per_node="${NUM_GPUS}" run_training.py \
    --config "${CONFIG}" \
    --run_name "${EXP_NAME}"