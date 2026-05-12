#!/bin/bash
# scripts/eval_job.sh — Evaluate a named experiment checkpoint.
#
# Mirrors the submit_job.sh interface but runs eval_checkpoint.py directly
# (no SLURM queue). Runs on the current node, so launch from a GPU node or
# an interactive session if you want CUDA. A typical full eval (loss + FID)
# takes ~15–30 min on one GPU; loss-only with --skip_fid takes ~2 min.
#
# Usage:
#   bash scripts/eval_job.sh <EXP_NAME> [extra eval_checkpoint.py flags]
#
# Examples:
#   bash scripts/eval_job.sh rope_v1
#   bash scripts/eval_job.sh rope_v1 --skip_fid
#   bash scripts/eval_job.sh rope_v1 --fid_samples 200 --fid_steps 32

set -euo pipefail

EXP_NAME="${1:?Error: missing experiment name. Usage: eval_job.sh <exp_name> [flags]}"
shift   # remaining args are forwarded to eval_checkpoint.py

# ── Resolve paths ─────────────────────────────────────────────────────────────

# ── Scratch storage (mirrors submit_job.sh) ───────────────────────────────────
SCRATCH_BASE="/scratch/${USER}/nano4M"

# Prefer scratch location; fall back to local outputs/ for legacy runs
if [[ -f "${SCRATCH_BASE}/${EXP_NAME}/checkpoint-final.safetensors" ]]; then
    CHECKPOINT="${SCRATCH_BASE}/${EXP_NAME}/checkpoint-final.safetensors"
else
    CHECKPOINT="outputs/${EXP_NAME}/checkpoint-final.safetensors"
fi
CONFIG_DIR="cfgs/nano4M/variants"

# Special case: baseline experiments always use the multiclevr config
if [[ "${EXP_NAME}" == baseline* ]]; then
    CONFIG="cfgs/nano4M/multiclevr_d6-6w512.yaml"
fi

# Try to find the matching config: variants/<exp_name>.yaml, else fall back to
# any yaml whose run_name matches, else ask the user.
CONFIG="${CONFIG_DIR}/${EXP_NAME}.yaml"
if [[ ! -f "${CONFIG}" ]]; then
    # Strip trailing version suffix (e.g. rope_v1 → rope)
    BASE="${EXP_NAME%_v*}"
    CONFIG="${CONFIG_DIR}/${BASE}.yaml"
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: could not find a config for '${EXP_NAME}'."
    echo "  Tried: ${CONFIG_DIR}/${EXP_NAME}.yaml"
    echo "  Tried: ${CONFIG_DIR}/${BASE}.yaml"
    echo "  Pass --config explicitly via eval_checkpoint.py directly."
    exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: checkpoint not found at '${CHECKPOINT}'."
    echo "  Checked: ${SCRATCH_BASE}/${EXP_NAME}/ and outputs/${EXP_NAME}/"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

export OMP_NUM_THREADS=1
export PYTHONPATH=$PWD

# ── Run ───────────────────────────────────────────────────────────────────────

echo "-----------------------------------------------------------------"
echo "  eval_job.sh - ${EXP_NAME}"
echo "  checkpoint : ${CHECKPOINT}"
echo "  config     : ${CONFIG}"
echo "  extra args : $*"
echo "-----------------------------------------------------------------"

python scripts/eval_checkpoint.py \
    --checkpoint "${CHECKPOINT}" \
    --config     "${CONFIG}" \
    "$@"

echo ""
echo "✓ Done: eval/${EXP_NAME}/report.log"
