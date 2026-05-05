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

CHECKPOINT="outputs/${EXP_NAME}/checkpoint-final.safetensors"
CONFIG_DIR="cfgs/nano4M/variants"

# Config resolution: check variants/ first, then top-level cfgs/nano4M/,
# then fall back to stripping version suffix (e.g. init_he_v1 → init_he).
_try_configs=(
    "${CONFIG_DIR}/${EXP_NAME}.yaml"
    "cfgs/nano4M/${EXP_NAME}.yaml"
)
BASE="${EXP_NAME%_v*}"
if [[ "${BASE}" != "${EXP_NAME}" ]]; then
    _try_configs+=(
        "${CONFIG_DIR}/${BASE}.yaml"
        "cfgs/nano4M/${BASE}.yaml"
    )
fi
CONFIG=""
for _c in "${_try_configs[@]}"; do
    if [[ -f "${_c}" ]]; then
        CONFIG="${_c}"
        break
    fi
done
if [[ -z "${CONFIG}" ]]; then
    echo "Error: could not find a config for '${EXP_NAME}'."
    for _c in "${_try_configs[@]}"; do echo "  Tried: ${_c}"; done
    echo "  Pass --config explicitly to eval_checkpoint.py directly."
    exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: checkpoint not found at '${CHECKPOINT}'."
    echo "  Make sure the training run completed and outputs/${EXP_NAME}/ exists."
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

export OMP_NUM_THREADS=1

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