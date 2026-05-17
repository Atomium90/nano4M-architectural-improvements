#!/bin/bash
# scripts/eval_job_slurm.sh — Submit a checkpoint evaluation to SLURM.
#
# Mirrors submit_job.sh but for evaluation. Useful when running many evals
# in parallel (e.g. all ablation checkpoints at once) without blocking a
# terminal. Logs go to slurm_logs/<EXP_NAME>_eval_<jobid>.out/.err and the
# report is written to eval/<EXP_NAME>/report.log.
#
# Usage:
#   bash scripts/eval_job_slurm.sh <EXP_NAME> [--skip_fid] [extra flags]
#
# Examples:
#   bash scripts/eval_job_slurm.sh rope_v1
#   bash scripts/eval_job_slurm.sh rope_v1 --skip_fid
#   bash scripts/eval_job_slurm.sh rope_v1 --fid_samples 200
#   # Compute-fair eval on an intermediate checkpoint:
#   bash scripts/eval_job_slurm.sh depth16_v1 \
#       --checkpoint /scratch/$USER/nano4M/depth16_v1/checkpoint-22889.safetensors \
#       --output eval/depth16_v1_cf/report.log

set -euo pipefail

EXP_NAME="${1:?Error: missing experiment name. Usage: eval_job_slurm.sh <exp_name> [flags]}"
shift   # remaining args are forwarded to eval_checkpoint.py

# ── Scratch storage (mirrors submit_job.sh) ───────────────────────────────────
SCRATCH_BASE="/scratch/${USER}/nano4M"

# ── Parse --checkpoint and --output overrides from extra args ─────────────────
# Allows targeting a specific checkpoint (e.g. compute-fair eval on step-22889)
# without the auto-resolved value shadowing validation and echo.
CHECKPOINT_OVERRIDE=""
OUTPUT_OVERRIDE=""
_args=("$@")
for i in "${!_args[@]}"; do
    if [[ "${_args[$i]}" == "--checkpoint" ]]; then
        CHECKPOINT_OVERRIDE="${_args[$((i+1))]}"
    fi
    if [[ "${_args[$i]}" == "--output" ]]; then
        OUTPUT_OVERRIDE="${_args[$((i+1))]}"
    fi
done

# ── Resolve checkpoint ────────────────────────────────────────────────────────
# Prefer explicit override, then scratch, then local outputs/ for legacy runs
if [[ -n "${CHECKPOINT_OVERRIDE}" ]]; then
    CHECKPOINT="${CHECKPOINT_OVERRIDE}"
elif [[ -f "${SCRATCH_BASE}/${EXP_NAME}/checkpoint-final.safetensors" ]]; then
    CHECKPOINT="${SCRATCH_BASE}/${EXP_NAME}/checkpoint-final.safetensors"
else
    CHECKPOINT="outputs/${EXP_NAME}/checkpoint-final.safetensors"
fi

REPORT_PATH="${OUTPUT_OVERRIDE:-eval/${EXP_NAME}/report.log}"

# ── Resolve config ────────────────────────────────────────────────────────────
CONFIG_DIR="cfgs/nano4M/variants"

# Special case: baseline experiments always use the multiclevr config
if [[ "${EXP_NAME}" == baseline* ]]; then
    CONFIG="cfgs/nano4M/multiclevr_d6-6w512.yaml"
else
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
        if [[ -f "${_c}" ]]; then CONFIG="${_c}"; break; fi
    done
    if [[ -z "${CONFIG}" ]]; then
        echo "Error: could not find a config for '${EXP_NAME}'."
        for _c in "${_try_configs[@]}"; do echo "  Tried: ${_c}"; done
        exit 1
    fi
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: checkpoint not found at '${CHECKPOINT}'."
    echo "  Checked: ${SCRATCH_BASE}/${EXP_NAME}/ and outputs/${EXP_NAME}/"
    exit 1
fi

mkdir -p slurm_logs

# ── Submit ────────────────────────────────────────────────────────────────────

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}_eval
#SBATCH --time=05:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/${EXP_NAME}_eval_%j.out
#SBATCH --error=slurm_logs/${EXP_NAME}_eval_%j.err
#SBATCH --partition=h100

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1

python scripts/eval_checkpoint.py \\
    --checkpoint ${CHECKPOINT} \\
    --config     ${CONFIG} \\
    --tokenizer_dir /scratch/${USER}/nano4M/tokenizers/Cosmos-0.1-Tokenizer-DI16x16 \\
    $@
EOF

echo "✓ Submitted eval: ${EXP_NAME}"
echo "  checkpoint → ${CHECKPOINT}"
echo "  config     → ${CONFIG}"
echo "  report     → ${REPORT_PATH}"
echo "  logs       → slurm_logs/${EXP_NAME}_eval_<jobid>.out"
