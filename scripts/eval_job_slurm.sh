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
#   # Evaluate all ablations at once:
#   for exp in baseline rope_v1 swiglu_v1 deepnorm_v1; do
#       bash scripts/eval_job_slurm.sh $exp
#   done
#
#   # Quick loss-only eval (no GPU queue pressure):
#   bash scripts/eval_job_slurm.sh rope_v1 --skip_fid
#
#   # Override FID sample count:
#   bash scripts/eval_job_slurm.sh rope_v1 --fid_samples 200

set -euo pipefail

EXP_NAME="${1:?Error: missing experiment name. Usage: eval_job_slurm.sh <exp_name> [flags]}"
shift   # remaining args are forwarded to eval_checkpoint.py

# ── Resolve checkpoint + config (same logic as eval_job.sh) ──────────────────

CHECKPOINT="outputs/${EXP_NAME}/checkpoint-final.safetensors"
CONFIG_DIR="cfgs/nano4M/variants"

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
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Error: checkpoint not found at '${CHECKPOINT}'."
    exit 1
fi

mkdir -p slurm_logs

# ── Submit ────────────────────────────────────────────────────────────────────

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${EXP_NAME}_eval
#SBATCH --time=01:00:00
#SBATCH --account=com-304
#SBATCH --qos=com-304
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/${EXP_NAME}_eval_%j.out
#SBATCH --error=slurm_logs/${EXP_NAME}_eval_%j.err
#SBATCH --partition=l40s

source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm

export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1

python scripts/eval_checkpoint.py \\
    --checkpoint ${CHECKPOINT} \\
    --config     ${CONFIG} \\
    $@
EOF

echo "✓ Submitted eval: ${EXP_NAME}"
echo "  checkpoint → ${CHECKPOINT}"
echo "  config     → ${CONFIG}"
echo "  report     → eval/${EXP_NAME}/report.log"
echo "  logs       → slurm_logs/${EXP_NAME}_eval_<jobid>.out"
