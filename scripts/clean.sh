#!/bin/bash
# scripts/clean.sh — Clean up experiment artifacts.
#
# Usage:
#   bash scripts/clean.sh <target> [<target> ...]
#
# Targets:
#   slurm    Wipe everything in slurm_logs/
#   models   Wipe outputs/ keeping only checkpoint-final.safetensors per run
#
# Examples:
#   bash scripts/clean.sh slurm
#   bash scripts/clean.sh models
#   bash scripts/clean.sh models slurm   (both at once, one confirmation each)

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: bash scripts/clean.sh <target> [<target> ...]"
    echo "  targets: slurm  models"
    echo ""
    echo "  slurm  → wipes slurm_logs/"
    echo "  models → wipes outputs/ keeping only checkpoint-final.safetensors"
    exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────

confirm() {
    local msg="$1"
    echo ""
    echo "  $msg"
    read -r -p "  Are you sure? [y/N] " ans
    [[ "${ans,,}" == "y" ]]
}

bytes_human() {
    du -sh "$1" 2>/dev/null | cut -f1
}

# ── Targets ───────────────────────────────────────────────────────────────────

clean_slurm() {
    if [[ ! -d slurm_logs ]] || [[ -z "$(ls -A slurm_logs 2>/dev/null)" ]]; then
        echo "  [slurm] slurm_logs/ is already empty."
        return
    fi
    local n; n=$(find slurm_logs -maxdepth 1 -type f | wc -l)
    local sz; sz=$(bytes_human slurm_logs)
    if confirm "[slurm] Delete ${n} files in slurm_logs/ (${sz})"; then
        rm -rf slurm_logs/*
        echo "  ✓ slurm_logs/ wiped."
    else
        echo "  ✗ slurm clean cancelled."
    fi
}

clean_models() {
    if [[ ! -d outputs ]] || [[ -z "$(ls -A outputs 2>/dev/null)" ]]; then
        echo "  [models] outputs/ is already empty."
        return
    fi

    # Count what would be deleted
    local keep=0 del=0 del_sz=0
    local del_files=()
    while IFS= read -r -d '' f; do
        if [[ "$(basename "$f")" == "checkpoint-final.safetensors" ]]; then
            keep=$((keep + 1))
        else
            del=$((del + 1))
            del_files+=("$f")
        fi
    done < <(find outputs -type f -print0)

    if [[ ${del} -eq 0 ]]; then
        echo "  [models] Nothing to clean (only final safetensors remain)."
        return
    fi

    # Estimate size
    local tmp_list; tmp_list=$(mktemp)
    printf '%s\n' "${del_files[@]}" > "${tmp_list}"
    local sz; sz=$(xargs du -ch < "${tmp_list}" 2>/dev/null | tail -1 | cut -f1)
    rm -f "${tmp_list}"

    echo "  [models] Will delete ${del} files (~${sz}) and keep ${keep} checkpoint-final.safetensors."
    echo "  Runs in outputs/: $(ls outputs/ | tr '\n' ' ')"

    if confirm "[models] Proceed with deletion?"; then
        for f in "${del_files[@]}"; do
            rm -f "$f"
        done
        # Remove any empty run dirs
        find outputs -type d -empty -delete 2>/dev/null || true
        echo "  ✓ outputs/ cleaned. Kept ${keep} final checkpoints."
    else
        echo "  ✗ models clean cancelled."
    fi
}

# ── Dispatch ──────────────────────────────────────────────────────────────────

for target in "$@"; do
    case "${target}" in
        slurm)   clean_slurm  ;;
        models)  clean_models ;;
        *)
            echo "Unknown target: '${target}'. Valid targets: slurm  models"
            exit 1
            ;;
    esac
done