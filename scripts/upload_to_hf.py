#!/usr/bin/env python3
"""
scripts/upload_to_hf.py — Upload all model checkpoints to HuggingFace.

Usage:
    python scripts/upload_to_hf.py --token hf_xxx --repo Atomium90/nano4M-architecture-ablations

Run once from the cluster where /scratch/ is accessible.
Uploads only .safetensors files (skips .pth).
For each model > 110M params, uploads both final and compute-fair checkpoint.
"""

import argparse
import getpass
from pathlib import Path

# ── Compute-fair checkpoint map ───────────────────────────────────────────────
# exp_name → step number of compute-fair checkpoint (None = no intermediate saved)
CF_STEPS = {
    # 122-125M params → step-30519
    "swiglu_v1":                                  30519,
    "rope_swiglu_v1":                             30519,
    "alibi_swiglu_v1":                            30519,
    "depth8":                                     30519,
    # 141M params → step-22889
    "swiglu_depth8":                              22889,
    # 154-179M params → step-22889
    "depth12":                                    22889,
    "swiglu_depth12":                             22889,
    "resi_scaled_depth12_v1":                     22889,
    "deepnorm_depth12_v1":                        22889,
    # 183M params → step-22889
    "depth16_v1":                                 22889,
    "resi_scaled_depth16_v1":                     22889,
    "deepnorm_depth16_v1":                        22889,
    "rope_depth16_v1":                            22889,
    # 217M params → step-15259
    "swiglu_depth16":                             15259,
    "swiglu_depth16_residual_depth":              15259,
    "swiglu_depth16_residual_depth_deepnorm":     15259,
    "swiglu_depth16_rope_deepnorm":               15259,
    "swiglu_depth16_rope_deepnorm_residual_depth":15259,
    "deepnorm_swiglu_depth16_v1":                 15259,
    "rope_swiglu_depth16_v1":                     15259,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--token",  required=True,
                   help="HuggingFace write token (hf_...)")
    p.add_argument("--repo",   default="Atomium90/nano4M-architecture-ablations",
                   help="HuggingFace repo id (org/name)")
    p.add_argument("--scratch", default=f"/scratch/{getpass.getuser()}/nano4M",
                   help="Scratch dir containing experiment folders")
    p.add_argument("--configs", default="cfgs/nano4M",
                   help="Root config dir (cfgs/nano4M)")
    p.add_argument("--dry_run", action="store_true",
                   help="Print what would be uploaded without uploading")
    args = p.parse_args()

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.token)

    # Create repo if it doesn't exist
    if not args.dry_run:
        create_repo(args.repo, repo_type="model", exist_ok=True,
                    token=args.token, private=False)
        print(f"✓ Repo ready: https://huggingface.co/{args.repo}")

    scratch = Path(args.scratch)
    configs_root = Path(args.configs)
    uploads = []   # list of (local_path, repo_path)

    for exp_dir in sorted(scratch.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == "tokenizers":
            continue
        exp = exp_dir.name

        # Final checkpoint
        final = exp_dir / "checkpoint-final.safetensors"
        if final.exists():
            uploads.append((final, f"{exp}/checkpoint-final.safetensors"))
        else:
            print(f"  [warn] no final checkpoint for {exp}")

        # Compute-fair checkpoint
        cf_step = CF_STEPS.get(exp)
        if cf_step:
            cf = exp_dir / f"checkpoint-{cf_step}.safetensors"
            if cf.exists():
                uploads.append((cf, f"{exp}/checkpoint-{cf_step}.safetensors"))
            else:
                print(f"  [warn] CF checkpoint not found: {cf}")

    # YAML configs
    for yaml_file in sorted(f for f in configs_root.rglob("*.yaml")
                             if ".ipynb_checkpoints" not in str(f)):
        repo_path = f"configs/{yaml_file.relative_to(configs_root)}"
        uploads.append((yaml_file, repo_path))

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Uploading {len(uploads)} files to {args.repo}\n")

    for i, (local, remote) in enumerate(uploads):
        size_mb = local.stat().st_size / 1e6
        print(f"  [{i+1:>3}/{len(uploads)}] {remote}  ({size_mb:.0f} MB)")
        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=args.repo,
                repo_type="model",
                token=args.token,
                commit_message=f"Upload {remote}",
            )

    if not args.dry_run:
        print(f"\n✓ Done → https://huggingface.co/{args.repo}")
    else:
        total_gb = sum(Path(l).stat().st_size for l, _ in uploads) / 1e9
        print(f"\nDry run complete. Would upload ~{total_gb:.1f} GB total.")


if __name__ == "__main__":
    main()