#!/usr/bin/env python3
"""
scripts/eval_checkpoint.py — Generate a performance report from a checkpoint.

Usage:
    python scripts/eval_checkpoint.py \
        --checkpoint outputs/rope_v1/checkpoint-final.safetensors \
        --config     cfgs/nano4M/variants/rope.yaml

Output: a .log file next to the checkpoint (or at --output) containing:
    • model param count
    • val loss (per-token cross-entropy) + perplexity
    • per-modality val loss
    • throughput (tokens/sec)
    • peak GPU memory
    • gradient norm (single backward pass)

Options:
    --output    PATH    Where to write the report (default: next to the checkpoint)
    --batches   INT     Number of val batches to evaluate (default: 50)
    --device    STR     cuda | cpu (default: cuda if available)
"""

import argparse
import datetime
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
import omegaconf
from hydra.utils import instantiate

from nanofm.utils.checkpoint import load_model_from_safetensors


# ── CLI ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to .safetensors checkpoint file")
    p.add_argument("--config",     required=True,
                   help="Path to the YAML config used for training")
    p.add_argument("--output",     default=None,
                   help="Output path for the .log report (default: next to checkpoint)")
    p.add_argument("--batches",    type=int, default=50,
                   help="Number of val batches to use (default: 50)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Metrics ───────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def val_metrics(model, loader, device, num_batches):
    """Run validation, return (mean_loss, per_modality_dict, tokens_per_sec)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    per_mod: dict = {}
    t0 = time.perf_counter()

    for i, data_dict in enumerate(loader):
        if i >= num_batches:
            break
        data_dict = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in data_dict.items()}
        loss, metrics = model(data_dict)
        total_loss += loss.item()
        if "dec_tokens" in data_dict:
            total_tokens += data_dict["dec_tokens"].numel()
        for mod, l in metrics.items():
            per_mod[mod] = per_mod.get(mod, 0.0) + l.item()

    n = i + 1
    elapsed = time.perf_counter() - t0
    tps = total_tokens / elapsed if elapsed > 0 else float("nan")
    return total_loss / n, {k: v / n for k, v in per_mod.items()}, tps


def measure_grad_norm(model, loader, device) -> str:
    """One backward pass to measure gradient norm."""
    model.train()
    try:
        data_dict = next(iter(loader))
        data_dict = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in data_dict.items()}
        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            loss, _ = model(data_dict)
        loss.backward()
        norm = sum(p.grad.norm(2).item() ** 2
                   for p in model.parameters() if p.grad is not None) ** 0.5
        model.zero_grad()
        return f"{norm:.4f}"
    except Exception as e:
        return f"error ({e})"


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(args, cfg, params, loss, per_mod, tps, peak_mem_mb, gnorm) -> str:
    mc = cfg.get("model_config", {})
    lines = [
        "=" * 65,
        "  nano4M — Evaluation Report",
        f"  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 65,
        "",
        f"  checkpoint : {args.checkpoint}",
        f"  config     : {args.config}",
        "",
        "── MODEL ────────────────────────────────────────────────────",
        f"  target     : {mc.get('_target_', '?')}",
        f"  params     : {params:,}  ({params / 1e6:.2f} M)",
        f"  enc_depth  : {mc.get('enc_depth', '?')}  "
        f"dec_depth : {mc.get('dec_depth', '?')}  "
        f"dim : {mc.get('dim', '?')}",
        "",
        "── VALIDATION ───────────────────────────────────────────────",
        f"  loss       : {loss:.6f}",
        f"  perplexity : {math.exp(loss):.4f}",
    ]
    if per_mod:
        lines.append("  per modality:")
        for mod, l in sorted(per_mod.items()):
            lines.append(f"    {mod:<30} {l:.6f}")
    lines += [
        "",
        "── EFFICIENCY ───────────────────────────────────────────────",
        f"  throughput : {tps:>10,.0f} tok/s",
        f"  peak GPU   : {peak_mem_mb:>10.1f} MB",
        f"  grad norm  : {gnorm}",
        "",
        "=" * 65,
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    # Load config
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    omegaconf.OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.create(raw), resolve=True)

    device = args.device
    out_path = Path(args.output) if args.output \
               else Path(args.checkpoint).with_suffix(".log")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model from safetensors
    print(f"[eval] loading model …")
    model = load_model_from_safetensors(args.checkpoint, device=device)
    model.eval()
    print(f"[eval] {count_params(model) / 1e6:.2f}M params")

    # Build val dataloader
    print("[eval] building val dataloader …")
    loader = instantiate(cfg["eval_loader_config"])

    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Val loss
    print(f"[eval] evaluating {args.batches} batches …")
    loss, per_mod, tps = val_metrics(model, loader, device, args.batches)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 \
               if device == "cuda" else float("nan")

    # Grad norm
    print("[eval] measuring gradient norm …")
    gnorm = measure_grad_norm(model, loader, device)

    # Build and write report
    report = build_report(args, cfg, count_params(model), loss, per_mod, tps, peak_mem, gnorm)
    print("\n" + report)
    out_path.write_text(report + "\n")
    print(f"\n[eval] saved → {out_path}")


if __name__ == "__main__":
    main()