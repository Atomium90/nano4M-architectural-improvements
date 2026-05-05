#!/usr/bin/env python3
"""
scripts/eval_checkpoint.py — Generate a performance report from a checkpoint.

Usage:
    python scripts/eval_checkpoint.py \
        --checkpoint outputs/rope_v1/checkpoint-final.safetensors \
        --config     cfgs/nano4M/variants/rope.yaml

Output: a .log file at eval/<exp_name>/report.log (or at --output) containing:
    • model param count
    • val loss (per-token cross-entropy) + perplexity
    • per-modality val loss
    • throughput (tokens/sec)
    • peak GPU memory
    • gradient norm (single backward pass)
    • FID score on generated RGB images

Options:
    --output        PATH    Where to write the report
                            (default: eval/<exp_name>/report.log, derived from checkpoint path)
    --batches       INT     Number of val batches for loss eval (default: 50)
    --device        STR     cuda | cpu (default: cuda if available)

    # FID-specific
    --fid_samples   INT     Number of images to generate for FID (default: 500)
    --fid_steps     INT     ROAR decoding steps per generation (default: 64)
    --fid_temp      FLOAT   Sampling temperature (default: 0.7)
    --fid_top_p     FLOAT   Nucleus sampling p (default: 0.9)
    --fid_input_mod STR     Conditioning modality for generation (default: scene_desc)
    --dataset_root  PATH    Root dir of the CLEVR dataset
                            (default: /work/com-304/datasets/clevr_com_304/)
    --tokenizer_dir PATH    Local dir of the Cosmos tokenizer
                            (default: /tmp/nvidia_<username>/Cosmos-0.1-Tokenizer-DI16x16,
                             created automatically per user if missing)
    --skip_fid              Skip FID evaluation (quick loss-only mode)
"""

import argparse
import datetime
import getpass
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
import omegaconf
from hydra.utils import instantiate

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None


# ── CLI ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to .safetensors checkpoint file")
    p.add_argument("--config",     required=True,
                   help="Path to the YAML config used for training")
    p.add_argument("--output",     default=None,
                   help="Output path for the .log report (default: eval/<exp_name>/report.log)")
    p.add_argument("--batches",    type=int, default=50,
                   help="Number of val batches for loss evaluation (default: 50)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")

    # FID options
    p.add_argument("--fid_samples",   type=int,   default=500,
                   help="Number of images to generate for FID (default: 500)")
    p.add_argument("--fid_steps",     type=int,   default=64,
                   help="ROAR decoding steps for generation (default: 64)")
    p.add_argument("--fid_temp",      type=float, default=0.7,
                   help="Sampling temperature for generation (default: 0.7)")
    p.add_argument("--fid_top_p",     type=float, default=0.9,
                   help="Nucleus sampling p (default: 0.9)")
    p.add_argument("--fid_input_mod", type=str,   default="scene_desc",
                   help="Conditioning modality for image generation (default: scene_desc)")
    p.add_argument("--dataset_root",  type=str,
                   default="/work/com-304/datasets/clevr_com_304/",
                   help="Root dir of the CLEVR dataset")
    _default_tok_dir = f"/tmp/nvidia_{getpass.getuser()}/Cosmos-0.1-Tokenizer-DI16x16"
    p.add_argument("--tokenizer_dir", type=str,
                   default=_default_tok_dir,
                   help="Local dir of the Cosmos tokenizer (encoder.jit / decoder.jit). "
                        "Defaults to /tmp/nvidia_<username>/... (auto-created if missing)")
    p.add_argument("--skip_fid",  action="store_true",
                   help="Skip FID evaluation (quick loss-only mode)")
    return p.parse_args()


# ── Loss / efficiency metrics ─────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def val_metrics(model, loader, device, num_batches):
    """Return (mean_loss, per_modality_dict, tokens_per_sec)."""
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


# ── FID helpers ───────────────────────────────────────────────────────────────

def _load_cosmos_tokenizer(tokenizer_dir: str, device: str):
    """Load the Cosmos DI16x16 image tokenizer (decoder only needed)."""
    try:
        from cosmos_tokenizer.image_lib import ImageTokenizer
    except ImportError:
        raise ImportError(
            "cosmos_tokenizer is not installed. "
            "Install it or use --skip_fid to skip FID evaluation."
        )
    tok_dir = Path(tokenizer_dir)
    if not tok_dir.exists():
        print(f"[fid] Cosmos tokenizer not found at {tok_dir}. Downloading …")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="nvidia/Cosmos-0.1-Tokenizer-DI16x16",
            local_dir=str(tok_dir),
        )
    return ImageTokenizer(
        checkpoint_enc=str(tok_dir / "encoder.jit"),
        checkpoint_dec=str(tok_dir / "decoder.jit"),
    ).to(device)


def _token_ids_to_uint8(token_ids: torch.Tensor,
                        image_tokenizer,
                        device: str) -> torch.Tensor:
    """Decode a flat token-id tensor → uint8 RGB tensor (3, H, W) in [0,255]."""
    import torchvision.transforms.functional as TF
    n_tokens = token_ids.numel()
    side = int(math.sqrt(n_tokens))
    token_ids = token_ids.reshape(1, side, side).to(device)
    with torch.no_grad():
        reconst = image_tokenizer.decode(token_ids)            # (1, 3, H, W)  [-1, 1]
    img = (reconst[0].clamp(-1, 1).float().cpu() + 1) / 2     # [0, 1]
    return (img * 255).to(torch.uint8)                         # (3, H, W)  uint8


def _load_val_dataset(dataset_root: str):
    """Load the unmasked val split of the multimodal CLEVR dataset."""
    from nanofm.data.multimodal.simple_multimodal_dataset import SimpleMultimodalDataset
    modalities     = ["tok_rgb@256", "tok_depth@256", "tok_normal@256", "scene_desc"]
    return SimpleMultimodalDataset(
        root_dir=dataset_root,
        split="val",
        modalities=modalities,
        sample_from_k_augmentations=1,
        text_tokenizer_path="gpt2",
        text_max_length=256,
        transforms=None,   # no masking — we want full tokens
    ), modalities


def compute_fid(model, args, device: str) -> str:
    """
    Generate `fid_samples` RGB images conditioned on `fid_input_mod`,
    then compute FID against the real val-set RGB images.

    Returns the FID score as a string (or an error message).
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        return "error (torchmetrics not installed; pip install torchmetrics[image])"

    print(f"[fid] loading val dataset from {args.dataset_root} …")
    try:
        val_dataset, modalities = _load_val_dataset(args.dataset_root)
    except Exception as e:
        return f"error loading dataset ({e})"

    print(f"[fid] loading Cosmos tokenizer from {args.tokenizer_dir} …")
    try:
        image_tokenizer = _load_cosmos_tokenizer(args.tokenizer_dir, device)
    except Exception as e:
        return f"error loading Cosmos tokenizer ({e})"

    rgb_mod   = "tok_rgb@256"
    input_mod = args.fid_input_mod
    mod_idx   = {m: i for i, m in enumerate(modalities)}

    if input_mod not in mod_idx:
        return f"error (unknown --fid_input_mod '{input_mod}'; choose from {modalities})"

    n = min(args.fid_samples, len(val_dataset))
    print(f"[fid] computing FID over {n} samples  "
          f"(cond: {input_mod} → {rgb_mod},  steps={args.fid_steps}, "
          f"T={args.fid_temp}, top_p={args.fid_top_p}) …")

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    model.eval()

    errors = 0
    for idx in range(n):
        if idx % 50 == 0:
            print(f"  [{idx}/{n}] …")

        try:
            sample = val_dataset[idx]

            # ── Real image ─────────────────────────────────────────────────
            real_tokens = sample[rgb_mod]                          # (N_rgb,)
            real_img_u8 = _token_ids_to_uint8(real_tokens, image_tokenizer, device)
            # FID expects (B, C, H, W) float in [0,1] when normalize=True
            real_img_f  = real_img_u8.float().unsqueeze(0) / 255.0
            fid_metric.update(real_img_f.to(device), real=True)

            # ── Generated image ────────────────────────────────────────────
            cond_tokens = sample[input_mod]                        # (N_cond,)
            n_cond = cond_tokens.shape[0]
            enc_tokens    = cond_tokens.unsqueeze(0).to(device)
            enc_positions = torch.arange(n_cond, device=device).unsqueeze(0)
            enc_modalities = (mod_idx[input_mod]
                              * torch.ones(1, n_cond, device=device, dtype=torch.long))

            with torch.no_grad():
                pred_tokens, _, _, _ = model.generate_one_modality_roar(
                    enc_tokens, enc_positions, enc_modalities,
                    target_mod=rgb_mod,
                    num_steps=args.fid_steps,
                    temp=args.fid_temp,
                    top_p=args.fid_top_p,
                    top_k=0.0,
                )

            gen_img_u8 = _token_ids_to_uint8(pred_tokens.squeeze(0), image_tokenizer, device)
            gen_img_f  = gen_img_u8.float().unsqueeze(0) / 255.0
            fid_metric.update(gen_img_f.to(device), real=False)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [fid] sample {idx} skipped: {e}")
            continue

    if errors:
        print(f"  [fid] {errors}/{n} samples skipped due to errors.")

    try:
        score = fid_metric.compute().item()
        return f"{score:.4f}  (n={n - errors})"
    except Exception as e:
        return f"error computing FID ({e})"


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(args, cfg, params, loss, per_mod, tps, peak_mem_mb, gnorm,
                 fid_score: str) -> str:
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
        "── VALIDATION LOSS ──────────────────────────────────────────",
        f"  loss       : {loss:.6f}",
        f"  perplexity : {math.exp(loss):.4f}",
    ]
    if per_mod:
        lines.append("  per modality:")
        for mod, l in sorted(per_mod.items()):
            lines.append(f"    {mod:<30} {l:.6f}")
    lines += [
        "",
        "── IMAGE QUALITY (FID) ───────────────────────────────────────",
        f"  FID score  : {fid_score}",
        f"  cond. mod  : {args.fid_input_mod} → tok_rgb@256",
        f"  gen steps  : {args.fid_steps}   temp={args.fid_temp}   top_p={args.fid_top_p}",
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
    if args.output:
        out_path = Path(args.output)
    else:
        # Derive exp name from checkpoint path: outputs/rope_v1/checkpoint-final.safetensors
        # → eval/rope_v1/report.log
        ckpt = Path(args.checkpoint)
        exp_name = ckpt.parent.name if ckpt.parent.name != "." else ckpt.stem
        out_path = Path("eval") / exp_name / "report.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model — instantiate from yaml config, then load weights from safetensors.
    # This avoids relying on the _target_ stored in checkpoint metadata, which can
    # fail if the module path changed (e.g. fourmUpgraded vs fourm_upgraded).
    print(f"[eval] loading model …")
    model = instantiate(cfg["model_config"]).to(device)
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.suffix == ".safetensors":
        if load_safetensors is None:
            raise ImportError("safetensors is not installed: pip install safetensors")
        state_dict = load_safetensors(str(ckpt_path), device=device)
        model.load_state_dict(state_dict, strict=True)
    else:
        # Fallback: plain .pth checkpoint
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[eval] {count_params(model) / 1e6:.2f}M params")

    # Val dataloader (masked, for loss)
    print("[eval] building val dataloader …")
    loader = instantiate(cfg["eval_loader_config"])

    # Reset memory stats
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Val loss + throughput
    print(f"[eval] evaluating {args.batches} batches …")
    loss, per_mod, tps = val_metrics(model, loader, device, args.batches)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 \
               if device == "cuda" else float("nan")

    # Gradient norm
    print("[eval] measuring gradient norm …")
    gnorm = measure_grad_norm(model, loader, device)

    # FID
    if args.skip_fid:
        fid_score = "skipped (--skip_fid)"
        print("[eval] FID skipped.")
    else:
        print("[eval] computing FID …")
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        fid_score = compute_fid(model, args, device)
        if device == "cuda":
            fid_peak = torch.cuda.max_memory_allocated(device) / 1024**2
            peak_mem = max(peak_mem, fid_peak)

    # Build and write report
    report = build_report(args, cfg, count_params(model), loss, per_mod,
                          tps, peak_mem, gnorm, fid_score)
    print("\n" + report)
    out_path.write_text(report + "\n")
    print(f"\n[eval] saved → {out_path}")


if __name__ == "__main__":
    main()