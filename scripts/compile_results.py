#!/usr/bin/env python3
"""
scripts/compile_results.py — Compile all eval/*/report*.log into a summary + CSV.

Usage:
    python scripts/compile_results.py
    python scripts/compile_results.py --eval_dir eval/ --output results_summary.txt
"""

import argparse
import csv
import re
from pathlib import Path


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")

    def get(pattern, cast=str, default=None):
        m = re.search(pattern, text)
        if m:
            try:
                return cast(m.group(1).strip())
            except Exception:
                return default
        return default

    ckpt_raw = get(r"checkpoint\s*:\s*(\S+)", default="")
    step_match = re.search(r"checkpoint-(\d+)\.(?:safetensors|pth)", ckpt_raw)
    ckpt_label = f"step-{step_match.group(1)}" if step_match else "final"
    is_cf = "_cf" in path.stem or ckpt_label != "final"

    return {
        "file":         path,
        "exp":          path.parent.name,
        "report":       path.name,
        "exp_id":       f"{path.parent.name}/{path.name}",
        "compute_fair": is_cf,
        "checkpoint":   ckpt_label,
        "params_m":     get(r"params\s*:.*?\((\S+)\s*M\)", float),
        "enc_depth":    get(r"enc_depth\s*:\s*(\d+)", int),
        "dec_depth":    get(r"dec_depth\s*:\s*(\d+)", int),
        "loss":         get(r"loss\s*:\s*([\d.]+)", float),
        "perplexity":   get(r"perplexity\s*:\s*([\d.]+)", float),
        "fid":          get(r"FID score\s*:\s*([\d.]+)", float),
        "fid_n":        get(r"FID score\s*:.*?\(n=([\d]+)\)", int),
        "scene_desc":   get(r"scene_desc\s+([\d.]+)", float),
        "tok_rgb":      get(r"tok_rgb@256\s+([\d.]+)", float),
        "tok_depth":    get(r"tok_depth@256\s+([\d.]+)", float),
        "tok_normal":   get(r"tok_normal@256\s+([\d.]+)", float),
        "throughput":   get(r"throughput\s*:\s*([\d,]+)",
                            lambda x: int(x.replace(",", ""))),
        "peak_gpu_mb":  get(r"peak GPU\s*:\s*([\d.]+)", float),
        "grad_norm":    get(r"grad norm\s*:\s*([\d.]+)", float),
    }


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt(val, spec=".4f", missing="—"):
    if val is None:
        return missing
    try:
        return format(val, spec)
    except Exception:
        return missing

def col(s, width, align="left"):
    s = str(s)
    if len(s) > width:
        s = s[:width - 1] + "…"
    return s.ljust(width) if align == "left" else s.rjust(width)


def make_table(headers, widths, aligns, rows):
    sep = "  ".join("-" * w for w in widths)
    head = "  ".join(col(h, w, a) for h, w, a in zip(headers, widths, aligns))
    lines = [head, sep]
    for r in rows:
        lines.append("  ".join(col(c, w, a) for c, w, a in zip(r, widths, aligns)))
    return "\n".join(lines)


# ── Build text summary ────────────────────────────────────────────────────────

def build_summary(records: list) -> str:
    records = sorted(records,
                     key=lambda r: (r["exp"], r["compute_fair"], r["report"]))
    baseline = next(
        (r for r in records if "baseline" in r["exp"] and not r["compute_fair"]),
        None
    )

    lines = [
        "=" * 115,
        "  nano4M — Results Summary",
        f"  {len(records)} reports found",
        "=" * 115, "",
    ]

    # ── Main table ────────────────────────────────────────────────────────────
    lines.append("── MAIN TABLE  (CF = compute-fair eval on intermediate checkpoint) "
                 + "─" * 35)

    h = ["experiment / report", "CF", "ckpt", "params M",
         "loss", "Δ baseline", "pplx", "FID", "rgb", "depth", "normal"]
    w = [44,                    3,    9,       8,
         7,     10,             7,     7,    8,     8,      8]
    a = ["left","left","left","right",
         "right","right","right","right","right","right","right"]

    data_rows = []
    for r in records:
        cf = "✓" if r["compute_fair"] else " "
        delta = ""
        if (baseline and r["loss"] is not None
                and baseline["loss"] is not None and not r["compute_fair"]):
            d = r["loss"] - baseline["loss"]
            delta = f"{'+' if d >= 0 else ''}{d:.4f}"
        data_rows.append([
            r["exp_id"],
            cf,
            r["checkpoint"],
            fmt(r["params_m"], ".2f"),
            fmt(r["loss"]),
            delta,
            fmt(r["perplexity"], ".3f"),
            fmt(r["fid"], ".2f"),
            fmt(r["tok_rgb"]),
            fmt(r["tok_depth"]),
            fmt(r["tok_normal"]),
        ])

    lines.append(make_table(h, w, a, data_rows))
    lines += ["", ""]

    # ── Per-modality ──────────────────────────────────────────────────────────
    lines.append("── PER-MODALITY LOSSES " + "─" * 75)
    h2 = ["experiment / report", "CF", "ckpt", "loss",
          "scene_desc", "tok_rgb", "tok_depth", "tok_normal"]
    w2 = [44, 3, 9, 7, 11, 9, 9, 10]
    a2 = ["left","left","left","right","right","right","right","right"]
    rows2 = [[r["exp_id"], "✓" if r["compute_fair"] else " ", r["checkpoint"],
              fmt(r["loss"]), fmt(r["scene_desc"]), fmt(r["tok_rgb"]),
              fmt(r["tok_depth"]), fmt(r["tok_normal"])] for r in records]
    lines.append(make_table(h2, w2, a2, rows2))
    lines += ["", ""]

    # ── Efficiency (all rows) ─────────────────────────────────────────────────
    lines.append("── EFFICIENCY " + "─" * 85)
    h3 = ["experiment / report", "CF", "ckpt",
          "throughput tok/s", "peak GPU MB", "grad norm"]
    w3 = [44, 3, 9, 17, 12, 10]
    a3 = ["left","left","left","right","right","right"]
    rows3 = []
    for r in records:
        tput = fmt(r["throughput"], ",d") if r["throughput"] is not None else "—"
        rows3.append([
            r["exp_id"],
            "✓" if r["compute_fair"] else " ",
            r["checkpoint"],
            tput,
            fmt(r["peak_gpu_mb"], ".1f"),
            fmt(r["grad_norm"]),
        ])
    lines.append(make_table(h3, w3, a3, rows3))
    lines += ["", "=" * 115]

    return "\n".join(lines)


# ── CSV ───────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "exp", "report", "exp_id", "compute_fair", "checkpoint",
    "params_m", "enc_depth", "dec_depth",
    "loss", "perplexity", "fid", "fid_n",
    "scene_desc", "tok_rgb", "tok_depth", "tok_normal",
    "throughput", "peak_gpu_mb", "grad_norm",
]

def write_csv(records: list, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default="eval")
    p.add_argument("--output",   default="results_summary.txt")
    args = p.parse_args()

    log_files = sorted(Path(args.eval_dir).rglob("*.log"))
    if not log_files:
        print(f"No .log files found in {args.eval_dir}/")
        return

    print(f"Found {len(log_files)} log files …")
    records, skipped = [], []
    for f in log_files:
        try:
            records.append(parse_log(f))
        except Exception as e:
            skipped.append((f, e))

    if skipped:
        for f, e in skipped:
            print(f"  [skip] {f}: {e}")

    summary = build_summary(records)
    print(summary)

    Path(args.output).write_text(summary + "\n", encoding="utf-8")
    print(f"\nSaved → {args.output}")

    csv_path = Path(args.output).with_suffix(".csv")
    write_csv(records, csv_path)
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()