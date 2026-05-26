"""
figures/plot_val_curves.py — Recreate the WandB val-loss curves locally.

If you prefer to pull data via the WandB API instead of exporting manually:
    pip install wandb
    wandb login

Then run:  python figures/plot_val_curves.py
Output:    figures/val_curves.pdf

Alternatively, export directly from WandB UI:
  1. Open epfl-com304-group16 / COM304_nano4M
  2. Charts → Line Chart
     X-axis: total_tokens_seen_m
     Y-axis: [Eval]/loss
     Smooth: 0
  3. Select runs (see RUNS below)
  4. Export PNG → convert to PDF or include directly
"""

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
})

# Runs to include (WandB run name → display label, color)
RUNS = {
    "baseline":                                 ("Baseline",          "#888888"),
    "swiglu_v1":                                ("SwiGLU",            "#4C72B0"),
    "depth16_v1":                               ("Depth-16",          "#55A868"),
    "swiglu_depth16_v2":                        ("SwD16",             "#C44E52"),
    "swiglu_depth16_residual_depth_deepnorm_v1":("SwD16+R+DN (best)","#8172B2"),
    "residual_rezero_v1":                       ("ReZero",            "#CCB974"),
}

# ── Pull data via WandB API ───────────────────────────────────────────────────
import wandb
api = wandb.Api()

fig, ax = plt.subplots(figsize=(3.5, 2.6))

for run_name, (label, color) in RUNS.items():
    try:
        runs = api.runs("epfl-com304-group16/COM304_nano4M",
                        filters={"display_name": run_name})
        run = next(iter(runs))
        hist = run.history(keys=["total_tokens_seen_m", "[Eval]/loss"],
                           samples=500, pandas=False)
        xs = [h["total_tokens_seen_m"] for h in hist if "[Eval]/loss" in h]
        ys = [h["[Eval]/loss"]         for h in hist if "[Eval]/loss" in h]
        ax.plot(xs, ys, label=label, color=color, linewidth=1.2)
    except Exception as e:
        print(f"  [skip] {run_name}: {e}")

# Compute-fair markers — slightly staggered to avoid overlap
ax.axvline(2540, color="black", linewidth=0.6, linestyle=":", alpha=0.7)
ax.axvline(3050, color="black", linewidth=0.6, linestyle=":", alpha=0.7)
ymin, ymax = ax.get_ylim()
label_y_base = ymin + (ymax - ymin) * 0.28
label_y_high = ymin + (ymax - ymin) * 0.35
ax.text(2540, label_y_base, "CF\n217M", fontsize=5.5, ha="center", va="bottom",
        linespacing=1.1)
ax.text(3050, label_y_high, "CF\n183M", fontsize=5.5, ha="center", va="bottom",
        linespacing=1.1)

ax.set_xlabel("Tokens seen (M)")
ax.set_ylabel("Val Loss")
ax.legend(loc="upper right", framealpha=0.8, ncol=1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(linewidth=0.3, alpha=0.5)

plt.tight_layout()
plt.savefig("figures/val_curves.pdf", bbox_inches="tight")
plt.savefig("figures/val_curves.png", bbox_inches="tight", dpi=200)
print("Saved figures/val_curves.pdf")