"""
figures/plot_convergence_gap.py — Generate the CF vs Final convergence figure (Appendix A).
Run from the project root:  python figures/plot_convergence_gap.py
Output: figures/convergence_gap.pdf / .png
"""

import csv
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
})

# ── Data from CSV ─────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open("results_summary.csv")))
final = {r["exp"]: r for r in rows if r["compute_fair"] == "False"}
cf    = {r["exp"]: r for r in rows if r["compute_fair"] == "True"}

BASELINE_CF    = 3.49857
BASELINE_FINAL = 3.49857

# Large combined models (217M)
MODELS = [
    ("swiglu_depth16",                          "SwD16"),
    ("swiglu_depth16_residual_depth",           "SwD16\n+ResScale"),
    ("swiglu_depth16_residual_depth_deepnorm",  "SwD16\n+RS+DN"),
    ("swiglu_depth16_rope_deepnorm",            "SwD16\n+RoPE+DN"),
    ("swiglu_depth16_rope_deepnorm_residual_depth", "RoPE+SwD16\n+DN+RS"),
]

cf_losses    = [float(cf[exp]["loss"])    for exp, _ in MODELS]
final_losses = [float(final[exp]["loss"]) for exp, _ in MODELS]
labels       = [label                     for _, label in MODELS]
x            = np.arange(len(MODELS))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.6))

COLOR_CF    = "#DD8452"
COLOR_FINAL = "#4C72B0"
COLOR_ARROW = "#888888"

# Bars: CF and Final side by side
w = 0.32
bars_cf    = ax.bar(x - w/2, cf_losses,    w, color=COLOR_CF,    alpha=0.9, label="Compute-fair")
bars_final = ax.bar(x + w/2, final_losses, w, color=COLOR_FINAL, alpha=0.9, label="Full budget")

# Arrows showing the convergence gap
for xi, (cf_l, fin_l) in enumerate(zip(cf_losses, final_losses)):
    ax.annotate("", xy=(xi + w/2, fin_l + 0.003), xytext=(xi - w/2, cf_l - 0.003),
                arrowprops=dict(arrowstyle="-|>", color=COLOR_ARROW, lw=0.8))

# Baseline lines
ax.axhline(BASELINE_FINAL, color="black", linewidth=0.9, linestyle="--", alpha=0.7,
           label="Baseline")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=6.5)
ax.set_ylabel("Val Loss")
ax.set_ylim(3.24, 3.72)
ax.legend(loc="upper right", framealpha=0.85, ncol=1)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linewidth=0.3, alpha=0.5)

# Annotate gap for the best model
best_idx = 2  # SwD16+RS+DN
gap = cf_losses[best_idx] - final_losses[best_idx]
ax.text(best_idx, (cf_losses[best_idx] + final_losses[best_idx]) / 2 + 0.005,
        f"Δ={gap:.3f}", ha="center", fontsize=6, color=COLOR_ARROW)

plt.tight_layout()
plt.savefig("figures/convergence_gap.pdf", bbox_inches="tight")
plt.savefig("figures/convergence_gap.png", bbox_inches="tight", dpi=200)
print("Saved figures/convergence_gap.pdf")