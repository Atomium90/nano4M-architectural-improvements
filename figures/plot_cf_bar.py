"""
figures/plot_cf_bar.py — Generate the compute-fair comparison bar chart.
Run from the project root:  python figures/plot_cf_bar.py
Output: figures/cf_bar.pdf
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

# ── Data ──────────────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open("results_summary.csv")))
final_loss = {r["exp"]: float(r["loss"]) for r in rows if r["compute_fair"] == "False"}
cf_loss    = {r["exp"]: float(r["loss"]) for r in rows
              if r["compute_fair"] == "True" and r["checkpoint"] != "final"}

BASELINE = 3.4986

MODELS = [
    ("Baseline",       "baseline_v1",                         None),
    ("RoPE",           "rope_v1",                             None),
    ("SwiGLU",         "swiglu_v1",                           "swiglu_v1"),
    ("RoPE+SwiGLU",    "rope_swiglu_v1",                      "rope_swiglu_v1"),
    ("Depth-8",        "depth8",                              "depth8"),
    ("Depth-12",       "depth12",                             "depth12"),
    ("Depth-16",       "depth16_v1",                          "depth16_v1"),
    ("SwD12",          "swiglu_depth12",                      "swiglu_depth12"),
    ("SwD16",          "swiglu_depth16",                      "swiglu_depth16"),
    ("SwD16+R+DN",     "swiglu_depth16_residual_depth_deepnorm",
                       "swiglu_depth16_residual_depth_deepnorm"),
]

labels   = [m[0] for m in MODELS]
f_losses = [final_loss.get(m[1], BASELINE) for m in MODELS]
c_losses = [cf_loss.get(m[2], None) if m[2] else None for m in MODELS]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.4))

x = np.arange(len(labels))
w = 0.35

COLOR_FINAL = "#4C72B0"
COLOR_CF    = "#DD8452"

bars_f = ax.bar(x - w/2, f_losses, w, label="Final (5000M tok)", color=COLOR_FINAL, alpha=0.9)
bars_c = ax.bar(
    [xi + w/2 for xi, m in zip(x, MODELS) if m[2] is not None],
    [l for l, m in zip(c_losses, MODELS) if m[2] is not None],
    w, label="Compute-fair", color=COLOR_CF, alpha=0.9,
)

ax.axhline(BASELINE, color="black", linewidth=0.8, linestyle="--", label="Baseline")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_ylabel("Val Loss")
ax.set_ylim(3.25, 3.75)
ax.legend(loc="upper left", framealpha=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", linewidth=0.4, alpha=0.5)

plt.tight_layout()
plt.savefig("figures/cf_bar.pdf", bbox_inches="tight")
plt.savefig("figures/cf_bar.png", bbox_inches="tight", dpi=200)
print("Saved figures/cf_bar.pdf")