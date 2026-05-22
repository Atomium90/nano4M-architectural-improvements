import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
RESULTS_CSV = Path("results_summary.csv")
OUT_FIG = Path("docs/assets/figures")
OUT_TABLE = Path("docs/assets/tables")
OUT_RAW = Path("docs/assets/raw")

OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TABLE.mkdir(parents=True, exist_ok=True)
OUT_RAW.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------
df = pd.read_csv(RESULTS_CSV)
df.columns = [c.strip() for c in df.columns]

print("Columns:", list(df.columns))

# Clean / standardize
df["exp"] = df["exp"].astype(str)
df["report"] = df["report"].astype(str)
df["compute_fair"] = df["compute_fair"].astype(bool)

# ---------------------------------------------------------------------
# Select compute-fair website rows
# Each row is identified by (exp, report)
# ---------------------------------------------------------------------
selected = [
    ("baseline_v1", "report.log", "Baseline"),
    ("swiglu_v1", "report_cf.log", "SwiGLU"),
    ("alibi_swiglu_v1", "report_cf.log", "ALiBi + SwiGLU"),
    ("rope_swiglu_v1", "report_cf.log", "RoPE + SwiGLU"),
    ("swiglu_depth8", "report_cf.log", "SwiGLU depth-8"),
    ("resi_scaled_depth16_v1", "report_cf.log", "Residual-scaled depth-16"),
    ("swiglu_depth16_residual_depth_deepnorm", "report_cf.log", "Combined depth-16"),
]

rows = []
missing = []

for exp, report, label in selected:
    match = df[(df["exp"] == exp) & (df["report"] == report)].copy()

    if len(match) == 0:
        missing.append((exp, report))
        continue

    if len(match) > 1:
        print(f"WARNING: multiple matches for {exp}/{report}, taking first.")

    row = match.iloc[0].copy()
    row["label"] = label
    row["selection_order"] = len(rows)
    rows.append(row)

if missing:
    print("WARNING: Missing rows:")
    for exp, report in missing:
        print(f"  exp={exp}, report={report}")

plot_df = pd.DataFrame(rows)

if plot_df.empty:
    raise RuntimeError(
        "No selected rows found. Check exp/report names in results_summary.csv."
    )

plot_df = plot_df.sort_values("selection_order")

print("\nSelected rows:")
print(plot_df[["label", "exp", "report", "checkpoint", "params_m", "loss", "fid"]])

# ---------------------------------------------------------------------
# Save compact table as CSV
# ---------------------------------------------------------------------
table_df = plot_df[
    ["label", "exp", "report", "checkpoint", "params_m", "loss", "perplexity", "fid",
     "tok_rgb", "tok_depth", "tok_normal", "throughput", "peak_gpu_mb", "grad_norm"]
].copy()

table_df.to_csv(OUT_TABLE / "compute_fair_main_table.csv", index=False)

# Also save raw selected rows
plot_df.to_csv(OUT_RAW / "selected_compute_fair_rows.csv", index=False)

# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def save_bar(data, y, ylabel, title, filename):
    plt.figure(figsize=(10, 5))
    plt.bar(data["label"], data[y])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_FIG / filename, dpi=220)
    plt.close()


def save_scatter(data, y, ylabel, title, filename):
    plt.figure(figsize=(8, 5.5))
    plt.scatter(data["params_m"], data[y], s=90)

    for _, row in data.iterrows():
        plt.annotate(
            row["label"],
            (row["params_m"], row[y]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Parameters (M)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_FIG / filename, dpi=220)
    plt.close()


# ---------------------------------------------------------------------
# Quantitative assets
# ---------------------------------------------------------------------
save_bar(
    plot_df,
    y="loss",
    ylabel="Validation cross-entropy loss ↓",
    title="Compute-fair validation loss",
    filename="compute_fair_loss_bar.png",
)

save_bar(
    plot_df,
    y="fid",
    ylabel="FID ↓",
    title="Compute-fair RGB generation quality",
    filename="compute_fair_fid_bar.png",
)

save_scatter(
    plot_df,
    y="loss",
    ylabel="Validation cross-entropy loss ↓",
    title="Compute-fair loss vs model size",
    filename="compute_fair_pareto_loss_vs_params.png",
)

save_scatter(
    plot_df,
    y="fid",
    ylabel="FID ↓",
    title="Compute-fair FID vs model size",
    filename="compute_fair_pareto_fid_vs_params.png",
)

# ---------------------------------------------------------------------
# Table image
# ---------------------------------------------------------------------
table_img_df = plot_df[["label", "params_m", "loss", "fid", "perplexity"]].copy()
table_img_df = table_img_df.rename(
    columns={
        "label": "Model",
        "params_m": "Params (M)",
        "loss": "Val. loss ↓",
        "fid": "FID ↓",
        "perplexity": "PPL ↓",
    }
)

table_img_df["Params (M)"] = table_img_df["Params (M)"].map(lambda x: f"{x:.2f}")
table_img_df["Val. loss ↓"] = table_img_df["Val. loss ↓"].map(lambda x: f"{x:.4f}")
table_img_df["FID ↓"] = table_img_df["FID ↓"].map(lambda x: f"{x:.2f}")
table_img_df["PPL ↓"] = table_img_df["PPL ↓"].map(lambda x: f"{x:.2f}")

fig, ax = plt.subplots(figsize=(11, 3.0))
ax.axis("off")

tbl = ax.table(
    cellText=table_img_df.values,
    colLabels=table_img_df.columns,
    cellLoc="center",
    loc="center",
)

tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.45)

plt.tight_layout()
plt.savefig(OUT_TABLE / "compute_fair_main_table.png", dpi=220)
plt.close()

print("\nSaved assets:")
print(f"  {OUT_FIG / 'compute_fair_loss_bar.png'}")
print(f"  {OUT_FIG / 'compute_fair_fid_bar.png'}")
print(f"  {OUT_FIG / 'compute_fair_pareto_loss_vs_params.png'}")
print(f"  {OUT_FIG / 'compute_fair_pareto_fid_vs_params.png'}")
print(f"  {OUT_TABLE / 'compute_fair_main_table.png'}")
print(f"  {OUT_TABLE / 'compute_fair_main_table.csv'}")