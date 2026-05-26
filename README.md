# nano4M — Architectural Improvements

> COM-304 Foundation Models Project | EPFL VILab | Spring 2026  
> Edouard Fousson · Thomas Picart · Jonathan Balli

**Pretrained checkpoints** → [huggingface.co/Atomium90/nano4M-architecture-ablations](https://huggingface.co/Atomium90/nano4M-architecture-ablations)

We systematically ablate five architectural modifications to the [nano4M](https://github.com/EPFL-VILAB/com-304-FM-project-2026) baseline Transformer trained on multimodal CLEVR (RGB, depth, normals, captions). 
The best combined model achieves **loss 3.286** and **FID 29.2** vs the baseline's 3.499 / 47.2.  
Read the [extension proposal](report/extension-proposal.pdf) and [final report](report/final-report.pdf) for the full analysis.

---

## Quick start

### 1. Environment setup

```bash
bash setup_env.sh
conda activate nanofm
```

This creates a `nanofm` conda environment (Python 3.10), installs all dependencies from `pyproject.toml`, and registers the Jupyter kernel.

### 2. Try the demo notebook

```bash
jupyter notebook demo.ipynb
```

The notebook loads pretrained checkpoints directly from HuggingFace (no cluster required) and lets you:
- Generate any modality conditioned on any other (Caption → RGB → Depth → Normals, …)
- Compare baseline vs best variant side-by-side
- Load compute-fair intermediate checkpoints

---

## Repository structure

```
nano4M-architectural-improvements/
│
├── cfgs/nano4M/
│   ├── multiclevr_d6-6w512.yaml   ← unmodified baseline config
│   └── variants/                  ← one YAML per experiment (29 configs)
│       ├── TEMPLATE.yaml          ← copy this for every new experiment
│       ├── swiglu.yaml
│       ├── rope.yaml
│       └── ...
│
├── nanofm/                        ← core library (installed as a package)
│   ├── data/multimodal/           ← dataset + dataloader
│   ├── modeling/
│   │   └── transformer_layers.py  ← all architectural building blocks
│   ├── models/
│   │   └── fourmUpgraded.py       ← FourM model with all variant flags
│   └── utils/                     ← training utilities
│
├── scripts/
│   ├── submit_job.sh              ← submit a training run to SLURM
│   ├── run_local.sh               ← quick local debug run (no SLURM)
│   ├── eval_job.sh                ← run eval interactively
│   ├── eval_job_slurm.sh          ← submit eval to SLURM
│   ├── eval_checkpoint.py         ← generate loss + FID report from checkpoint
│   ├── compile_results.py         ← aggregate all eval/*.log into a summary table
│   ├── watch_and_submit.sh        ← auto-submit jobs as SLURM slots open
│   ├── upload_to_hf.py            ← upload checkpoints to HuggingFace
│   └── clean.sh                   ← wipe slurm logs / intermediate checkpoints
│
├── figures/
│   ├── plot_cf_bar.py             ← compute-fair bar chart (paper Fig. 2)
│   └── plot_val_curves.py         ← WandB training curves (paper Fig. 1)
│
├── demo.ipynb                     ← interactive generation demo
├── run_training.py                ← training entry point
├── setup_env.sh                   ← one-shot environment setup
└── pyproject.toml                 ← package definition + dependencies
```

---

## Running an experiment

### 1. Create a config

```bash
cp cfgs/nano4M/variants/TEMPLATE.yaml cfgs/nano4M/variants/my_variant.yaml
# Edit only the model flags:
#   pos_encoding, use_swiglu, init_strategy, enc_depth, dec_depth, residual_scaling
# Leave run_name / output_dir / wandb_run_name on 'auto'
```

### 2. Submit to SLURM

```bash
bash scripts/submit_job.sh cfgs/nano4M/variants/my_variant.yaml
```

| Argument | Description | Default |
|---|---|---|
| config | Path to YAML | required |
| exp_name | Unique run name — auto-versioned (`_v1`, `_v2`, …) if omitted | auto |
| partition | SLURM partition | `l40s` |
| num_gpus | Number of GPUs | `2` |
| wandb_key | WandB API key | — |

> The script enforces `batch_size × num_gpus = 512` and fails fast if the YAML is misconfigured.

Outputs → `/scratch/$USER/nano4M/<exp_name>/`  
SLURM logs → `slurm_logs/<exp_name>_<jobid>.out/.err`  
WandB → `epfl-com304-group16 / COM304_nano4M / <exp_name>`

### 3. Quick local debug

```bash
bash scripts/run_local.sh cfgs/nano4M/variants/my_variant.yaml my_variant_debug 1
```

---

## Evaluating a checkpoint

```bash
# Single eval on a GPU node (loss + FID on full val set, ~30 min)
bash scripts/eval_job.sh my_variant_v1

# Submit to SLURM queue
bash scripts/eval_job_slurm.sh my_variant_v1

# Loss only, no FID (~2 min)
bash scripts/eval_job.sh my_variant_v1 --skip_fid

# Evaluate all experiments in parallel
for exp in baseline_v1 swiglu_v1 rope_v1; do
    bash scripts/eval_job_slurm.sh $exp
done
```

Reports are saved to `eval/<exp_name>/report.log`.

### Compute-fair evaluation

Models with more parameters use more FLOPs per step. To compare at equal compute:

```
fair_tokens = 5000M × (N_baseline / N_variant)    where N_baseline = 110M
```

Pass an intermediate checkpoint to override the default final one:

```bash
# depth16 (183M params): fair ≈ step 22889
bash scripts/eval_job_slurm.sh depth16_v1 \
    --checkpoint /scratch/$USER/nano4M/depth16_v1/checkpoint-22889.safetensors \
    --output eval/depth16_v1/report_cf.log
```

### Compile all results

```bash
python scripts/compile_results.py
# → results_summary.txt   (human-readable)
# → results_summary.csv   (for analysis / plotting)
```

---

## Key results

| Model | Params | Loss | Δ baseline | FID |
|---|---|---|---|---|
| Baseline | 110M | 3.499 | — | 47.2 |
| SwiGLU | 122M | 3.446 | −0.053 | 59.7 |
| RoPE | 110M | 3.492 | −0.007 | 41.3 |
| SwiGLU + Depth 16 | 217M | 3.313 | −0.186 | 38.9 |
| **SwiGLU + D16 + ResScale + DeepNorm** | **217M** | **3.286** | **−0.213** | **29.2** |

All 30 checkpoints (final + compute-fair intermediate) are available on HuggingFace.  
See [`scripts/compile_results.py`](scripts/compile_results.py) and [`results_summary.csv`](results_summary.csv) for the full table.  
For a detailed analysis of all experiments and findings, see [`report/final-report.pdf`](report/final-report.pdf).

---

## Storage layout (only for EPFL cluster)

```
/scratch/$USER/nano4M/<exp_name>/   ← checkpoints (scratch, 30-day retention)
eval/<exp_name>/report.log          ← eval reports (home)
slurm_logs/                         ← SLURM stdout/stderr (home)
```

The scratch directory is created automatically on first `submit_job.sh` run.  
Run `bash scripts/clean.sh slurm` or `bash scripts/clean.sh models` to free space.

---

## Adding a new architectural modification

1. **Implement** the building block in `nanofm/modeling/transformer_layers.py`
2. **Add a flag** to `FourMUpgraded.__init__` in `nanofm/models/fourmUpgraded.py`
3. **Create a config** by copying `TEMPLATE.yaml` and setting the new flag
4. **Submit** with `submit_job.sh`

Each person works in their own classes → no merge conflicts on `transformer_layers.py`.