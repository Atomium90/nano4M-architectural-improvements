# nano4M — Architectural Improvements

> COM-304 project | EPFL VILab
> Edouard Fousson · Thomas Picart · Jonathan Balli

---

## Project overview

We implement and ablate five targeted architectural modifications to the nano4M baseline Transformer.

Planned variants:

1. Positional encoding — RoPE / ALiBi instead of learned absolute embeddings
2. Feed-forward block — SwiGLU instead of GELU
3. Weight initialization — He / Xavier / DeepNorm
4. Model depth — ± encoder/decoder layers
5. Residual scaling — α·F(x) with α = 1/√N

Full details in [`report/extension-proposal.pdf`](report/extension-proposal.pdf).

---

## Repository structure

```
nano4M-architectural-improvements/
├── cfgs/nano4M/
│   ├── multiclevr_d6-6w512.yam         ← unmodified baseline config
│   └── variants/                       ← one YAML per experiment
│       └── TEMPLATE.yaml               ← copy this for every new experiment
│       ├── rope.yaml                   ← example: RoPE variant
│       └── ...                         ← one yaml per experiment
│
├── nanofm/
│   ├── modeling/
│   │   ├── transformer_layers.py   ← add new building blocks here
│   └── models/
│       └── fourm.py                ← add variant flags here
│
├── scripts/
│   ├── submit_job.sh      ← submit to SLURM with an experiment name
│   ├── run_local.sh       ← quick local debug run
│   └── eval_checkpoint.py ← generate a .log report from a checkpoint
│
├── outputs/               ← auto-created, one subfolder per run name
├── slurm_logs/            ← SLURM stdout/stderr, named by experiment
├── run_training.py        ← training entry point
└── report/
    └── extension-proposal.pdf
```

---

## How code changes are organized
 
**`nanofm/modeling/transformer_layers.py`**  
Add new classes here (e.g. `RoPEAttention`, `SwiGLU`, `ScaledResidualBlock`).  
Each person adds their own classes without touching others → no merge conflicts.
 
**`nanofm/models/fourm.py`**  
One file, with constructor flags to switch between implementations:
```python
class FourM(nn.Module):
    def __init__(self, ..., pos_encoding='none', use_swiglu=False, ...):
        ...
        if pos_encoding == 'rope':
            self.encoder = TransformerTrunkRoPE(...)
        else:
            self.encoder = TransformerTrunk(...)
```
The YAML passes the flags via `model_config`.
 
---

## How to run an experiment
 
### 1. Create a config
 
```bash
cp cfgs/nano4M/variants/TEMPLATE.yaml cfgs/nano4M/variants/my_variant.yaml
# Edit: run_name, output_dir, _target_ class, and any model flags
```
 
### 2. Submit to SLURM
 
```bash
bash scripts/submit_job.sh cfgs/nano4M/variants/my_variant.yaml my_variant 2 $WANDB_API_KEY
```
 
| Argument | Description | Example |
|---|---|---|
| config | Path to YAML | `cfgs/nano4M/variants/rope.yaml` |
| exp_name | Unique run name (used for output dir, SLURM logs, WandB run) | `rope_v1` |
| num_gpus | GPUs to allocate (default: 2) | `2` |
| wandb_key | WandB API key (optional) | `$WANDB_API_KEY` |
 
Outputs → `./outputs/my_variant/`  
SLURM logs → `./slurm_logs/my_variant_<jobid>.out/.err`  
WandB → `epfl-com304-group16 / COM304_nano4M / my_variant`
 
> Each run gets its own WandB entry — passing a different `exp_name` is enough, no runs override each other.
 
### 3. Quick local debug (no SLURM)
 
```bash
bash scripts/run_local.sh cfgs/nano4M/variants/my_variant.yaml my_variant_debug 1
```
 
---

## Evaluating a checkpoint

```bash
python scripts/eval_checkpoint.py \
    --checkpoint outputs/<EXP_NAME>/checkpoint-best.pth \
    --config     cfgs/nano4M/variants/<name>.yaml
```

This generates `outputs/<EXP_NAME>/eval.log` containing:

- Model info (param count, architecture, variant keys)
- Validation cross-entropy loss (per-token, per-modality)
- Perplexity
- Throughput (tokens/sec)
- Peak GPU memory
- Gradient norm snapshot

Optional FID computation (slow):
```bash
python scripts/eval_checkpoint.py ... --fid
```

---

## Environment setup

```bash
bash setup_env.sh        # create conda env (once)
conda activate nanofm    # before any run
```
