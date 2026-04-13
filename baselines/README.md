# Baselines

External repos needed for teacher, baselines, and benchmark eval.
All `git clone` into this directory. They are gitignored from the parent repo.

## Required (for Phase 1/2 on 5090)

```bash
# Teacher backbone (PyTorch Flashzoi port, Apache-2.0)
git clone https://github.com/johahi/borzoi-pytorch.git baselines/borzoi-pytorch

# Official Borzoi eval scripts (reference for metric definitions)
git clone https://github.com/calico/borzoi-paper.git baselines/borzoi-paper

# westminster — the metric code the Borzoi paper actually uses
git clone https://github.com/calico/westminster.git baselines/westminster
```

## Recommended (for DEGU baseline + benchmarks)

```bash
# Direct competitor — complete reproduction needs this
git clone https://github.com/zrcjessica/ensemble_distillation.git baselines/ensemble_distillation

# TraitGym benchmark (Snakemake pipeline)
git clone https://github.com/songlab-cal/TraitGym.git baselines/TraitGym

# DART-Eval (variant effect task 5)
git clone https://github.com/kundajelab/DART-Eval.git baselines/DART-Eval
```

## Optional (reference / scooby follow-up)

```bash
git clone https://github.com/calico/borzoi.git baselines/borzoi-tf     # original Keras
git clone https://github.com/gagneurlab/scooby.git baselines/scooby    # single-cell extension
```

## Install

After cloning, install `borzoi-pytorch` into the 5090 env:

```bash
conda activate edgpp_5090
pip install -e baselines/borzoi-pytorch/
```

The other baselines are reference-only (no install needed).
