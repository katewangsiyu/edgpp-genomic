# EDG++ Genomic

Reliability-gated selective distillation from frozen Borzoi to compact student for variant effect prediction.

## Layout

```
edgpp-genomic/
├── configs/                  3 YAML configs (T4 debug / 5090 smoke / 5090 full)
├── src/edgpp_genomic/        main package
│   ├── models/               teacher (Fake + Flashzoi), student, reliability estimator
│   ├── data/                 VCF + FASTA + Dataset
│   ├── training/             SelectiveDistillLoss (EDG++ core)
│   └── evaluation/           SED / QTL metrics / calibration
├── scripts/                  runnable entry points
├── baselines/                external repos (gitignored; clone here)
└── data/                     raw + cached (gitignored)
```

## Pipeline

| Phase | Hardware | Script | Config |
|---|---|---|---|
| 0 (Day 1–3) | T4 | `scripts/03_smoke.py` | `configs/t4_debug.yaml` |
| 1 (Day 4)   | 5090 | `scripts/03_smoke.py` | `configs/gpu5090_smoke.yaml` |
| 2 (Day 5–10) | 5090 | `scripts/05_teacher_precompute.py` → `06_train.py` → `07_eval.py` | `configs/gpu5090_full.yaml` |

Same `03_smoke.py` runs on both T4 and 5090 — only the config changes.

## Setup (you run these)

### 1. Clone this repo's baselines
See `baselines/README.md` for the exact `git clone` commands.

### 2. Download data
See `scripts/download_data.sh`. Requires `gcloud` for the Borzoi QTL VCFs (Requester Pays).

### 3. Create conda env (T4 first)
```
bash scripts/env_t4.sh
conda activate edgpp_t4
pip install -e .
```

### 4. Prepare debug SNP subset
```
python scripts/02_prepare_debug_snp.py --n 100
```

### 5. Phase 0 smoke test
```
python scripts/03_smoke.py --config configs/t4_debug.yaml
```

Should see ~50 training steps with decreasing `distill` loss, `mean_w` drifting, and no NaN.

## Design notes

- **Attention backend is hardware-agnostic**: `FlashzoiTeacher` wraps `borzoi-pytorch` which uses `F.scaled_dot_product_attention`. PyTorch dispatches to `math`/`efficient` on T4 and to `flash` on 5090 automatically.
- **Phase 0 uses `FakeTeacher`**: a small random CNN. T4 is only for pipeline/loss validation, not teacher accuracy.
- **Teacher cache**: Phase 2 precomputes Flashzoi SED scores once (~1.5 h on 5090) to a parquet; all downstream training reads from cache.
