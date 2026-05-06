# Open Targets H2H Adapter — Design Document (Step 4)

**Goal**: run HCCP + 6 contemporary CP baselines (split, σ̂-Mondrian, class-Mondrian, RLCP, weighted CP, SC-CP) on the Open Targets GWAS polygenic dataset produced by `T_tools/open_targets_subsample.py`, with a protocol aligned to the existing TraitGym H2H in `T_tools/cp_baselines_h2h.py` so the per-method coverage / σ̂-bin gap numbers are directly comparable.

This document is for protocol review **before** writing the adapter. Once you sign off on the seven decisions below, I write `T_tools/cp_baselines_h2h_open_targets.py` (or merge into the existing `cp_baselines_h2h.py` via a `--dataset open_targets` flag — see Decision 7).

---

## What's the same as TraitGym

1. **chrom-LOO protocol**: test = chr {17, 18, 19, 20, 21, 22, X}, calibration + train = remaining 16 autosomes. Identical to TraitGym `06_experiments.tex` setup.
2. **HCCP construction**: same `mondrian_calibrate` from `src/edgpp_genomic.hccp.conformal`, same nonconformity score `|y − p̂(x)| / σ̂(x)`, same nested chrom-LOO K-selection on $\hat K(c_{\mathrm{outer}})$ via `T_tools/nested_kcv_helpers.py`.
3. **Six baselines unchanged**: B1 vanilla split CP, B2 σ̂-Mondrian (Boström 2020), B3 class-Mondrian (Sadinle 2019), RLCP (Hore 2025), weighted CP (Tibshirani 2019), SC-CP (van der Laan 2024). Implementations in `cp_baselines_h2h.py:rlcp_chrom_loo / weighted_cp_chrom_loo / sccp_chrom_loo` reused without modification.
4. **Bootstrap CIs**: $B = 200$ chrom-level paired bootstrap, identical to `T_tools/paired_bootstrap_h2h.py`.
5. **K_eval = 5**: matches TraitGym Complex headline. Per-cell minority @ K=5 with $n=11{,}400$, $\pi_+=0.10$ ⇒ ≈ 228, well above the per-cell floor of 100. Ablation $K_{\mathrm{eval}} \in \{2, 3, 5, 8, 10\}$ optional once headline numbers are in.

## What's different (7 decisions)

### Decision 1 — Feature set

TraitGym uses **CADD + GPN-MSA + Borzoi** (pre-computed pathogenicity scores). OT does not have these.

**OT-native feature set** (from the parsed parquet, all numerical):
- `pip` — posterior inclusion probability (DROP from features — this is essentially the label)
- `beta`, `standardError` — GWAS effect size
- `pValueMantissa`, `pValueExponent` — GWAS significance
- `log10BF` — variant-level Bayes factor
- `r2Overall` — LD r² with the credible-set lead
- `credibleSetlog10BF`, `purityMeanR2`, `purityMinR2` — set-level statistics
- `sampleSize`, `locus_size` — sample / set scale
- Position normalized within chromosome (0-1 fraction)

**Drop**: `pip` itself (it's nearly the label), `studyLocusId` / `studyId` (identifiers, would leak set membership).

→ **Net feature dim ≈ 10 numerical + a few categorical (`finemappingMethod`, `studyType`)**.

**Protocol question 1** (sign off needed): are you OK with this feature set? Alternatives: (a) add gnomAD AF if I download `variant/` parquet (extra 3.75 GB); (b) compute hg38 sequence features (≥hours of additional work); (c) cross-reference TraitGym features for the variants in OT (likely <10% overlap, dataset-level integrity hit).

### Decision 2 — Aggregator $\hat p(x)$

TraitGym uses **HistGradientBoostingClassifier** (matches `scripts/11_aggregator_gbm.py`). OT-native: same HGB (no domain reason to differ).

**Protocol question 2**: same HGB, `max_depth=3`, `max_iter=300`, `class_weight="balanced"` to match TraitGym. OK?

### Decision 3 — Heteroscedastic head $\hat\sigma(x)$

TraitGym uses HGB regressor on $|y - \hat p(x)|$ residuals fit on the calibration fold. OT-native: same.

**Protocol question 3**: identical recipe? OK?

### Decision 4 — Calibration / training fold structure

TraitGym chrom-LOO: for each test chromosome $c$, train + calibrate on the other 22. The `mondrian_calibrate` function builds K bins on calibration σ̂-quantiles.

OT subsample (per `T_tools/open_targets_subsample.py`): test split = chr {17, 18, 19, 20, 21, 22, X} (TraitGym convention). Train + calibrate on chr 1–16. **Note**: this is *split-CP* not chrom-LOO, because we evaluate **the union** of test chroms as one test fold rather than per-chrom out-of-fold.

→ Two sub-options:
- **(A) Single-fold split-CP** (simpler): train+cal on chr 1–16, test on chr 17–X as one block. Bootstrap on test chroms for CIs.
- **(B) Per-test-chrom chrom-LOO** (matches TraitGym verbatim): for each $c \in \{17, 18, ..., X\}$, train + cal on the *other 22* chroms (i.e., chr 1–22 + X, leave $c$ out). 7 outer folds.

**Protocol question 4** (sign off needed): do we want (A) split-CP simplicity or (B) full chrom-LOO match? (B) is the strictly stronger protocol but 7× slower; it's also what cp_baselines_h2h.py already implements. **Recommend (B)** for direct comparability with TraitGym Tab.cp_baselines numbers.

### Decision 5 — K_eval choice + nested CV

For TraitGym Complex, K_eval=5 with K_HCCP per outer fold via nested chrom-LOO (modal $\hat K = 5$). Per-cell minority floor binds at K=5: $\pi_{\min} n / K = 0.1 \times 11{,}400 / 5 = 228$.

For OT subsampled, $\pi_{\min} n / K = 0.1 \times 11{,}400 / 5 = 228$ (same n, same π_+). → **Keep K_eval = 5**.

Nested K-grid: $\{2, 3, 5, 8, 10\}$ same as TraitGym.

**Protocol question 5**: K_eval=5 + nested K-grid {2,3,5,8,10}? OK?

### Decision 6 — Output schema (for paired-bootstrap reuse)

`T_tools/paired_bootstrap_h2h.py` consumes the per-fold per-method (cov, gap) outputs from `cp_baselines_h2h.py`. To reuse it, the OT adapter must produce the **identical** JSON schema:

```json
{
  "dataset": "open_targets",
  "K_eval": 5,
  "K_hccp_per_fold": {"17": 5, "18": 8, ...},
  "K_sccp_per_fold": {...},
  "raw_per_replicate": {
    "HCCP": [...], "RLCP": [...], "WeightedCP": [...], "SCCP": [...]
  },
  "marginal_stats": {...},
  "paired_vs_HCCP": {...}
}
```

Stored at `R_raw/cp_baselines_h2h/open_targets_paired_bootstrap_Keval5_B200.json` for direct downstream reuse.

**Protocol question 6**: replicate exact schema? OK?

### Decision 7 — Code organization

Two options:
- **(α) Add `--dataset open_targets` flag to `T_tools/cp_baselines_h2h.py`**: minimal new code, swap `load_scores` to a new branch reading `data/processed/open_targets/gwas_complex_aligned.parquet` and computing $(\hat p, \hat\sigma)$ on-the-fly via the same pipelines as scripts/38 ProteinGym.
- **(β) New file `T_tools/cp_baselines_h2h_open_targets.py`**: forks the TraitGym version with OT-specific load + feature-build steps.

**Protocol question 7**: (α) or (β)? **Recommend (α)** — single source of truth; touch points are `load_scores()` (line 44) and a new `--feature-set open_targets` branch in feature extraction.

---

## After sign-off: execution plan

| Step | Code | Time | Output |
|---|---|---|---|
| Adapter implementation per decisions 1–7 | edits to `T_tools/cp_baselines_h2h.py` (or new file) | ~3-4h | runnable adapter |
| Per-fold HCCP + 6 baselines, K_eval=5, B=200 paired bootstrap | `python T_tools/cp_baselines_h2h.py --dataset open_targets --K-mode nested-cv --bootstrap 200` | ~30-60 min CPU | `R_raw/cp_baselines_h2h/open_targets_paired_bootstrap_Keval5_B200.json` |
| K_eval ∈ {2,3,5,8,10} sweep (matches Tab.keval_sensitivity) | same script with `--K-eval-grid` | ~3-5h CPU | per-K JSON |
| π_+ ∈ {0.05, 0.10, 0.20, 0.30, 0.50} imbalance sweep (matches Tab.imbsweep) | re-run subsample.py × 5 with `--target-pi-pos`, then adapter on each | ~1-2h | per-π JSON |
| Trait-LOO (analog of TraitGym §6.3 OOD) | study-LOO via `--loo-on study` flag | ~1h | study-LOO JSON |
| Paper writing — new §6.x "Open Targets cross-platform replication" | manual | 1-2 days | LaTeX subsection |

**Total compute: ~6-10h CPU**, parallelizable. **Total wall: ~1 week** (mostly writing).

---

## Three open questions for you to answer before adapter coding

1. **Feature set** (Decision 1): OT-native 10-d (recommend) | OT-native + gnomAD AF (extra 3.75 GB download) | other?
2. **Fold structure** (Decision 4): full chrom-LOO 7 outer folds (recommend, matches TraitGym verbatim) | single-fold split-CP?
3. **Code organization** (Decision 7): `--dataset open_targets` flag in existing file (recommend) | new file?

Other 4 decisions are mechanical (HGB, residual σ̂, K_eval=5, paired bootstrap schema) — defaulting to "match TraitGym" unless you flag.
