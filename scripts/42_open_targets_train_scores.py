"""Train OOF chrom-LOO HCCP scores on the Open Targets subsampled dataset.

Pipeline (matches the implicit assumptions of T_tools/cp_baselines_h2h.py):

  1. For each chromosome c, fit HistGradientBoostingClassifier on rows with
     chrom != c, predict p_hat[chrom == c]. Out-of-fold predictions over all
     22 + X chromosomes ⇒ each row has a p_hat that did not see its own chrom.
  2. Fit HistGradientBoostingRegressor on |y - p_hat| residuals using the
     same OOF protocol — for each chrom c, fit on rows where chrom != c
     (residuals are valid OOF since p_hat is OOF), predict sigma[chrom == c].
  3. Save a parquet at the path that load_scores() in
     T_tools/cp_baselines_h2h.py expects (open_targets branch):
         outputs/hetero_head/open_targets/scores_with_sigma.parquet
     with columns: chrom, p_hat, sigma, label, plus passthrough id columns.

The chrom-LOO loop here precomputes per-row scores; cp_baselines_h2h.py then
runs an *outer* chrom-LOO conformal calibration on top, which is the same
two-tier structure used for TraitGym (see scripts/11_aggregator_gbm.py +
scripts/13_hetero_head.py).

Feature set (per OT adapter design Q1: native, sign-off):
    beta, standardError, pValueMantissa, pValueExponent, log10BF, r2Overall,
    credibleSetlog10BF, purityMeanR2, purityMinR2, sampleSize, locus_size,
    position_norm.
We deliberately drop pip (the label proxy) and identifiers
(studyLocusId, studyId).

Usage:
    python scripts/42_open_targets_train_scores.py \\
        --in data/processed/open_targets/gwas_complex_aligned.parquet \\
        --out outputs/hetero_head/open_targets/scores_with_sigma.parquet \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm


FEATURE_SETS: dict[str, list[str]] = {
    # Circular (label-leaking) — kept only for the negative result documented
    # in 06_experiments §6.x. SuSiE's PIP is computed FROM these columns, so
    # using them to predict PIP gives an artificial AUPRC ≈ 0.98.
    "circular": [
        "beta", "standardError",
        "pValueMantissa", "pValueExponent",
        "logBF", "r2Overall",
        "credibleSetlog10BF", "purityMeanR2", "purityMinR2",
        "sampleSize", "locus_size",
        "position_norm",
    ],
    # Leak-free, matched-9 dataset features (output of
    # scripts/44_open_targets_matched_controls.py).  Mirrors TraitGym Complex's
    # external-pathogenicity feature pipeline (CADD analogues from gnomAD VEP).
    "matched9": [
        "af_overall", "af_nfe", "af_max",
        "gerp", "alphamissense", "sift", "foldx",
        "is_lof_loftee", "is_lof_vep",
        "consequence_severity",
        "has_alphamissense", "has_sift", "has_foldx",
    ],
}
DEFAULT_FEATURE_SET = "circular"


def clf_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("hgb", HistGradientBoostingClassifier(
            max_depth=3, max_iter=300, learning_rate=0.1,
            class_weight="balanced", random_state=seed,
            early_stopping=False)),
    ])


def reg_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("hgb", HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=3, max_iter=300, learning_rate=0.1,
            random_state=seed, early_stopping=False)),
    ])


def chrom_loo_oof_classifier(X: np.ndarray, y: np.ndarray, chroms: np.ndarray,
                             seed: int = 42) -> np.ndarray:
    """Out-of-fold p_hat per chromosome."""
    p_hat = np.full(len(y), np.nan)
    unique_chroms = sorted(set(chroms))
    for c in tqdm(unique_chroms, desc="OOF p_hat"):
        mask_te = chroms == c
        mask_tr = ~mask_te
        if y[mask_tr].sum() == 0 or y[mask_te].sum() == 0:
            print(f"  [skip] chrom {c}: zero positives in train or test")
            continue
        clf = clf_pipeline(seed)
        clf.fit(X[mask_tr], y[mask_tr])
        p_hat[mask_te] = clf.predict_proba(X[mask_te])[:, 1]
    return p_hat


def chrom_loo_oof_sigma(X: np.ndarray, y: np.ndarray, p_hat: np.ndarray,
                        chroms: np.ndarray, seed: int = 42) -> np.ndarray:
    """Out-of-fold sigma_hat per chromosome, fit on |y - p_hat| residuals."""
    sigma = np.full(len(y), np.nan)
    residual = np.abs(y - p_hat)
    unique_chroms = sorted(set(chroms))
    for c in tqdm(unique_chroms, desc="OOF sigma"):
        mask_te = chroms == c
        mask_tr = ~mask_te
        if np.isnan(residual[mask_tr]).any():
            print(f"  [warn] chrom {c}: residuals contain NaN in train pool, skipping")
            continue
        reg = reg_pipeline(seed)
        reg.fit(X[mask_tr], residual[mask_tr])
        sigma[mask_te] = np.clip(reg.predict(X[mask_te]), 1e-4, None)
    return sigma


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="Subsample output (circular features) or matched-9 output (matched9 features)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Path to write scores_with_sigma.parquet")
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS.keys()),
                    default=DEFAULT_FEATURE_SET,
                    help="Which feature list to use; see FEATURE_SETS in module")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    feature_cols = FEATURE_SETS[args.feature_set]
    print(f"[feature-set] {args.feature_set} ⇒ {len(feature_cols)} cols")

    print(f"[load] {args.inp}")
    df = pd.read_parquet(args.inp).copy()
    chrom_col = "chromosome" if "chromosome" in df.columns else "chrom"
    df[chrom_col] = df[chrom_col].astype(str)
    pos_col = "position" if "position" in df.columns else "pos"

    if "position_norm" in feature_cols and "position_norm" not in df.columns:
        # Add normalized position feature (0-1 fraction within chromosome).
        pos_min = df.groupby(chrom_col)[pos_col].transform("min")
        pos_max = df.groupby(chrom_col)[pos_col].transform("max")
        pos_range = (pos_max - pos_min).replace(0, 1)
        df["position_norm"] = (df[pos_col] - pos_min) / pos_range

    for col in feature_cols:
        if col not in df.columns:
            raise KeyError(f"missing feature column: {col} (have {list(df.columns)})")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].astype(int).to_numpy()
    chroms = df[chrom_col].to_numpy()

    print(f"[shape] {len(df):,} rows × {len(feature_cols)} features  "
          f"π_+={y.mean():.4f}  chroms={sorted(set(chroms))}")

    # OOF stage 1 — classifier
    print("\n[stage 1/2] OOF classifier (chrom-LOO)")
    p_hat = chrom_loo_oof_classifier(X, y, chroms, seed=args.seed)
    valid = ~np.isnan(p_hat)
    if valid.sum() == 0:
        raise RuntimeError("All chroms skipped — check label balance.")
    auprc = float(average_precision_score(y[valid], p_hat[valid]))
    auroc = float(roc_auc_score(y[valid], p_hat[valid]))
    print(f"  full-pool OOF AUPRC = {auprc:.4f}, AUROC = {auroc:.4f}")

    # OOF stage 2 — sigma regressor (only on rows with valid p_hat)
    print("\n[stage 2/2] OOF sigma_hat (chrom-LOO on residuals)")
    sigma = np.full(len(y), np.nan)
    sigma[valid] = chrom_loo_oof_sigma(X[valid], y[valid], p_hat[valid],
                                        chroms[valid], seed=args.seed)
    sigma_valid = ~np.isnan(sigma) & valid
    print(f"  σ̂ range: [{sigma[sigma_valid].min():.4f}, {sigma[sigma_valid].max():.4f}]  "
          f"R = {sigma[sigma_valid].max() - sigma[sigma_valid].min():.4f}")

    # Build output DataFrame in cp_baselines_h2h.py schema
    out_df = pd.DataFrame({
        "chrom": df[chrom_col].astype(str),
        "p_hat": p_hat,
        "sigma": sigma,
        "label": y,
        "position": df[pos_col].astype(int),
        "variantId": df["variantId"].astype(str) if "variantId" in df.columns else "",
    })
    if "studyLocusId" in df.columns:
        out_df["studyLocusId"] = df["studyLocusId"].astype(str)
    if "match_group" in df.columns:
        out_df["match_group"] = df["match_group"].astype(int)
    if "split" in df.columns:
        out_df["split"] = df["split"].astype(str)
    out_df = out_df[sigma_valid].reset_index(drop=True)
    print(f"\n[output] {len(out_df):,} rows after dropping rows with NaN p_hat or sigma")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"saved: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")

    if args.summary_out is not None:
        chroms_present = sorted(out_df["chrom"].unique())
        per_chrom = (out_df.groupby("chrom", as_index=False)
                            .agg(n=("label", "size"),
                                 pi_pos=("label", "mean")))
        summary = {
            "n_rows": int(len(out_df)),
            "n_chroms": int(out_df["chrom"].nunique()),
            "auprc_oof_full": auprc,
            "auroc_oof_full": auroc,
            "sigma_min": float(sigma[sigma_valid].min()),
            "sigma_max": float(sigma[sigma_valid].max()),
            "sigma_range": float(sigma[sigma_valid].max() - sigma[sigma_valid].min()),
            "per_chrom": per_chrom.to_dict("records"),
            "feature_cols": feature_cols,
            "feature_set": args.feature_set,
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"saved: {args.summary_out}")


if __name__ == "__main__":
    main()
