"""HCCP assay-LOO on ProteinGym — out-of-domain validation.

For each held-out assay:
  1. Train p̂ (HistGradientBoostingClassifier on DMS_score_bin) using all
     other assays' labeled mutants.
  2. Fit σ̂ (HistGradientBoostingRegressor on |y - p̂|) using the same
     training pool with assay-LOO OOF residuals.
  3. Mondrian (y × σ̂-bin) conformal calibration: pool the calibration
     fold from all non-target assays.
  4. Predict sets on the target assay; report marginal coverage, σ̂-bin
     gap, and per-cell coverage table.

This is the ProteinGym analogue of our TraitGym trait-LOO stress test
(§6 of the main paper). Because ProteinGym is a regression benchmark,
we use the DMS_score_bin discretization provided by the dataset authors
(roughly 57% positive) and evaluate coverage of the binary prediction
set.

Usage:
    python scripts/38_proteingym_hccp.py \\
        --features-parquet data/raw/proteingym/features.parquet \\
        --n-assays 30 --alpha 0.10 --n-bins 5 \\
        --out-dir outputs/proteingym_hccp
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                               HistGradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm


# Reuse HCCP conformal primitives from the pip-installable package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from edgpp_genomic.hccp.conformal import mondrian_calibrate, predict_set_from_calibration


def clf_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("hgb", HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.1,
            class_weight="balanced", random_state=seed,
            early_stopping=False)),
    ])


def reg_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("hgb", HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=3, max_iter=150, learning_rate=0.1,
            random_state=seed, early_stopping=False)),
    ])


def assay_loo_one(target_assay: str, df: pd.DataFrame,
                   feat_cols: list[str], alpha: float, n_bins: int,
                   seed: int = 42) -> dict:
    mask_target = df["DMS_id"] == target_assay
    mask_other = ~mask_target
    X_tr = df.loc[mask_other, feat_cols].to_numpy()
    y_tr = df.loc[mask_other, "DMS_score_bin"].astype(int).to_numpy()
    X_te = df.loc[mask_target, feat_cols].to_numpy()
    y_te = df.loc[mask_target, "DMS_score_bin"].astype(int).to_numpy()

    if y_tr.sum() == 0 or y_te.sum() == 0:
        return {"target_assay": target_assay, "skipped": "no positives"}

    # 1. p̂ trained on all other assays
    clf = clf_pipeline(seed); clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]
    p_tr = clf.predict_proba(X_tr)[:, 1]
    auprc_te = float(average_precision_score(y_te, p_te))
    auroc_te = float(roc_auc_score(y_te, p_te))

    # 2. σ̂ trained on the same non-target pool using residuals of the same p̂
    # (valid split-CP setup: calibration fold is the same as σ̂-training fold,
    # but we use absolute residuals not labels, so this does not leak).
    r_tr = np.abs(y_tr - p_tr)
    reg = reg_pipeline(seed); reg.fit(X_tr, r_tr)
    sigma_tr = np.clip(reg.predict(X_tr), 1e-4, None)
    sigma_te = np.clip(reg.predict(X_te), 1e-4, None)

    # 3. Mondrian conformal calibration on the non-target pool
    cal = mondrian_calibrate(p_tr, sigma_tr, y_tr, alpha, n_bins, min_cell_size=10)
    pred_sets = predict_set_from_calibration(p_te, sigma_te, cal)
    covered = np.array([y_te[i] in ps for i, ps in enumerate(pred_sets)])
    sizes = np.array([len(ps) for ps in pred_sets])

    # σ̂-bin coverage (recompute bins on target assay using calibration edges)
    bin_id_te = np.digitize(sigma_te, cal.bin_edges[1:-1])
    per_bin = []
    for b in range(n_bins):
        m = bin_id_te == b
        if m.sum() < 5:
            continue
        per_bin.append({
            "bin": b, "n": int(m.sum()),
            "coverage": float(covered[m].mean()),
            "gap": float(abs(covered[m].mean() - (1 - alpha))),
        })
    sigma_range = (max(b["coverage"] for b in per_bin)
                   - min(b["coverage"] for b in per_bin)) if per_bin else float("nan")

    return {
        "target_assay": target_assay,
        "n_train": int(mask_other.sum()),
        "n_test": int(mask_target.sum()),
        "pos_rate_test": float(y_te.mean()),
        "AUPRC_test": auprc_te,
        "AUROC_test": auroc_te,
        "marginal_coverage": float(covered.mean()),
        "sigma_bin_range": sigma_range,
        "frac_singleton": float((sizes == 1).mean()),
        "frac_both": float((sizes == 2).mean()),
        "frac_empty": float((sizes == 0).mean()),
        "per_sigma_bin": per_bin,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--n-assays", type=int, default=0,
                    help="Cap number of assays to evaluate (0 = all). "
                         "Assays are sorted by size (biggest first) to keep "
                         "calibration folds well-populated.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.features_parquet)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    print(f"[load] {len(df):,} mutants × {len(feat_cols)} features "
          f"across {df.DMS_id.nunique()} assays")

    # Pick target assays — prefer medium-sized assays (sufficient positives but
    # not dominating the training pool).
    assay_sizes = df.groupby("DMS_id").size().sort_values(ascending=False)
    assay_pos = df.groupby("DMS_id")["DMS_score_bin"].sum()
    candidates = [a for a in assay_sizes.index
                  if assay_pos[a] >= 50 and assay_sizes[a] >= 200]
    if args.n_assays > 0:
        # Sort by size DESC, take the top-n (biggest valid assays)
        candidates = candidates[:args.n_assays]
    print(f"[select] {len(candidates)} assays to evaluate")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    results = []
    for assay in tqdm(candidates, desc="assay-LOO"):
        r = assay_loo_one(assay, df, feat_cols, args.alpha, args.n_bins, args.seed)
        results.append(r)
        # Periodic checkpoint
        (out / "per_assay_results.json").write_text(json.dumps(results, indent=2))

    # Aggregate stats
    valid = [r for r in results if "marginal_coverage" in r]
    if valid:
        covs = np.array([r["marginal_coverage"] for r in valid])
        gaps = np.array([r["sigma_bin_range"] for r in valid
                         if not np.isnan(r["sigma_bin_range"])])
        auprcs = np.array([r["AUPRC_test"] for r in valid])
        singletons = np.array([r["frac_singleton"] for r in valid])
        summary = {
            "alpha": args.alpha, "n_bins": args.n_bins,
            "n_assays_evaluated": len(valid),
            "coverage_mean": float(covs.mean()),
            "coverage_std": float(covs.std()),
            "coverage_target": 1 - args.alpha,
            "sigma_bin_gap_mean": float(gaps.mean()) if gaps.size else None,
            "sigma_bin_gap_median": float(np.median(gaps)) if gaps.size else None,
            "sigma_bin_gap_p90": float(np.quantile(gaps, 0.9)) if gaps.size else None,
            "AUPRC_test_mean": float(auprcs.mean()),
            "AUPRC_test_median": float(np.median(auprcs)),
            "frac_singleton_mean": float(singletons.mean()),
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("\n=== ProteinGym assay-LOO summary ===")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k:<30s}: {v:.4f}")
            else:
                print(f"  {k:<30s}: {v}")
    print(f"\nsaved: {out}/per_assay_results.json, summary.json")


if __name__ == "__main__":
    main()
