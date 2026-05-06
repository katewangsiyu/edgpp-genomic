"""Per-assay L_F estimation for ProteinGym predict-verify (Eq.(3) Phase 1.1).

For each ProteinGym assay evaluated by scripts/38_proteingym_hccp.py, replicates
the (clf, reg) training pipeline to obtain (p_hat, sigma, label) on the test
fold, then runs the LCLS estimator (T_tools/asl_audit:lcls_pairwise_ks) to
fit L_F via least-squares on KS = beta * gap (no intercept), the convention
used in §5.1's Eq.(3).

Skips Mondrian conformal calibration (already done in scripts/38). Joins the
resulting per-assay (n, pi_min, sigma_range_R, L_F) with the observed gap
from outputs/proteingym_hccp_n50/per_assay_results.json so downstream
predict-verify can score Eq.(3) per-assay without a second compute.

Usage:
    python T_tools/proteingym_lf_per_assay.py \\
        --features-parquet data/raw/proteingym/features.parquet \\
        --hccp-results outputs/proteingym_hccp_n50/per_assay_results.json \\
        --out R_raw/proteingym_lf/per_assay_LF.json
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
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from T_tools.asl_audit import lcls_pairwise_ks  # noqa: E402


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


def fit_lf_lcls(p: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                n_bins: int = 30, min_per_bin: int = 20) -> tuple[float, int, dict]:
    """LCLS L_F estimator: least-squares fit of KS = beta * gap (no intercept).

    Returns (beta, n_pairs, summary). beta is NaN if fewer than 2 pairs survive
    the per-bin minimum-count filter.
    """
    pairs = lcls_pairwise_ks(p, sigma, y, n_bins=n_bins, min_per_bin=min_per_bin)
    g = pairs["sigma_gaps"]
    k = pairs["ks_stats"]
    if g.size < 2:
        return float("nan"), int(g.size), {
            "n_pairs": int(g.size),
            "median_ks": float("nan"),
            "median_gap": float("nan"),
        }
    denom = float(np.sum(g * g))
    if denom < 1e-12:
        return float("nan"), int(g.size), {
            "n_pairs": int(g.size),
            "median_ks": float(np.median(k)),
            "median_gap": float(np.median(g)),
        }
    beta = float(np.sum(k * g) / denom)
    return beta, int(g.size), {
        "n_pairs": int(g.size),
        "median_ks": float(np.median(k)),
        "median_gap": float(np.median(g)),
        "max_ks": float(k.max()),
        "max_gap": float(g.max()),
    }


def per_assay_extract(target_assay: str, df: pd.DataFrame,
                      feat_cols: list[str], seed: int = 42,
                      n_test_cap: int = 3000) -> dict | None:
    """Replicate scripts/38 training to get (p_hat, sigma, label) on test fold.

    Returns None if the assay has zero positives in train or test (would skip
    in scripts/38).
    """
    mask_target = df["DMS_id"] == target_assay
    mask_other = ~mask_target

    X_tr = df.loc[mask_other, feat_cols].to_numpy()
    y_tr = df.loc[mask_other, "DMS_score_bin"].astype(int).to_numpy()
    X_te = df.loc[mask_target, feat_cols].to_numpy()
    y_te = df.loc[mask_target, "DMS_score_bin"].astype(int).to_numpy()

    if y_tr.sum() == 0 or y_te.sum() == 0:
        return None

    n_test_full = int(len(y_te))
    if n_test_full > n_test_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_test_full, n_test_cap, replace=False)
        X_te = X_te[idx]
        y_te = y_te[idx]

    clf = clf_pipeline(seed)
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]
    p_tr = clf.predict_proba(X_tr)[:, 1]

    r_tr = np.abs(y_tr - p_tr)
    reg = reg_pipeline(seed)
    reg.fit(X_tr, r_tr)
    sigma_te = np.clip(reg.predict(X_te), 1e-4, None)

    lf, n_pairs, lcls_summary = fit_lf_lcls(p_te, sigma_te, y_te)

    return {
        "target_assay": target_assay,
        "n_train": int(mask_other.sum()),
        "n_test": int(len(y_te)),
        "n_test_full": n_test_full,
        "pos_rate_test": float(y_te.mean()),
        "pi_min": float(min(y_te.mean(), 1 - y_te.mean())),
        "sigma_min": float(sigma_te.min()),
        "sigma_max": float(sigma_te.max()),
        "sigma_range_R": float(sigma_te.max() - sigma_te.min()),
        "L_F_lcls": lf,
        "n_pairs_lcls": n_pairs,
        "lcls_summary": lcls_summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", required=True, type=Path)
    ap.add_argument("--hccp-results", required=True, type=Path,
                    help="Path to outputs/proteingym_hccp_n50/per_assay_results.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-test-cap", type=int, default=3000,
                    help="Match scripts/38 default to keep observed gap comparable.")
    ap.add_argument("--n-assays", type=int, default=0,
                    help="Cap number of assays (0 = all). Smoke-test with --n-assays 5.")
    ap.add_argument("--lcls-n-bins", type=int, default=30)
    ap.add_argument("--lcls-min-per-bin", type=int, default=20)
    args = ap.parse_args()

    hccp_records = json.loads(args.hccp_results.read_text())
    target_assays = [r["target_assay"] for r in hccp_records
                     if "marginal_coverage" in r]
    if args.n_assays > 0:
        target_assays = target_assays[:args.n_assays]
    print(f"[plan] {len(target_assays)} target assays "
          f"(of {len(hccp_records)} in HCCP results)")

    df = pd.read_parquet(args.features_parquet)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    print(f"[load] features: {len(df):,} rows × {len(feat_cols)} cols, "
          f"{df.DMS_id.nunique()} assays")

    results: list[dict] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for assay in tqdm(target_assays, desc="L_F per-assay"):
        try:
            r = per_assay_extract(assay, df, feat_cols, args.seed, args.n_test_cap)
        except Exception as e:
            print(f"\n[error] {assay}: {e}")
            continue
        if r is None:
            print(f"\n[skip] {assay}: zero positives in train or test")
            continue

        h = next((h for h in hccp_records
                  if h["target_assay"] == assay), None)
        if h is not None:
            r["observed_marginal_coverage"] = h["marginal_coverage"]
            r["observed_sigma_bin_range"] = h["sigma_bin_range"]
            r["observed_AUPRC"] = h["AUPRC_test"]
            r["observed_singleton"] = h["frac_singleton"]
            cells = h.get("per_sigma_bin", [])
            if cells:
                r["observed_cell_worst_gap"] = float(max(c["gap"] for c in cells))
                r["observed_n_cells"] = len(cells)
            else:
                r["observed_cell_worst_gap"] = float("nan")
                r["observed_n_cells"] = 0

        results.append(r)
        # Periodic checkpoint after each assay.
        args.out.write_text(json.dumps(results, indent=2))

    print(f"\nsaved: {args.out}  ({len(results)} assays)")
    if results:
        lfs = [r["L_F_lcls"] for r in results
               if not np.isnan(r["L_F_lcls"])]
        if lfs:
            print(f"[L_F] min={min(lfs):.3f}  median={np.median(lfs):.3f}  "
                  f"max={max(lfs):.3f}  n_valid={len(lfs)}/{len(results)}")


if __name__ == "__main__":
    main()
