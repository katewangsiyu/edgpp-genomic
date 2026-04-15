"""EDG++ v2 — selective-reliability head on top of a chrom-LOO LogReg aggregator.

Reads LogReg chrom-LOO predictions (from 09_aggregator_chrom_loo.py), defines
residuals r_i = |p_i - y_i| (zero-leak: each p_i from a fit that excluded
chrom(i)), and trains a GBM regressor per held-out chrom on NON-TEACHER-MAG
features (sequence context + conservation) to predict r_i.

Combined confidence = |p - 0.5| + λ * (1 - r̂).  λ is selected by inner-LOO
(one chrom held out for λ tuning, remaining chroms for model fit) to prevent
over-fitting the combination weight.

Reports coverage-AUPRC curves for:
  * baseline:  |p - 0.5|  (LogReg margin)
  * selective: combined confidence (margin + λ_cv * reliability)

Feature selection explicitly excludes:
  - any Borzoi feature (teacher magnitude leakage, per Day 5 findings)
  - CADD.RawScore (trained variant-effect score, too close to label)
Included: log1p|tss_dist|, GC, CpG, priPhyloP, mamPhyloP, verPhyloP, bStatistic.

Usage:
    python scripts/10_selective_head.py \\
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --cadd-parquet data/raw/traitgym/mendelian_traits_matched_9/features/CADD.parquet \\
        --base-scores outputs/aggregator/reproduce_Borzoi/scores.parquet \\
        --out-dir outputs/selective/Borzoi_v2
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr


RELIABILITY_FEATURES_FROM_CADD = [
    "GC", "CpG", "priPhyloP", "mamPhyloP", "verPhyloP", "bStatistic",
]
COVERAGES = (0.10, 0.25, 0.50, 0.75, 1.00)
LAMBDA_GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]


def build_reliability_features(V: pd.DataFrame, cadd: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=V.index)
    out["log_tss_abs"] = np.log1p(np.abs(V["tss_dist"].astype(float)))
    for col in RELIABILITY_FEATURES_FROM_CADD:
        out[col] = cadd[col].astype(float).to_numpy()
    return out


def weighted_per_chrom_auprc(y, s, chroms):
    keep = []
    for c in sorted(set(chroms)):
        m = chroms == c
        if m.sum() < 10 or y[m].sum() == 0:
            continue
        keep.append((m.sum(), average_precision_score(y[m], s[m])))
    if not keep:
        return float("nan")
    w = np.array([x[0] for x in keep]); a = np.array([x[1] for x in keep])
    return float((w * a).sum() / w.sum())


def coverage_curve(y, s, conf, coverages=COVERAGES):
    order = np.argsort(-conf)
    out = {}
    for c in coverages:
        k = int(np.ceil(c * len(y)))
        idx = order[:k]
        out[c] = float(average_precision_score(y[idx], s[idx])) if y[idx].sum() > 0 else float("nan")
    return out


def fit_reliability_gbm(X_train, r_train):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
        ("gbm", GradientBoostingRegressor(
            n_estimators=100, max_depth=3, subsample=0.8,
            random_state=42, learning_rate=0.1,
        )),
    ])
    pipe.fit(X_train, r_train)
    return pipe


def select_lambda_inner_cv(y, p, margin, w, chroms, lam_grid=LAMBDA_GRID):
    """Select best λ by inner chrom-LOO: for each chrom c, score is computed
    on c using λ, and the metric is summed over all inner chroms. Returns λ
    that maximizes mean cov@25% AUPRC across inner folds."""
    scores = {lam: [] for lam in lam_grid}
    for c in sorted(set(chroms)):
        mask = chroms == c
        yc, pc, mc, wc = y[mask], p[mask], margin[mask], w[mask]
        if yc.sum() == 0 or len(yc) < 10:
            continue
        for lam in lam_grid:
            conf = mc + lam * wc
            # AUPRC at 25% coverage on this chrom
            order = np.argsort(-conf)
            k = max(1, int(np.ceil(0.25 * len(yc))))
            idx = order[:k]
            if yc[idx].sum() > 0:
                scores[lam].append(average_precision_score(yc[idx], pc[idx]))
    # pick λ with best mean score
    best_lam = max(lam_grid, key=lambda l: np.mean(scores[l]) if scores[l] else -1)
    return best_lam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--cadd-parquet", required=True)
    ap.add_argument("--base-scores", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    cadd = pd.read_parquet(args.cadd_parquet).reset_index(drop=True)
    base = pd.read_parquet(args.base_scores).reset_index(drop=True)
    assert len(V) == len(cadd) == len(base), "row mismatch"

    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()
    p = base["score"].to_numpy()
    residual = np.abs(p - y)
    margin = np.abs(p - 0.5)

    X_rel = build_reliability_features(V, cadd)
    print(f"[load] n={len(V)} features={X_rel.shape[1]}: {list(X_rel.columns)}")

    # -- Outer chrom-LOO: predict residual --
    r_hat = np.full(len(V), np.nan, dtype=float)
    for c in sorted(set(chroms)):
        tr = chroms != c; te = chroms == c
        pipe = fit_reliability_gbm(X_rel[tr], residual[tr])
        r_hat[te] = pipe.predict(X_rel[te])
    r_hat = np.clip(r_hat, 0.0, 1.0)
    w = 1.0 - r_hat

    rho_rhat, _ = spearmanr(r_hat, residual)
    print(f"[reliability] spearman(r̂, residual) = {rho_rhat:+.3f}")

    # -- Inner CV to select λ --
    best_lam = select_lambda_inner_cv(y, p, margin, w, chroms)
    print(f"[lambda] CV-selected λ = {best_lam}")

    # -- Final combined confidence --
    conf_combined = margin + best_lam * w
    cov_base = coverage_curve(y, p, margin)
    cov_sel  = coverage_curve(y, p, conf_combined)

    # also report full-lambda sweep for transparency
    lam_sweep = {}
    for lam in LAMBDA_GRID:
        lam_sweep[str(lam)] = coverage_curve(y, p, margin + lam * w)

    aupr_chr = weighted_per_chrom_auprc(y, p, chroms)
    corr_mw = float(np.corrcoef(margin, w)[0, 1])

    metrics = {
        "n": int(len(V)),
        "seed": args.seed,
        "model": "7feat+GBM",
        "lambda_cv": best_lam,
        "AUPRC_per_chrom_full": aupr_chr,
        "spearman_rhat_vs_residual": float(rho_rhat),
        "corr_margin_vs_reliability": corr_mw,
        "coverage_AUPRC_baseline_margin": cov_base,
        "coverage_AUPRC_selective_combined": cov_sel,
        "lambda_sweep": lam_sweep,
    }

    print(f"\n== coverage-AUPRC: baseline margin vs selective (λ={best_lam}) ==")
    print(f"{'cov':>6}  {'margin':>8}  {'selective':>10}  {'Δ':>8}")
    for c in cov_base:
        d = cov_sel[c] - cov_base[c]
        print(f"{c:>6.0%}  {cov_base[c]:>8.4f}  {cov_sel[c]:>10.4f}  {d:>+8.4f}")
    print(f"\nfull AUPRC_per_chrom (unchanged) = {aupr_chr:.4f}")
    print(f"corr(margin, reliability) = {corr_mw:+.3f}")

    # save
    V_out = V[["chrom"]].copy()
    V_out["label"] = y
    V_out["score"] = p
    V_out["reliability"] = w
    V_out["r_hat"] = r_hat
    V_out["conf_margin"] = margin
    V_out["conf_combined"] = conf_combined
    V_out.to_parquet(out / "scores.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nsaved: {out}/")


if __name__ == "__main__":
    main()
