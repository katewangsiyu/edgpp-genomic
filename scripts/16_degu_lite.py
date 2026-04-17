"""DEGU-lite — M seed-ensemble of GBMs (feature-level adaptation).

Faithful translation of Zhou et al. (2026, npj AI) "Deep Ensemble with Gaussian
Uncertainty" idea to our feature-level pipeline:

    1. Train M GBMs with different random seeds + bootstrap resampling.
    2. For each test variant x:
       - p_mean = mean_{i=1..M} p_i(x)   → the ranking score
       - p_std  = std_{i=1..M} p_i(x)    → ensemble-disagreement uncertainty
    3. Output scores_ensemble.parquet with [chrom, label, p_mean, p_std,
       p_logvar, p_0, ..., p_{M-1}].

Why bootstrap (not just seed): HistGradientBoostingClassifier with default
hyperparameters is deterministic — `random_state` only affects feature
subsampling (which is off by default). Bootstrap resampling of the training
set is the canonical way to inject ensemble diversity without changing the
inductive bias.

Contract: scores_ensemble.parquet is drop-in compatible with
scripts/14_conformal_hetero.py (same [chrom, label, p_hat, sigma] schema,
where `sigma = p_std`). This lets us plug DEGU-lite σ̂ into the same
Mondrian(y×σ̂-bin) conformal calibrator and compare σ̂-bin local-coverage gap
against our supervised σ̂ (from scripts/13_hetero_head.py).

Usage:
    python scripts/16_degu_lite.py \\
        --feature-set CADD+Borzoi \\
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --features-dir data/raw/traitgym/mendelian_traits_matched_9/features \\
        --out-dir outputs/degu_lite/CADD+Borzoi_mendelian \\
        --n-ensemble 10
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))
# Reuse feature-set dict and loader from aggregator
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "agg11", Path(__file__).parent / "11_aggregator_gbm.py"
)
_agg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_agg)
FEATURE_SETS = _agg.FEATURE_SETS
load_feature_matrix = _agg.load_feature_matrix
weighted_per_chrom_auprc = _agg.weighted_per_chrom_auprc
coverage_auprc = _agg.coverage_auprc


def build_pipeline(seed=42, max_depth=2, max_iter=100, learning_rate=0.1):
    return Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("hgb", HistGradientBoostingClassifier(
            max_depth=max_depth, max_iter=max_iter,
            learning_rate=learning_rate,
            class_weight="balanced",
            random_state=seed,
            early_stopping=False,
        )),
    ])


def chrom_loo_ensemble(X, y, chroms, n_ensemble, base_seed=42,
                       max_depth=2, max_iter=100, learning_rate=0.1,
                       bootstrap=True):
    """For each chrom c, train M GBMs on (other chroms, bootstrap-resampled)
    and predict the held-out chrom. Returns a (n_variants, M) matrix of p_i(x)."""
    n = len(y)
    uniq_chroms = sorted(set(chroms))
    preds = np.full((n, n_ensemble), np.nan, dtype=np.float32)

    rng_seeds = np.random.RandomState(base_seed).randint(1, 1_000_000, size=n_ensemble)

    for c in tqdm(uniq_chroms, desc="chrom-LOO"):
        mask_test = chroms == c
        mask_train = ~mask_test
        X_tr_full = X.loc[mask_train].to_numpy()
        y_tr_full = y[mask_train]
        X_te = X.loc[mask_test].to_numpy()
        n_tr = len(y_tr_full)

        for m in range(n_ensemble):
            seed = int(rng_seeds[m])
            pipe = build_pipeline(
                seed=seed, max_depth=max_depth,
                max_iter=max_iter, learning_rate=learning_rate,
            )
            if bootstrap:
                # Standard bootstrap resample with replacement
                rng = np.random.RandomState(seed)
                idx = rng.randint(0, n_tr, size=n_tr)
                X_fit = X_tr_full[idx]
                y_fit = y_tr_full[idx]
                # Guard: ensure at least one positive/negative in bootstrap
                if y_fit.sum() == 0 or y_fit.sum() == n_tr:
                    X_fit = X_tr_full
                    y_fit = y_tr_full
            else:
                X_fit = X_tr_full
                y_fit = y_tr_full
            pipe.fit(X_fit, y_fit)
            preds[mask_test, m] = pipe.predict_proba(X_te)[:, 1]
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--feature-set", required=True, choices=list(FEATURE_SETS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-ensemble", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="Disable bootstrap (ensemble collapses to identical unless "
                         "max_features<1 — mostly for debug).")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    feat_names = FEATURE_SETS[args.feature_set]

    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    X = load_feature_matrix(Path(args.features_dir), feat_names).reset_index(drop=True)
    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()
    print(f"[load] feature_set={args.feature_set} n={len(V)} dim={X.shape[1]} "
          f"chroms={len(sorted(set(chroms)))} pos={int(y.sum())} M={args.n_ensemble}")

    preds = chrom_loo_ensemble(
        X, y, chroms, n_ensemble=args.n_ensemble,
        base_seed=args.base_seed, max_depth=args.max_depth,
        max_iter=args.max_iter, learning_rate=args.learning_rate,
        bootstrap=not args.no_bootstrap,
    )

    p_mean = preds.mean(axis=1)
    p_std = preds.std(axis=1, ddof=0)
    p_var = preds.var(axis=1, ddof=0)
    p_logvar = np.log(p_var + 1e-12)

    # ---- metrics ----
    aupr = float(average_precision_score(y, p_mean))
    auroc = float(roc_auc_score(y, p_mean))
    aupr_chr, nc = weighted_per_chrom_auprc(y, p_mean, chroms)
    cov = coverage_auprc(y, p_mean)

    # Per-seed AUPRC (sanity: if all identical, bootstrap isn't injecting diversity)
    per_seed_aupr = [float(average_precision_score(y, preds[:, m])) for m in range(args.n_ensemble)]
    # Disagreement summary
    mean_std = float(p_std.mean())
    mean_absdiff_from_mean = float(np.mean(np.abs(preds - p_mean[:, None])))

    metrics = {
        "model": "DEGU-lite (HGB M-bootstrap ensemble)",
        "feature_set": args.feature_set,
        "n": len(V), "n_chroms": nc,
        "n_ensemble": args.n_ensemble,
        "base_seed": args.base_seed,
        "bootstrap": not args.no_bootstrap,
        "AUPRC_ensemble_mean": aupr,
        "AUROC_ensemble_mean": auroc,
        "AUPRC_per_chrom": aupr_chr,
        "coverage_AUPRC": cov,
        "per_seed_AUPRC": per_seed_aupr,
        "p_std_mean": mean_std,
        "mean_abs_diff_from_mean": mean_absdiff_from_mean,
    }

    print(f"\n=== DEGU-lite {args.feature_set} (M={args.n_ensemble}, bootstrap={not args.no_bootstrap}) ===")
    print(f"Ensemble-mean  AUPRC={aupr:.4f}  AUROC={auroc:.4f}  AUPRC_per_chrom={aupr_chr:.4f}")
    print(f"Per-seed AUPRC: min={min(per_seed_aupr):.4f}  "
          f"max={max(per_seed_aupr):.4f}  std={float(np.std(per_seed_aupr)):.4f}")
    print(f"Ensemble disagreement: mean(p_std)={mean_std:.4f}  "
          f"mean|p_i - p_mean|={mean_absdiff_from_mean:.4f}")
    print(f"coverage-AUPRC: " + "  ".join(f"@{c:.2f}={v:.3f}" for c, v in cov.items()))

    # ---- save outputs ----
    # 1. Aggregator-compatible scores.parquet (for downstream re-use with 13_hetero_head.py)
    V_base = V[["chrom"]].copy()
    V_base["label"] = y
    V_base["score"] = p_mean
    V_base.to_parquet(out / "scores.parquet", index=False)

    # 2. Ensemble output: drop-in for 14_conformal_hetero.py (sigma = p_std)
    V_ens = V[["chrom"]].copy()
    V_ens["label"] = y
    V_ens["p_hat"] = p_mean
    V_ens["sigma"] = p_std  # DEGU-lite's σ̂_ensemble
    V_ens["p_std"] = p_std
    V_ens["p_logvar"] = p_logvar
    for m in range(args.n_ensemble):
        V_ens[f"p_{m}"] = preds[:, m]
    V_ens.to_parquet(out / "scores_ensemble.parquet", index=False)
    # Also save `scores_with_sigma.parquet` so that 14_conformal_hetero.py can consume directly
    V_sig = V_ens[["chrom", "label", "p_hat", "sigma"]].copy()
    V_sig.to_parquet(out / "scores_with_sigma.parquet", index=False)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"saved: {out}/scores.parquet, scores_ensemble.parquet, "
          f"scores_with_sigma.parquet, metrics.json")


if __name__ == "__main__":
    main()
