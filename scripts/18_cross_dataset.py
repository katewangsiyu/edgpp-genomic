"""Cross-dataset shift evaluation — Day 14 Plan B2.

Tests T4 (chrom-shift robustness) in a harder, more realistic regime: train on
one TraitGym subset, evaluate on the other. Mendelian → Complex is a shift in
biology (monogenic disease → common-trait polygenic signal); Complex →
Mendelian is the reverse.

For each direction (A → B):
  1. Train base classifier p̂ on A (ALL variants, no chrom hold-out).
  2. Train σ̂ head on |y - p̂| residuals of A (in-sample).
  3. Predict p̂, σ̂ on B.
  4. Conformal: calibrate Mondrian (y × σ̂-bin) on A, evaluate on B.
  5. Estimate d_TV(score_A, score_B) empirically for T4's bound.

Metrics:
  - AUPRC on B (using p̂ trained on A)
  - Marginal / class-cond coverage on B at target 1-α
  - Per-σ̂-bin coverage gap on B
  - Empirical d_TV between score distributions
  - Barber 2023 Thm 2 predicted lower bound vs. observed coverage

Usage:
    python scripts/18_cross_dataset.py \\
        --mendelian-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --complex-parquet data/raw/traitgym/complex_traits_matched_9/test.parquet \\
        --mendelian-features data/raw/traitgym/mendelian_traits_matched_9/features \\
        --complex-features data/raw/traitgym/complex_traits_matched_9/features \\
        --feature-set CADD+Borzoi \\
        --out-dir outputs/cross_dataset/CADD+Borzoi \\
        --alpha 0.10
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
mod_11 = __import__("11_aggregator_gbm")
FEATURE_SETS = mod_11.FEATURE_SETS
load_feature_matrix = mod_11.load_feature_matrix


def build_clf(seed=42):
    return Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("hgb", HistGradientBoostingClassifier(
            max_depth=2, max_iter=100, learning_rate=0.1,
            class_weight="balanced", random_state=seed, early_stopping=False)),
    ])


def build_reg(seed=42):
    return Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("hgb", HistGradientBoostingRegressor(
            loss="squared_error", max_depth=2, max_iter=100, learning_rate=0.1,
            random_state=seed, early_stopping=False)),
    ])


def load_dataset(parquet_path, features_dir, feat_names):
    V = pd.read_parquet(parquet_path).reset_index(drop=True)
    X = load_feature_matrix(Path(features_dir), feat_names).reset_index(drop=True)
    assert len(V) == len(X)
    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()
    return V, X, y, chroms


def empirical_tv(s_a: np.ndarray, s_b: np.ndarray, n_bins: int = 50,
                 edges: np.ndarray | None = None):
    """Empirical TV between two 1-D samples via binned histograms.

    d_TV = 0.5 * sum_i |p_a[i] - p_b[i]|
    """
    if edges is None:
        lo = min(s_a.min(), s_b.min())
        hi = max(s_a.max(), s_b.max())
        edges = np.linspace(lo, hi, n_bins + 1)
    h_a, _ = np.histogram(s_a, bins=edges, density=False)
    h_b, _ = np.histogram(s_b, bins=edges, density=False)
    p_a = h_a / max(h_a.sum(), 1)
    p_b = h_b / max(h_b.sum(), 1)
    return float(0.5 * np.abs(p_a - p_b).sum())


def mondrian_cross_conformal(p_a, sigma_a, y_a,
                              p_b, sigma_b, y_b,
                              alpha, edges_sigma, eps=1e-4, min_cell=5):
    """Calibrate on A (full), predict sets on B.

    σ̂-bin edges are determined from A's calibration scores and applied to B.
    """
    # Bin assignment using A's edges (test-time σ̂ on B).
    sigma_bin_a = np.digitize(sigma_a, edges_sigma[1:-1])
    sigma_bin_b = np.digitize(sigma_b, edges_sigma[1:-1])
    n_bins = len(edges_sigma) - 1

    def score_fn(p, sig, k):
        return np.abs(k - p) / (sig + eps)

    q = {}
    for k in (0, 1):
        for b in range(n_bins):
            cal_kb = (y_a == k) & (sigma_bin_a == b)
            n_kb = int(cal_kb.sum())
            if n_kb < min_cell:
                # Fall back to pooled class-cond
                cal_k = (y_a == k)
                n_k = int(cal_k.sum())
                if n_k == 0:
                    q[(k, b)] = np.inf
                    continue
                s_cal = score_fn(p_a[cal_k], sigma_a[cal_k], np.full(n_k, k))
                level = min(1.0, np.ceil((n_k + 1) * (1 - alpha)) / n_k)
                q[(k, b)] = float(np.quantile(s_cal, level, method="higher"))
            else:
                s_cal = score_fn(p_a[cal_kb], sigma_a[cal_kb], np.full(n_kb, k))
                level = min(1.0, np.ceil((n_kb + 1) * (1 - alpha)) / n_kb)
                q[(k, b)] = float(np.quantile(s_cal, level, method="higher"))

    # Predict sets for B
    n_b = len(y_b)
    pred_sets = [set() for _ in range(n_b)]
    for i in range(n_b):
        b = int(sigma_bin_b[i])
        for k in (0, 1):
            s_ik = score_fn(np.array([p_b[i]]), np.array([sigma_b[i]]), np.array([k]))[0]
            if s_ik <= q[(k, b)]:
                pred_sets[i].add(k)
    return pred_sets, q, sigma_bin_a, sigma_bin_b


def evaluate_cross(pred_sets, y_b, sigma_b, sigma_bin_b, n_bins):
    covered = np.array([y_b[i] in ps for i, ps in enumerate(pred_sets)])
    sizes = np.array([len(ps) for ps in pred_sets])

    bin_cov = []
    for b in range(n_bins):
        m = sigma_bin_b == b
        if m.sum() < 5:
            continue
        bin_cov.append({"bin": b, "n": int(m.sum()),
                        "sigma_mean": float(sigma_b[m].mean()),
                        "coverage": float(covered[m].mean())})
    bin_gap = (max(x["coverage"] for x in bin_cov)
               - min(x["coverage"] for x in bin_cov)) if bin_cov else float("nan")
    return {
        "marginal_coverage": float(covered.mean()),
        "coverage_pos": float(covered[y_b == 1].mean()) if (y_b == 1).any() else float("nan"),
        "coverage_neg": float(covered[y_b == 0].mean()) if (y_b == 0).any() else float("nan"),
        "frac_empty": float((sizes == 0).mean()),
        "frac_singleton": float((sizes == 1).mean()),
        "frac_both": float((sizes == 2).mean()),
        "sigma_bin_coverage": bin_cov,
        "sigma_bin_gap": float(bin_gap),
    }, covered


def run_one_direction(V_a, X_a, y_a, chroms_a, name_a,
                       V_b, X_b, y_b, chroms_b, name_b,
                       alpha, n_sigma_bins, seed):
    print(f"\n{'='*72}")
    print(f"  Direction: train on {name_a} (n={len(y_a)}, pos={int(y_a.sum())}) → "
          f"eval on {name_b} (n={len(y_b)}, pos={int(y_b.sum())})")
    print(f"{'='*72}")

    # Step 1: base classifier on A
    clf = build_clf(seed=seed)
    clf.fit(X_a.to_numpy(), y_a)
    p_a = clf.predict_proba(X_a.to_numpy())[:, 1]  # in-sample
    p_b = clf.predict_proba(X_b.to_numpy())[:, 1]  # OOD

    # Step 2: σ̂ on A's residuals
    r_a = np.abs(y_a - p_a)
    reg = build_reg(seed=seed)
    reg.fit(X_a.to_numpy(), r_a)
    sigma_a = np.clip(reg.predict(X_a.to_numpy()), 1e-4, None)
    sigma_b = np.clip(reg.predict(X_b.to_numpy()), 1e-4, None)

    # Step 3: metrics on B
    aup_b = float(average_precision_score(y_b, p_b))
    auroc_b = float(roc_auc_score(y_b, p_b))
    # Weighted-per-chrom AUPRC on B (match TraitGym metric)
    aup_per_chrom_b, nchr_b = mod_11.weighted_per_chrom_auprc(y_b, p_b, chroms_b)
    # Within-A sanity
    aup_a = float(average_precision_score(y_a, p_a))

    print(f"[p̂ train on {name_a}] in-sample AUPRC={aup_a:.4f}")
    print(f"[p̂ eval on {name_b}]  AUPRC={aup_b:.4f}  AUROC={auroc_b:.4f}  "
          f"AUPRC_per_chrom={aup_per_chrom_b:.4f}")

    # Step 4: TV distance between A and B score distributions (for T4 bound)
    tv_score = empirical_tv(p_a, p_b, n_bins=50)
    tv_sigma = empirical_tv(sigma_a, sigma_b, n_bins=50)
    print(f"  d_TV(p̂_A, p̂_B) = {tv_score:.4f}")
    print(f"  d_TV(σ̂_A, σ̂_B) = {tv_sigma:.4f}")

    # Step 5: conformal Mondrian — calibrate on A, evaluate on B
    edges_sigma = np.quantile(sigma_a, np.linspace(0, 1, n_sigma_bins + 1))
    edges_sigma[0] -= 1e-9
    edges_sigma[-1] += 1e-9

    pred_sets_b, q, sigma_bin_a, sigma_bin_b = mondrian_cross_conformal(
        p_a, sigma_a, y_a, p_b, sigma_b, y_b, alpha, edges_sigma,
    )
    cross_eval, _ = evaluate_cross(pred_sets_b, y_b, sigma_b, sigma_bin_b, n_sigma_bins)

    print(f"\n[Conformal A→B] target {1-alpha:.2f}, observed:")
    print(f"  marginal cov = {cross_eval['marginal_coverage']:.4f}")
    print(f"  Cov|pos = {cross_eval['coverage_pos']:.4f}  Cov|neg = {cross_eval['coverage_neg']:.4f}")
    print(f"  sizes: empty={cross_eval['frac_empty']:.3f}  "
          f"single={cross_eval['frac_singleton']:.3f}  both={cross_eval['frac_both']:.3f}")
    print(f"  σ̂-bin gap = {cross_eval['sigma_bin_gap']:.3f}")
    for row in cross_eval["sigma_bin_coverage"]:
        print(f"    bin {row['bin']}: n={row['n']:>5}  σ̂={row['sigma_mean']:.3f}  "
              f"cov={row['coverage']:.3f}")

    # Sanity vs Barber 2023 Thm 2 bound: observed cov ≥ (1-α) - d_TV ?
    predicted_lower = (1 - alpha) - tv_score
    gap_vs_bound = cross_eval["marginal_coverage"] - predicted_lower
    print(f"\n  Barber 2023 Thm 2 lower bound: (1-α) - d_TV(p̂_A,p̂_B) = {predicted_lower:.4f}")
    print(f"  Observed - bound = {gap_vs_bound:+.4f}  "
          f"({'above ✓' if gap_vs_bound >= -0.01 else 'below ✗'})")

    return {
        "direction": f"{name_a} → {name_b}",
        "n_train": int(len(y_a)),
        "n_test": int(len(y_b)),
        "AUPRC_in_sample_train": aup_a,
        "AUPRC_OOD_test": aup_b,
        "AUROC_OOD_test": auroc_b,
        "AUPRC_per_chrom_OOD_test": aup_per_chrom_b,
        "tv_score": tv_score,
        "tv_sigma": tv_sigma,
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "barber2023_lower_bound": predicted_lower,
        "observed_above_bound": bool(gap_vs_bound >= -0.01),
        **cross_eval,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mendelian-parquet", required=True)
    ap.add_argument("--complex-parquet", required=True)
    ap.add_argument("--mendelian-features", required=True)
    ap.add_argument("--complex-features", required=True)
    ap.add_argument("--feature-set", required=True, choices=list(FEATURE_SETS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-sigma-bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    feat_names = FEATURE_SETS[args.feature_set]

    # Load both
    V_m, X_m, y_m, chroms_m = load_dataset(
        args.mendelian_parquet, args.mendelian_features, feat_names)
    V_c, X_c, y_c, chroms_c = load_dataset(
        args.complex_parquet, args.complex_features, feat_names)

    print(f"[load] feature_set={args.feature_set}")
    print(f"  Mendelian: n={len(V_m)} pos={int(y_m.sum())} ({y_m.mean():.1%})")
    print(f"  Complex  : n={len(V_c)} pos={int(y_c.sum())} ({y_c.mean():.1%})")

    # Align feature columns across datasets (should be same due to FEATURE_SETS)
    common_cols = sorted(set(X_m.columns) & set(X_c.columns))
    missing_m = set(X_c.columns) - set(X_m.columns)
    missing_c = set(X_m.columns) - set(X_c.columns)
    if missing_m or missing_c:
        print(f"  [warn] mendelian missing: {len(missing_m)}  complex missing: {len(missing_c)}")
    X_m = X_m[common_cols].reset_index(drop=True)
    X_c = X_c[common_cols].reset_index(drop=True)
    print(f"  common feature dim: {X_m.shape[1]}")

    results = {
        "feature_set": args.feature_set,
        "common_feature_dim": int(X_m.shape[1]),
        "n_mendelian": int(len(V_m)),
        "n_complex": int(len(V_c)),
        "alpha": args.alpha,
    }

    r_mc = run_one_direction(
        V_m, X_m, y_m, chroms_m, "Mendelian",
        V_c, X_c, y_c, chroms_c, "Complex",
        args.alpha, args.n_sigma_bins, args.seed,
    )
    results["mendelian_to_complex"] = r_mc

    r_cm = run_one_direction(
        V_c, X_c, y_c, chroms_c, "Complex",
        V_m, X_m, y_m, chroms_m, "Mendelian",
        args.alpha, args.n_sigma_bins, args.seed,
    )
    results["complex_to_mendelian"] = r_cm

    (out / "cross_dataset_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out}/cross_dataset_results.json")


if __name__ == "__main__":
    main()
