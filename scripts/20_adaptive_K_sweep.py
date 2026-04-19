"""T5 Adaptive K validation — K-sweep + L_F estimation + theory comparison.

Runs the Mondrian-by-(y × σ̂-bin) conformal pipeline for multiple K values,
estimates the score-Lipschitz constant L_F, and compares the empirical
coverage-gap curve to the T5 theoretical prediction.

Usage:
    conda run -n edgpp_t4 python scripts/20_adaptive_K_sweep.py \
        --sigma-scores outputs/hetero_head/CADD+GPN-MSA+Borzoi_mendelian_abs/scores_with_sigma.parquet \
        --out-dir outputs/adaptive_K/CADD+GPN-MSA+Borzoi_mendelian
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


EPS = 1e-4


def hetero_score(p, sigma, y_cand):
    return np.abs(y_cand - p) / (sigma + EPS)


def run_mondrian_conformal(p, sigma, y, chroms, alpha, K):
    """Run Mondrian-by-(y × σ̂-bin) chrom-LOO conformal for a given K.

    Returns per-variant coverage indicator and per-cell metadata.
    """
    n = len(y)
    edges = np.quantile(sigma, np.linspace(0, 1, K + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    sigma_bin = np.digitize(sigma, edges[1:-1])

    covered = np.zeros(n, dtype=bool)
    cell_counts = {}  # (k, b) -> n_kb across all folds

    for c in sorted(set(chroms)):
        mask_test = chroms == c
        mask_cal = ~mask_test
        q = {}
        for k in (0, 1):
            for b in range(K):
                cal_kb = mask_cal & (y == k) & (sigma_bin == b)
                n_kb = int(cal_kb.sum())
                cell_counts[(k, b)] = cell_counts.get((k, b), [])
                cell_counts[(k, b)].append(n_kb)
                if n_kb < max(5, int(np.ceil(1 / alpha))):
                    # Fall back to class-pooled
                    cal_k = mask_cal & (y == k)
                    n_k = int(cal_k.sum())
                    if n_k == 0:
                        q[(k, b)] = np.inf
                        continue
                    s_cal = hetero_score(p[cal_k], sigma[cal_k], np.full(n_k, k))
                    level = min(1.0, np.ceil((n_k + 1) * (1 - alpha)) / n_k)
                    q[(k, b)] = float(np.quantile(s_cal, level, method="higher"))
                else:
                    s_cal = hetero_score(p[cal_kb], sigma[cal_kb], np.full(n_kb, k))
                    level = min(1.0, np.ceil((n_kb + 1) * (1 - alpha)) / n_kb)
                    q[(k, b)] = float(np.quantile(s_cal, level, method="higher"))

        for i in np.where(mask_test)[0]:
            b_i = int(sigma_bin[i])
            y_i = int(y[i])
            s_true = hetero_score(p[i:i + 1], sigma[i:i + 1], np.array([y_i]))[0]
            if s_true <= q[(y_i, b_i)]:
                covered[i] = True

    # Per-cell coverage
    cell_coverage = {}
    for k in (0, 1):
        for b in range(K):
            mask = (y == k) & (sigma_bin == b)
            n_cell = int(mask.sum())
            if n_cell >= 5:
                cell_coverage[(k, b)] = {
                    "n": n_cell,
                    "coverage": float(covered[mask].mean()),
                    "gap": abs(float(covered[mask].mean()) - (1 - alpha)),
                    "n_cal_mean": float(np.mean(cell_counts.get((k, b), [0]))),
                }

    worst_gap = max(c["gap"] for c in cell_coverage.values()) if cell_coverage else 0.0
    mean_gap = float(np.mean([c["gap"] for c in cell_coverage.values()])) if cell_coverage else 0.0
    n_min = min(c["n"] for c in cell_coverage.values()) if cell_coverage else 0

    return {
        "K": K,
        "marginal_coverage": float(covered.mean()),
        "worst_cell_gap": worst_gap,
        "mean_cell_gap": mean_gap,
        "n_min_cell": n_min,
        "n_cells_valid": len(cell_coverage),
        "n_cells_fallback": 2 * K - len(cell_coverage),
        "cell_details": {f"{k}_{b}": v for (k, b), v in cell_coverage.items()},
    }


def estimate_L_F(p, sigma, y, K_pilot=10):
    """Estimate score-Lipschitz constant L_F from adjacent-bin KS distances.

    Uses a pilot partition with K_pilot bins, computes KS distance between
    score distributions in adjacent bins for each class.
    """
    edges = np.quantile(sigma, np.linspace(0, 1, K_pilot + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    sigma_bin = np.digitize(sigma, edges[1:-1])

    L_estimates = []
    for k in (0, 1):
        for b in range(K_pilot - 1):
            mask_b = (y == k) & (sigma_bin == b)
            mask_b1 = (y == k) & (sigma_bin == b + 1)
            n_b, n_b1 = int(mask_b.sum()), int(mask_b1.sum())
            if n_b < 10 or n_b1 < 10:
                continue
            s_b = hetero_score(p[mask_b], sigma[mask_b], np.full(n_b, k))
            s_b1 = hetero_score(p[mask_b1], sigma[mask_b1], np.full(n_b1, k))
            ks_stat, _ = ks_2samp(s_b, s_b1)
            sigma_gap = abs(sigma[mask_b1].mean() - sigma[mask_b].mean())
            if sigma_gap > 1e-8:
                L_estimates.append(ks_stat / sigma_gap)

    if not L_estimates:
        return 1.0  # fallback
    return float(np.max(L_estimates))


def compute_theoretical_curve(L_F, R, pi_min, n, K_values):
    """Compute the T5 theoretical upper bound for each K."""
    theory = []
    for K in K_values:
        bias = L_F * R / K
        variance = K / (pi_min * n)
        theory.append({
            "K": K,
            "bias": bias,
            "variance": variance,
            "total_bound": bias + variance,
        })
    return theory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma-scores", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--K-values", type=str, default="2,3,5,8,10,15,20,30")
    parser.add_argument("--K-pilot", type=int, default=10,
                        help="Pilot K for L_F estimation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    K_values = [int(k) for k in args.K_values.split(",")]

    # Load data
    df = pd.read_parquet(args.sigma_scores)
    p = df["p_hat"].values
    sigma = df["sigma"].values
    y = df["label"].values
    chroms = df["chrom"].values
    n = len(y)
    pi_min = min((y == 0).mean(), (y == 1).mean())
    R = sigma.max() - sigma.min()

    print(f"n={n}, pi_min={pi_min:.4f}, R={R:.4f}, alpha={args.alpha}")

    # Step 1: Estimate L_F
    L_F = estimate_L_F(p, sigma, y, K_pilot=args.K_pilot)
    print(f"Estimated L_F = {L_F:.4f}")

    # Step 2: Compute oracle K*
    K_star = max(2, int(np.floor(np.sqrt(L_F * R * pi_min * n))))
    G_star = 2 * np.sqrt(L_F * R / (pi_min * n))
    print(f"Oracle K* = {K_star}, G(K*) = {G_star:.6f}")

    # Step 3: K-sweep — run Mondrian conformal for each K
    sweep_results = []
    for K in K_values:
        print(f"  Running K={K}...", end=" ", flush=True)
        res = run_mondrian_conformal(p, sigma, y, chroms, args.alpha, K)
        sweep_results.append(res)
        print(f"worst_gap={res['worst_cell_gap']:.4f}, "
              f"mean_gap={res['mean_cell_gap']:.4f}, "
              f"n_min={res['n_min_cell']}, "
              f"coverage={res['marginal_coverage']:.4f}")

    # Step 4: Theoretical curve
    theory = compute_theoretical_curve(L_F, R, pi_min, n, K_values)

    # Step 5: CV-based K selection
    cv_gaps = {r["K"]: r["worst_cell_gap"] for r in sweep_results}
    K_cv = min(cv_gaps, key=cv_gaps.get)

    # Step 6: Plug-in K selection
    K_plugin = max(2, min(
        int(np.floor(np.sqrt(L_F * R * pi_min * n))),
        int(np.floor(pi_min * n / max(10, int(np.ceil(1 / args.alpha)))))
    ))

    summary = {
        "n": n,
        "pi_min": pi_min,
        "R": R,
        "alpha": args.alpha,
        "L_F_estimated": L_F,
        "K_star_oracle": K_star,
        "G_star_oracle": G_star,
        "K_cv": K_cv,
        "K_plugin": K_plugin,
        "K_current_fixed": 5,
        "gap_at_K5": cv_gaps.get(5, None),
        "gap_at_K_cv": cv_gaps.get(K_cv, None),
        "gap_improvement_vs_K5": (
            (cv_gaps.get(5, 0) - cv_gaps.get(K_cv, 0)) / cv_gaps.get(5, 1)
            if 5 in cv_gaps and cv_gaps.get(5, 0) > 0 else None
        ),
        "sweep": sweep_results,
        "theory": theory,
    }

    out_file = out_dir / "adaptive_K_results.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Print summary table
    print(f"\n{'K':>4} | {'Worst Gap':>10} | {'Mean Gap':>9} | {'Theory':>8} | {'n_min':>6}")
    print("-" * 50)
    for res, th in zip(sweep_results, theory):
        print(f"{res['K']:>4} | {res['worst_cell_gap']:>10.4f} | "
              f"{res['mean_cell_gap']:>9.4f} | {th['total_bound']:>8.4f} | "
              f"{res['n_min_cell']:>6}")
    print(f"\nOracle K* = {K_star} (theory), K_CV = {K_cv} (empirical), "
          f"K_plugin = {K_plugin}")


if __name__ == "__main__":
    main()
