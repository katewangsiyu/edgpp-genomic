"""Robust L_F estimators for T5 oracle K derivation.

The legacy estimator (scripts/20_adaptive_K_sweep.py:105) takes the MAX of
adjacent-bin KS/σ-gap ratios across 10 pilot bins × 2 classes, which inflates
L_F by ~100× on TraitGym because the KS statistic saturates at 1 while σ-gaps
can be arbitrarily small. We compare three alternatives and report the
median-based estimator as the T5 plug-in.

Outputs:
    outputs/L_F_audit/{feature_set}_{dataset}_L_F.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def hetero_score(p: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    eps = 1e-6
    return np.abs(y.astype(float) - p) / (sigma + eps)


def lf_max_adjacent(p, sigma, y, K_pilot=10, min_n=10):
    """Legacy estimator: max over adjacent-bin pairs (scripts/20)."""
    return _lf_adjacent(p, sigma, y, K_pilot, min_n, reducer=np.max)


def lf_median_adjacent(p, sigma, y, K_pilot=10, min_n=10):
    """Median over adjacent-bin pairs. More robust to KS saturation."""
    return _lf_adjacent(p, sigma, y, K_pilot, min_n, reducer=np.median)


def lf_p90_adjacent(p, sigma, y, K_pilot=10, min_n=10):
    """90th percentile — tradeoff between max (noisy) and median (possibly biased low)."""
    return _lf_adjacent(p, sigma, y, K_pilot, min_n,
                        reducer=lambda a: float(np.quantile(a, 0.90)))


def _lf_adjacent(p, sigma, y, K_pilot, min_n, reducer):
    edges = np.quantile(sigma, np.linspace(0, 1, K_pilot + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    sigma_bin = np.digitize(sigma, edges[1:-1])
    estimates: list[float] = []
    for k in (0, 1):
        for b in range(K_pilot - 1):
            m_b = (y == k) & (sigma_bin == b)
            m_b1 = (y == k) & (sigma_bin == b + 1)
            n_b, n_b1 = int(m_b.sum()), int(m_b1.sum())
            if n_b < min_n or n_b1 < min_n:
                continue
            s_b = hetero_score(p[m_b], sigma[m_b], np.full(n_b, k))
            s_b1 = hetero_score(p[m_b1], sigma[m_b1], np.full(n_b1, k))
            ks, _ = ks_2samp(s_b, s_b1)
            sigma_gap = abs(sigma[m_b1].mean() - sigma[m_b].mean())
            if sigma_gap > 1e-8:
                estimates.append(float(ks / sigma_gap))
    if not estimates:
        return float("nan")
    return float(reducer(estimates))


def lf_from_empirical_inversion(sweep: list[dict], R: float, pi_min: float, n: int,
                                 alpha: float = 0.10) -> dict:
    """Invert G(K) = L_F R / K + K/(π_min n) at each K, solve for L_F.

    Uses empirical worst-cell gap as G(K). Requires K/(π_min n) < G(K).
    """
    implied = {}
    for entry in sweep:
        K = entry["K"]
        G_emp = entry["worst_cell_gap"]
        var_term = K / (pi_min * n)
        bias_term = max(G_emp - var_term, 0.0)
        implied[K] = {
            "G_emp": G_emp,
            "var_term": var_term,
            "bias_term": bias_term,
            "implied_L_F": bias_term * K / R if R > 0 else float("nan"),
        }
    vals = [v["implied_L_F"] for v in implied.values()
            if not np.isnan(v["implied_L_F"]) and v["implied_L_F"] > 0]
    return {
        "per_K": implied,
        "median_implied_L_F": float(np.median(vals)) if vals else float("nan"),
        "min_implied_L_F": float(np.min(vals)) if vals else float("nan"),
        "max_implied_L_F": float(np.max(vals)) if vals else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-parquet", required=True,
                    help="conformal_hetero_scores.parquet (has p_hat, sigma, label)")
    ap.add_argument("--adaptive-K-results", required=True,
                    help="outputs/adaptive_K/*/adaptive_K_results.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.scores_parquet)
    p = df["p_hat"].astype(float).to_numpy()
    sigma = df["sigma"].astype(float).to_numpy()
    y = df["label"].astype(int).to_numpy()

    K_results = json.load(open(args.adaptive_K_results))
    R, pi_min, n = K_results["R"], K_results["pi_min"], K_results["n"]

    estimators = {
        "legacy_max":   lf_max_adjacent(p, sigma, y),
        "median":       lf_median_adjacent(p, sigma, y),
        "p90":          lf_p90_adjacent(p, sigma, y),
    }
    inversion = lf_from_empirical_inversion(K_results["sweep"], R, pi_min, n)

    # Recompute oracle K for each estimator and report
    def oracle(L_F):
        K_star = int(np.floor(np.sqrt(max(L_F, 1e-6) * R * pi_min * n)))
        G_star = 2 * np.sqrt(max(L_F, 1e-6) * R / (pi_min * n))
        return {"L_F": L_F, "K_star": K_star, "G_star_bound": G_star}

    summary = {
        "R": R, "pi_min": pi_min, "n": n,
        "estimators": {name: oracle(v) for name, v in estimators.items()},
        "empirical_inversion": {
            "median_implied_L_F": inversion["median_implied_L_F"],
            "oracle_from_median": oracle(inversion["median_implied_L_F"]),
            "range_implied_L_F": [inversion["min_implied_L_F"], inversion["max_implied_L_F"]],
            "per_K": inversion["per_K"],
        },
        "K_cv": K_results.get("K_cv"),
    }

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"=== L_F estimator audit ===")
    print(f"  {'Estimator':<25s} {'L_F':>8s} {'K*':>5s} {'G(K*) bound':>12s}")
    for name, o in summary["estimators"].items():
        print(f"  {name:<25s} {o['L_F']:>8.3f} {o['K_star']:>5d} {o['G_star_bound']:>12.4f}")
    ei = summary["empirical_inversion"]
    print(f"  {'empirical_inversion_med':<25s} {ei['median_implied_L_F']:>8.3f} "
          f"{ei['oracle_from_median']['K_star']:>5d} {ei['oracle_from_median']['G_star_bound']:>12.4f}")
    print(f"\n  K_CV (selected via cross-validation) = {summary['K_cv']}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
