"""Nonconformity score function ablation.

Compare three scores under identical Mondrian-(y×σ̂-bin) calibration:
  (A) HCCP default:   s(x, y) = |y - p̂(x)| / (σ̂(x) + ε)
  (B) Squared:        s(x, y) = (y - p̂(x))² / (σ̂(x)² + ε)
  (C) CDF-based:      s(x, y) = Φ(|y - p̂(x)| / σ̂(x))  — score as Gaussian CDF tail
  (D) No σ̂ (baseline): s(x, y) = |y - p̂(x)| (DEGU-lite split CP style)

Uses the saved p̂/σ̂ from the HCCP pipeline so the only varying quantity is the
score function.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
mod_14 = __import__("14_conformal_hetero")
mondrian_class_sigma_conformal = mod_14.mondrian_class_sigma_conformal
coverage_by_sigma_bin = mod_14.coverage_by_sigma_bin
per_chrom_coverage = mod_14.per_chrom_coverage
EPS_DEFAULT = mod_14.EPS_DEFAULT


def score_abs(p: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(y.astype(float) - p) / (sigma + EPS_DEFAULT)


def score_squared(p: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (y.astype(float) - p) ** 2 / (sigma ** 2 + EPS_DEFAULT)


def score_cdf(p: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Gaussian CDF tail: larger → more anomalous
    return norm.cdf(np.abs(y.astype(float) - p) / (sigma + EPS_DEFAULT))


def score_no_sigma(p: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(y.astype(float) - p)


SCORES = {
    "abs_over_sigma":     score_abs,
    "squared_over_var":   score_squared,
    "cdf_abs_over_sigma": score_cdf,
    "no_sigma":           score_no_sigma,
}


def evaluate(pred_sets, y, sigma, chroms, n_bins: int) -> dict:
    covered = np.array([y[i] in ps for i, ps in enumerate(pred_sets)])
    sizes = np.array([len(ps) for ps in pred_sets])
    cov_by_bin = coverage_by_sigma_bin(covered, sigma, n_bins=n_bins)
    cov_range = (max(b["coverage"] for b in cov_by_bin)
                 - min(b["coverage"] for b in cov_by_bin))
    per_chrom_max = max(
        abs(v - 0.9) for v in per_chrom_coverage(covered, chroms).values()
    ) if len(set(chroms)) > 1 else 0.0
    return {
        "coverage": float(covered.mean()),
        "sigma_bin_range": float(cov_range),
        "per_chrom_max_gap": float(per_chrom_max),
        "frac_singleton": float((sizes == 1).mean()),
        "frac_both": float((sizes == 2).mean()),
        "frac_empty": float((sizes == 0).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-scores", required=True)
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.sigma_scores).reset_index(drop=True)
    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    assert len(df) == len(V)
    p = df["p_hat"].astype(float).to_numpy()
    sigma = df["sigma"].astype(float).to_numpy()
    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()

    results = {}
    for name, fn in SCORES.items():
        pred_sets = mondrian_class_sigma_conformal(
            p, sigma, y, chroms, args.alpha,
            n_sigma_bins=args.n_bins, score_fn=fn)
        metrics = evaluate(pred_sets, y, sigma, chroms, args.n_bins)
        results[name] = metrics
        print(f"  {name:<25s}: cov={metrics['coverage']:.3f} "
              f"σ̂-range={metrics['sigma_bin_range']:.3f} "
              f"per-chrom-max={metrics['per_chrom_max_gap']:.3f} "
              f"singleton={metrics['frac_singleton']:.2f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "alpha": args.alpha, "n_bins": args.n_bins,
        "n": int(len(V)),
        "scores": results,
    }, indent=2))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
