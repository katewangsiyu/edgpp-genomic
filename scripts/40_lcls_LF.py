"""LCLS-style L_F estimator with bootstrap confidence interval.

Adapts the Lipschitz Constant Least Squares (LCLS, ICLR 2024) framework to the
score-CDF Lipschitz problem of HCCP's Theorem 7. We estimate

    L_F = sup_{s, k, σ_1 ≠ σ_2}  |F_{k,σ_1}(s) - F_{k,σ_2}(s)| / |σ_1 - σ_2|

where F_{k,σ}(s) is the conditional CDF of the nonconformity score given class
k and σ̂(X) = σ. Three estimators are computed for cross-checking:

  1. Pointwise max-of-pairs (legacy, biased high)
  2. LCLS-style: regression of pairwise KS distances on σ-gap
  3. Robust isotonic upper envelope

A non-parametric bootstrap (B = 500 resamples of variant indices) gives a
finite-sample 95% CI on each estimator.

Usage:
    python scripts/40_lcls_LF.py \\
        --scores-parquet outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/conformal_hetero_scores.parquet \\
        --out outputs/L_F_audit/CADD+GPN-MSA+Borzoi_mendelian_LF_LCLS.json
"""
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class LFEstimate:
    """One estimator's point + bootstrap CI."""
    name: str
    L_F: float
    ci_lo: float
    ci_hi: float
    n_pairs_used: int


def _bin_pairs(sigma: np.ndarray, scores: np.ndarray, mask: np.ndarray,
               n_bins: int, min_per_bin: int) -> list[tuple[float, np.ndarray, int]]:
    """Quantile-bin σ̂ inside the masked subset and return (σ̂-mean, scores, n) per bin."""
    if mask.sum() < min_per_bin * 2:
        return []
    sub_sigma = sigma[mask]
    sub_scores = scores[mask]
    edges = np.quantile(sub_sigma, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_id = np.digitize(sub_sigma, edges[1:-1])
    out: list[tuple[float, np.ndarray, int]] = []
    for b in range(n_bins):
        m_b = bin_id == b
        if m_b.sum() < min_per_bin:
            continue
        out.append((float(sub_sigma[m_b].mean()),
                    sub_scores[m_b].copy(),
                    int(m_b.sum())))
    return out


def lcls_pairwise_ks(p: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                     n_bins: int = 30, min_per_bin: int = 20,
                     eps: float = 1e-6) -> dict:
    """Compute pairwise (KS, σ-gap) across many bin pairs, per class.

    Returns dict with all pair observations for downstream regression.
    """
    score = np.abs(y.astype(float) - p) / (sigma + eps)
    pairs: list[tuple[float, float]] = []  # (sigma_gap, KS)
    for k in (0, 1):
        bins_k = _bin_pairs(sigma, score, y == k, n_bins, min_per_bin)
        for (sig_a, sc_a, _), (sig_b, sc_b, _) in combinations(bins_k, 2):
            ks_stat, _ = ks_2samp(sc_a, sc_b)
            gap = abs(sig_b - sig_a)
            if gap > 1e-8:
                pairs.append((gap, float(ks_stat)))
    if not pairs:
        return {"sigma_gaps": np.array([]), "ks_stats": np.array([])}
    arr = np.asarray(pairs)
    return {"sigma_gaps": arr[:, 0], "ks_stats": arr[:, 1]}


def estimator_max_ratio(pairs: dict) -> float:
    """Legacy estimator: max of KS/sigma_gap across all pairs."""
    if pairs["sigma_gaps"].size == 0:
        return float("nan")
    return float((pairs["ks_stats"] / pairs["sigma_gaps"]).max())


def estimator_lcls_regression(pairs: dict) -> float:
    """LCLS-style: least-squares regression of KS on sigma_gap, no intercept.

    Under L-Lipschitz F, KS ≤ L · sigma_gap, so regressing KS = β · gap
    estimates β as a tight upper bound on L. Constraining the slope to be
    non-negative is automatic via abs values.
    """
    g = pairs["sigma_gaps"]
    k = pairs["ks_stats"]
    if g.size < 5:
        return float("nan")
    # Least-squares slope through origin
    beta = float((g * k).sum() / (g * g).sum())
    return max(beta, 0.0)


def estimator_isotonic_envelope(pairs: dict) -> float:
    """Fit isotonic upper envelope of KS as a function of sigma_gap.

    The Lipschitz constant is the slope of the envelope at small gaps; we
    take the slope on the smallest decile of gaps (most local) as the
    estimator. This is robust to heteroscedastic noise in pairs.
    """
    g = pairs["sigma_gaps"]
    k = pairs["ks_stats"]
    if g.size < 10:
        return float("nan")
    order = np.argsort(g)
    g_s, k_s = g[order], k[order]
    # Use lower-decile pairs only (smallest σ-gaps where the bound is tightest)
    cutoff = max(int(0.1 * len(g_s)), 5)
    g_local = g_s[:cutoff]
    k_local = k_s[:cutoff]
    if g_local.sum() < 1e-9:
        return float("nan")
    # Isotonic-regress k_local against g_local then read slope
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    k_iso = iso.fit_transform(g_local, k_local)
    # Slope estimator: total rise over total run on local set
    if g_local.max() - g_local.min() < 1e-9:
        return float("nan")
    return float((k_iso.max() - k_iso.min()) / (g_local.max() - g_local.min()))


def bootstrap_ci(p, sigma, y, B: int, seed: int,
                  n_bins: int, min_per_bin: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(p)
    estimators = {"max_ratio": [], "lcls_regression": [], "isotonic_envelope": []}
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        pairs = lcls_pairwise_ks(p[idx], sigma[idx], y[idx],
                                  n_bins=n_bins, min_per_bin=min_per_bin)
        if pairs["sigma_gaps"].size == 0:
            continue
        estimators["max_ratio"].append(estimator_max_ratio(pairs))
        estimators["lcls_regression"].append(estimator_lcls_regression(pairs))
        estimators["isotonic_envelope"].append(estimator_isotonic_envelope(pairs))
    out = {}
    for name, vals in estimators.items():
        v = np.array([x for x in vals if not np.isnan(x)])
        out[name] = {
            "B_used": int(len(v)),
            "ci_lo": float(np.quantile(v, 0.025)) if v.size else float("nan"),
            "ci_hi": float(np.quantile(v, 0.975)) if v.size else float("nan"),
            "median": float(np.median(v)) if v.size else float("nan"),
            "mean": float(v.mean()) if v.size else float("nan"),
            "std": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bins", type=int, default=30,
                    help="Number of σ̂-bins per class for pair sampling")
    ap.add_argument("--min-per-bin", type=int, default=20)
    ap.add_argument("--B", type=int, default=500,
                    help="Bootstrap replicates for CI")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.scores_parquet)
    p = df["p_hat"].astype(float).to_numpy()
    sigma = df["sigma"].astype(float).to_numpy()
    y = df["label"].astype(int).to_numpy()
    print(f"[load] n={len(df)}  pos={int(y.sum())} ({y.mean()*100:.1f}%)")

    # Point estimate
    pairs = lcls_pairwise_ks(p, sigma, y, args.n_bins, args.min_per_bin)
    n_pairs = int(pairs["sigma_gaps"].size)
    print(f"[pairs] {n_pairs} (σ̂-bin × σ̂-bin) pairs collected across two classes")

    estimators = [
        LFEstimate("max_ratio (legacy)",
                   estimator_max_ratio(pairs), float("nan"), float("nan"), n_pairs),
        LFEstimate("lcls_regression",
                   estimator_lcls_regression(pairs), float("nan"), float("nan"), n_pairs),
        LFEstimate("isotonic_envelope",
                   estimator_isotonic_envelope(pairs), float("nan"), float("nan"), n_pairs),
    ]

    # Bootstrap CI
    print(f"[bootstrap] B={args.B} replicates ...")
    ci = bootstrap_ci(p, sigma, y, args.B, args.seed, args.n_bins, args.min_per_bin)

    print(f"\n=== L_F estimators (bootstrap B={args.B}) ===")
    print(f"  {'estimator':<22s} {'point':>8s}  {'95% CI':<22s} {'B_used':>7s}")
    rows = []
    for est in estimators:
        c = ci[est.name.split(" ")[0] if "lcls" not in est.name else "lcls_regression"]
        ci_str = f"[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]"
        print(f"  {est.name:<22s} {est.L_F:>8.3f}  {ci_str:<22s} {c['B_used']:>7d}")
        rows.append({
            "estimator": est.name,
            "point_estimate": est.L_F,
            "ci_lo_95": c["ci_lo"],
            "ci_hi_95": c["ci_hi"],
            "median_bootstrap": c["median"],
            "B_used": c["B_used"],
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_variants": int(len(df)),
        "n_pairs": n_pairs,
        "n_bins_per_class": args.n_bins,
        "min_per_bin": args.min_per_bin,
        "bootstrap_B": args.B,
        "estimators": rows,
        "bootstrap_raw": ci,
    }, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
