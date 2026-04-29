"""Bootstrap CI on Tab tab:cp_baselines: B=200 chromosome-level resamples.

Wraps cp_baselines_h2h to compute B=200 chrom-resamples for HCCP/RLCP/wCP/SCCP
on a fixed K_eval reporting metric. Reports mean ± std for σ̂-bin gap,
marginal coverage, and minority-class coverage. Resamples test chromosomes
with replacement (same protocol as bootstrap CIs in Tab tab:main).

Usage:
    python T_tools/bootstrap_cp_baselines.py --dataset complex --K-eval 5 --B 200
    python T_tools/bootstrap_cp_baselines.py --dataset mendelian --K-eval 3 --B 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cp_baselines_h2h import (  # type: ignore
    ALPHA, EPS, load_scores, hccp_chrom_loo, rlcp_chrom_loo,
    weighted_cp_chrom_loo, sccp_chrom_loo, select_K_hccp_nested_cv,
    select_K_sccp_nested_cv, compute_metric_bin_idx,
)


def coverage_on_subset(
    in0: np.ndarray, in1: np.ndarray, y: np.ndarray, sigma: np.ndarray,
    metric_bin_idx: np.ndarray, K_eval: int, idx: np.ndarray,
) -> dict[str, float]:
    """Compute coverage metrics on a subset (idx) of variants."""
    in_set = (y == 0) * in0 + (y == 1) * in1
    n = len(idx)
    if n == 0:
        return {"marginal_coverage": float("nan"), "coverage_pos": float("nan"),
                "sigma_bin_gap": float("nan")}
    cov_marg = float(in_set[idx].mean())
    pos = (y[idx] == 1)
    cov_pos = float(in_set[idx][pos].mean()) if pos.any() else float("nan")

    bin_ids = metric_bin_idx[idx]
    y_sub = y[idx]
    in_sub = in_set[idx]
    bin_gaps = []
    for k in (0, 1):
        for b in range(K_eval):
            mask = (y_sub == k) & (bin_ids == b)
            if mask.sum() < 5:
                continue
            cov_kb = in_sub[mask].mean()
            bin_gaps.append(abs(cov_kb - (1 - ALPHA)))
    sig_gap = float(max(bin_gaps)) if bin_gaps else float("nan")
    return {"marginal_coverage": cov_marg, "coverage_pos": cov_pos,
            "sigma_bin_gap": sig_gap}


def bootstrap_method(
    in0: np.ndarray, in1: np.ndarray, y: np.ndarray, sigma: np.ndarray,
    chroms: np.ndarray, metric_bin_idx: np.ndarray, K_eval: int,
    B: int, seed: int,
) -> dict[str, dict[str, float]]:
    """B=B chromosome-level resamples; return mean / std / 95% CI per metric."""
    rng = np.random.default_rng(seed)
    unique_chroms = np.array(sorted(set(chroms)))
    n_chroms = len(unique_chroms)
    samples: list[dict[str, float]] = []
    for _ in range(B):
        sampled = rng.choice(unique_chroms, size=n_chroms, replace=True)
        idx_parts = [np.where(chroms == c)[0] for c in sampled]
        idx = np.concatenate(idx_parts)
        samples.append(coverage_on_subset(in0, in1, y, sigma, metric_bin_idx,
                                           K_eval, idx))
    out: dict[str, dict[str, float]] = {}
    for metric in ("marginal_coverage", "coverage_pos", "sigma_bin_gap"):
        vals = np.array([s[metric] for s in samples if not np.isnan(s[metric])])
        if vals.size == 0:
            out[metric] = {"mean": float("nan"), "std": float("nan"),
                            "ci_lo": float("nan"), "ci_hi": float("nan"),
                            "B_used": 0}
        else:
            out[metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "ci_lo": float(np.quantile(vals, 0.025)),
                "ci_hi": float(np.quantile(vals, 0.975)),
                "B_used": int(vals.size),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mendelian", "complex"], required=True)
    ap.add_argument("--K-eval", type=int, required=True)
    ap.add_argument("--K-grid", type=str, default="2,3,5,8,10")
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_scores(args.dataset)
    chroms = df["chrom"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values
    K_eval = args.K_eval

    K_grid = [int(s) for s in args.K_grid.split(",")]
    print(f"Loaded {args.dataset}: n={len(df)}, K_eval={K_eval}, B={args.B}")

    print(f"\n[1/4] Nested-CV K selection ...")
    hccp_sel = select_K_hccp_nested_cv(df, K_grid, K_eval=K_eval)
    sccp_sel = select_K_sccp_nested_cv(df, K_grid, K_eval=K_eval)
    K_hccp = hccp_sel["K_per_fold"]
    K_sccp = sccp_sel["K_per_fold"]

    metric_bin_idx = compute_metric_bin_idx(sigma, chroms, K_eval)

    print(f"\n[2/4] Computing chrom-LOO predictions for 4 methods ...")
    print("  HCCP ...")
    i0_h, i1_h, _ = hccp_chrom_loo(df, K_hccp)
    print("  RLCP ...")
    i0_r, i1_r, _ = rlcp_chrom_loo(df, K_eval)
    print("  Weighted CP ...")
    i0_w, i1_w, _ = weighted_cp_chrom_loo(df, K_eval)
    print("  SC-CP ...")
    i0_s, i1_s, _ = sccp_chrom_loo(df, K_sccp, K_eval=K_eval)

    methods = {
        "HCCP": (i0_h, i1_h),
        "RLCP": (i0_r, i1_r),
        "WeightedCP": (i0_w, i1_w),
        "SCCP": (i0_s, i1_s),
    }

    print(f"\n[3/4] Bootstrap B={args.B} chrom-resamples ...")
    results: dict[str, dict] = {}
    for name, (i0, i1) in methods.items():
        boot = bootstrap_method(i0, i1, y, sigma, chroms, metric_bin_idx,
                                  K_eval, args.B, args.seed)
        results[name] = boot
        gap = boot["sigma_bin_gap"]
        cp = boot["coverage_pos"]
        print(f"  {name:12} sig-gap = {gap['mean']:.4f} ± {gap['std']:.4f}  "
              f"CI [{gap['ci_lo']:.4f}, {gap['ci_hi']:.4f}]  |  "
              f"cov+ = {cp['mean']:.3f} ± {cp['std']:.3f}")

    out_dir = Path("/home/lzeng/workspace/edgpp-genomic/R_raw/cp_baselines_h2h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_bootstrap_Keval{K_eval}_B{args.B}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "K_eval": K_eval,
        "B": args.B,
        "K_hccp_per_fold": K_hccp,
        "K_sccp_per_fold": K_sccp,
        "methods": results,
    }, indent=2, default=str))
    print(f"\n[4/4] saved: {out_path}")


if __name__ == "__main__":
    main()
