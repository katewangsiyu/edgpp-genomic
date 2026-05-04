"""Paired bootstrap test: HCCP vs.\\ each baseline on the same chrom-resamples.

Why this exists: Tab tab:cp_baselines reports per-method marginal CIs (mean +/- std
across B=200 chrom-resamples). When two methods' marginal CIs overlap (e.g.\\ HCCP
0.173 +/- 0.192 vs.\\ weighted CP 0.252 +/- 0.153 on Mendelian K_eval=3), this does
NOT imply they are statistically equivalent --- the bootstrap noise is highly
correlated across methods (same chrom resample), so paired diffs can have much
tighter std. This script computes the paired statistic directly.

Per resample b:
  - draw chrom indices once (seeded RNG)
  - compute sigma-bin gap for each method on the SAME subsample
  - record diff_b[method] = gap_HCCP_b - gap_method_b

Output: per-replicate raw values (200 floats per method per metric) plus paired
mean / std / 95% CI / win-rate for HCCP-vs-method.

Usage:
    python T_tools/paired_bootstrap_h2h.py --dataset mendelian --K-eval 3 --B 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cp_baselines_h2h import (  # type: ignore
    ALPHA, EPS, load_scores, hccp_chrom_loo, rlcp_chrom_loo,
    weighted_cp_chrom_loo, sccp_chrom_loo, select_K_hccp_nested_cv,
    select_K_sccp_nested_cv, compute_metric_bin_idx,
)


def gap_on_subset(in_set: np.ndarray, y: np.ndarray, metric_bin_idx: np.ndarray,
                  K_eval: int, idx: np.ndarray) -> float:
    """Cell-level worst-(k,b) coverage deviation from 1-alpha."""
    if len(idx) == 0:
        return float("nan")
    bin_ids = metric_bin_idx[idx]
    y_sub = y[idx]
    in_sub = in_set[idx]
    bin_gaps: list[float] = []
    for k in (0, 1):
        for b in range(K_eval):
            mask = (y_sub == k) & (bin_ids == b)
            if mask.sum() < 5:
                continue
            cov_kb = float(in_sub[mask].mean())
            bin_gaps.append(abs(cov_kb - (1 - ALPHA)))
    return float(max(bin_gaps)) if bin_gaps else float("nan")


def in_set_for(in0: np.ndarray, in1: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (y == 0) * in0 + (y == 1) * in1


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
    print(f"Loaded {args.dataset}: n={len(df)}, K_eval={K_eval}, B={args.B}, seed={args.seed}")

    print("\n[1/3] Nested-CV K selection ...")
    hccp_sel = select_K_hccp_nested_cv(df, K_grid, K_eval=K_eval)
    sccp_sel = select_K_sccp_nested_cv(df, K_grid, K_eval=K_eval)
    K_hccp = hccp_sel["K_per_fold"]
    K_sccp = sccp_sel["K_per_fold"]
    metric_bin_idx = compute_metric_bin_idx(sigma, chroms, K_eval)

    print("\n[2/3] Computing chrom-LOO predictions for 4 methods ...")
    print("  HCCP ...")
    i0_h, i1_h, _ = hccp_chrom_loo(df, K_hccp)
    print("  RLCP ...")
    i0_r, i1_r, _ = rlcp_chrom_loo(df, K_eval)
    print("  Weighted CP ...")
    i0_w, i1_w, _ = weighted_cp_chrom_loo(df, K_eval)
    print("  SC-CP ...")
    i0_s, i1_s, _ = sccp_chrom_loo(df, K_sccp, K_eval=K_eval)

    methods = {
        "HCCP": in_set_for(i0_h, i1_h, y),
        "RLCP": in_set_for(i0_r, i1_r, y),
        "WeightedCP": in_set_for(i0_w, i1_w, y),
        "SCCP": in_set_for(i0_s, i1_s, y),
    }

    print(f"\n[3/3] Paired bootstrap B={args.B} chrom-resamples (seed={args.seed}) ...")
    rng = np.random.default_rng(args.seed)
    unique_chroms = np.array(sorted(set(chroms)))
    n_chroms = len(unique_chroms)
    raw: dict[str, list[float]] = {name: [] for name in methods}
    for _ in range(args.B):
        sampled = rng.choice(unique_chroms, size=n_chroms, replace=True)
        idx = np.concatenate([np.where(chroms == c)[0] for c in sampled])
        for name, in_set in methods.items():
            raw[name].append(gap_on_subset(in_set, y, metric_bin_idx, K_eval, idx))

    raw_arrays = {name: np.array(vals) for name, vals in raw.items()}

    def stats(vals: np.ndarray) -> dict[str, float]:
        v = vals[~np.isnan(vals)]
        return {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)),
            "ci_lo": float(np.quantile(v, 0.025)),
            "ci_hi": float(np.quantile(v, 0.975)),
            "B_used": int(v.size),
        }

    paired: dict[str, dict] = {}
    hccp_arr = raw_arrays["HCCP"]
    print(f"\n--- Paired diffs (HCCP - baseline), {args.dataset} K_eval={K_eval} ---")
    print(f"{'method':12} {'paired_mean':>11} {'paired_std':>10} {'CI_lo':>8} {'CI_hi':>8} "
          f"{'win%':>6} {'p_one':>7}")
    for name, arr in raw_arrays.items():
        if name == "HCCP":
            continue
        diff = hccp_arr - arr
        diff_v = diff[~np.isnan(diff)]
        n_wins = int((diff_v < 0).sum())
        p_one = float(((diff_v >= 0).sum() + 1) / (diff_v.size + 1))
        s = stats(diff)
        paired[name] = {**s, "n_wins": n_wins, "p_one_sided": p_one,
                        "interpretation": (
                            f"HCCP gap < {name} gap on {n_wins}/{diff_v.size} resamples; "
                            f"paired one-sided p = {p_one:.4f}"
                        )}
        print(f"{name:12} {s['mean']:>+11.4f} {s['std']:>10.4f} "
              f"{s['ci_lo']:>+8.4f} {s['ci_hi']:>+8.4f} "
              f"{100*n_wins/diff_v.size:>5.1f}% {p_one:>7.4f}")

    out_dir = Path("/home/lzeng/workspace/edgpp-genomic/R_raw/cp_baselines_h2h")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_paired_bootstrap_Keval{K_eval}_B{args.B}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset, "K_eval": K_eval, "B": args.B, "seed": args.seed,
        "K_hccp_per_fold": K_hccp, "K_sccp_per_fold": K_sccp,
        "raw_per_replicate": {name: arr.tolist() for name, arr in raw_arrays.items()},
        "marginal_stats": {name: stats(arr) for name, arr in raw_arrays.items()},
        "paired_vs_HCCP": paired,
    }, indent=2, default=str))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
