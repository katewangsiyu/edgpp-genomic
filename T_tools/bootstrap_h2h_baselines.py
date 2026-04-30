"""Bootstrap CI on Tab tab:h2h: B=200 chromosome-level resamples for B1/B2/B3.

Mirrors `bootstrap_cp_baselines.py` but covers the four single-axis Mondrian
variants in tab:h2h:

  B1 vanilla split CP        (no Mondrian)
  B2 sigma-hat Mondrian      (Bostrom 2020, K=K_eval)
  B3 class-Mondrian          (Sadinle 2019)
  B4 HCCP                    (loaded from cp_baselines_h2h for consistency)

Same nonconformity score s(x, y) = |y - p(x)| / (sigma(x) + eps), same
chrom-LOO test split. B=200 chrom-level bootstrap returns mean / std / 95% CI
on (marginal_coverage, coverage_pos, sigma_bin_gap).

Usage:
    python T_tools/bootstrap_h2h_baselines.py --dataset mendelian --K-eval 3 --B 200
    python T_tools/bootstrap_h2h_baselines.py --dataset complex   --K-eval 5 --B 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "T_tools"))

from cp_baselines_h2h import (  # type: ignore
    ALPHA, EPS, load_scores, hccp_chrom_loo, select_K_hccp_nested_cv,
    compute_metric_bin_idx,
)
from bootstrap_cp_baselines import bootstrap_method  # type: ignore


def split_chrom_loo(df, alpha: float = ALPHA):
    """B1: vanilla split CP, no Mondrian. Single qhat from |y-p|/sigma."""
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values
    n = len(df)
    in0 = np.zeros(n, dtype=bool)
    in1 = np.zeros(n, dtype=bool)
    for c in sorted(set(chroms)):
        te = chroms == c
        cal = ~te
        s_cal = np.abs(y[cal] - p[cal]) / sigma[cal]
        n_cal = int(cal.sum())
        q = np.quantile(s_cal, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal)
        in0[te] = np.abs(0 - p[te]) <= q * sigma[te]
        in1[te] = np.abs(1 - p[te]) <= q * sigma[te]
    return in0, in1


def sigmamond_chrom_loo(df, K: int, alpha: float = ALPHA):
    """B2: sigma-hat Mondrian CP (Bostrom 2020). K bins on sigma quantiles."""
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values
    n = len(df)
    in0 = np.zeros(n, dtype=bool)
    in1 = np.zeros(n, dtype=bool)
    floor = int(np.ceil(1.0 / alpha))
    for c in sorted(set(chroms)):
        te = chroms == c
        cal = ~te
        s_cal = np.abs(y[cal] - p[cal]) / sigma[cal]
        edges = np.quantile(sigma[cal], np.linspace(0, 1, K + 1)[1:-1])
        b_cal = np.searchsorted(edges, sigma[cal])
        b_te = np.searchsorted(edges, sigma[te])
        for b in range(K):
            cell_cal = b_cal == b
            n_b = int(cell_cal.sum())
            if n_b >= floor:
                q = np.quantile(s_cal[cell_cal],
                                np.ceil((n_b + 1) * (1 - alpha)) / n_b)
            else:
                q = np.quantile(s_cal, np.ceil((len(s_cal) + 1) * (1 - alpha))
                                / len(s_cal))
            cell_te = (b_te == b)
            te_idx = np.where(te)[0][cell_te]
            in0[te_idx] = np.abs(0 - p[te_idx]) <= q * sigma[te_idx]
            in1[te_idx] = np.abs(1 - p[te_idx]) <= q * sigma[te_idx]
    return in0, in1


def classmond_chrom_loo(df, alpha: float = ALPHA):
    """B3: class-conditional Mondrian (Sadinle 2019). Per-class qhat."""
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values
    n = len(df)
    in0 = np.zeros(n, dtype=bool)
    in1 = np.zeros(n, dtype=bool)
    floor = int(np.ceil(1.0 / alpha))
    for c in sorted(set(chroms)):
        te = chroms == c
        cal = ~te
        s_cal = np.abs(y[cal] - p[cal]) / sigma[cal]
        for k in (0, 1):
            cell = (y[cal] == k)
            n_k = int(cell.sum())
            if n_k >= floor:
                q = np.quantile(s_cal[cell],
                                np.ceil((n_k + 1) * (1 - alpha)) / n_k)
            else:
                q = np.quantile(s_cal, np.ceil((len(s_cal) + 1) * (1 - alpha))
                                / len(s_cal))
            te_idx = np.where(te)[0]
            if k == 0:
                in0[te_idx] = np.abs(0 - p[te_idx]) <= q * sigma[te_idx]
            else:
                in1[te_idx] = np.abs(1 - p[te_idx]) <= q * sigma[te_idx]
    return in0, in1


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
    print(f"Loaded {args.dataset}: n={len(df)}, K_eval={K_eval}, B={args.B}")

    K_grid = [int(s) for s in args.K_grid.split(",")]
    print(f"\n[1/3] Nested-CV K selection (HCCP) ...")
    hccp_sel = select_K_hccp_nested_cv(df, K_grid, K_eval=K_eval)
    K_hccp = hccp_sel["K_per_fold"]

    metric_bin_idx = compute_metric_bin_idx(sigma, chroms, K_eval)

    print(f"\n[2/3] Computing chrom-LOO predictions for B1/B2/B3/B4 ...")
    print("  B1 vanilla split CP ...")
    i0_1, i1_1 = split_chrom_loo(df)
    print("  B2 sigma-Mondrian (K=K_eval) ...")
    i0_2, i1_2 = sigmamond_chrom_loo(df, K=K_eval)
    print("  B3 class-Mondrian ...")
    i0_3, i1_3 = classmond_chrom_loo(df)
    print("  B4 HCCP ...")
    i0_4, i1_4, _ = hccp_chrom_loo(df, K_hccp)

    methods = {
        "B1_split":      (i0_1, i1_1),
        "B2_sigmaMond":  (i0_2, i1_2),
        "B3_classMond":  (i0_3, i1_3),
        "B4_HCCP":       (i0_4, i1_4),
    }

    print(f"\n[3/3] Bootstrap B={args.B} chrom-resamples ...")
    results: dict[str, dict] = {}
    for name, (i0, i1) in methods.items():
        boot = bootstrap_method(i0, i1, y, sigma, chroms, metric_bin_idx,
                                  K_eval, args.B, args.seed)
        results[name] = boot
        gap = boot["sigma_bin_gap"]
        cp = boot["coverage_pos"]
        print(f"  {name:14} sig-gap = {gap['mean']:.4f} ± {gap['std']:.4f}  "
              f"CI [{gap['ci_lo']:.4f}, {gap['ci_hi']:.4f}]  |  "
              f"cov+ = {cp['mean']:.3f} ± {cp['std']:.3f}")

    out_dir = REPO / "R_raw" / "cp_baselines_h2h"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_h2h_baselines_Keval{K_eval}_B{args.B}.json"
    out_path.write_text(json.dumps({
        "dataset": args.dataset,
        "K_eval": K_eval,
        "B": args.B,
        "K_hccp_per_fold": K_hccp,
        "methods": results,
    }, indent=2, default=str))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
