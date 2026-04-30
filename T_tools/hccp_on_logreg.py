"""HCCP applied on top of the published LogReg aggregator instead of GBM.

Decouples the +14.8 pp AUPRC improvement (a property of the GBM aggregator)
from the conformal-side coverage gap improvement (a property of HCCP).

For each chrom-LOO fold:
  - p_hat_LR = LogReg score from outputs/aggregator/reproduce_CADD+GPN-MSA+Borzoi
  - sigma_hat = the existing GBM-residual variance head sigma_hat(x) (Day 13);
    sigma_hat is fit on features X, not on p_hat_GBM, so it estimates per-x
    aleatoric scale independent of the base predictor. Reusing it with p_hat_LR
    is the natural plug-in for the question "does HCCP improvement transfer
    when only the base predictor changes?"
  - Apply HCCP Mondrian (y x sigma-bin) calibration with K = K_eval (3 mendelian /
    5 complex), B = 200 chrom-bootstrap.

Saves: R_raw/cp_baselines_h2h/{ds}_hccp_logreg_Keval{K}_B200.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "T_tools"))

from cp_baselines_h2h import (  # type: ignore
    ALPHA, EPS, hccp_chrom_loo, compute_metric_bin_idx,
)
from bootstrap_cp_baselines import bootstrap_method  # type: ignore


def load_sigma(dataset: str) -> pd.DataFrame:
    suffix = "complex" if dataset == "complex" else "mendelian"
    p = REPO / "outputs" / "hetero_head" / f"CADD+GPN-MSA+Borzoi_{suffix}_abs" / "scores_with_sigma.parquet"
    return pd.read_parquet(p)


def load_logreg_scores(dataset: str) -> pd.DataFrame:
    if dataset == "complex":
        # No CADD+GPN-MSA+Borzoi reproduce dir for complex; fall back to
        # reproduce_CADD+Borzoi_complex (same TraitGym LogReg pipeline,
        # one feature short of the GBM backbone — sufficient for the
        # base-predictor decoupling demo).
        p = REPO / "outputs" / "aggregator" / "reproduce_CADD+Borzoi_complex"
    else:
        p = REPO / "outputs" / "aggregator" / "reproduce_CADD+GPN-MSA+Borzoi"
    return pd.read_parquet(p / "scores.parquet")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mendelian", "complex"], required=True)
    ap.add_argument("--K-eval", type=int, required=True)
    ap.add_argument("--B", type=int, default=200)
    args = ap.parse_args()

    df_sigma = load_sigma(args.dataset)
    df_lr = load_logreg_scores(args.dataset)
    df_lr["chrom"] = df_lr["chrom"].astype(str)

    assert len(df_sigma) == len(df_lr), f"{len(df_sigma)} vs {len(df_lr)}"
    p_lr = df_lr["score"].values
    sigma_lr = df_sigma["sigma"].values

    print(f"Loaded {args.dataset}: n={len(df_sigma)}, K_eval={args.K_eval}, B={args.B}")
    print(f"  p_hat_LR range:    [{p_lr.min():.4f}, {p_lr.max():.4f}], mean {p_lr.mean():.4f}")
    print(f"  sigma_hat (reused): [{sigma_lr.min():.4f}, {sigma_lr.max():.4f}], mean {sigma_lr.mean():.4f}")

    df = pd.DataFrame({
        "chrom": df_sigma["chrom"].astype(str).values,
        "label": df_sigma["label"].values,
        "p_hat": p_lr,
        "sigma": sigma_lr + EPS,
    })

    print(f"\n[1/3] HCCP chrom-LOO at fixed K = K_eval = {args.K_eval} ...")
    in0, in1, _ = hccp_chrom_loo(df, args.K_eval)

    print(f"\n[2/3] Bootstrap B={args.B} chrom-resamples ...")
    chroms = df["chrom"].values
    sigma = df["sigma"].values
    y = df["label"].values
    metric_bin_idx = compute_metric_bin_idx(sigma, chroms, args.K_eval)
    boot = bootstrap_method(in0, in1, y, sigma, chroms, metric_bin_idx,
                              args.K_eval, args.B, seed=42)
    gap = boot["sigma_bin_gap"]
    cp = boot["coverage_pos"]
    mc = boot["marginal_coverage"]
    print(f"  HCCP-on-LogReg  marg = {mc['mean']:.3f} ± {mc['std']:.3f}")
    print(f"                  cov+ = {cp['mean']:.3f} ± {cp['std']:.3f}")
    print(f"                  gap  = {gap['mean']:.3f} ± {gap['std']:.3f}  CI [{gap['ci_lo']:.3f}, {gap['ci_hi']:.3f}]")

    out = REPO / "R_raw" / "cp_baselines_h2h" / f"{args.dataset}_hccp_logreg_Keval{args.K_eval}_B{args.B}.json"
    out.write_text(json.dumps({
        "dataset": args.dataset,
        "aggregator": "LogReg (CADD+GPN-MSA+Borzoi)",
        "K_eval": args.K_eval,
        "B": args.B,
        "method": "HCCP",
        "metrics": boot,
    }, indent=2, default=str))
    print(f"\n[3/3] saved: {out}")


if __name__ == "__main__":
    main()
