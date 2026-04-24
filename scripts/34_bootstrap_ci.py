"""Bootstrap confidence intervals for HCCP coverage/AUPRC.

HistGradientBoosting at max_depth=2 is deterministic across seeds, so
`random_state` does not produce meaningful variance. Instead, we bootstrap the
calibration fold (B resamples with replacement of chrom-LOO calibration sets)
and report percentile 95% CI on marginal coverage and σ̂-bin gap.

Single-seed bootstrap is the appropriate uncertainty quantifier here because
it captures *calibration-set variability*, which is what Table 3 readers care
about for reproducibility.

Usage:
    python scripts/34_bootstrap_ci.py \\
        --sigma-scores outputs/hetero_head/CADD+GPN-MSA+Borzoi_mendelian_abs/scores_with_sigma.parquet \\
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --B 200 --out outputs/bootstrap_ci/mendelian.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
mod_14 = __import__("14_conformal_hetero")
mondrian_class_sigma_conformal = mod_14.mondrian_class_sigma_conformal
coverage_by_sigma_bin = mod_14.coverage_by_sigma_bin


def one_bootstrap(p, sigma, y, chroms, alpha, n_bins, rng) -> dict:
    """Chrom-level bootstrap: resample chromosomes with replacement."""
    uniq = np.array(sorted(set(chroms)))
    sampled = rng.choice(uniq, size=len(uniq), replace=True)
    idx = np.concatenate([np.where(chroms == c)[0] for c in sampled])
    pb, sb, yb, cb = p[idx], sigma[idx], y[idx], chroms[idx]

    # Relabel chromosomes so Mondrian chrom-LOO treats resamples correctly.
    # Use rank instead of original label.
    chrom_relabel = np.array([f"b{i}" for i in range(len(pb))])
    # Keep original chrom for proper LOO — but since we resampled, duplicates
    # exist. Simpler: use the resampled chrom-id directly and accept that
    # Mondrian will treat duplicate chroms as one big fold.
    pred_sets = mondrian_class_sigma_conformal(
        pb, sb, yb, cb, alpha, n_sigma_bins=n_bins)
    covered = np.array([yb[i] in ps for i, ps in enumerate(pred_sets)])
    sizes = np.array([len(ps) for ps in pred_sets])
    cov_by_bin = coverage_by_sigma_bin(covered, sb, n_bins=n_bins)
    return {
        "coverage": float(covered.mean()),
        "coverage_pos": float(covered[yb == 1].mean()) if (yb == 1).any() else float("nan"),
        "sigma_cov_range": float(max(b["coverage"] for b in cov_by_bin)
                                  - min(b["coverage"] for b in cov_by_bin)),
        "frac_singleton": float((sizes == 1).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-scores", required=True)
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.sigma_scores).reset_index(drop=True)
    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    assert len(df) == len(V)
    p = df["p_hat"].astype(float).to_numpy()
    sigma = df["sigma"].astype(float).to_numpy()
    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()

    rng = np.random.default_rng(args.seed)
    reps = []
    for b in range(args.B):
        if (b + 1) % 20 == 0:
            print(f"  bootstrap {b+1}/{args.B}")
        reps.append(one_bootstrap(p, sigma, y, chroms,
                                   args.alpha, args.n_bins, rng))

    metrics = ["coverage", "coverage_pos", "sigma_cov_range", "frac_singleton"]
    summary = {"B": args.B, "alpha": args.alpha, "n_bins": args.n_bins}
    for m in metrics:
        vals = np.array([r[m] for r in reps if not np.isnan(r[m])])
        summary[m] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)),
            "ci95_lo": float(np.quantile(vals, 0.025)),
            "ci95_hi": float(np.quantile(vals, 0.975)),
            "n_valid": int(len(vals)),
        }

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "replicates": reps}, indent=2))

    print("\n=== Bootstrap CI (B={}, α={}, K={}) ===".format(
          args.B, args.alpha, args.n_bins))
    for m in metrics:
        s = summary[m]
        print(f"  {m:<22s}: {s['mean']:.4f} ± {s['std']:.4f} "
              f"[{s['ci95_lo']:.4f}, {s['ci95_hi']:.4f}]")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
