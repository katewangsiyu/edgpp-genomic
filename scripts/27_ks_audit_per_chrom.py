"""Per-chromosome A2-cell KS audit.

For each Mondrian cell (k, b) with sufficient samples, compare the score distribution
of each chromosome against all other chromosomes' pooled scores. Report:
  - per-cell KS stats and rejection rates
  - per-chromosome worst-case KS across cells

Produces outputs/ks_audit/{feature_set}_{dataset}_ks_per_chrom.json for Appendix B.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-parquet", required=True,
                    help="conformal_hetero_scores.parquet from 14_conformal_hetero.py")
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--min-per-chrom-cell", type=int, default=5)
    ap.add_argument("--alpha-ks", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.scores_parquet).reset_index(drop=True)
    y = df["label"].astype(int).to_numpy()
    p = df["p_hat"].astype(float).to_numpy()
    sigma = df["sigma"].astype(float).to_numpy()
    chroms = df["chrom"].astype(str).to_numpy()
    eps = 1e-6
    score = np.abs(y - p) / (sigma + eps)

    edges = np.quantile(sigma, np.linspace(0, 1, args.n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_id = np.digitize(sigma, edges[1:-1])

    all_chroms = sorted(set(chroms))
    ks_results: list[dict] = []
    per_chrom_worst: dict[str, dict] = {}
    total_tests = 0
    n_reject = 0
    ks_stats: list[float] = []

    for k in (0, 1):
        for b in range(args.n_bins):
            cell_mask = (y == k) & (bin_id == b)
            for c in all_chroms:
                m_c = cell_mask & (chroms == c)
                m_other = cell_mask & (chroms != c)
                if m_c.sum() < args.min_per_chrom_cell or m_other.sum() < args.min_per_chrom_cell:
                    continue
                res = stats.ks_2samp(score[m_c], score[m_other])
                ks, pval = float(res.statistic), float(res.pvalue)
                ks_results.append({
                    "class": k, "bin": b, "chrom": c,
                    "n_chrom": int(m_c.sum()), "n_other": int(m_other.sum()),
                    "ks_stat": ks, "p_value": pval,
                    "reject_at_005": pval < args.alpha_ks,
                })
                ks_stats.append(ks)
                total_tests += 1
                if pval < args.alpha_ks:
                    n_reject += 1
                per = per_chrom_worst.setdefault(
                    c, {"chrom": c, "worst_ks": 0.0, "worst_cell": None,
                        "n_tests": 0, "n_reject": 0})
                per["n_tests"] += 1
                per["n_reject"] += int(pval < args.alpha_ks)
                if ks > per["worst_ks"]:
                    per["worst_ks"] = ks
                    per["worst_cell"] = f"(y={k},b={b})"

    summary = {
        "n_tests": total_tests,
        "n_reject": n_reject,
        "rejection_rate": n_reject / total_tests if total_tests else 0.0,
        "ks_stat_max": max(ks_stats) if ks_stats else None,
        "ks_stat_mean": float(np.mean(ks_stats)) if ks_stats else None,
        "ks_stat_median": float(np.median(ks_stats)) if ks_stats else None,
        "per_chrom": sorted(per_chrom_worst.values(),
                            key=lambda d: -d["worst_ks"]),
        "all_tests": ks_results,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"=== per-chrom KS audit ===")
    print(f"  tests: {total_tests}  reject@{args.alpha_ks:.2f}: {n_reject} "
          f"({100*n_reject/total_tests:.1f}%)")
    print(f"  KS stat: max={summary['ks_stat_max']:.3f}  "
          f"mean={summary['ks_stat_mean']:.3f}  median={summary['ks_stat_median']:.3f}")
    print(f"\n  Worst 5 chromosomes by max KS:")
    for row in summary["per_chrom"][:5]:
        print(f"    chr{row['chrom']:<3s}: worst KS={row['worst_ks']:.3f} "
              f"at {row['worst_cell']}, reject {row['n_reject']}/{row['n_tests']}")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
