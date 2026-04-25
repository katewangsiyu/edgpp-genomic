"""E1 — Imbalance sweep: validate T5's pi_min-explicit rate.

Question: as pi_pos varies in {0.05, 0.10, 0.20, 0.50}, does single-axis
sigma-Mondrian (B2) lose minority-class coverage while joint HCCP (B4) holds?

This is the empirical realization of T5's pi_min-explicit bound:
    G(K*) <= 2 sqrt(L_F R / (pi_min n))

We subsample positives from the full TraitGym test set to hit each target
pi_pos, run B1/B2/B3/B4 chrom-LOO, and report cov|pos and sigma-bin gap.

Usage:
    conda run -n edgpp_t4 --no-capture-output python scripts/22_imbalance_sweep.py \
        --sigma-scores outputs/hetero_head/CADD+GPN-MSA+Borzoi_complex_abs/scores_with_sigma.parquet \
        --test-parquet data/raw/traitgym/complex_traits_matched_9/test.parquet \
        --out-dir outputs/imbalance_sweep/CADD+GPN-MSA+Borzoi_complex \
        --alpha 0.10 --K 2 --seed 42 \
        --pi-pos-grid 0.05 0.10 0.20 0.30 0.50
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
crepes_mod = import_module("21_crepes_baseline")
chrom_loo_crepes_per_class_pred_sets = crepes_mod.chrom_loo_crepes_per_class_pred_sets
evaluate_pred_sets = crepes_mod.evaluate_pred_sets


def subsample_to_target_pi(
    p: np.ndarray, sigma: np.ndarray, y: np.ndarray, chroms: np.ndarray,
    target_pi: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subsample to hit target positive rate.

    Strategy: keep ALL positives (preserve scarce minority signal), then
    subsample negatives to give n_neg = n_pos * (1 - target_pi) / target_pi.
    If that requires more negatives than available, instead subsample
    positives to match (rare in our setting).
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    obs_pi = n_pos / (n_pos + n_neg)

    if abs(obs_pi - target_pi) < 1e-3:
        return p, sigma, y, chroms

    if target_pi <= obs_pi:
        # need to drop positives
        target_n_pos = int(round(target_pi * n_neg / (1 - target_pi)))
        target_n_pos = min(target_n_pos, n_pos)
        keep_pos = rng.choice(pos_idx, size=target_n_pos, replace=False)
        keep_neg = neg_idx
    else:
        # need to drop negatives
        target_n_neg = int(round(n_pos * (1 - target_pi) / target_pi))
        target_n_neg = min(target_n_neg, n_neg)
        keep_pos = pos_idx
        keep_neg = rng.choice(neg_idx, size=target_n_neg, replace=False)

    keep = np.concatenate([keep_pos, keep_neg])
    keep.sort()
    return p[keep], sigma[keep], y[keep], chroms[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-scores", required=True)
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pi-pos-grid", type=float, nargs="+",
                    default=[0.05, 0.10, 0.20, 0.30, 0.50])
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    sig = pd.read_parquet(args.sigma_scores).reset_index(drop=True)
    assert len(V) == len(sig)

    y_full = V["label"].astype(int).to_numpy()
    chroms_full = V["chrom"].astype(str).to_numpy()
    p_full = sig["p_hat"].to_numpy()
    sigma_full = sig["sigma"].to_numpy()

    rng = np.random.default_rng(args.seed)
    obs_pi = float(y_full.mean())
    print(f"[load] n={len(V)} n_pos={int(y_full.sum())} observed pi={obs_pi:.4f}")

    all_results = {
        "alpha": args.alpha, "K": args.K, "seed": args.seed,
        "observed_pi": obs_pi, "n_full": int(len(V)),
        "sweep": [],
    }

    methods = [
        ("B1_split", "none"),
        ("B2_sigma", "sigma"),
        ("B3_class", "class"),
        ("B4_HCCP", "class_sigma"),
    ]

    print(f"\n{'pi':>6} {'n':>6} {'n+':>5} | {'method':<10} {'cov':>7} {'cov|+':>7} {'sig-gap':>9}")
    print("-" * 70)
    for target_pi in args.pi_pos_grid:
        p, sigma, y, chroms = subsample_to_target_pi(
            p_full, sigma_full, y_full, chroms_full, target_pi, rng)
        achieved_pi = float(y.mean())
        n = len(y); n_pos = int(y.sum())

        per_method = {}
        for method_name, mode in methods:
            ps = chrom_loo_crepes_per_class_pred_sets(
                p, sigma, y, chroms, args.alpha, mode, args.K)
            metrics = evaluate_pred_sets(
                ps, y, sigma, chroms, args.alpha, f"{method_name}_pi{target_pi:.2f}")
            per_method[method_name] = {
                "marginal_coverage": metrics["marginal_coverage"],
                "coverage_pos": metrics["coverage_pos"],
                "coverage_neg": metrics["coverage_neg"],
                "sigma_bin_gap": metrics["sigma_bin_gap"],
                "per_chrom_gap": metrics["per_chrom_gap"],
                "frac_singleton": metrics["frac_singleton"],
                "frac_both": metrics["frac_both"],
            }
            print(f"{achieved_pi:>6.3f} {n:>6} {n_pos:>5} | {method_name:<10} "
                  f"{metrics['marginal_coverage']:>7.4f} {metrics['coverage_pos']:>7.4f} "
                  f"{metrics['sigma_bin_gap']:>9.4f}")

        all_results["sweep"].append({
            "target_pi": target_pi,
            "achieved_pi": achieved_pi,
            "n": n,
            "n_pos": n_pos,
            "methods": per_method,
        })
        print()

    (out / "imbalance_sweep_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nsaved: {out}/imbalance_sweep_results.json")

    # CSV summary
    rows = []
    for s in all_results["sweep"]:
        for m, vals in s["methods"].items():
            rows.append({
                "pi": s["achieved_pi"], "n": s["n"], "n_pos": s["n_pos"],
                "method": m, **vals,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "imbalance_sweep_summary.csv", index=False)
    print(f"summary: {out}/imbalance_sweep_summary.csv")

    # Key takeaway: cov|pos vs pi for each method
    print(f"\n{'='*70}")
    print(f"{'method':<10} | " + "  ".join(f"pi={s['achieved_pi']:.2f}" for s in all_results["sweep"]))
    print(f"{'-'*70}")
    print(f"{'cov|pos':<10}")
    for m, _ in methods:
        line = f"{m:<10} | "
        for s in all_results["sweep"]:
            line += f"  {s['methods'][m]['coverage_pos']:.3f} "
        print(line)
    print(f"\n{'sig-gap':<10}")
    for m, _ in methods:
        line = f"{m:<10} | "
        for s in all_results["sweep"]:
            line += f"  {s['methods'][m]['sigma_bin_gap']:.3f} "
        print(line)


if __name__ == "__main__":
    main()
