"""E2 — K-at-pi sweep: validate T5's K* = O(sqrt(L_F R pi_min n)) scaling.

For each pi_pos in {0.05, 0.10, 0.20, 0.30, 0.50}, sweep K in {2,3,5,8,10,15}
and find K_argmin = K minimizing the worst sigma-bin gap on HCCP.

T5 predicts K* should scale as sqrt(pi_min). With pi=0.05 -> 0.50 (10x range),
K* should grow by sqrt(10) ~ 3.16x. We expect K_argmin to monotonically
increase with pi_pos.

Usage:
    conda run -n edgpp_t4 --no-capture-output python scripts/23_K_at_pi_sweep.py \
        --sigma-scores outputs/hetero_head/CADD+GPN-MSA+Borzoi_complex_abs/scores_with_sigma.parquet \
        --test-parquet data/raw/traitgym/complex_traits_matched_9/test.parquet \
        --out-dir outputs/K_at_pi/CADD+GPN-MSA+Borzoi_complex \
        --alpha 0.10 --seed 42
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
sweep_mod = import_module("22_imbalance_sweep")
chrom_loo_crepes_per_class_pred_sets = crepes_mod.chrom_loo_crepes_per_class_pred_sets
evaluate_pred_sets = crepes_mod.evaluate_pred_sets
subsample_to_target_pi = sweep_mod.subsample_to_target_pi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-scores", required=True)
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pi-pos-grid", type=float, nargs="+",
                    default=[0.05, 0.10, 0.20, 0.30, 0.50])
    ap.add_argument("--K-grid", type=int, nargs="+",
                    default=[2, 3, 5, 8, 10, 15])
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    sig = pd.read_parquet(args.sigma_scores).reset_index(drop=True)

    y_full = V["label"].astype(int).to_numpy()
    chroms_full = V["chrom"].astype(str).to_numpy()
    p_full = sig["p_hat"].to_numpy()
    sigma_full = sig["sigma"].to_numpy()

    rng = np.random.default_rng(args.seed)
    print(f"[load] n_full={len(V)} obs_pi={float(y_full.mean()):.3f}")

    rows = []
    print(f"\n{'pi':>6} {'K':>4} {'cov|+':>7} {'sig-gap':>9}")
    print("-" * 40)
    for target_pi in args.pi_pos_grid:
        p, sigma, y, chroms = subsample_to_target_pi(
            p_full, sigma_full, y_full, chroms_full, target_pi, rng)
        achieved_pi = float(y.mean())
        n = len(y)

        per_K = []
        for K in args.K_grid:
            ps = chrom_loo_crepes_per_class_pred_sets(
                p, sigma, y, chroms, args.alpha, "class_sigma", K)
            # IMPORTANT: evaluate sigma-bin gap on the SAME K used for calibration.
            # This matches T5's worst-cell-gap quantity. Using a fixed eval K=10
            # creates a trivial alignment artifact when K_cal == 10.
            metrics = evaluate_pred_sets(
                ps, y, sigma, chroms, args.alpha, f"HCCP_pi{target_pi:.2f}_K{K}",
                n_sigma_bins=K)
            row = {
                "target_pi": target_pi,
                "achieved_pi": achieved_pi,
                "n": n,
                "K": K,
                "marginal_coverage": metrics["marginal_coverage"],
                "coverage_pos": metrics["coverage_pos"],
                "sigma_bin_gap": metrics["sigma_bin_gap"],
                "per_chrom_gap": metrics["per_chrom_gap"],
            }
            per_K.append(row)
            rows.append(row)
            print(f"{achieved_pi:>6.3f} {K:>4} {metrics['coverage_pos']:>7.4f} "
                  f"{metrics['sigma_bin_gap']:>9.4f}")
        # find argmin
        K_argmin = min(per_K, key=lambda r: r["sigma_bin_gap"])["K"]
        print(f"  -> pi={achieved_pi:.3f}: K_argmin = {K_argmin}")
        print()

    df = pd.DataFrame(rows)
    df.to_csv(out / "K_at_pi_sweep.csv", index=False)
    (out / "K_at_pi_sweep.json").write_text(json.dumps(rows, indent=2))

    # T5 scaling check: K_argmin vs sqrt(pi_min)
    print(f"\n{'='*50}")
    print(f"T5 prediction: K* = sqrt(L_F R pi_min n)")
    print(f"For fixed n, K* should scale ~ sqrt(pi_min)")
    print(f"{'='*50}")
    print(f"{'pi':>6} {'n':>6} {'K_argmin':>10} {'best_gap':>10}")
    print("-" * 50)
    for pi in args.pi_pos_grid:
        sub = df[df["target_pi"] == pi]
        if len(sub) == 0:
            continue
        best = sub.loc[sub["sigma_bin_gap"].idxmin()]
        print(f"{best['achieved_pi']:>6.3f} {int(best['n']):>6} {int(best['K']):>10} "
              f"{best['sigma_bin_gap']:>10.4f}")
    print(f"\nsaved: {out}/K_at_pi_sweep.csv, K_at_pi_sweep.json")


if __name__ == "__main__":
    main()
