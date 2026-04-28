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
in0_in1_to_pred_sets = crepes_mod.in0_in1_to_pred_sets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "T_tools"))
import nested_kcv_helpers as nkv


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
    ap.add_argument("--K-mode", choices=["fixed", "nested-cv"], default="fixed")
    ap.add_argument("--K", type=int, default=2,
                    help="Fixed K for B2/B4 (used when --K-mode fixed)")
    ap.add_argument("--K-grid", type=str, default="2,3,5,8,10,15,20",
                    help="K candidates for nested CV")
    ap.add_argument("--K-eval", type=int, default=5,
                    help="sigma-bin count for metric reporting")
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
    print(f"[load] n={len(V)} n_pos={int(y_full.sum())} observed pi={obs_pi:.4f} "
          f"K-mode={args.K_mode} K_eval={args.K_eval}")

    all_results = {
        "alpha": args.alpha, "K_mode": args.K_mode, "K_eval": args.K_eval,
        "K_fixed": args.K, "seed": args.seed,
        "observed_pi": obs_pi, "n_full": int(len(V)),
        "sweep": [],
    }

    K_grid = [int(s) for s in args.K_grid.split(",")] if args.K_mode == "nested-cv" else None

    print(f"\n{'pi':>6} {'n':>6} {'n+':>5} | {'method':<10} {'cov':>7} {'cov|+':>7} {'sig-gap':>9}")
    print("-" * 70)
    for target_pi in args.pi_pos_grid:
        p, sigma, y, chroms = subsample_to_target_pi(
            p_full, sigma_full, y_full, chroms_full, target_pi, rng)
        achieved_pi = float(y.mean())
        n = len(y); n_pos = int(y.sum())

        if args.K_mode == "nested-cv":
            sel_b4 = nkv.select_K_nested_chrom_loo(p, sigma, y, chroms, K_grid,
                                                   method="b4_hccp",
                                                   K_eval=args.K_eval, alpha=args.alpha)
            sel_b2 = nkv.select_K_nested_chrom_loo(p, sigma, y, chroms, K_grid,
                                                   method="b2_sigma",
                                                   K_eval=args.K_eval, alpha=args.alpha)
            K_b4 = sel_b4["K_per_fold"]
            K_b2 = sel_b2["K_per_fold"]
        else:
            K_b4 = args.K
            K_b2 = args.K

        per_method: dict = {}
        kcv_log: dict = {}
        if args.K_mode == "nested-cv":
            kcv_log = {"K_b4_per_fold": K_b4, "K_b2_per_fold": K_b2}

        # B1 (no Mondrian) and B3 (class-Mondrian): no partition K, use crepes API.
        ps_b1 = chrom_loo_crepes_per_class_pred_sets(
            p, sigma, y, chroms, args.alpha, "none", args.K_eval)
        m_b1 = evaluate_pred_sets(ps_b1, y, sigma, chroms, args.alpha,
                                  f"B1_split_pi{target_pi:.2f}", n_sigma_bins=args.K_eval)
        ps_b3 = chrom_loo_crepes_per_class_pred_sets(
            p, sigma, y, chroms, args.alpha, "class", args.K_eval)
        m_b3 = evaluate_pred_sets(ps_b3, y, sigma, chroms, args.alpha,
                                  f"B3_class_pi{target_pi:.2f}", n_sigma_bins=args.K_eval)

        # B2 sigma-Mondrian, B4 HCCP: use nkv (partition K via nested CV or fixed).
        in0, in1, _ = nkv.chrom_loo_predict(p, sigma, y, chroms, K_b2, "b2_sigma", alpha=args.alpha)
        m_b2 = evaluate_pred_sets(in0_in1_to_pred_sets(in0, in1), y, sigma, chroms,
                                  args.alpha, f"B2_sigma_pi{target_pi:.2f}",
                                  n_sigma_bins=args.K_eval)
        in0, in1, _ = nkv.chrom_loo_predict(p, sigma, y, chroms, K_b4, "b4_hccp", alpha=args.alpha)
        m_b4 = evaluate_pred_sets(in0_in1_to_pred_sets(in0, in1), y, sigma, chroms,
                                  args.alpha, f"B4_HCCP_pi{target_pi:.2f}",
                                  n_sigma_bins=args.K_eval)

        for method_name, metrics in (("B1_split", m_b1), ("B2_sigma", m_b2),
                                     ("B3_class", m_b3), ("B4_HCCP", m_b4)):
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
            "K_selection": kcv_log,
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

    method_names = ("B1_split", "B2_sigma", "B3_class", "B4_HCCP")
    print(f"\n{'='*70}")
    print(f"{'method':<10} | " + "  ".join(f"pi={s['achieved_pi']:.2f}" for s in all_results["sweep"]))
    print(f"{'-'*70}")
    print(f"{'cov|pos':<10}")
    for m in method_names:
        line = f"{m:<10} | "
        for s in all_results["sweep"]:
            line += f"  {s['methods'][m]['coverage_pos']:.3f} "
        print(line)
    print(f"\n{'sig-gap':<10}")
    for m in method_names:
        line = f"{m:<10} | "
        for s in all_results["sweep"]:
            line += f"  {s['methods'][m]['sigma_bin_gap']:.3f} "
        print(line)


if __name__ == "__main__":
    main()
