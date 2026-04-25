"""crepes off-the-shelf normalized + Mondrian baseline for HCCP H2H comparison.

Goal: address Reviewer-level objection "this is just normalized Mondrian CP from
Bostrom 2020 / crepes 2022". We run crepes' ConformalRegressor with the same
(p_hat, sigma) we feed HCCP, and compare:

  B1: crepes normalized split CP        (no Mondrian)
  B2: crepes normalized + sigma-Mondrian (Bostrom 2020 default; bins along sigma only)
  B3: crepes normalized + class-Mondrian (Mondrian by label only)
  B4: HCCP (class x sigma-bin Mondrian, K=K_CV)         <-- already in 14_conformal_hetero.py

We rerun B4 here so all four numbers come out of the same pipeline + parquet for fair
comparison.

Usage:
    conda run -n edgpp_t4 --no-capture-output python scripts/21_crepes_baseline.py \
        --sigma-scores outputs/hetero_head/CADD+GPN-MSA+Borzoi_mendelian_abs/scores_with_sigma.parquet \
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \
        --out-dir outputs/crepes_baseline/CADD+GPN-MSA+Borzoi_mendelian_abs \
        --alpha 0.10 --K 3
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from crepes import ConformalRegressor


def chrom_loo_crepes(
    p: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    chroms: np.ndarray,
    alpha: float,
    mondrian_mode: str,  # "none" | "sigma" | "class" | "class_sigma"
    K: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Run crepes ConformalRegressor in chrom-LOO with optional Mondrian categorizer.

    Returns (lo, hi) prediction interval per variant. We then convert intervals
    to label sets via {k in {0,1} : lo <= k <= hi}.

    crepes API contract (v0.9):
      cr = ConformalRegressor()
      cr.fit(residuals=cal_residuals, sigmas=cal_sigmas, bins=cal_bins)
      lo, hi = cr.predict(y_hat=test_p, sigmas=test_sigmas, bins=test_bins,
                          confidence=1-alpha, lower_percentiles=..., higher_percentiles=...).T
    For normalized + Mondrian, fit accepts `bins` (categorical id per cal sample)
    and predict accepts `bins` for test samples; per-bin quantile is used.
    """
    n = len(y)
    lo_all = np.full(n, -np.inf)
    hi_all = np.full(n, np.inf)

    # Global sigma-bin edges (calibration-fold quantiles, computed inside loop)
    for c in sorted(set(chroms)):
        mask_test = chroms == c
        mask_cal = ~mask_test

        cal_residuals = np.abs(y[mask_cal].astype(float) - p[mask_cal])
        cal_sigmas = sigma[mask_cal] + 1e-4
        test_sigmas = sigma[mask_test] + 1e-4
        test_p = p[mask_test]

        if mondrian_mode == "none":
            cal_bins = None
            test_bins = None
        elif mondrian_mode == "sigma":
            edges = np.quantile(cal_sigmas, np.linspace(0, 1, K + 1))
            edges[0] -= 1e-9
            edges[-1] += 1e-9
            cal_bins = np.digitize(cal_sigmas, edges[1:-1]).astype(int)
            test_bins = np.digitize(test_sigmas, edges[1:-1]).astype(int)
        elif mondrian_mode == "class":
            cal_bins = y[mask_cal].astype(int)
            test_bins = np.zeros(int(mask_test.sum()), dtype=int)  # placeholder; we set per-class below
        elif mondrian_mode == "class_sigma":
            edges = np.quantile(cal_sigmas, np.linspace(0, 1, K + 1))
            edges[0] -= 1e-9
            edges[-1] += 1e-9
            cal_sb = np.digitize(cal_sigmas, edges[1:-1]).astype(int)
            test_sb = np.digitize(test_sigmas, edges[1:-1]).astype(int)
            cal_bins = y[mask_cal].astype(int) * K + cal_sb
            test_bins = np.zeros(int(mask_test.sum()), dtype=int)  # placeholder
            # for class_sigma we also need test bins; we resolve them per-test-class below
            _test_sb_cache = test_sb
        else:
            raise ValueError(mondrian_mode)

        cr = ConformalRegressor()
        cr.fit(residuals=cal_residuals, sigmas=cal_sigmas, bins=cal_bins)

        if mondrian_mode in ("none", "sigma"):
            intervals = cr.predict_int(
                y_hat=test_p, sigmas=test_sigmas, bins=test_bins,
                confidence=1 - alpha,
            )
            lo_all[mask_test] = intervals[:, 0]
            hi_all[mask_test] = intervals[:, 1]
        elif mondrian_mode == "class":
            # Run predict twice with bins set to the candidate class k
            for k in (0, 1):
                bins_k = np.full(int(mask_test.sum()), k, dtype=int)
                intervals = cr.predict_int(
                    y_hat=test_p, sigmas=test_sigmas, bins=bins_k,
                    confidence=1 - alpha,
                )
                # For label k inclusion, we only care about whether k in [lo_k, hi_k].
                # Simplest: store per-class interval and combine after loop.
                if k == 0:
                    lo0 = intervals[:, 0]; hi0 = intervals[:, 1]
                else:
                    lo1 = intervals[:, 0]; hi1 = intervals[:, 1]
            # Combined interval: union of per-class intervals (since each gives the
            # CI for residual under that class hypothesis). For prediction-set
            # extraction we use per-class bins below; here we record marginalized
            # interval = (min(lo), max(hi)) for the eval.
            lo_all[mask_test] = np.minimum(lo0, lo1)
            hi_all[mask_test] = np.maximum(hi0, hi1)
        elif mondrian_mode == "class_sigma":
            # Same per-class strategy
            for k in (0, 1):
                bins_k = (k * K + _test_sb_cache).astype(int)
                intervals = cr.predict_int(
                    y_hat=test_p, sigmas=test_sigmas, bins=bins_k,
                    confidence=1 - alpha,
                )
                if k == 0:
                    lo0 = intervals[:, 0]; hi0 = intervals[:, 1]
                else:
                    lo1 = intervals[:, 0]; hi1 = intervals[:, 1]
            lo_all[mask_test] = np.minimum(lo0, lo1)
            hi_all[mask_test] = np.maximum(hi0, hi1)

    return lo_all, hi_all


def chrom_loo_crepes_per_class_pred_sets(
    p: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    chroms: np.ndarray,
    alpha: float,
    mondrian_mode: str,
    K: int = 5,
) -> list[set]:
    """Returns prediction sets {k in {0,1} : k in [lo_k, hi_k]} via per-class crepes calls.

    For each test point and each candidate label k, we ask crepes for the
    residual interval under cal-bin label k (when mondrian uses class), then
    include k iff k in [p - hi, p - lo]_residual  i.e. iff |k - p| in [lo, hi]
    where [lo, hi] is the cal-residual interval at confidence 1-alpha.

    This is the cleanest binary-classification adaptation of crepes' ConformalRegressor.
    """
    n = len(y)
    pred_sets: list[set] = [set() for _ in range(n)]

    for c in sorted(set(chroms)):
        mask_test = chroms == c
        mask_cal = ~mask_test

        cal_residuals = np.abs(y[mask_cal].astype(float) - p[mask_cal])
        cal_sigmas = sigma[mask_cal] + 1e-4
        test_sigmas = sigma[mask_test] + 1e-4
        test_p = p[mask_test]
        test_idx = np.where(mask_test)[0]

        if mondrian_mode == "sigma":
            edges = np.quantile(cal_sigmas, np.linspace(0, 1, K + 1))
            edges[0] -= 1e-9; edges[-1] += 1e-9
            cal_bins = np.digitize(cal_sigmas, edges[1:-1]).astype(int)
            test_bins_base = np.digitize(test_sigmas, edges[1:-1]).astype(int)
        elif mondrian_mode == "class":
            cal_bins = y[mask_cal].astype(int)
            test_bins_base = None
        elif mondrian_mode == "class_sigma":
            edges = np.quantile(cal_sigmas, np.linspace(0, 1, K + 1))
            edges[0] -= 1e-9; edges[-1] += 1e-9
            cal_sb = np.digitize(cal_sigmas, edges[1:-1]).astype(int)
            test_sb = np.digitize(test_sigmas, edges[1:-1]).astype(int)
            cal_bins = (y[mask_cal].astype(int) * K + cal_sb).astype(int)
            test_bins_base = test_sb
        else:
            cal_bins = None
            test_bins_base = None

        cr = ConformalRegressor()
        cr.fit(residuals=cal_residuals, sigmas=cal_sigmas, bins=cal_bins)

        for k in (0, 1):
            if mondrian_mode == "class":
                bins_k = np.full(int(mask_test.sum()), k, dtype=int)
            elif mondrian_mode == "class_sigma":
                bins_k = (k * K + test_bins_base).astype(int)
            elif mondrian_mode == "sigma":
                bins_k = test_bins_base
            else:
                bins_k = None

            intervals = cr.predict_int(
                y_hat=test_p, sigmas=test_sigmas, bins=bins_k,
                confidence=1 - alpha,
            )
            lo = intervals[:, 0]
            hi = intervals[:, 1]
            include = (k >= lo) & (k <= hi)
            for j, ti in enumerate(test_idx):
                if include[j]:
                    pred_sets[ti].add(k)

    return pred_sets


def evaluate_pred_sets(
    pred_sets: list[set],
    y: np.ndarray,
    sigma: np.ndarray,
    chroms: np.ndarray,
    alpha: float,
    label: str,
    n_sigma_bins: int = 10,
) -> dict:
    n = len(y)
    covered = np.array([y[i] in pred_sets[i] for i in range(n)])
    sizes = np.array([len(ps) for ps in pred_sets])

    pos = y == 1
    neg = y == 0
    cov_pos = float(covered[pos].mean()) if pos.any() else float("nan")
    cov_neg = float(covered[neg].mean()) if neg.any() else float("nan")

    # sigma-bin gap (the headline T3 metric)
    bins = pd.qcut(sigma, q=n_sigma_bins, labels=False, duplicates="drop")
    bin_covs = []
    for b in sorted(set(bins[~pd.isna(bins)])):
        m = bins == b
        if m.sum() < 5:
            continue
        bin_covs.append(float(covered[m].mean()))
    sigma_gap = float(max(bin_covs) - min(bin_covs)) if bin_covs else float("nan")

    # per-chrom gap
    chrom_covs = []
    for c in sorted(set(chroms)):
        m = chroms == c
        if m.sum() < 5:
            continue
        chrom_covs.append(float(covered[m].mean()))
    chrom_gap = float(max(chrom_covs) - min(chrom_covs)) if chrom_covs else float("nan")

    print(f"\n--- {label} ---")
    print(f"  marginal cov={covered.mean():.4f}  cov|pos={cov_pos:.4f}  cov|neg={cov_neg:.4f}")
    print(f"  singleton={float((sizes==1).mean()):.3f}  both={float((sizes==2).mean()):.3f}  empty={float((sizes==0).mean()):.3f}")
    print(f"  sigma-bin gap (K={n_sigma_bins}) = {sigma_gap:.4f}")
    print(f"  per-chrom gap = {chrom_gap:.4f}")

    return {
        "label": label,
        "alpha": alpha,
        "marginal_coverage": float(covered.mean()),
        "coverage_pos": cov_pos,
        "coverage_neg": cov_neg,
        "frac_singleton": float((sizes == 1).mean()),
        "frac_both": float((sizes == 2).mean()),
        "frac_empty": float((sizes == 0).mean()),
        "sigma_bin_gap": sigma_gap,
        "per_chrom_gap": chrom_gap,
        "sigma_bin_coverage": bin_covs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-scores", required=True)
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--K", type=int, default=5,
                    help="sigma-bin count for B2/B4")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    sig = pd.read_parquet(args.sigma_scores).reset_index(drop=True)
    assert len(V) == len(sig)
    assert (V["chrom"].astype(str).values == sig["chrom"].astype(str).values).all()
    assert (V["label"].astype(int).values == sig["label"].astype(int).values).all()

    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()
    p = sig["p_hat"].to_numpy()
    sigma = sig["sigma"].to_numpy()

    print(f"[load] n={len(V)} pos={int(y.sum())} alpha={args.alpha} K={args.K}")

    results = {"alpha": args.alpha, "K": args.K, "n": int(len(V))}

    # B1: crepes normalized split CP (no Mondrian)
    ps_b1 = chrom_loo_crepes_per_class_pred_sets(p, sigma, y, chroms, args.alpha, "none", args.K)
    results["B1_crepes_normalized_split"] = evaluate_pred_sets(
        ps_b1, y, sigma, chroms, args.alpha, "B1: crepes normalized split (no Mondrian)")

    # B2: crepes normalized + sigma-Mondrian (Bostrom 2020 default)
    ps_b2 = chrom_loo_crepes_per_class_pred_sets(p, sigma, y, chroms, args.alpha, "sigma", args.K)
    results["B2_crepes_sigma_mondrian"] = evaluate_pred_sets(
        ps_b2, y, sigma, chroms, args.alpha, f"B2: crepes normalized + sigma-Mondrian K={args.K}")

    # B3: crepes normalized + class-Mondrian
    ps_b3 = chrom_loo_crepes_per_class_pred_sets(p, sigma, y, chroms, args.alpha, "class", args.K)
    results["B3_crepes_class_mondrian"] = evaluate_pred_sets(
        ps_b3, y, sigma, chroms, args.alpha, "B3: crepes normalized + class-Mondrian")

    # B4: HCCP (class x sigma-bin Mondrian) via crepes API for fair comparison
    ps_b4 = chrom_loo_crepes_per_class_pred_sets(p, sigma, y, chroms, args.alpha, "class_sigma", args.K)
    results["B4_HCCP_via_crepes"] = evaluate_pred_sets(
        ps_b4, y, sigma, chroms, args.alpha, f"B4: HCCP (class x sigma-bin Mondrian) K={args.K}")

    (out / "crepes_baseline_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out}/crepes_baseline_results.json")

    # Summary table
    print(f"\n{'='*92}")
    print(f"{'method':<48} {'cov':>7} {'cov|+':>7} {'sig-gap':>9} {'chr-gap':>9}")
    print(f"{'='*92}")
    for key in ("B1_crepes_normalized_split", "B2_crepes_sigma_mondrian",
                "B3_crepes_class_mondrian", "B4_HCCP_via_crepes"):
        r = results[key]
        print(f"{r['label']:<48} {r['marginal_coverage']:>7.4f} {r['coverage_pos']:>7.4f} "
              f"{r['sigma_bin_gap']:>9.4f} {r['per_chrom_gap']:>9.4f}")
    print(f"{'='*92}")


if __name__ == "__main__":
    main()
