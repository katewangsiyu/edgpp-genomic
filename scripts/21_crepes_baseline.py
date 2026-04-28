"""crepes off-the-shelf normalized + Mondrian baseline for HCCP H2H comparison.

Goal: address Reviewer-level objection "this is just normalized Mondrian CP from
Bostrom 2020 / crepes 2022". We run crepes' ConformalRegressor with the same
(p_hat, sigma) we feed HCCP, and compare:

  B1: crepes normalized split CP        (no Mondrian)
  B2: crepes normalized + sigma-Mondrian (Bostrom 2020 default; bins along sigma only)
  B3: crepes normalized + class-Mondrian (Mondrian by label only)
  B4: HCCP (class x sigma-bin Mondrian)

K-selection modes:
  --K-mode fixed --K {int}   legacy fixed K for all outer folds.
  --K-mode nested-cv         proper nested chrom-LOO CV for B2 (sigma-Mondrian)
                             and B4 (HCCP) partition K, scored on K_eval-fair
                             sigma-bin metric. B1 (no Mondrian) and B3
                             (class-only Mondrian) do not use a partition K, so
                             nested CV does not apply.

Usage:
    conda run -n edgpp_t4 --no-capture-output python scripts/21_crepes_baseline.py \
        --sigma-scores outputs/hetero_head/CADD+GPN-MSA+Borzoi_mendelian_abs/scores_with_sigma.parquet \
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \
        --out-dir outputs/crepes_baseline/CADD+GPN-MSA+Borzoi_mendelian_abs \
        --alpha 0.10 --K-mode nested-cv --K-grid 2,3,5,8,10,15,20 --K-eval 3
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from crepes import ConformalRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "T_tools"))
import nested_kcv_helpers as nkv


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
    """Evaluate prediction sets. sigma_bin_gap is the (y, sigma-bin)-conditional
    worst-cell |cov - (1-alpha)|, matching the metric used in cp_baselines_h2h.py
    and paper hero claim. The per-outer-fold sigma-bin index is computed via
    nkv.compute_metric_bin_idx for cross-method-fair reporting."""
    n = len(y)
    covered = np.array([y[i] in pred_sets[i] for i in range(n)])
    sizes = np.array([len(ps) for ps in pred_sets])

    pos = y == 1
    neg = y == 0
    cov_pos = float(covered[pos].mean()) if pos.any() else float("nan")
    cov_neg = float(covered[neg].mean()) if neg.any() else float("nan")

    bin_idx = nkv.compute_metric_bin_idx(sigma, chroms, n_sigma_bins)
    cell_gaps: list[float] = []
    bin_covs: list[float] = []
    for k in (0, 1):
        for b in range(n_sigma_bins):
            mask = (y == k) & (bin_idx == b)
            if mask.sum() >= 5:
                cov_kb = float(covered[mask].mean())
                cell_gaps.append(abs(cov_kb - (1 - alpha)))
    for b in range(n_sigma_bins):
        m = bin_idx == b
        if m.sum() >= 5:
            bin_covs.append(float(covered[m].mean()))
    sigma_gap = float(max(cell_gaps)) if cell_gaps else float("nan")
    sigma_marg_gap = float(max(bin_covs) - min(bin_covs)) if bin_covs else float("nan")

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
    print(f"  sigma-bin gap (y-conditional, K_eval={n_sigma_bins}) = {sigma_gap:.4f}")
    print(f"  sigma-bin gap (marginal-over-y, K_eval={n_sigma_bins}) = {sigma_marg_gap:.4f}")
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
        "sigma_bin_gap_marg_over_y": sigma_marg_gap,
        "per_chrom_gap": chrom_gap,
        "sigma_bin_coverage": bin_covs,
    }


def in0_in1_to_pred_sets(in0: np.ndarray, in1: np.ndarray) -> list:
    """Convert (in0, in1) bool arrays to list of label sets."""
    out = [set() for _ in range(len(in0))]
    for i in range(len(in0)):
        if in0[i]:
            out[i].add(0)
        if in1[i]:
            out[i].add(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-scores", required=True)
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--K-mode", choices=["fixed", "nested-cv"], default="fixed")
    ap.add_argument("--K", type=int, default=5,
                    help="Fixed K for B2/B4 (used when --K-mode fixed)")
    ap.add_argument("--K-grid", type=str, default="2,3,5,8,10,15,20",
                    help="K candidates for nested CV")
    ap.add_argument("--K-eval", type=int, default=5,
                    help="sigma-bin count for metric reporting and inner-fold scoring")
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

    print(f"[load] n={len(V)} pos={int(y.sum())} alpha={args.alpha} "
          f"K-mode={args.K_mode} K_eval={args.K_eval}")

    K_b2: int | dict
    K_b4: int | dict
    kcv_log: dict = {}
    if args.K_mode == "fixed":
        K_b2 = args.K
        K_b4 = args.K
        K_label = f"fixed_K{args.K}_Keval{args.K_eval}"
    else:
        K_grid = [int(s) for s in args.K_grid.split(",")]
        print(f"\n[nested-cv] selecting K_B4 (HCCP) on K_eval={args.K_eval} fair score...")
        sel_b4 = nkv.select_K_nested_chrom_loo(p, sigma, y, chroms, K_grid,
                                                method="b4_hccp", K_eval=args.K_eval,
                                                alpha=args.alpha)
        K_b4 = sel_b4["K_per_fold"]
        print(f"  K_B4 mode = {pd.Series(list(K_b4.values())).mode().iloc[0]}")
        print(f"  K_B4 per-fold: {K_b4}")
        print(f"\n[nested-cv] selecting K_B2 (sigma-Mondrian) on K_eval={args.K_eval} fair score...")
        sel_b2 = nkv.select_K_nested_chrom_loo(p, sigma, y, chroms, K_grid,
                                                method="b2_sigma", K_eval=args.K_eval,
                                                alpha=args.alpha)
        K_b2 = sel_b2["K_per_fold"]
        print(f"  K_B2 mode = {pd.Series(list(K_b2.values())).mode().iloc[0]}")
        print(f"  K_B2 per-fold: {K_b2}")
        K_label = f"nestedcv_Keval{args.K_eval}"
        kcv_log = {
            "K_grid": K_grid,
            "K_b4_per_fold": K_b4,
            "K_b2_per_fold": K_b2,
            "b4_inner_avg_gap": sel_b4["inner_avg_gap"],
            "b2_inner_avg_gap": sel_b2["inner_avg_gap"],
        }

    results = {"alpha": args.alpha, "K_mode": args.K_mode, "K_eval": args.K_eval,
               "n": int(len(V))}

    # B1 (no Mondrian) and B3 (class-only) do not use partition K. K passed to
    # crepes here is just the API placeholder; they do not consume it.
    ps_b1 = chrom_loo_crepes_per_class_pred_sets(p, sigma, y, chroms, args.alpha, "none", args.K_eval)
    results["B1_crepes_normalized_split"] = evaluate_pred_sets(
        ps_b1, y, sigma, chroms, args.alpha,
        "B1: crepes normalized split (no Mondrian)", n_sigma_bins=args.K_eval)

    ps_b3 = chrom_loo_crepes_per_class_pred_sets(p, sigma, y, chroms, args.alpha, "class", args.K_eval)
    results["B3_crepes_class_mondrian"] = evaluate_pred_sets(
        ps_b3, y, sigma, chroms, args.alpha,
        "B3: crepes normalized + class-Mondrian", n_sigma_bins=args.K_eval)

    # B2 / B4 use the nested-cv-selected K_per_fold (or fixed K).
    in0, in1, _ = nkv.chrom_loo_predict(p, sigma, y, chroms, K_b2, "b2_sigma", alpha=args.alpha)
    ps_b2 = in0_in1_to_pred_sets(in0, in1)
    results["B2_crepes_sigma_mondrian"] = evaluate_pred_sets(
        ps_b2, y, sigma, chroms, args.alpha,
        f"B2: sigma-Mondrian (K-mode {args.K_mode})", n_sigma_bins=args.K_eval)

    in0, in1, _ = nkv.chrom_loo_predict(p, sigma, y, chroms, K_b4, "b4_hccp", alpha=args.alpha)
    ps_b4 = in0_in1_to_pred_sets(in0, in1)
    results["B4_HCCP_via_crepes"] = evaluate_pred_sets(
        ps_b4, y, sigma, chroms, args.alpha,
        f"B4: HCCP class x sigma-bin Mondrian (K-mode {args.K_mode})",
        n_sigma_bins=args.K_eval)

    if kcv_log:
        results["K_selection_log"] = kcv_log

    (out / f"crepes_baseline_{K_label}.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved: {out}/crepes_baseline_{K_label}.json")

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
