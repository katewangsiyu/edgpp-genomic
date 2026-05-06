"""Empirical H2H comparison: HCCP vs RLCP, Weighted CP, Self-Calibrating CP.

Reads precomputed (p_hat, sigma, y, chrom) from hetero_head/scores_with_sigma.parquet
and runs four CP variants in chrom-LOO with the same nonconformity score
s(x, y) = |y - p(x)| / sigma(x):

  HCCP    : Mondrian (y x sigma-bin), K=K_cv (reproduces existing tab:h2h)
  RLCP    : random localization in (p_hat, sigma) 2D space, Gaussian kernel,
            bandwidth via Silverman's rule (Hore & Barber 2025 random-localized
            CP, simplified to deterministic kernel weights for reproducibility).
  WCP     : weighted CP with density-ratio weights from a logistic discriminator
            predicting source-vs-target chromosome (Tibshirani et al. 2019).
  SC-CP   : Self-Calibrating CP (Van der Laan 2024), binning along the
            calibrated point prediction p_hat with K=K_cv equiprobable bins.

K-selection modes:
  --K-mode fixed --K {int}  use a preset K for all outer folds (legacy).
  --K-mode nested-cv        proper nested chrom-LOO CV. For each outer fold
                            c_outer, run inner chrom-LOO on (chroms - c_outer)
                            to select K(c_outer) = argmin_K mean inner-fold
                            worst-cell gap. K is a deterministic function of
                            the calibration set alone, so CP exchangeability
                            holds. Fixes the test-set-tuning leak in the
                            previous K_cv = argmin chrom-LOO test-gap protocol.

Usage:
    python T_tools/cp_baselines_h2h.py --dataset complex --K-mode fixed --K 2
    python T_tools/cp_baselines_h2h.py --dataset mendelian --K-mode nested-cv \
        --K-grid 2,3,5,8,10,15,20
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
ALPHA = 0.10
EPS = 1e-6


def load_scores(dataset: str) -> pd.DataFrame:
    if dataset == "open_targets":
        # OT-native scores from scripts/42_open_targets_train_scores.py.
        # Schema matches TraitGym (chrom, p_hat, sigma, label) plus pass-through
        # id columns ignored by the H2H pipeline.
        path = REPO / "outputs" / "hetero_head" / "open_targets" / "scores_with_sigma.parquet"
    elif dataset in ("complex", "mendelian"):
        suffix = dataset
        path = REPO / "outputs" / "hetero_head" / f"CADD+GPN-MSA+Borzoi_{suffix}_abs" / "scores_with_sigma.parquet"
    else:
        raise ValueError(f"unknown --dataset {dataset!r}")
    df = pd.read_parquet(path)
    df["chrom"] = df["chrom"].astype(str)
    return df


def coverage_metrics(in_set_0, in_set_1, y, sigma, K, bin_idx, label="m", chroms=None):
    covered = np.where(y == 0, in_set_0, in_set_1)
    marg = float(covered.mean())
    cov_y1 = float(covered[y == 1].mean()) if (y == 1).any() else np.nan
    cov_y0 = float(covered[y == 0].mean()) if (y == 0).any() else np.nan

    cell_gaps = []
    for k in (0, 1):
        for b in range(K):
            mask = (y == k) & (bin_idx == b)
            if mask.sum() >= 5:
                cov_kb = float(covered[mask].mean())
                cell_gaps.append(abs(cov_kb - (1 - ALPHA)))
    sigma_bin_gap = float(max(cell_gaps)) if cell_gaps else float("nan")

    chrom_gap = float("nan")
    if chroms is not None:
        chrom_covs = []
        for c in sorted(set(chroms)):
            m = chroms == c
            if m.sum() >= 5:
                chrom_covs.append(float(covered[m].mean()))
        if chrom_covs:
            chrom_gap = float(max(chrom_covs) - min(chrom_covs))

    return dict(label=label, marginal_coverage=marg, coverage_pos=cov_y1,
                coverage_neg=cov_y0, sigma_bin_gap=sigma_bin_gap,
                per_chrom_gap=chrom_gap,
                n_test=int(len(y)), frac_singleton=float((in_set_0 ^ in_set_1).mean()))


def _hccp_calibrate_predict_one(p_cal, sigma_cal, y_cal, p_te, sigma_te, K: int, alpha: float):
    """One Mondrian (y x sigma-bin) calibrate-and-predict step.

    Bin edges are calibration-set quantiles (1..K-1 inner edges via searchsorted).
    Cell quantiles fall back to pooled-class quantile if cell is too small.
    """
    s_cal = np.abs(y_cal - p_cal) / sigma_cal
    edges = np.quantile(sigma_cal, np.linspace(0, 1, K + 1)[1:-1])
    b_cal = np.searchsorted(edges, sigma_cal)
    b_te = np.searchsorted(edges, sigma_te)

    n_te = len(p_te)
    in0 = np.zeros(n_te, dtype=bool)
    in1 = np.zeros(n_te, dtype=bool)

    for k in (0, 1):
        for b in range(K):
            cell = (y_cal == k) & (b_cal == b)
            n_kb = int(cell.sum())
            if n_kb >= int(np.ceil(1.0 / alpha)):
                qhat = np.quantile(s_cal[cell], np.ceil((n_kb + 1) * (1 - alpha)) / n_kb)
            else:
                cell_p = (y_cal == k)
                n_k = int(cell_p.sum())
                if n_k == 0:
                    qhat = np.inf
                else:
                    qhat = np.quantile(s_cal[cell_p], np.ceil((n_k + 1) * (1 - alpha)) / n_k)
            te_cell_mask = (b_te == b)
            if k == 0:
                in0[te_cell_mask] = np.abs(0 - p_te[te_cell_mask]) <= qhat * sigma_te[te_cell_mask]
            else:
                in1[te_cell_mask] = np.abs(1 - p_te[te_cell_mask]) <= qhat * sigma_te[te_cell_mask]
    return in0, in1, b_te


def _worst_cell_gap_partition(in0, in1, y, b_te, K: int, alpha: float, min_cell: int = 5):
    """Worst (over (k, b) cells) absolute coverage gap relative to (1 - alpha)."""
    covered = np.where(y == 0, in0, in1)
    cell_gaps = []
    for k in (0, 1):
        for b in range(K):
            mask = (y == k) & (b_te == b)
            if mask.sum() >= min_cell:
                cell_gaps.append(abs(float(covered[mask].mean()) - (1 - alpha)))
    return float(max(cell_gaps)) if cell_gaps else float("nan")


def select_K_hccp_nested_cv(df: pd.DataFrame, K_grid: list[int], alpha: float = ALPHA,
                            K_eval: int = 10) -> dict:
    """Nested chrom-LOO K-selection for HCCP partition, scored on K_eval-fair metric.

    For each outer chrom c_outer, run inner chrom-LOO on (chroms - c_outer)
    and pick K(c_outer) = argmin_K mean inner-fold worst-cell gap. The inner-fold
    gap is computed on the K_eval sigma-bin partition (NOT on K_part itself),
    so that different K candidates are scored on a comparable, K-independent
    metric. Otherwise small K_part would be artificially favoured (fewer cells
    => smaller worst-cell gap by definition).

    K is a deterministic function of the outer calibration set alone, preserving
    CP exchangeability under chrom-group A1.
    """
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values

    K_per_fold: dict[str, int] = {}
    inner_score_log: dict[str, dict[int, float]] = {}

    for c_outer in sorted(set(chroms)):
        mask_outer_cal = chroms != c_outer
        cal_chroms = sorted(set(chroms[mask_outer_cal]))

        K_score = {K: [] for K in K_grid}
        for K in K_grid:
            for c_inner in cal_chroms:
                mask_inner_test = mask_outer_cal & (chroms == c_inner)
                mask_inner_cal = mask_outer_cal & (chroms != c_inner)
                if mask_inner_cal.sum() < K * 4 or mask_inner_test.sum() < 5:
                    continue
                in0, in1, _b_part = _hccp_calibrate_predict_one(
                    p[mask_inner_cal], sigma[mask_inner_cal], y[mask_inner_cal],
                    p[mask_inner_test], sigma[mask_inner_test], K, alpha,
                )
                # K_eval-fair scoring: build sigma-bin partition at K_eval using
                # inner-cal sigma quantiles, place inner-test sigmas in bins,
                # compute worst (k, b) cell gap.
                eval_edges = np.quantile(sigma[mask_inner_cal],
                                         np.linspace(0, 1, K_eval + 1)[1:-1])
                b_eval = np.searchsorted(eval_edges, sigma[mask_inner_test])
                gap = _worst_cell_gap_partition(in0, in1, y[mask_inner_test],
                                                b_eval, K_eval, alpha)
                if not np.isnan(gap):
                    K_score[K].append(gap)
        K_avg = {K: (float(np.mean(v)) if v else float("inf")) for K, v in K_score.items()}
        K_per_fold[c_outer] = int(min(K_avg, key=K_avg.get))
        inner_score_log[c_outer] = K_avg

    return {"K_per_fold": K_per_fold, "inner_avg_gap": inner_score_log}


def hccp_chrom_loo(df: pd.DataFrame, K, alpha: float = ALPHA):
    """HCCP chrom-LOO. K may be int (fixed) or dict[str, int] (per-outer-fold)."""
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values

    in0 = np.zeros(len(df), dtype=bool)
    in1 = np.zeros(len(df), dtype=bool)
    bin_idx_all = np.zeros(len(df), dtype=int)

    for c in sorted(set(chroms)):
        mask_te = chroms == c
        mask_cal = ~mask_te
        K_c = K[c] if isinstance(K, dict) else int(K)

        in0_te, in1_te, b_te = _hccp_calibrate_predict_one(
            p[mask_cal], sigma[mask_cal], y[mask_cal],
            p[mask_te], sigma[mask_te], K_c, alpha,
        )
        in0[mask_te] = in0_te
        in1[mask_te] = in1_te
        bin_idx_all[mask_te] = b_te

    return in0, in1, bin_idx_all


def rlcp_chrom_loo(df: pd.DataFrame, K: int, alpha: float = ALPHA, h: float | None = None):
    """Random/kernel-localized CP in (p_hat, sigma) 2D space.

    For each test point, weight calibration scores by w_i = exp(-||z_test - z_i||^2 / 2h^2)
    where z = (p_hat, sigma), and use the weighted (1-alpha)-quantile as the threshold.
    Bandwidth via Silverman's rule on calibration z.
    """
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values
    z = np.column_stack([p, sigma])

    in0 = np.zeros(len(df), dtype=bool)
    in1 = np.zeros(len(df), dtype=bool)
    bin_idx_all = np.zeros(len(df), dtype=int)  # placeholder for sigma-bin metric

    for c in sorted(set(chroms)):
        mask_te = chroms == c
        mask_cal = ~mask_te
        s_cal = np.abs(y[mask_cal] - p[mask_cal]) / sigma[mask_cal]
        z_cal = z[mask_cal]
        z_te = z[mask_te]

        edges = np.quantile(sigma[mask_cal], np.linspace(0, 1, K + 1)[1:-1])
        bin_idx_all[mask_te] = np.searchsorted(edges, sigma[mask_te])

        # Silverman bandwidth on z_cal (per-coord then mean)
        n_cal = len(z_cal)
        h_use = h if h is not None else np.mean(np.std(z_cal, axis=0)) * (4.0 / (3.0 * n_cal)) ** 0.2

        # For each test point, compute kernel-weighted (1-alpha)-quantile of s_cal
        for j_te, idx in enumerate(np.where(mask_te)[0]):
            d2 = np.sum((z_cal - z_te[j_te]) ** 2, axis=1)
            w = np.exp(-d2 / (2 * h_use ** 2))
            # Weighted quantile (Hore-Barber adjustment: include test point with weight 1)
            order = np.argsort(s_cal)
            w_sorted = w[order]
            s_sorted = s_cal[order]
            cum = np.cumsum(w_sorted) / (w_sorted.sum() + 1.0)
            target = 1 - alpha
            qi = np.searchsorted(cum, target)
            qi = min(qi, len(s_sorted) - 1)
            qhat = s_sorted[qi]
            in0[idx] = abs(0 - p[idx]) <= qhat * sigma[idx]
            in1[idx] = abs(1 - p[idx]) <= qhat * sigma[idx]
    return in0, in1, bin_idx_all


def weighted_cp_chrom_loo(df: pd.DataFrame, K: int, alpha: float = ALPHA):
    """Weighted CP with density-ratio weights via logistic discriminator
    predicting test-chrom (target) vs other chroms (source).
    """
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values
    Z = np.column_stack([p, sigma])

    in0 = np.zeros(len(df), dtype=bool)
    in1 = np.zeros(len(df), dtype=bool)
    bin_idx_all = np.zeros(len(df), dtype=int)

    for c in sorted(set(chroms)):
        mask_te = chroms == c
        mask_cal = ~mask_te
        s_cal = np.abs(y[mask_cal] - p[mask_cal]) / sigma[mask_cal]

        edges = np.quantile(sigma[mask_cal], np.linspace(0, 1, K + 1)[1:-1])
        bin_idx_all[mask_te] = np.searchsorted(edges, sigma[mask_te])

        # Discriminator: target=test chrom (label 1), source=other chroms (label 0)
        y_disc = mask_te.astype(int)
        clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=42)
        clf.fit(Z, y_disc)
        # Likelihood ratio = p(target|z) / p(source|z); on calibration set
        prob_cal = clf.predict_proba(Z[mask_cal])[:, 1]
        prob_cal = np.clip(prob_cal, 0.05, 0.95)
        w_cal = prob_cal / (1.0 - prob_cal)

        # Weighted (1-alpha) quantile of s_cal w.r.t. w_cal
        order = np.argsort(s_cal)
        w_sorted = w_cal[order]
        s_sorted = s_cal[order]
        cum = np.cumsum(w_sorted) / (w_sorted.sum() + 1.0)
        qi = np.searchsorted(cum, 1 - alpha)
        qi = min(qi, len(s_sorted) - 1)
        qhat = s_sorted[qi]

        in0[mask_te] = np.abs(0 - p[mask_te]) <= qhat * sigma[mask_te]
        in1[mask_te] = np.abs(1 - p[mask_te]) <= qhat * sigma[mask_te]
    return in0, in1, bin_idx_all


def _sccp_calibrate_predict_one(p_cal, sigma_cal, y_cal, p_te, sigma_te, K_part: int,
                                 K_eval: int, alpha: float):
    """SC-CP one fold: bin along p_hat with K_part equiprobable bins; return
    sigma-bin index at K_eval for reporting metric."""
    s_cal = np.abs(y_cal - p_cal) / sigma_cal

    p_edges = np.quantile(p_cal, np.linspace(0, 1, K_part + 1)[1:-1])
    b_cal = np.searchsorted(p_edges, p_cal)
    b_te = np.searchsorted(p_edges, p_te)

    sig_edges = np.quantile(sigma_cal, np.linspace(0, 1, K_eval + 1)[1:-1])
    sig_b_te = np.searchsorted(sig_edges, sigma_te)

    n_te = len(p_te)
    in0 = np.zeros(n_te, dtype=bool)
    in1 = np.zeros(n_te, dtype=bool)

    for b in range(K_part):
        cell_cal = (b_cal == b)
        n_b = int(cell_cal.sum())
        if n_b >= int(np.ceil(1.0 / alpha)):
            qhat = np.quantile(s_cal[cell_cal], np.ceil((n_b + 1) * (1 - alpha)) / n_b)
        else:
            qhat = np.quantile(s_cal, np.ceil((len(s_cal) + 1) * (1 - alpha)) / len(s_cal))
        te_cell = (b_te == b)
        in0[te_cell] = np.abs(0 - p_te[te_cell]) <= qhat * sigma_te[te_cell]
        in1[te_cell] = np.abs(1 - p_te[te_cell]) <= qhat * sigma_te[te_cell]
    return in0, in1, sig_b_te, b_te


def select_K_sccp_nested_cv(df: pd.DataFrame, K_grid: list[int], alpha: float = ALPHA,
                            K_eval: int = 10) -> dict:
    """Nested chrom-LOO K-selection for SC-CP, scored on K_eval sigma-bin metric.

    Uses the same K_eval-fair scoring as select_K_hccp_nested_cv: the candidate
    K_part determines the calibration partition (along p_hat axis), but the
    inner-fold worst-cell gap is measured on the K_eval sigma-bin axis (the
    cross-method reporting metric). This gives SC-CP a fair shot at the metric
    HCCP optimises for.
    """
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values

    K_per_fold: dict[str, int] = {}
    inner_score_log: dict[str, dict[int, float]] = {}

    for c_outer in sorted(set(chroms)):
        mask_outer_cal = chroms != c_outer
        cal_chroms = sorted(set(chroms[mask_outer_cal]))

        K_score = {K: [] for K in K_grid}
        for K in K_grid:
            for c_inner in cal_chroms:
                mask_inner_test = mask_outer_cal & (chroms == c_inner)
                mask_inner_cal = mask_outer_cal & (chroms != c_inner)
                if mask_inner_cal.sum() < K * 4 or mask_inner_test.sum() < 5:
                    continue
                in0, in1, sig_b_eval, _b_p = _sccp_calibrate_predict_one(
                    p[mask_inner_cal], sigma[mask_inner_cal], y[mask_inner_cal],
                    p[mask_inner_test], sigma[mask_inner_test], K, K_eval, alpha,
                )
                # Score on K_eval sigma-bin axis (cross-method-fair).
                gap = _worst_cell_gap_partition(in0, in1, y[mask_inner_test],
                                                sig_b_eval, K_eval, alpha)
                if not np.isnan(gap):
                    K_score[K].append(gap)
        K_avg = {K: (float(np.mean(v)) if v else float("inf")) for K, v in K_score.items()}
        K_per_fold[c_outer] = int(min(K_avg, key=K_avg.get))
        inner_score_log[c_outer] = K_avg

    return {"K_per_fold": K_per_fold, "inner_avg_gap": inner_score_log}


def sccp_chrom_loo(df: pd.DataFrame, K, alpha: float = ALPHA, K_eval: int | None = None):
    """SC-CP chrom-LOO. K may be int or dict[str, int] (per-outer-fold partition).

    K_eval is the sigma-bin count used for the reporting metric (defaults to
    fixed metric K = 10 when partition K is dict; otherwise matches partition K
    for backward compatibility).
    """
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values

    in0 = np.zeros(len(df), dtype=bool)
    in1 = np.zeros(len(df), dtype=bool)
    bin_idx_all = np.zeros(len(df), dtype=int)

    for c in sorted(set(chroms)):
        mask_te = chroms == c
        mask_cal = ~mask_te
        K_c = K[c] if isinstance(K, dict) else int(K)
        K_e = K_eval if K_eval is not None else K_c

        in0_te, in1_te, sig_b_te, _b_te = _sccp_calibrate_predict_one(
            p[mask_cal], sigma[mask_cal], y[mask_cal],
            p[mask_te], sigma[mask_te], K_c, K_e, alpha,
        )
        in0[mask_te] = in0_te
        in1[mask_te] = in1_te
        bin_idx_all[mask_te] = sig_b_te

    return in0, in1, bin_idx_all


def compute_metric_bin_idx(sigma: np.ndarray, chroms: np.ndarray, K_eval: int) -> np.ndarray:
    """Per-outer-fold K_eval sigma-bin index for cross-method-fair metric reporting.

    Each outer fold's calibration sigma is used to define K_eval equiprobable
    quantile bins; test sigmas are placed in those bins. This decouples the
    reporting metric from each method's partition K, enabling fair head-to-head
    sigma-bin gap comparisons (HCCP / SC-CP partition K may differ per fold).
    """
    bin_idx = np.zeros(len(sigma), dtype=int)
    for c in sorted(set(chroms)):
        mask_te = chroms == c
        mask_cal = ~mask_te
        edges = np.quantile(sigma[mask_cal], np.linspace(0, 1, K_eval + 1)[1:-1])
        bin_idx[mask_te] = np.searchsorted(edges, sigma[mask_te])
    return bin_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    choices=["mendelian", "complex", "open_targets"],
                    required=True)
    ap.add_argument("--K-mode", choices=["fixed", "nested-cv"], default="fixed",
                    help="K-selection protocol. fixed: legacy (CLI --K). "
                         "nested-cv: proper inner-CV K per outer fold.")
    ap.add_argument("--K", type=int, default=5,
                    help="Fixed K (used when --K-mode fixed)")
    ap.add_argument("--K-grid", type=str, default="2,3,5,8,10,15,20",
                    help="K candidates for nested CV")
    ap.add_argument("--K-eval", type=int, default=10,
                    help="Sigma-bin count for metric reporting (cross-method fair)")
    args = ap.parse_args()

    df = load_scores(args.dataset)
    chroms = df["chrom"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values
    K_eval = args.K_eval

    if args.K_mode == "fixed":
        K_hccp: int | dict = args.K
        K_sccp: int | dict = args.K
        K_label = f"fixed_K{args.K}"
        kcv_log: dict = {}
        print(f"Loaded {args.dataset}: {len(df)} variants, mode=fixed K={args.K}, K_eval={K_eval}")
    else:
        K_grid = [int(s) for s in args.K_grid.split(",")]
        print(f"Loaded {args.dataset}: {len(df)} variants, mode=nested-cv "
              f"K_grid={K_grid}, K_eval={K_eval}")
        print(f"\n[nested-cv] Selecting K_HCCP per outer fold (K_eval={K_eval} fair scoring)...")
        hccp_sel = select_K_hccp_nested_cv(df, K_grid, K_eval=K_eval)
        K_hccp = hccp_sel["K_per_fold"]
        print(f"  K_HCCP per fold: {K_hccp}")
        print(f"  K_HCCP mode = {pd.Series(list(K_hccp.values())).mode().iloc[0]}")
        print(f"\n[nested-cv] Selecting K_SCCP per outer fold (K_eval={K_eval} fair scoring)...")
        sccp_sel = select_K_sccp_nested_cv(df, K_grid, K_eval=K_eval)
        K_sccp = sccp_sel["K_per_fold"]
        print(f"  K_SCCP per fold: {K_sccp}")
        print(f"  K_SCCP mode = {pd.Series(list(K_sccp.values())).mode().iloc[0]}")
        K_label = f"nestedcv_Keval{K_eval}"
        kcv_log = {
            "K_grid": K_grid,
            "K_hccp_per_fold": K_hccp,
            "K_sccp_per_fold": K_sccp,
            "hccp_inner_avg_gap": hccp_sel["inner_avg_gap"],
            "sccp_inner_avg_gap": sccp_sel["inner_avg_gap"],
        }

    metric_bin_idx = compute_metric_bin_idx(sigma, chroms, K_eval)
    results: dict = {"K_eval": K_eval, "K_mode": args.K_mode}

    print("\n--- HCCP (B4) ---")
    i0, i1, _ = hccp_chrom_loo(df, K_hccp)
    m = coverage_metrics(i0, i1, y, sigma, K_eval, metric_bin_idx, label="HCCP", chroms=chroms)
    print(m)
    results["HCCP"] = m

    print("\n--- RLCP (kernel localization in (p, sigma) 2D) ---")
    i0, i1, _ = rlcp_chrom_loo(df, K_eval)
    m = coverage_metrics(i0, i1, y, sigma, K_eval, metric_bin_idx, label="RLCP", chroms=chroms)
    print(m)
    results["RLCP"] = m

    print("\n--- Weighted CP (chrom-shift density ratio) ---")
    i0, i1, _ = weighted_cp_chrom_loo(df, K_eval)
    m = coverage_metrics(i0, i1, y, sigma, K_eval, metric_bin_idx, label="WeightedCP", chroms=chroms)
    print(m)
    results["WeightedCP"] = m

    print("\n--- Self-Calibrating CP (p_hat-bin Mondrian) ---")
    i0, i1, _ = sccp_chrom_loo(df, K_sccp, K_eval=K_eval)
    m = coverage_metrics(i0, i1, y, sigma, K_eval, metric_bin_idx, label="SCCP", chroms=chroms)
    print(m)
    results["SCCP"] = m

    if kcv_log:
        results["K_selection_log"] = kcv_log

    out_dir = REPO / "R_raw" / "cp_baselines_h2h"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{K_label}.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {out_path}")
    print("\n=== summary ===")
    print(f"{'method':<14} {'cov':>7} {'cov|+':>7} {'sig-gap':>9} {'chr-gap':>9} {'singleton':>10}")
    for key in ("HCCP", "RLCP", "WeightedCP", "SCCP"):
        r = results[key]
        print(f"{r['label']:<14} {r['marginal_coverage']:>7.4f} {r['coverage_pos']:>7.4f} "
              f"{r['sigma_bin_gap']:>9.4f} {r['per_chrom_gap']:>9.4f} {r['frac_singleton']:>10.3f}")


if __name__ == "__main__":
    main()
