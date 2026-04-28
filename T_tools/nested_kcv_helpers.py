"""Shared nested chrom-LOO K-selection helpers for HCCP / Mondrian-K methods.

This is the single source of truth for the K_partition selection protocol used
across the empirical pipeline. The legacy approach (K_cv = argmin chrom-LOO
test-set worst-cell gap, in scripts/20_adaptive_K_sweep.py) selects K on the
very metric being reported, breaking CP exchangeability and inflating the
HCCP-vs-baseline gap by an order of magnitude.

Protocol (canonical nested CV, K_eval-fair scoring):

    For each outer chrom c_outer in chroms:
        cal_outer = data[chrom != c_outer]
        For each candidate K in K_grid:
            For each inner chrom c_inner in cal_outer:
                cal_inner = cal_outer[chrom != c_inner]
                test_inner = cal_outer[chrom == c_inner]
                Calibrate Mondrian conformal with K_partition = K on cal_inner,
                predict on test_inner.
                Score: worst-cell gap on the K_eval sigma-bin partition (NOT
                K_partition itself), giving K candidates a comparable, K-
                independent metric.
            inner_avg_gap[K] = mean over inner folds.
        K(c_outer) = argmin K inner_avg_gap[K].
    Outer fold: use K(c_outer) on cal_outer => predict on test_outer.

K is a deterministic function of cal_outer alone, so calibration-set
exchangeability holds; CP coverage guarantees apply on the outer fold.

Why K_eval-fair scoring matters: a candidate K_partition that is too small
trivially has fewer cells, so its self-consistent "worst cell gap" is
artificially small. Using K_eval (typically chosen for cell statistical
stability, e.g. K_eval = floor(n * pi_min / 30)) decouples K-selection from
metric granularity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-6


def _calibrate_predict_b4_hccp(p_cal, sigma_cal, y_cal, p_te, sigma_te, K, alpha):
    """B4 = HCCP: Mondrian (y x sigma-bin) at K bins. Returns (in0, in1, b_te)."""
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
            te_cell = (b_te == b)
            if k == 0:
                in0[te_cell] = np.abs(0 - p_te[te_cell]) <= qhat * sigma_te[te_cell]
            else:
                in1[te_cell] = np.abs(1 - p_te[te_cell]) <= qhat * sigma_te[te_cell]
    return in0, in1, b_te


def _calibrate_predict_b2_sigma(p_cal, sigma_cal, y_cal, p_te, sigma_te, K, alpha):
    """B2 = sigma-only Mondrian (no class split). Returns (in0, in1, b_te)."""
    s_cal = np.abs(y_cal - p_cal) / sigma_cal
    edges = np.quantile(sigma_cal, np.linspace(0, 1, K + 1)[1:-1])
    b_cal = np.searchsorted(edges, sigma_cal)
    b_te = np.searchsorted(edges, sigma_te)

    n_te = len(p_te)
    in0 = np.zeros(n_te, dtype=bool)
    in1 = np.zeros(n_te, dtype=bool)

    for b in range(K):
        cell = (b_cal == b)
        n_b = int(cell.sum())
        if n_b >= int(np.ceil(1.0 / alpha)):
            qhat = np.quantile(s_cal[cell], np.ceil((n_b + 1) * (1 - alpha)) / n_b)
        else:
            qhat = np.quantile(s_cal, np.ceil((len(s_cal) + 1) * (1 - alpha)) / len(s_cal))
        te_cell = (b_te == b)
        in0[te_cell] = np.abs(0 - p_te[te_cell]) <= qhat * sigma_te[te_cell]
        in1[te_cell] = np.abs(1 - p_te[te_cell]) <= qhat * sigma_te[te_cell]
    return in0, in1, b_te


def _calibrate_predict_sccp(p_cal, sigma_cal, y_cal, p_te, sigma_te, K, alpha):
    """SC-CP: Mondrian on p_hat-bin axis at K bins. Returns (in0, in1, sig_b_te_unused).

    Note: returned b_te is the sigma-bin index at K (placeholder); caller
    typically computes a separate K_eval-based sigma-bin index for scoring.
    """
    s_cal = np.abs(y_cal - p_cal) / sigma_cal
    p_edges = np.quantile(p_cal, np.linspace(0, 1, K + 1)[1:-1])
    b_cal = np.searchsorted(p_edges, p_cal)
    b_te = np.searchsorted(p_edges, p_te)

    n_te = len(p_te)
    in0 = np.zeros(n_te, dtype=bool)
    in1 = np.zeros(n_te, dtype=bool)

    for b in range(K):
        cell = (b_cal == b)
        n_b = int(cell.sum())
        if n_b >= int(np.ceil(1.0 / alpha)):
            qhat = np.quantile(s_cal[cell], np.ceil((n_b + 1) * (1 - alpha)) / n_b)
        else:
            qhat = np.quantile(s_cal, np.ceil((len(s_cal) + 1) * (1 - alpha)) / len(s_cal))
        te_cell = (b_te == b)
        in0[te_cell] = np.abs(0 - p_te[te_cell]) <= qhat * sigma_te[te_cell]
        in1[te_cell] = np.abs(1 - p_te[te_cell]) <= qhat * sigma_te[te_cell]
    return in0, in1, b_te


def _worst_cell_gap_y_sigma(in0, in1, y, b_eval, K_eval, alpha, min_cell=5):
    """Worst (y, sigma-bin)-cell gap on K_eval partition."""
    covered = np.where(y == 0, in0, in1)
    cell_gaps = []
    for k in (0, 1):
        for b in range(K_eval):
            mask = (y == k) & (b_eval == b)
            if mask.sum() >= min_cell:
                cell_gaps.append(abs(float(covered[mask].mean()) - (1 - alpha)))
    return float(max(cell_gaps)) if cell_gaps else float("nan")


METHOD_FNS = {
    "b4_hccp": _calibrate_predict_b4_hccp,
    "b2_sigma": _calibrate_predict_b2_sigma,
    "sccp": _calibrate_predict_sccp,
}


def select_K_nested_chrom_loo(
    p: np.ndarray, sigma: np.ndarray, y: np.ndarray, chroms: np.ndarray,
    K_grid: list[int], method: str = "b4_hccp",
    K_eval: int = 5, alpha: float = 0.10,
) -> dict:
    """Nested chrom-LOO K-selection with K_eval-fair scoring.

    Args:
        p, sigma, y, chroms: 1-D numpy arrays of equal length.
        K_grid: candidate K_partition values.
        method: which calibrate-predict function to use ("b4_hccp", "b2_sigma",
            "sccp"). All methods are scored on the K_eval sigma-bin axis for
            cross-method-fair comparison.
        K_eval: sigma-bin partition for inner-fold scoring (default 5; see
            module docstring for stability heuristic).
        alpha: target miscoverage rate.

    Returns:
        {"K_per_fold": dict[chrom -> K], "inner_avg_gap": dict[chrom -> {K: gap}]}
    """
    if method not in METHOD_FNS:
        raise ValueError(f"unknown method {method!r}; use one of {list(METHOD_FNS)}")
    calib_fn = METHOD_FNS[method]
    sigma_safe = sigma + EPS

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
                in0, in1, _b = calib_fn(
                    p[mask_inner_cal], sigma_safe[mask_inner_cal], y[mask_inner_cal],
                    p[mask_inner_test], sigma_safe[mask_inner_test], K, alpha,
                )
                eval_edges = np.quantile(sigma_safe[mask_inner_cal],
                                         np.linspace(0, 1, K_eval + 1)[1:-1])
                b_eval = np.searchsorted(eval_edges, sigma_safe[mask_inner_test])
                gap = _worst_cell_gap_y_sigma(in0, in1, y[mask_inner_test],
                                              b_eval, K_eval, alpha)
                if not np.isnan(gap):
                    K_score[K].append(gap)
        K_avg = {K: (float(np.mean(v)) if v else float("inf")) for K, v in K_score.items()}
        K_per_fold[str(c_outer)] = int(min(K_avg, key=K_avg.get))
        inner_score_log[str(c_outer)] = K_avg

    return {"K_per_fold": K_per_fold, "inner_avg_gap": inner_score_log}


def compute_metric_bin_idx(sigma: np.ndarray, chroms: np.ndarray, K_eval: int) -> np.ndarray:
    """Per-outer-fold K_eval sigma-bin index for cross-method-fair metric reporting.

    Each outer fold's calibration sigma is used to define K_eval equiprobable
    quantile bins; test sigmas are placed in those bins. Decouples reporting
    metric from per-method partition K.
    """
    sigma_safe = sigma + EPS
    bin_idx = np.zeros(len(sigma_safe), dtype=int)
    for c in sorted(set(chroms)):
        mask_te = chroms == c
        mask_cal = ~mask_te
        edges = np.quantile(sigma_safe[mask_cal], np.linspace(0, 1, K_eval + 1)[1:-1])
        bin_idx[mask_te] = np.searchsorted(edges, sigma_safe[mask_te])
    return bin_idx


def stable_K_eval(n: int, pi_min: float, min_per_cell: int = 30) -> int:
    """Heuristic K_eval = floor(n * pi_min / min_per_cell), clamped to [2, 10].

    Ensures at least min_per_cell minority-class samples per cell on average.
    n=3380, pi_min=0.10, min=30 -> K=3 (Mendelian).
    n=11400, pi_min=0.10, min=30 -> K=10 (Complex, but capped at 10).
    """
    K = int(np.floor(n * pi_min / min_per_cell))
    return max(2, min(10, K))


def chrom_loo_predict(
    p: np.ndarray, sigma: np.ndarray, y: np.ndarray, chroms: np.ndarray,
    K, method: str, alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run chrom-LOO with a given method and K (int or dict[chrom->K]).

    Returns (in0, in1, b_te_partition_dependent). The bin-idx return is in the
    method's native partition; callers should compute a separate K_eval index
    via compute_metric_bin_idx for cross-method-fair metric reporting.
    """
    if method not in METHOD_FNS:
        raise ValueError(f"unknown method {method!r}")
    calib_fn = METHOD_FNS[method]
    sigma_safe = sigma + EPS

    n = len(y)
    in0 = np.zeros(n, dtype=bool)
    in1 = np.zeros(n, dtype=bool)
    b_idx = np.zeros(n, dtype=int)

    for c in sorted(set(chroms)):
        mask_te = chroms == c
        mask_cal = ~mask_te
        K_c = K[str(c)] if isinstance(K, dict) else int(K)
        in0_te, in1_te, b_te = calib_fn(
            p[mask_cal], sigma_safe[mask_cal], y[mask_cal],
            p[mask_te], sigma_safe[mask_te], K_c, alpha,
        )
        in0[mask_te] = in0_te
        in1[mask_te] = in1_te
        b_idx[mask_te] = b_te
    return in0, in1, b_idx
