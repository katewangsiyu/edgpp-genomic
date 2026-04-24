"""T5 adaptive K selection — bias-variance tradeoff for Mondrian bin count."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .conformal import mondrian_calibrate, predict_set_from_calibration


def oracle_K(L_F: float, R: float, pi_min: float, n: int) -> int:
    """T5.1 oracle bin count: K* = floor(sqrt(L_F R π_min n)).

    Args:
        L_F: score-Lipschitz constant (upper bound on |F_{k,σ1} - F_{k,σ2}| / |σ1 - σ2|)
        R: σ̂ range = σ_max - σ_min
        pi_min: minority-class prevalence
        n: total calibration size

    Returns:
        Integer K* ≥ 2.
    """
    K_cont = math.sqrt(max(L_F * R * pi_min * n, 4.0))
    return max(2, int(math.floor(K_cont)))


def select_K_cv(p: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                alpha: float,
                candidate_K: Iterable[int] = (2, 3, 5, 8, 10, 15, 20),
                n_folds: int = 5,
                min_cell_size: int = 5) -> dict:
    """Cross-validated worst-cell coverage gap K-selector.

    For each candidate K, split the data into n_folds, compute conformal
    thresholds on all folds but one, evaluate worst-cell coverage gap on
    the held-out fold, and pick the K with smallest mean worst-cell gap.

    Args:
        p: base predictions
        sigma: σ̂ values
        y: binary labels
        alpha: miscoverage
        candidate_K: iterable of K values to compare
        n_folds: CV folds
        min_cell_size: fallback threshold

    Returns:
        dict with keys:
            K_cv: best K
            per_K: dict mapping K → dict(mean_worst_gap, std_worst_gap)
    """
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(p))
    fold_idx = np.array_split(idx, n_folds)

    results: dict[int, dict] = {}
    for K in candidate_K:
        gaps = []
        for f in range(n_folds):
            test_idx = fold_idx[f]
            train_idx = np.concatenate([fold_idx[g] for g in range(n_folds) if g != f])
            cal = mondrian_calibrate(p[train_idx], sigma[train_idx], y[train_idx],
                                      alpha, K, min_cell_size)
            pred_sets = predict_set_from_calibration(
                p[test_idx], sigma[test_idx], cal)
            covered = np.array([y[test_idx][i] in ps
                                for i, ps in enumerate(pred_sets)])
            # Worst-cell gap on test fold
            edges = cal.bin_edges
            bin_id_te = np.digitize(sigma[test_idx], edges[1:-1])
            worst = 0.0
            for k in (0, 1):
                for b in range(K):
                    m = (y[test_idx] == k) & (bin_id_te == b)
                    if m.sum() < 5:
                        continue
                    worst = max(worst, abs(covered[m].mean() - (1 - alpha)))
            gaps.append(worst)
        results[K] = {
            "mean_worst_gap": float(np.mean(gaps)),
            "std_worst_gap": float(np.std(gaps, ddof=1)),
        }

    K_cv = min(results, key=lambda k: results[k]["mean_worst_gap"])
    return {"K_cv": K_cv, "per_K": results}
