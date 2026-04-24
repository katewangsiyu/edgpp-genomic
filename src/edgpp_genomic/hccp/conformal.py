"""Mondrian (y × σ̂-bin) conformal calibration."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CalibrationResult:
    """Output of mondrian_calibrate."""
    thresholds: dict[tuple[int, int], float]
    bin_edges: np.ndarray
    alpha: float
    n_bins: int


def hetero_score(p: np.ndarray, sigma: np.ndarray,
                 y_candidate: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """HCCP nonconformity score: |y - p̂| / (σ̂ + ε)."""
    return np.abs(y_candidate.astype(float) - p) / (sigma + eps)


def mondrian_calibrate(p: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                        alpha: float, n_bins: int,
                        min_cell_size: int = 5) -> CalibrationResult:
    """Compute Mondrian-(y × σ̂-bin) conformal quantile thresholds.

    Uses the *entire* input as the calibration fold; callers that need
    chrom-LOO should repeat the call with each held-out chromosome.

    Args:
        p: base probabilistic predictions in [0, 1]
        sigma: per-point σ̂ values
        y: binary labels {0, 1}
        alpha: miscoverage level
        n_bins: number of σ̂-bins K
        min_cell_size: fall back to pooled class-conditional threshold if
            a cell has fewer than this many calibration points

    Returns:
        CalibrationResult with per-cell thresholds and shared bin edges.
    """
    edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_id = np.digitize(sigma, edges[1:-1])

    thresholds: dict[tuple[int, int], float] = {}
    for k in (0, 1):
        mask_k = y == k
        for b in range(n_bins):
            mask_kb = mask_k & (bin_id == b)
            n_kb = int(mask_kb.sum())
            if n_kb < min_cell_size:
                # Pooled class-cond fallback
                n_k = int(mask_k.sum())
                s_cal = hetero_score(p[mask_k], sigma[mask_k], np.full(n_k, k))
                level = min(1.0, np.ceil((n_k + 1) * (1 - alpha)) / max(n_k, 1))
                thresholds[(k, b)] = float(np.quantile(s_cal, level, method="higher")) \
                                   if n_k > 0 else np.inf
            else:
                s_cal = hetero_score(p[mask_kb], sigma[mask_kb], np.full(n_kb, k))
                level = min(1.0, np.ceil((n_kb + 1) * (1 - alpha)) / n_kb)
                thresholds[(k, b)] = float(np.quantile(s_cal, level, method="higher"))
    return CalibrationResult(
        thresholds=thresholds, bin_edges=edges,
        alpha=alpha, n_bins=n_bins,
    )


def predict_set_from_calibration(p: np.ndarray, sigma: np.ndarray,
                                  cal: CalibrationResult) -> list[set[int]]:
    """Return prediction set per test point using pre-computed thresholds."""
    bin_id = np.digitize(sigma, cal.bin_edges[1:-1])
    pred_sets: list[set[int]] = []
    for i, b in enumerate(bin_id):
        ps: set[int] = set()
        for k in (0, 1):
            s_ik = hetero_score(p[i:i+1], sigma[i:i+1], np.array([k]))[0]
            q = cal.thresholds.get((k, int(b)), np.inf)
            if s_ik <= q:
                ps.add(k)
        pred_sets.append(ps)
    return pred_sets
