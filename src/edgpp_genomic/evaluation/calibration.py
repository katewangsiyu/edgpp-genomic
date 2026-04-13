"""Calibration metrics for the student: ECE / NLL / Brier."""
from __future__ import annotations
import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    N = len(probs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / N) * abs(labels[mask].mean() - probs[mask].mean())
    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(((probs - labels) ** 2).mean())


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-8) -> float:
    p = np.clip(probs, eps, 1 - eps)
    return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())
