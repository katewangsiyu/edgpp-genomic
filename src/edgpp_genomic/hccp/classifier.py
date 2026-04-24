"""High-level HCCP classifier combining σ̂ head + Mondrian calibration."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .adaptive_K import oracle_K, select_K_cv
from .conformal import (
    CalibrationResult, mondrian_calibrate, predict_set_from_calibration,
)
from .sigma_head import SigmaHead, SigmaHeadConfig


@dataclass
class HCCPClassifier:
    """Post-hoc HCCP wrapper around any probabilistic classifier.

    The user provides pre-computed p̂(x) scores and raw features X; HCCP trains
    the σ̂ head on chrom-LOO residuals and then calibrates Mondrian conformal
    thresholds.

    For convenience we expose .predict_set() and .evaluate_coverage().
    """

    alpha: float = 0.10
    n_sigma_bins: int | str = "auto"  # int, or "auto" for K_CV selection
    sigma_config: SigmaHeadConfig = field(default_factory=SigmaHeadConfig)
    min_cell_size: int = 5

    def __post_init__(self) -> None:
        self._sigma_head: SigmaHead | None = None
        self._calibration: CalibrationResult | None = None
        self._K_cv_selection: dict | None = None

    def fit(self, X_cal: np.ndarray, p_cal: np.ndarray, y_cal: np.ndarray,
            chroms_cal: np.ndarray) -> "HCCPClassifier":
        """Fit σ̂ and Mondrian calibration using chrom-LOO residuals."""
        residuals = y_cal.astype(float) - p_cal
        self._sigma_head = SigmaHead(self.sigma_config)
        sigma_cal = self._sigma_head.fit_predict_chrom_loo(X_cal, residuals, chroms_cal)

        if self.n_sigma_bins == "auto":
            sel = select_K_cv(p_cal, sigma_cal, y_cal, self.alpha)
            K = sel["K_cv"]
            self._K_cv_selection = sel
        else:
            K = int(self.n_sigma_bins)

        self._calibration = mondrian_calibrate(
            p_cal, sigma_cal, y_cal, self.alpha, K, self.min_cell_size)
        return self

    def predict_set(self, X_test: np.ndarray, p_test: np.ndarray,
                    chrom_key: str | None = None) -> list[set[int]]:
        self._assert_fit()
        sigma_test = self._sigma_head.predict(X_test, chrom_key)
        return predict_set_from_calibration(p_test, sigma_test, self._calibration)

    def evaluate_coverage(self, X_test: np.ndarray, p_test: np.ndarray,
                          y_test: np.ndarray, chrom_key: str | None = None) -> dict:
        pred_sets = self.predict_set(X_test, p_test, chrom_key)
        covered = np.array([y_test[i] in ps for i, ps in enumerate(pred_sets)])
        sizes = np.array([len(ps) for ps in pred_sets])
        return {
            "marginal_coverage": float(covered.mean()),
            "target": 1 - self.alpha,
            "gap": float(abs(covered.mean() - (1 - self.alpha))),
            "frac_singleton": float((sizes == 1).mean()),
            "frac_both": float((sizes == 2).mean()),
            "frac_empty": float((sizes == 0).mean()),
            "coverage_pos": float(covered[y_test == 1].mean()) if (y_test == 1).any() else float("nan"),
            "coverage_neg": float(covered[y_test == 0].mean()) if (y_test == 0).any() else float("nan"),
        }

    @property
    def calibration(self) -> CalibrationResult:
        self._assert_fit()
        return self._calibration

    @property
    def K_cv_selection(self) -> dict | None:
        return self._K_cv_selection

    def _assert_fit(self) -> None:
        if self._calibration is None:
            raise RuntimeError("HCCPClassifier is not fit. Call .fit() first.")
