"""σ̂ heteroscedastic reliability head."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class SigmaHeadConfig:
    max_depth: int = 2
    max_iter: int = 200
    learning_rate: float = 0.05
    target_mode: str = "abs_residual"  # or "log_variance"
    sigma_floor: float = 1e-4
    seed: int = 42


class SigmaHead:
    """Gradient-boosted regressor predicting |residual| or log-variance.

    Trained on chrom-LOO out-of-fold residuals from the base classifier,
    so σ̂(x) is honest w.r.t. the held-out chromosome.
    """

    def __init__(self, config: SigmaHeadConfig | None = None) -> None:
        self.config = config or SigmaHeadConfig()
        self._per_chrom_models: dict[str, Pipeline] = {}

    def _pipeline(self) -> Pipeline:
        c = self.config
        return Pipeline([
            ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("hgb", HistGradientBoostingRegressor(
                loss="squared_error",
                max_depth=c.max_depth, max_iter=c.max_iter,
                learning_rate=c.learning_rate,
                random_state=c.seed,
                early_stopping=False,
            )),
        ])

    def fit_predict_chrom_loo(self, X: np.ndarray, residuals: np.ndarray,
                              chroms: np.ndarray) -> np.ndarray:
        """Fit chrom-LOO σ̂ and return σ̂(x) for every variant."""
        sigma = np.full(len(X), np.nan, dtype=float)
        uniq_chroms = sorted(set(chroms))
        for c in uniq_chroms:
            mask_te = chroms == c
            mask_tr = ~mask_te
            target = self._target(residuals[mask_tr])
            pipe = self._pipeline()
            pipe.fit(X[mask_tr], target)
            pred = pipe.predict(X[mask_te])
            sigma[mask_te] = self._invert(pred)
            self._per_chrom_models[str(c)] = pipe
        return np.clip(sigma, self.config.sigma_floor, None)

    def predict(self, X: np.ndarray, chrom_key: str | None = None) -> np.ndarray:
        """Predict σ̂ using a specific chrom-LOO model (or mean if None)."""
        if chrom_key is None:
            preds = np.stack([p.predict(X) for p in self._per_chrom_models.values()])
            return np.clip(self._invert(preds.mean(axis=0)), self.config.sigma_floor, None)
        pred = self._per_chrom_models[str(chrom_key)].predict(X)
        return np.clip(self._invert(pred), self.config.sigma_floor, None)

    def _target(self, r: np.ndarray) -> np.ndarray:
        if self.config.target_mode == "abs_residual":
            return np.abs(r)
        if self.config.target_mode == "log_variance":
            return np.log(r ** 2 + 1e-6)
        raise ValueError(self.config.target_mode)

    def _invert(self, pred: np.ndarray) -> np.ndarray:
        if self.config.target_mode == "abs_residual":
            return np.maximum(pred, self.config.sigma_floor)
        return np.exp(0.5 * pred)
