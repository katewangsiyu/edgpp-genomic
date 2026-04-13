"""QTL benchmark metrics mirroring calico/westminster."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr


def compute_eqtl_metrics(scores: np.ndarray, coefs: np.ndarray | None = None,
                         labels: np.ndarray | None = None) -> dict:
    """
    scores: (N,)   predicted SED (signed)
    coefs:  (N,)   GTEx measured coef (optional; for sign AUROC / Spearman)
    labels: (N,)   1=causal, 0=control (optional; for classification)

    Mirrors westminster_eqtl_gtexg.py metric rows.
    """
    out = {}
    if coefs is not None:
        out["sign_auroc"] = float(roc_auc_score(coefs > 0, scores))
        rho, p = spearmanr(coefs, scores)
        out["spearman"] = float(rho)
        out["spearman_p"] = float(p)
    if labels is not None:
        abs_scores = np.abs(scores)
        out["class_auroc"] = float(roc_auc_score(labels, abs_scores))
        out["class_auprc"] = float(average_precision_score(labels, abs_scores))
    return out


def per_tissue_macro_mean(per_tissue_metrics: list[dict]) -> dict:
    """Average metric values across tissues (macro-mean, as in Borzoi paper)."""
    if not per_tissue_metrics:
        return {}
    keys = set().union(*[m.keys() for m in per_tissue_metrics])
    out = {}
    for k in keys:
        vals = [m[k] for m in per_tissue_metrics if k in m]
        if vals:
            out[k] = float(np.mean(vals))
    return out
