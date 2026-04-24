"""HCCP — Heteroscedastic Class-Conditional Conformal Prediction.

Reference: "Conformalized Heteroscedastic Variant Effect Prediction with Local
Coverage Guarantees" (anonymous, NeurIPS 2027 submission).

Usage::

    from edgpp_genomic.hccp import HCCPClassifier

    clf = HCCPClassifier(alpha=0.10, n_sigma_bins=5)
    clf.fit(X_cal, y_cal, chroms_cal)
    pred_sets = clf.predict_set(X_test)  # list of {0}, {1}, {0,1}, or set()
    cov = clf.evaluate_coverage(X_test, y_test)

See scripts/ for end-to-end reproduction scripts that use these primitives.
"""

from .classifier import HCCPClassifier
from .sigma_head import SigmaHead
from .conformal import mondrian_calibrate, predict_set_from_calibration
from .adaptive_K import select_K_cv, oracle_K

__all__ = [
    "HCCPClassifier",
    "SigmaHead",
    "mondrian_calibrate",
    "predict_set_from_calibration",
    "select_K_cv",
    "oracle_K",
]

__version__ = "0.1.0"
