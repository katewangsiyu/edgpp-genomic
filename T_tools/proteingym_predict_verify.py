"""Predict-verify on ProteinGym (Phase 1.2).

For each assay, computes Eq.(3)'s K-fixed bound:
    predicted_gap(K) = L_F * R / K + K / (pi_min * n)

evaluated at K=5 (the App E reporting K), then scores against the observed
sigma_bin gap (both the marginal-bin metric in App E Tab 11 and the cell-
level worst-(k,b) metric stored as max-of-cell-gaps in per_sigma_bin).

Reports the Phase 1.3 decision signal:
  - phase2_go     : Spearman(pred, observed_cell) > 0.5 AND ROC(outlier) > 0.65
  - reframe_only  : 0.3 < Spearman <= 0.5 OR ROC in [0.55, 0.65]
  - abort         : Spearman <= 0.3 (Eq.(3) does not calibrate ProteinGym)

Usage:
    python T_tools/proteingym_predict_verify.py \\
        --lf-json R_raw/proteingym_lf/per_assay_LF.json \\
        --out R_raw/proteingym_predict_verify/scores.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def predicted_gap_eq3(L_F: float, R: float, K: int,
                      pi_min: float, n: int) -> float:
    """Eq.(3) of §5.1: G(K) <= L_F R / K + K / (pi_min n).

    Use the K-fixed form (not the K* optimum) because we want predictions at
    the same K=5 that App E reports.
    """
    return L_F * R / K + K / (pi_min * n)


def predicted_gap_oracle(L_F: float, R: float, pi_min: float, n: int) -> float:
    """Theorem 5.1 oracle: G(K*) <= 2 sqrt(L_F R / (pi_min n))."""
    return 2.0 * float(np.sqrt(max(L_F * R / max(pi_min * n, 1e-12), 0.0)))


def classify(rho: float, roc: float | None) -> str:
    """Phase 1.3 decision rule: see module docstring."""
    if rho > 0.5 and (roc is not None and roc > 0.65):
        return "phase2_go"
    if rho > 0.3 or (roc is not None and roc > 0.55):
        return "reframe_only"
    return "abort"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lf-json", required=True, type=Path,
                    help="Output of T_tools/proteingym_lf_per_assay.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--K", type=int, default=5,
                    help="Reporting K for Eq.(3) prediction (matches App E).")
    ap.add_argument("--outlier-cov-threshold", type=float, default=0.85,
                    help="Coverage below this counts as outlier (matches App E).")
    args = ap.parse_args()

    data = json.loads(args.lf_json.read_text())
    df = pd.DataFrame(data)

    n_total = len(df)
    df = df.dropna(subset=["L_F_lcls", "observed_sigma_bin_range"])
    print(f"[load] {len(df)}/{n_total} assays with valid L_F + observed gap")

    df["predicted_gap_eq3"] = df.apply(
        lambda r: predicted_gap_eq3(
            L_F=r["L_F_lcls"], R=r["sigma_range_R"], K=args.K,
            pi_min=r["pi_min"], n=r["n_test"]),
        axis=1)
    df["predicted_gap_oracle"] = df.apply(
        lambda r: predicted_gap_oracle(
            L_F=r["L_F_lcls"], R=r["sigma_range_R"],
            pi_min=r["pi_min"], n=r["n_test"]),
        axis=1)

    summary: dict[str, object] = {
        "n_assays": int(len(df)),
        "K_eval": int(args.K),
        "outlier_cov_threshold": float(args.outlier_cov_threshold),
    }

    for pred_col in ("predicted_gap_eq3", "predicted_gap_oracle"):
        for obs_col, label in (("observed_sigma_bin_range", "marginalbin"),
                               ("observed_cell_worst_gap", "cellworst")):
            sub = df.dropna(subset=[pred_col, obs_col])
            if len(sub) >= 5:
                rho, p_rho = spearmanr(sub[pred_col], sub[obs_col])
                r_p, p_p = pearsonr(sub[pred_col], sub[obs_col])
                summary[f"spearman_{pred_col}_vs_{label}"] = {
                    "rho": float(rho), "p": float(p_rho), "n": int(len(sub))
                }
                summary[f"pearson_{pred_col}_vs_{label}"] = {
                    "r": float(r_p), "p": float(p_p), "n": int(len(sub))
                }
            else:
                summary[f"spearman_{pred_col}_vs_{label}"] = None
                summary[f"pearson_{pred_col}_vs_{label}"] = None

    df["outlier"] = (df["observed_marginal_coverage"]
                     < args.outlier_cov_threshold).astype(int)
    n_out = int(df["outlier"].sum())
    summary["n_outliers"] = n_out
    summary["outlier_assays"] = df.loc[df["outlier"] == 1,
                                       "target_assay"].tolist()

    if n_out >= 3 and (len(df) - n_out) >= 3:
        roc_eq3 = float(roc_auc_score(df["outlier"], df["predicted_gap_eq3"]))
        roc_oracle = float(roc_auc_score(df["outlier"],
                                         df["predicted_gap_oracle"]))
        summary["roc_outlier_from_predicted_gap_eq3"] = roc_eq3
        summary["roc_outlier_from_predicted_gap_oracle"] = roc_oracle

        feats = pd.DataFrame({
            "log_n": np.log(df["n_test"]),
            "inv_sqrt_pi_min": 1 / np.sqrt(df["pi_min"]),
            "sqrt_L_F": np.sqrt(np.maximum(df["L_F_lcls"], 1e-9)),
            "R": df["sigma_range_R"],
        })
        lr = LogisticRegression(max_iter=2000)
        lr.fit(feats.values, df["outlier"].values)
        prob = lr.predict_proba(feats.values)[:, 1]
        roc_logreg = float(roc_auc_score(df["outlier"], prob))
        summary["roc_outlier_logreg_in_sample"] = roc_logreg
        summary["logreg_coefs"] = dict(zip(feats.columns,
                                            map(float, lr.coef_[0])))
    else:
        roc_eq3 = roc_oracle = roc_logreg = None
        summary["roc_outlier_from_predicted_gap_eq3"] = None
        summary["roc_outlier_from_predicted_gap_oracle"] = None
        summary["roc_outlier_logreg_in_sample"] = None
        summary["logreg_coefs"] = None

    rho_cell_eq3 = (
        summary["spearman_predicted_gap_eq3_vs_cellworst"]["rho"]
        if isinstance(summary.get("spearman_predicted_gap_eq3_vs_cellworst"),
                       dict) else 0.0
    )
    rho_marg_eq3 = (
        summary["spearman_predicted_gap_eq3_vs_marginalbin"]["rho"]
        if isinstance(summary.get("spearman_predicted_gap_eq3_vs_marginalbin"),
                       dict) else 0.0
    )

    decision_cell = classify(rho_cell_eq3, roc_logreg)
    decision_marg = classify(rho_marg_eq3, roc_logreg)
    summary["decision_cell_metric"] = decision_cell
    summary["decision_marginal_metric"] = decision_marg
    summary["decision"] = (decision_cell
                           if decision_cell == "phase2_go"
                           else decision_marg)
    summary["decision_rule"] = (
        "phase2_go: Spearman > 0.5 AND ROC > 0.65; "
        "reframe_only: Spearman > 0.3 OR ROC > 0.55; "
        "abort otherwise."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "summary": summary,
        "per_assay": df.to_dict("records"),
    }, indent=2, default=str))
    print(f"saved: {args.out}")
    print("\n=== Phase 1.3 decision signal ===")
    for k, v in summary.items():
        if k in ("outlier_assays", "logreg_coefs"):
            continue
        if isinstance(v, dict):
            print(f"  {k}: " + ", ".join(f"{kk}={vv:.4f}"
                                         if isinstance(vv, float) else f"{kk}={vv}"
                                         for kk, vv in v.items()))
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\n  >>> DECISION: {summary['decision']}  <<<")


if __name__ == "__main__":
    main()
