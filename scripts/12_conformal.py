"""Conformal prediction for genomic variant effect scoring.

Implements multiple conformal methods on top of chrom-LOO predictions:

1. **Split conformal**: Hold out a subset of chromosomes for calibration.
2. **CV+ conformal** (Barber et al. 2021): Uses chrom-LOO residuals directly,
   no data splitting waste. Each variant's conformal score comes from its
   leave-one-chrom-out prediction.
3. **Mondrian conformal**: Per-group (e.g., per-consequence-type) calibration
   for group-conditional coverage guarantees.

For binary classification, produces:
  - Conformal p-values for each class
  - Prediction sets {0}, {1}, {0,1} at user-specified α
  - Coverage metrics: empirical coverage, set size distribution, conditional coverage

Usage:
    python scripts/12_conformal.py \\
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --base-scores outputs/aggregator_gbm/CADD+Borzoi_mendelian/scores.parquet \\
        --out-dir outputs/conformal/CADD+Borzoi_gbm_mendelian \\
        --alpha 0.10
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# ---------- nonconformity scores ----------

def lac_score(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least Ambiguous set-valued Classifier (LAC) nonconformity score.
    s(x, y) = 1 - f_y(x), where f_y is the predicted probability of true class y."""
    return 1.0 - np.where(y == 1, prob, 1.0 - prob)


def aps_score(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Adaptive Prediction Sets (APS) nonconformity score (Romano et al. 2020).
    Accumulate probability mass from most likely to least likely class until
    we include the true label. For binary: simpler formula."""
    # For binary: if true class has highest prob, score = prob_true
    # if true class has lowest prob, score = 1.0
    # More precisely: score = sum of probs of classes ranked above true + prob_true
    prob_true = np.where(y == 1, prob, 1.0 - prob)
    prob_other = 1.0 - prob_true
    # If true class is top-ranked (prob_true >= prob_other): score = prob_true (just itself)
    # If true class is second: score = prob_other + prob_true = 1.0
    # We add U(0, prob_true) randomization for exact coverage
    return np.where(prob_true >= prob_other, prob_true, 1.0)


# ---------- CV+ conformal (Barber et al. 2021) ----------

def cv_plus_conformal(prob: np.ndarray, y: np.ndarray, chroms: np.ndarray,
                      alpha: float = 0.10, score_fn=lac_score):
    """CV+ conformal prediction using chrom-LOO predictions.

    For each test point i (on chrom c_i), the calibration set is all points
    NOT on chrom c_i. The conformal quantile is computed from calibration scores.

    Returns prediction sets and metrics.
    """
    n = len(y)
    uniq_chroms = sorted(set(chroms))

    # Compute nonconformity scores for all points
    scores = score_fn(prob, y)

    # For each point, compute conformal p-values for each candidate label
    # p-value for label y_hat: fraction of calibration scores >= s(x_i, y_hat)
    pvals_0 = np.zeros(n)  # p-value for y=0
    pvals_1 = np.zeros(n)  # p-value for y=1

    for c in uniq_chroms:
        test_mask = chroms == c
        cal_mask = ~test_mask

        # Calibration scores (from their own LOO predictions)
        cal_scores = scores[cal_mask]
        n_cal = len(cal_scores)

        # For test points on this chrom, compute hypothetical scores
        test_idx = np.where(test_mask)[0]
        for i in test_idx:
            # Score if true label were 0
            s0 = score_fn(prob[i:i+1], np.array([0]))[0]
            # Score if true label were 1
            s1 = score_fn(prob[i:i+1], np.array([1]))[0]

            # Conformal p-values (with finite-sample correction)
            pvals_0[i] = (np.sum(cal_scores >= s0) + 1) / (n_cal + 1)
            pvals_1[i] = (np.sum(cal_scores >= s1) + 1) / (n_cal + 1)

    # Prediction sets at level alpha
    pred_sets = []
    for i in range(n):
        pset = set()
        if pvals_0[i] > alpha:
            pset.add(0)
        if pvals_1[i] > alpha:
            pset.add(1)
        pred_sets.append(pset)

    return pvals_0, pvals_1, pred_sets


def class_conditional_conformal(prob: np.ndarray, y: np.ndarray, chroms: np.ndarray,
                                alpha: float = 0.10, score_fn=lac_score):
    """Class-conditional conformal: calibrate separately for each class.

    This guarantees P(y ∈ C(x) | y=k) ≥ 1-α for EACH class k, not just marginally.
    Critical for imbalanced genomic data where pathogenic variants are rare (~10%).

    Trade-off: prediction sets are larger (more {0,1} sets) but coverage is balanced.
    """
    n = len(y)
    scores = score_fn(prob, y)

    pvals_0 = np.zeros(n)
    pvals_1 = np.zeros(n)

    for c in sorted(set(chroms)):
        test_mask = chroms == c
        cal_mask = ~test_mask

        # Separate calibration scores by class
        cal_pos = scores[cal_mask & (y == 1)]
        cal_neg = scores[cal_mask & (y == 0)]

        for i in np.where(test_mask)[0]:
            s0 = score_fn(prob[i:i+1], np.array([0]))[0]
            s1 = score_fn(prob[i:i+1], np.array([1]))[0]

            # P-value for y=0: calibrate against negative examples
            n_neg = len(cal_neg)
            pvals_0[i] = (np.sum(cal_neg >= s0) + 1) / (n_neg + 1) if n_neg > 0 else 1.0

            # P-value for y=1: calibrate against positive examples
            n_pos = len(cal_pos)
            pvals_1[i] = (np.sum(cal_pos >= s1) + 1) / (n_pos + 1) if n_pos > 0 else 1.0

    pred_sets = []
    for i in range(n):
        pset = set()
        if pvals_0[i] > alpha:
            pset.add(0)
        if pvals_1[i] > alpha:
            pset.add(1)
        pred_sets.append(pset)

    return pvals_0, pvals_1, pred_sets


def mondrian_conformal(prob: np.ndarray, y: np.ndarray, chroms: np.ndarray,
                       groups: np.ndarray, alpha: float = 0.10, score_fn=lac_score):
    """Mondrian conformal: separate calibration per group for group-conditional coverage."""
    n = len(y)
    uniq_groups = sorted(set(groups))
    scores = score_fn(prob, y)

    pvals_0 = np.zeros(n)
    pvals_1 = np.zeros(n)

    for g in uniq_groups:
        g_mask = groups == g
        for c in sorted(set(chroms)):
            test_mask = (chroms == c) & g_mask
            if not test_mask.any():
                continue
            # Calibration: same group, different chrom
            cal_mask = (chroms != c) & g_mask
            if not cal_mask.any():
                # Fallback to all chroms if group too small
                cal_mask = chroms != c
            cal_scores = scores[cal_mask]
            n_cal = len(cal_scores)

            for i in np.where(test_mask)[0]:
                s0 = score_fn(prob[i:i+1], np.array([0]))[0]
                s1 = score_fn(prob[i:i+1], np.array([1]))[0]
                pvals_0[i] = (np.sum(cal_scores >= s0) + 1) / (n_cal + 1)
                pvals_1[i] = (np.sum(cal_scores >= s1) + 1) / (n_cal + 1)

    pred_sets = []
    for i in range(n):
        pset = set()
        if pvals_0[i] > alpha:
            pset.add(0)
        if pvals_1[i] > alpha:
            pset.add(1)
        pred_sets.append(pset)

    return pvals_0, pvals_1, pred_sets


# ---------- evaluation ----------

def evaluate_conformal(y, pred_sets, pvals_0, pvals_1, chroms, alpha, label=""):
    n = len(y)
    # Coverage: fraction of variants where true label is in prediction set
    covered = np.array([y[i] in ps for i, ps in enumerate(pred_sets)])
    coverage = covered.mean()

    # Set sizes
    sizes = np.array([len(ps) for ps in pred_sets])
    empty = (sizes == 0).mean()
    singleton = (sizes == 1).mean()
    both = (sizes == 2).mean()

    # Conditional coverage by class
    pos_mask = y == 1
    neg_mask = y == 0
    cov_pos = covered[pos_mask].mean() if pos_mask.any() else float("nan")
    cov_neg = covered[neg_mask].mean() if neg_mask.any() else float("nan")

    # Per-chrom coverage
    chrom_cov = {}
    for c in sorted(set(chroms)):
        m = chroms == c
        if m.sum() < 5:
            continue
        chrom_cov[c] = float(covered[m].mean())

    # Selective prediction using conformal confidence
    # Confidence = max p-value (how strongly model believes in its top prediction)
    conf = np.maximum(pvals_0, pvals_1)

    # Coverage-AUPRC using conformal confidence
    p_hat = pvals_1 / (pvals_0 + pvals_1 + 1e-12)  # calibrated probability
    cov_auprc = {}
    for frac in (0.10, 0.25, 0.50, 0.75, 1.00):
        k = int(np.ceil(frac * n))
        idx = np.argsort(-conf)[:k]
        if y[idx].sum() > 0:
            cov_auprc[frac] = float(average_precision_score(y[idx], p_hat[idx]))
        else:
            cov_auprc[frac] = float("nan")

    metrics = {
        "alpha": alpha,
        "n": n,
        "coverage": float(coverage),
        "target_coverage": 1.0 - alpha,
        "coverage_gap": float(coverage - (1.0 - alpha)),
        "coverage_pos": float(cov_pos),
        "coverage_neg": float(cov_neg),
        "frac_empty": float(empty),
        "frac_singleton": float(singleton),
        "frac_both": float(both),
        "per_chrom_coverage": chrom_cov,
        "coverage_AUPRC_conformal": cov_auprc,
    }

    print(f"\n{'='*60}")
    print(f"  {label} (α={alpha})")
    print(f"{'='*60}")
    print(f"  Coverage: {coverage:.4f} (target ≥ {1-alpha:.2f})")
    print(f"  Coverage|pos: {cov_pos:.4f}  Coverage|neg: {cov_neg:.4f}")
    print(f"  Sets: empty={empty:.3f}  singleton={singleton:.3f}  both={both:.3f}")
    print(f"  Per-chrom coverage range: [{min(chrom_cov.values()):.3f}, {max(chrom_cov.values()):.3f}]")
    print(f"  Coverage-AUPRC (conformal conf): "
          + "  ".join(f"@{c:.0%}={v:.3f}" for c, v in cov_auprc.items()))

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--base-scores", required=True,
                    help="scores.parquet from 09/11 aggregator (must have chrom, label, score)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="Miscoverage rate (default 0.10 → target 90%% coverage)")
    ap.add_argument("--score-fn", choices=["lac", "aps"], default="lac")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    base = pd.read_parquet(args.base_scores).reset_index(drop=True)
    assert len(V) == len(base), "row mismatch"

    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()
    prob = base["score"].to_numpy()

    score_fn = lac_score if args.score_fn == "lac" else aps_score
    print(f"[load] n={len(V)} score_fn={args.score_fn} alpha={args.alpha}")

    # 1. CV+ conformal
    pvals_0, pvals_1, pred_sets = cv_plus_conformal(
        prob, y, chroms, alpha=args.alpha, score_fn=score_fn
    )
    m_cvplus = evaluate_conformal(y, pred_sets, pvals_0, pvals_1, chroms,
                                   args.alpha, label=f"CV+ conformal ({args.score_fn})")

    # 2. Class-conditional conformal
    p0cc, p1cc, pscc = class_conditional_conformal(
        prob, y, chroms, alpha=args.alpha, score_fn=score_fn
    )
    m_classcond = evaluate_conformal(y, pscc, p0cc, p1cc, chroms,
                                      args.alpha, label=f"Class-conditional conformal ({args.score_fn})")

    # 3. Mondrian conformal by consequence type (if available)
    m_mondrian = None
    if "consequence" in V.columns:
        groups = V["consequence"].astype(str).to_numpy()
        p0m, p1m, psm = mondrian_conformal(
            prob, y, chroms, groups, alpha=args.alpha, score_fn=score_fn
        )
        m_mondrian = evaluate_conformal(y, psm, p0m, p1m, chroms,
                                         args.alpha, label=f"Mondrian conformal ({args.score_fn})")

    # 3. Sweep alpha for coverage-alpha curve
    alpha_sweep = {}
    for a in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        _, _, ps_a = cv_plus_conformal(prob, y, chroms, alpha=a, score_fn=score_fn)
        covered_a = np.mean([y[i] in ps for i, ps in enumerate(ps_a)])
        sizes_a = np.array([len(ps) for ps in ps_a])
        alpha_sweep[str(a)] = {
            "coverage": float(covered_a),
            "target": float(1.0 - a),
            "frac_singleton": float((sizes_a == 1).mean()),
            "frac_both": float((sizes_a == 2).mean()),
        }
        print(f"  α={a:.2f}: coverage={covered_a:.4f} (target ≥ {1-a:.2f}), "
              f"singleton={float((sizes_a == 1).mean()):.3f}, "
              f"both={float((sizes_a == 2).mean()):.3f}")

    # Save results
    results = {
        "cv_plus": m_cvplus,
        "class_conditional": m_classcond,
        "mondrian": m_mondrian,
        "alpha_sweep": alpha_sweep,
    }
    (out / "conformal_results.json").write_text(json.dumps(results, indent=2))

    # Save per-variant results
    V_out = V[["chrom"]].copy()
    V_out["label"] = y
    V_out["score"] = prob
    V_out["pval_0"] = pvals_0
    V_out["pval_1"] = pvals_1
    V_out["pred_set_size"] = [len(ps) for ps in pred_sets]
    V_out["covered"] = [y[i] in ps for i, ps in enumerate(pred_sets)]
    V_out.to_parquet(out / "conformal_scores.parquet", index=False)

    print(f"\nsaved: {out}/")


if __name__ == "__main__":
    main()
