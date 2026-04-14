"""Chrom-leave-one-out supervised aggregator — matches TraitGym's reference
pipeline in baselines/TraitGym/workflow/rules/common.smk (`train_logistic_regression`).

For each chromosome c in test.parquet:
    mask_train = V.chrom != c
    GridSearchCV over C ∈ logspace(-8, 0, 10) using GroupKFold(chrom)
        pipeline: SimpleImputer(mean) → StandardScaler → LogisticRegression(balanced)
    predict_proba(V[V.chrom == c])
Assemble all predictions and compute AUPRC_by_chrom_weighted_average
(matches the TraitGym leaderboard exactly).

Use --assert-leaderboard to enforce the known-reference value as a sanity gate.

Usage:
    python scripts/09_aggregator_chrom_loo.py \\
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --features-dir data/raw/traitgym/mendelian_traits_matched_9/features \\
        --feature-set Borzoi \\
        --assert-leaderboard 0.493
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


FEATURE_SETS = {
    "CADD":                ["CADD"],
    "GPN-MSA":             ["GPN-MSA_LLR", "GPN-MSA_absLLR", "GPN-MSA_InnerProducts"],
    "Borzoi":              ["Borzoi_L2", "Borzoi_L2_L2"],
    "Borzoi_L2_L2":        ["Borzoi_L2_L2"],
    "CADD+Borzoi":         ["CADD", "Borzoi_L2", "Borzoi_L2_L2"],
    "CADD+GPN-MSA+Borzoi": ["CADD", "GPN-MSA_LLR", "GPN-MSA_absLLR", "GPN-MSA_InnerProducts",
                            "Borzoi_L2", "Borzoi_L2_L2"],
    "CADD+GPN-MSA":        ["CADD", "GPN-MSA_LLR", "GPN-MSA_absLLR", "GPN-MSA_InnerProducts"],
    "GPN-MSA+Borzoi":      ["GPN-MSA_LLR", "GPN-MSA_absLLR", "GPN-MSA_InnerProducts",
                            "Borzoi_L2", "Borzoi_L2_L2"],
}


def build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean",
                                  keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("linear", LogisticRegression(class_weight="balanced", random_state=42,
                                      max_iter=1000)),
    ])


def train_predict(X_train, y_train, groups_train, X_test):
    pipe = build_pipeline()
    Cs = np.logspace(-8, 0, 10)
    grid = GridSearchCV(
        pipe, {"linear__C": Cs},
        scoring="average_precision",
        cv=GroupKFold(n_splits=min(5, len(set(groups_train)))),
        n_jobs=-1,
    )
    grid.fit(X_train, y_train, groups=groups_train)
    return grid.predict_proba(X_test)[:, 1], grid.best_params_["linear__C"]


def weighted_per_chrom_auprc(y, s, chroms):
    keep = []
    for c in sorted(set(chroms)):
        m = chroms == c
        if m.sum() < 10 or y[m].sum() == 0:
            continue
        keep.append((m.sum(), average_precision_score(y[m], s[m])))
    if not keep:
        return float("nan"), 0
    w = np.array([x[0] for x in keep], dtype=float)
    a = np.array([x[1] for x in keep], dtype=float)
    return float((w * a).sum() / w.sum()), len(keep)


def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e, N = 0.0, len(y)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        e += m.sum() / N * abs(y[m].mean() - p[m].mean())
    return float(e)


def coverage_auprc(y, s, coverages=(0.1, 0.25, 0.5, 0.75, 1.0)):
    """At each coverage level, keep top-k by score magnitude (high = confident),
    compute AUPRC on that subset. Returns dict coverage→AUPRC."""
    order = np.argsort(-np.abs(s - 0.5))  # farther-from-decision-boundary = more confident
    out = {}
    for c in coverages:
        k = int(np.ceil(c * len(y)))
        idx = order[:k]
        if y[idx].sum() == 0:
            out[c] = float("nan")
        else:
            out[c] = float(average_precision_score(y[idx], s[idx]))
    return out


def load_feature_matrix(features_dir: Path, feature_names: list[str]):
    dfs = []
    for name in feature_names:
        path = features_dir / f"{name}.parquet"
        df = pd.read_parquet(path)
        df.columns = [f"{name}__{c}" for c in df.columns]
        dfs.append(df)
    return pd.concat(dfs, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--feature-set", required=True, choices=list(FEATURE_SETS))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--assert-leaderboard", type=float, default=None,
                    help="If provided, require |AUPRC_per_chrom - ref| < 0.015")
    ap.add_argument("--tol", type=float, default=0.015)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    feat_names = FEATURE_SETS[args.feature_set]

    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    X = load_feature_matrix(Path(args.features_dir), feat_names).reset_index(drop=True)
    assert len(V) == len(X), f"len mismatch V={len(V)} X={len(X)}"
    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()
    uniq_chroms = sorted(set(chroms))
    print(f"[load] feature_set={args.feature_set} n={len(V)} dim={X.shape[1]} "
          f"chroms={len(uniq_chroms)} pos={int(y.sum())} ({y.mean():.1%})")

    scores = np.full(len(V), np.nan, dtype=float)
    best_Cs = []
    for c in tqdm(uniq_chroms, desc="chrom-LOO"):
        mask_test = chroms == c
        mask_train = ~mask_test
        X_tr = X.loc[mask_train].to_numpy()
        y_tr = y[mask_train]
        g_tr = chroms[mask_train]
        X_te = X.loc[mask_test].to_numpy()
        preds, bestC = train_predict(X_tr, y_tr, g_tr, X_te)
        scores[mask_test] = preds
        best_Cs.append((c, bestC))

    # metrics
    aupr = float(average_precision_score(y, scores))
    auroc = float(roc_auc_score(y, scores))
    aupr_chr, nc = weighted_per_chrom_auprc(y, scores, chroms)
    p_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    brier = float(brier_score_loss(y, p_norm))
    e = ece(y, p_norm)
    cov = coverage_auprc(y, scores)

    metrics = {
        "feature_set": args.feature_set, "n": len(V), "n_chroms": nc,
        "AUPRC": aupr, "AUROC": auroc, "AUPRC_per_chrom": aupr_chr,
        "Brier": brier, "ECE": e,
        "coverage_AUPRC": cov,
        "best_Cs_per_chrom_loo": {c: v for c, v in best_Cs},
    }

    print(f"\n=== {args.feature_set} ===")
    print(f"AUPRC={aupr:.4f}  AUROC={auroc:.4f}  AUPRC_per_chrom={aupr_chr:.4f}  "
          f"Brier={brier:.4f}  ECE={e:.4f}")
    print(f"coverage-AUPRC: " + "  ".join(f"@{c:.2f}={v:.3f}" for c, v in cov.items()))

    # save scores + metrics
    V_out = V[["chrom"]].copy()
    V_out["label"] = y
    V_out["score"] = scores
    V_out.to_parquet(out / "scores.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"saved: {out}/scores.parquet, metrics.json")

    # sanity gate
    if args.assert_leaderboard is not None:
        diff = abs(aupr_chr - args.assert_leaderboard)
        status = "PASS" if diff < args.tol else "FAIL"
        print(f"\n[sanity] AUPRC_per_chrom {aupr_chr:.4f} vs leaderboard "
              f"{args.assert_leaderboard:.4f} (|Δ|={diff:.4f}, tol={args.tol}) → {status}")
        if status == "FAIL":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
