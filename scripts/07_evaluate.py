"""Evaluate a scores parquet with TraitGym-style metrics.

SCOPE WARNING: this script evaluates on whatever variants are in the input
scores parquet. When that parquet comes from 06_train.py, it contains only
the val-chrom split (chr{17,18,19,20,21,22,X} ≈ 1030/3380 variants for
mendelian_traits_matched_9). The resulting AUPRC_per_chrom is NOT on the
same scale as the TraitGym leaderboard — which computes the metric across
ALL chroms of the test set.

For leaderboard-scale numbers on zero-shot features (no training), use
scripts/08_zero_shot_eval.py. For distilled students, the full-test
comparison requires chrom leave-one-out, not a single train/val split.

Expected columns: label (0/1), chrom (str), plus one or more score columns.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def weighted_per_chrom_auprc(y, s, chroms):
    keep = []
    for c in sorted(set(chroms)):
        m = chroms == c
        if m.sum() < 10 or y[m].sum() == 0:
            continue
        keep.append((m.sum(), average_precision_score(y[m], s[m])))
    if not keep:
        return float("nan")
    w = np.array([x[0] for x in keep], dtype=float)
    a = np.array([x[1] for x in keep], dtype=float)
    return float((w * a).sum() / w.sum())


def ece(y, p, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e, N = 0.0, len(y)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        e += m.sum() / N * abs(y[m].mean() - p[m].mean())
    return float(e)


def compute(df: pd.DataFrame, score_col: str) -> dict:
    y = df["label"].to_numpy().astype(int)
    s = df[score_col].to_numpy().astype(float)
    if len(set(y)) < 2:
        return {"AUPRC": float("nan")}
    p = (s - s.min()) / (s.max() - s.min() + 1e-12)
    return {
        "AUPRC": float(average_precision_score(y, s)),
        "AUROC": float(roc_auc_score(y, s)),
        "AUPRC_per_chrom": weighted_per_chrom_auprc(y, s, df["chrom"].to_numpy()),
        "Brier": float(brier_score_loss(y, p)),
        "ECE": ece(y, p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--score-cols", nargs="+", default=None,
                    help="Default: all columns starting with 'score_'")
    args = ap.parse_args()

    df = pd.read_parquet(args.scores)
    cols = args.score_cols or [c for c in df.columns if c.startswith("score_")]
    assert cols, "no score columns found"

    out = {}
    n_chr = len(set(df["chrom"].astype(str)))
    scope = "full-test" if n_chr >= 18 else f"val-chroms only ({n_chr} chroms, INTERNAL — not leaderboard-comparable)"
    print(f"\n=== {args.scores} | n={len(df)} pos={df.label.sum()} | scope: {scope} ===")
    for col in cols:
        m = compute(df, col)
        out[col] = m
        row = " | ".join(f"{k}={v:.4f}" for k, v in m.items())
        print(f"  {col:<16} {row}")

    out_path = Path(args.scores).parent / "metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
