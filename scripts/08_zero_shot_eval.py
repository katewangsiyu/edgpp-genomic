"""Zero-shot evaluation under TraitGym leaderboard methodology.

Unlike 07_evaluate.py (which scores only val chroms — needed for distilled
students that saw train chroms), this script evaluates scores computable
from features alone (no training) on the *full* TraitGym test set, matching
the leaderboard's AUPRC_by_chrom_weighted_average convention.

Built-in sanity gate: Borzoi_L2_L2 L2-norm must give AUPRC_per_chrom = 0.4356
on mendelian_traits_matched_9 (matches official leaderboard exactly).

Usage:
    python scripts/08_zero_shot_eval.py \\
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --teacher-parquet data/raw/traitgym/mendelian_traits_matched_9/features/Borzoi_L2_L2.parquet
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

BORZOI_L2L2_COLS = ["CAGE", "DNASE", "ATAC", "CHIP", "RNA", "all"]
# songlab/TraitGym HF dataset, AUPRC_by_chrom_weighted_average / all / Borzoi_L2_L2.plus.all
LEADERBOARD_BORZOI_L2_L2 = {
    "mendelian_traits_matched_9": 0.4356,
    "complex_traits_matched_9":   0.2357,
}


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


def compute(y, s, chroms):
    p = (s - s.min()) / (s.max() - s.min() + 1e-12)
    aupr_chr, nc = weighted_per_chrom_auprc(y, s, chroms)
    return {
        "AUPRC": float(average_precision_score(y, s)),
        "AUROC": float(roc_auc_score(y, s)),
        "AUPRC_per_chrom": aupr_chr,
        "n_chroms": nc,
        "Brier": float(brier_score_loss(y, p)),
        "ECE": ece(y, p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--teacher-parquet", required=True)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--sanity-tol", type=float, default=0.005,
                    help="tolerance for Borzoi_L2_L2 leaderboard sanity gate")
    args = ap.parse_args()

    test = pd.read_parquet(args.test_parquet)
    teacher = pd.read_parquet(args.teacher_parquet)
    assert len(test) == len(teacher), f"row mismatch: {len(test)} vs {len(teacher)}"
    df = pd.concat([test.reset_index(drop=True),
                    teacher[BORZOI_L2L2_COLS].reset_index(drop=True)], axis=1)

    y = df["label"].astype(int).to_numpy()
    chroms = df["chrom"].astype(str).to_numpy()
    T = df[BORZOI_L2L2_COLS].to_numpy().astype(float)

    print(f"test n={len(df)} pos={int(y.sum())} ({y.mean():.2%}) chroms={len(set(chroms))}")

    scores = {
        "Borzoi_L2_L2":      np.linalg.norm(T, axis=1),
        "Borzoi_L1_sum":     np.abs(T).sum(axis=1),
        "Borzoi_max_abs":    np.abs(T).max(axis=1),
        "Borzoi_all_abs":    np.abs(T[:, BORZOI_L2L2_COLS.index("all")]),
        "Borzoi_CAGE_abs":   np.abs(T[:, BORZOI_L2L2_COLS.index("CAGE")]),
        "Borzoi_DNASE_abs":  np.abs(T[:, BORZOI_L2L2_COLS.index("DNASE")]),
    }
    if "tss_dist" in df.columns:
        scores["neg_log_tss_dist"] = -np.log1p(np.abs(df["tss_dist"].astype(float).to_numpy()))

    out = {}
    print(f"\n{'score':<20} {'AUPRC':>8} {'AUROC':>8} {'AUPRC/chr':>10} {'Brier':>8} {'ECE':>8}")
    for name, s in scores.items():
        m = compute(y, s, chroms)
        out[name] = m
        print(f"{name:<20} {m['AUPRC']:>8.4f} {m['AUROC']:>8.4f} "
              f"{m['AUPRC_per_chrom']:>10.4f} {m['Brier']:>8.4f} {m['ECE']:>8.4f}")

    # Leaderboard sanity gate
    ref = out["Borzoi_L2_L2"]["AUPRC_per_chrom"]
    dataset_name = Path(args.test_parquet).parent.name
    if dataset_name in LEADERBOARD_BORZOI_L2_L2:
        lb = LEADERBOARD_BORZOI_L2_L2[dataset_name]
        diff = abs(ref - lb)
        status = "PASS" if diff < args.sanity_tol else "FAIL"
        print(f"\n[sanity] {dataset_name} Borzoi_L2_L2 AUPRC_per_chrom={ref:.4f} "
              f"vs leaderboard {lb:.4f} (|Δ|={diff:.4f}, tol={args.sanity_tol}) → {status}")
        if status == "FAIL":
            raise SystemExit(1)
    else:
        print(f"\n[sanity] no leaderboard ref for dataset '{dataset_name}' — gate skipped")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
