"""Empirical H2H comparison: HCCP vs RLCP, Weighted CP, Self-Calibrating CP.

Reads precomputed (p_hat, sigma, y, chrom) from hetero_head/scores_with_sigma.parquet
and runs four CP variants in chrom-LOO with the same nonconformity score
s(x, y) = |y - p(x)| / sigma(x):

  HCCP    : Mondrian (y x sigma-bin), K=K_cv (reproduces existing tab:h2h)
  RLCP    : random localization in (p_hat, sigma) 2D space, Gaussian kernel,
            bandwidth via Silverman's rule (Hore & Barber 2025 random-localized
            CP, simplified to deterministic kernel weights for reproducibility).
  WCP     : weighted CP with density-ratio weights from a logistic discriminator
            predicting source-vs-target chromosome (Tibshirani et al. 2019).
  SC-CP   : Self-Calibrating CP (Van der Laan 2024), binning along the
            calibrated point prediction p_hat with K=K_cv equiprobable bins.

Usage:
    python T_tools/cp_baselines_h2h.py --dataset complex --K 2
    python T_tools/cp_baselines_h2h.py --dataset mendelian --K 3
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
ALPHA = 0.10
EPS = 1e-6


def load_scores(dataset: str) -> pd.DataFrame:
    name = "complex" if dataset == "complex" else "mendelian"
    suffix = "complex" if name == "complex" else "mendelian"
    path = REPO / "outputs" / "hetero_head" / f"CADD+GPN-MSA+Borzoi_{suffix}_abs" / "scores_with_sigma.parquet"
    df = pd.read_parquet(path)
    df["chrom"] = df["chrom"].astype(str)
    return df


def coverage_metrics(in_set_0, in_set_1, y, sigma, K, bin_idx, label="m"):
    covered = np.where(y == 0, in_set_0, in_set_1)
    marg = float(covered.mean())
    cov_y1 = float(covered[y == 1].mean()) if (y == 1).any() else np.nan
    cov_y0 = float(covered[y == 0].mean()) if (y == 0).any() else np.nan

    cell_gaps = []
    for k in (0, 1):
        for b in range(K):
            mask = (y == k) & (bin_idx == b)
            if mask.sum() >= 5:
                cov_kb = float(covered[mask].mean())
                cell_gaps.append(abs(cov_kb - (1 - ALPHA)))
    sigma_bin_gap = float(max(cell_gaps)) if cell_gaps else np.nan
    return dict(label=label, marginal_coverage=marg, coverage_pos=cov_y1,
                coverage_neg=cov_y0, sigma_bin_gap=sigma_bin_gap,
                n_test=int(len(y)), frac_singleton=float((in_set_0 ^ in_set_1).mean()))


def hccp_chrom_loo(df: pd.DataFrame, K: int, alpha: float = ALPHA):
    """Reproduce HCCP B4 via direct Mondrian (y x sigma-bin) calibration."""
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values

    in0 = np.zeros(len(df), dtype=bool)
    in1 = np.zeros(len(df), dtype=bool)
    bin_idx_all = np.zeros(len(df), dtype=int)

    for c in sorted(set(chroms)):
        mask_te = chroms == c
        mask_cal = ~mask_te
        s_cal = np.abs(y[mask_cal] - p[mask_cal]) / sigma[mask_cal]
        edges = np.quantile(sigma[mask_cal], np.linspace(0, 1, K + 1)[1:-1])
        b_cal = np.searchsorted(edges, sigma[mask_cal])
        b_te = np.searchsorted(edges, sigma[mask_te])
        bin_idx_all[mask_te] = b_te

        for k in (0, 1):
            for b in range(K):
                cell = (y[mask_cal] == k) & (b_cal == b)
                n_kb = int(cell.sum())
                if n_kb >= int(np.ceil(1.0 / alpha)):
                    qhat = np.quantile(s_cal[cell], np.ceil((n_kb + 1) * (1 - alpha)) / n_kb)
                else:  # fallback to pooled class
                    cell_p = (y[mask_cal] == k)
                    n_k = int(cell_p.sum())
                    qhat = np.quantile(s_cal[cell_p], np.ceil((n_k + 1) * (1 - alpha)) / n_k)
                te_cell = mask_te.copy()
                te_cell[mask_te] = (b_te == b)
                if k == 0:
                    in0[te_cell] = np.abs(0 - p[te_cell]) <= qhat * sigma[te_cell]
                else:
                    in1[te_cell] = np.abs(1 - p[te_cell]) <= qhat * sigma[te_cell]
    return in0, in1, bin_idx_all


def rlcp_chrom_loo(df: pd.DataFrame, K: int, alpha: float = ALPHA, h: float | None = None):
    """Random/kernel-localized CP in (p_hat, sigma) 2D space.

    For each test point, weight calibration scores by w_i = exp(-||z_test - z_i||^2 / 2h^2)
    where z = (p_hat, sigma), and use the weighted (1-alpha)-quantile as the threshold.
    Bandwidth via Silverman's rule on calibration z.
    """
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values
    z = np.column_stack([p, sigma])

    in0 = np.zeros(len(df), dtype=bool)
    in1 = np.zeros(len(df), dtype=bool)
    bin_idx_all = np.zeros(len(df), dtype=int)  # placeholder for sigma-bin metric

    for c in sorted(set(chroms)):
        mask_te = chroms == c
        mask_cal = ~mask_te
        s_cal = np.abs(y[mask_cal] - p[mask_cal]) / sigma[mask_cal]
        z_cal = z[mask_cal]
        z_te = z[mask_te]

        edges = np.quantile(sigma[mask_cal], np.linspace(0, 1, K + 1)[1:-1])
        bin_idx_all[mask_te] = np.searchsorted(edges, sigma[mask_te])

        # Silverman bandwidth on z_cal (per-coord then mean)
        n_cal = len(z_cal)
        h_use = h if h is not None else np.mean(np.std(z_cal, axis=0)) * (4.0 / (3.0 * n_cal)) ** 0.2

        # For each test point, compute kernel-weighted (1-alpha)-quantile of s_cal
        for j_te, idx in enumerate(np.where(mask_te)[0]):
            d2 = np.sum((z_cal - z_te[j_te]) ** 2, axis=1)
            w = np.exp(-d2 / (2 * h_use ** 2))
            # Weighted quantile (Hore-Barber adjustment: include test point with weight 1)
            order = np.argsort(s_cal)
            w_sorted = w[order]
            s_sorted = s_cal[order]
            cum = np.cumsum(w_sorted) / (w_sorted.sum() + 1.0)
            target = 1 - alpha
            qi = np.searchsorted(cum, target)
            qi = min(qi, len(s_sorted) - 1)
            qhat = s_sorted[qi]
            in0[idx] = abs(0 - p[idx]) <= qhat * sigma[idx]
            in1[idx] = abs(1 - p[idx]) <= qhat * sigma[idx]
    return in0, in1, bin_idx_all


def weighted_cp_chrom_loo(df: pd.DataFrame, K: int, alpha: float = ALPHA):
    """Weighted CP with density-ratio weights via logistic discriminator
    predicting test-chrom (target) vs other chroms (source).
    """
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values
    Z = np.column_stack([p, sigma])

    in0 = np.zeros(len(df), dtype=bool)
    in1 = np.zeros(len(df), dtype=bool)
    bin_idx_all = np.zeros(len(df), dtype=int)

    for c in sorted(set(chroms)):
        mask_te = chroms == c
        mask_cal = ~mask_te
        s_cal = np.abs(y[mask_cal] - p[mask_cal]) / sigma[mask_cal]

        edges = np.quantile(sigma[mask_cal], np.linspace(0, 1, K + 1)[1:-1])
        bin_idx_all[mask_te] = np.searchsorted(edges, sigma[mask_te])

        # Discriminator: target=test chrom (label 1), source=other chroms (label 0)
        y_disc = mask_te.astype(int)
        clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=42)
        clf.fit(Z, y_disc)
        # Likelihood ratio = p(target|z) / p(source|z); on calibration set
        prob_cal = clf.predict_proba(Z[mask_cal])[:, 1]
        prob_cal = np.clip(prob_cal, 0.05, 0.95)
        w_cal = prob_cal / (1.0 - prob_cal)

        # Weighted (1-alpha) quantile of s_cal w.r.t. w_cal
        order = np.argsort(s_cal)
        w_sorted = w_cal[order]
        s_sorted = s_cal[order]
        cum = np.cumsum(w_sorted) / (w_sorted.sum() + 1.0)
        qi = np.searchsorted(cum, 1 - alpha)
        qi = min(qi, len(s_sorted) - 1)
        qhat = s_sorted[qi]

        in0[mask_te] = np.abs(0 - p[mask_te]) <= qhat * sigma[mask_te]
        in1[mask_te] = np.abs(1 - p[mask_te]) <= qhat * sigma[mask_te]
    return in0, in1, bin_idx_all


def sccp_chrom_loo(df: pd.DataFrame, K: int, alpha: float = ALPHA):
    """Self-Calibrating CP: bin along p_hat with K equiprobable bins.

    This is the orthogonal-axis variant of HCCP: bin along the calibrated point
    prediction p_hat rather than the variance head sigma. Per Van der Laan 2024.
    """
    chroms = df["chrom"].values
    p = df["p_hat"].values
    sigma = df["sigma"].values + EPS
    y = df["label"].values

    in0 = np.zeros(len(df), dtype=bool)
    in1 = np.zeros(len(df), dtype=bool)
    bin_idx_all = np.zeros(len(df), dtype=int)

    for c in sorted(set(chroms)):
        mask_te = chroms == c
        mask_cal = ~mask_te
        s_cal = np.abs(y[mask_cal] - p[mask_cal]) / sigma[mask_cal]

        # Bin along p_hat (SC-CP axis); use K equiprobable bins
        p_edges = np.quantile(p[mask_cal], np.linspace(0, 1, K + 1)[1:-1])
        b_cal = np.searchsorted(p_edges, p[mask_cal])
        b_te = np.searchsorted(p_edges, p[mask_te])

        # Also compute sigma-bin index for the metric (orthogonal)
        sig_edges = np.quantile(sigma[mask_cal], np.linspace(0, 1, K + 1)[1:-1])
        bin_idx_all[mask_te] = np.searchsorted(sig_edges, sigma[mask_te])

        for b in range(K):
            cell_cal = (b_cal == b)
            n_b = int(cell_cal.sum())
            if n_b >= int(np.ceil(1.0 / alpha)):
                qhat = np.quantile(s_cal[cell_cal], np.ceil((n_b + 1) * (1 - alpha)) / n_b)
            else:
                qhat = np.quantile(s_cal, np.ceil((len(s_cal) + 1) * (1 - alpha)) / len(s_cal))
            te_cell = mask_te.copy()
            te_cell[mask_te] = (b_te == b)
            in0[te_cell] = np.abs(0 - p[te_cell]) <= qhat * sigma[te_cell]
            in1[te_cell] = np.abs(1 - p[te_cell]) <= qhat * sigma[te_cell]
    return in0, in1, bin_idx_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mendelian", "complex"], required=True)
    ap.add_argument("--K", type=int, required=True)
    args = ap.parse_args()

    df = load_scores(args.dataset)
    print(f"Loaded {args.dataset}: {len(df)} variants, K={args.K}")
    K = args.K

    results = {}

    print("\n--- HCCP (B4 reproduction) ---")
    i0, i1, bi = hccp_chrom_loo(df, K)
    m = coverage_metrics(i0, i1, df["label"].values, df["sigma"].values, K, bi, label="HCCP")
    print(m)
    results["HCCP"] = m

    print("\n--- RLCP (kernel localization in (p, sigma) 2D) ---")
    i0, i1, bi = rlcp_chrom_loo(df, K)
    m = coverage_metrics(i0, i1, df["label"].values, df["sigma"].values, K, bi, label="RLCP")
    print(m)
    results["RLCP"] = m

    print("\n--- Weighted CP (chrom-shift density ratio) ---")
    i0, i1, bi = weighted_cp_chrom_loo(df, K)
    m = coverage_metrics(i0, i1, df["label"].values, df["sigma"].values, K, bi, label="WeightedCP")
    print(m)
    results["WeightedCP"] = m

    print("\n--- Self-Calibrating CP (p_hat-bin Mondrian) ---")
    i0, i1, bi = sccp_chrom_loo(df, K)
    m = coverage_metrics(i0, i1, df["label"].values, df["sigma"].values, K, bi, label="SCCP")
    print(m)
    results["SCCP"] = m

    out_dir = REPO / "R_raw" / "cp_baselines_h2h"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_K{K}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
