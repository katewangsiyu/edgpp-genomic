"""Heteroscedastic class-conditional conformal — Path A Phase 2 core.

New nonconformity score (formulation_v0.md §1.5):

    s(x, y) = |y - p̂(x)| / (σ̂(x) + ε)

For binary classification (y ∈ {0,1}):
    s(x, 0) = p̂(x) / (σ̂(x) + ε)
    s(x, 1) = (1 - p̂(x)) / (σ̂(x) + ε)

Class-conditional (Mondrian-by-y) split conformal:
    For each class k:
        S_k = {s(x_j, y_j) : y_j = k, j in calibration set}
        q̂_k = Quantile_⌈(n_k+1)(1-α)⌉/n_k  (S_k)
    Prediction set: C_α(x) = {k : s(x, k) ≤ q̂_k}

Chrom-LOO calibration: for each test chrom c, calibration set = variants on all
other chroms (stratified by label).

When σ̂ ≡ 1 this reduces to Day 10 class-cond LAC (sanity check baseline).

Usage:
    python scripts/14_conformal_hetero.py \\
        --sigma-scores outputs/hetero_head/CADD+GPN-MSA+Borzoi_mendelian_abs/scores_with_sigma.parquet \\
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --out-dir outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_mendelian_abs \\
        --alpha 0.10
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


EPS_DEFAULT = 1e-4


# ---------- nonconformity scores ----------

def make_hetero_score(eps: float):
    """Factory: returns a score_fn with the given denominator floor ε."""
    def score(p, sigma, y_candidate):
        return np.abs(y_candidate - p) / (sigma + eps)
    score.__name__ = f"hetero_eps{eps}"
    return score


def homosc_score(p: np.ndarray, sigma: np.ndarray, y_candidate: np.ndarray) -> np.ndarray:
    """σ̂≡1 special case — same as Day 10 LAC. For ablation."""
    return np.abs(y_candidate - p)


# ---------- class-conditional (Mondrian-by-y) chrom-LOO calibration ----------

def class_cond_hetero_conformal(p: np.ndarray, sigma: np.ndarray, y: np.ndarray,
                                chroms: np.ndarray, alpha: float,
                                score_fn=None):
    if score_fn is None:
        score_fn = make_hetero_score(EPS_DEFAULT)
    """Return per-variant prediction sets using chrom-LOO class-cond calibration.

    For each test chrom c:
        cal = variants on other chroms
        For k in {0, 1}:
            S_k = score_fn(p, σ, k) on variants in cal with y=k
            q̂_k(c) = (⌈(n_k+1)(1-α)⌉ / n_k)-quantile of S_k  (+∞ if empty)
        For each test variant i:
            include k in C(x_i) iff score_fn(p[i], σ[i], k) ≤ q̂_k(c)
    """
    n = len(y)
    uniq_chroms = sorted(set(chroms))
    pred_sets = [set() for _ in range(n)]
    q_hat_per_chrom = {c: {0: np.inf, 1: np.inf} for c in uniq_chroms}

    for c in uniq_chroms:
        mask_test = chroms == c
        mask_cal = ~mask_test
        for k in (0, 1):
            cal_k = mask_cal & (y == k)
            n_k = int(cal_k.sum())
            if n_k == 0:
                q = np.inf
            else:
                s_cal = score_fn(p[cal_k], sigma[cal_k], np.full(n_k, k))
                # Conformal quantile (finite-sample corrected)
                level = min(1.0, (np.ceil((n_k + 1) * (1 - alpha)) / n_k))
                q = float(np.quantile(s_cal, level, method="higher"))
            q_hat_per_chrom[c][k] = q

        for i in np.where(mask_test)[0]:
            for k in (0, 1):
                s_ik = score_fn(p[i:i+1], sigma[i:i+1], np.array([k]))[0]
                if s_ik <= q_hat_per_chrom[c][k]:
                    pred_sets[i].add(k)

    return pred_sets, q_hat_per_chrom


def mondrian_class_sigma_conformal(p, sigma, y, chroms, alpha,
                                   n_sigma_bins=5, score_fn=None):
    """Mondrian-by-(y × σ̂-bin) chrom-LOO calibration.

    Intuition: pool calibration only within class k and within a σ̂-stratum
    comparable to the test variant. This delivers LOCAL class-conditional
    coverage by construction (Vovk 2003 Mondrian argument).

    Uses global σ̂-quantile edges as bin boundaries (consistent across chroms).
    """
    if score_fn is None:
        score_fn = make_hetero_score(EPS_DEFAULT)
    n = len(y)
    pred_sets = [set() for _ in range(n)]

    # Global σ̂ quantile edges (label-free so calibration and test share bins).
    edges = np.quantile(sigma, np.linspace(0, 1, n_sigma_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    sigma_bin = np.digitize(sigma, edges[1:-1])  # integer 0..n_sigma_bins-1

    for c in sorted(set(chroms)):
        mask_test = chroms == c
        mask_cal = ~mask_test
        q = {}  # (k, bin) -> quantile
        for k in (0, 1):
            for b in range(n_sigma_bins):
                cal_kb = mask_cal & (y == k) & (sigma_bin == b)
                n_kb = int(cal_kb.sum())
                if n_kb < 5:
                    # Too few → fall back to pooled class-cond within chrom calibration
                    cal_k = mask_cal & (y == k)
                    n_k = int(cal_k.sum())
                    if n_k == 0:
                        q[(k, b)] = np.inf
                        continue
                    s_cal = score_fn(p[cal_k], sigma[cal_k], np.full(n_k, k))
                    level = min(1.0, np.ceil((n_k + 1) * (1 - alpha)) / n_k)
                    q[(k, b)] = float(np.quantile(s_cal, level, method="higher"))
                else:
                    s_cal = score_fn(p[cal_kb], sigma[cal_kb], np.full(n_kb, k))
                    level = min(1.0, np.ceil((n_kb + 1) * (1 - alpha)) / n_kb)
                    q[(k, b)] = float(np.quantile(s_cal, level, method="higher"))
        for i in np.where(mask_test)[0]:
            b = int(sigma_bin[i])
            for k in (0, 1):
                s_ik = score_fn(p[i:i+1], sigma[i:i+1], np.array([k]))[0]
                if s_ik <= q[(k, b)]:
                    pred_sets[i].add(k)
    return pred_sets


# ---------- evaluation ----------

def per_chrom_coverage(covered: np.ndarray, chroms: np.ndarray):
    out = {}
    for c in sorted(set(chroms)):
        m = chroms == c
        if m.sum() < 5:
            continue
        out[c] = float(covered[m].mean())
    return out


def coverage_by_sigma_bin(covered: np.ndarray, sigma: np.ndarray, n_bins: int = 10):
    """Local-coverage proxy: bin by σ̂ decile → empirical coverage per bin.

    This is an x-space partition by "difficulty" (σ̂). Uniform coverage across
    bins is empirical evidence for T3 (local conditional coverage).
    """
    try:
        bins = pd.qcut(sigma, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(sigma, bins=n_bins, labels=False)
    out = []
    for b in sorted(set(bins[~pd.isna(bins)])):
        m = bins == b
        if m.sum() < 5:
            continue
        out.append({
            "bin": int(b),
            "n": int(m.sum()),
            "sigma_mean": float(sigma[m].mean()),
            "coverage": float(covered[m].mean()),
        })
    return out


def coverage_by_phat_bin(covered: np.ndarray, p: np.ndarray):
    """Coverage by p̂ region: 0-10, 10-30, 30-70, 70-90, 90-100."""
    edges = [-0.01, 0.10, 0.30, 0.70, 0.90, 1.01]
    labels = ["0-10", "10-30", "30-70", "70-90", "90-100"]
    bins = pd.cut(p, bins=edges, labels=labels)
    out = []
    for lab in labels:
        m = bins == lab
        if m.sum() < 5:
            continue
        out.append({
            "phat_bin": lab,
            "n": int(m.sum()),
            "coverage": float(covered[m].mean()),
        })
    return out


def evaluate(pred_sets, y, sigma, p, chroms, alpha, label=""):
    n = len(y)
    covered = np.array([y[i] in ps for i, ps in enumerate(pred_sets)])
    sizes = np.array([len(ps) for ps in pred_sets])

    coverage = float(covered.mean())
    pos = y == 1
    neg = y == 0
    cov_pos = float(covered[pos].mean()) if pos.any() else float("nan")
    cov_neg = float(covered[neg].mean()) if neg.any() else float("nan")

    chrom_cov = per_chrom_coverage(covered, chroms)
    sigma_tbl = coverage_by_sigma_bin(covered, sigma)
    phat_tbl = coverage_by_phat_bin(covered, p)

    chrom_cov_values = list(chrom_cov.values())
    max_gap = float(max(chrom_cov_values) - min(chrom_cov_values)) if chrom_cov_values else float("nan")
    sigma_cov_values = [row["coverage"] for row in sigma_tbl]
    sigma_cov_range = (
        float(max(sigma_cov_values) - min(sigma_cov_values)) if sigma_cov_values else float("nan")
    )

    metrics = {
        "label": label,
        "alpha": alpha,
        "coverage": coverage,
        "target_coverage": 1 - alpha,
        "coverage_gap": coverage - (1 - alpha),
        "coverage_pos": cov_pos,
        "coverage_neg": cov_neg,
        "frac_empty": float((sizes == 0).mean()),
        "frac_singleton": float((sizes == 1).mean()),
        "frac_both": float((sizes == 2).mean()),
        "per_chrom_coverage": chrom_cov,
        "per_chrom_max_gap": max_gap,
        "coverage_by_sigma_bin": sigma_tbl,
        "sigma_cov_range": sigma_cov_range,
        "coverage_by_phat_bin": phat_tbl,
    }

    print(f"\n{'='*64}")
    print(f"  {label}  (α={alpha}, target ≥ {1-alpha:.2f})")
    print(f"{'='*64}")
    print(f"  marginal cov={coverage:.4f}  Cov|pos={cov_pos:.4f}  Cov|neg={cov_neg:.4f}")
    print(f"  sizes: empty={metrics['frac_empty']:.3f}  single={metrics['frac_singleton']:.3f}  "
          f"both={metrics['frac_both']:.3f}")
    print(f"  per-chrom cov range: [{min(chrom_cov_values):.3f}, {max(chrom_cov_values):.3f}] "
          f"(max gap {max_gap:.3f})")
    print(f"  σ̂-bin cov range: {sigma_cov_range:.3f}  (local-coverage empirical gap)")
    for row in sigma_tbl:
        print(f"    σ̂-bin {row['bin']}: n={row['n']:>4}  σ̂={row['sigma_mean']:.4f}  "
              f"cov={row['coverage']:.3f}")
    print(f"  p̂-bin coverage:")
    for row in phat_tbl:
        print(f"    p̂∈[{row['phat_bin']:>6}]: n={row['n']:>4}  cov={row['coverage']:.3f}")
    return metrics


def alpha_sweep(p, sigma, y, chroms, score_fn, alphas=(0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50), skip_signature=None):
    out = {}
    for a in alphas:
        pred_sets, _ = class_cond_hetero_conformal(p, sigma, y, chroms, a, score_fn)
        covered = np.array([y[i] in ps for i, ps in enumerate(pred_sets)])
        sizes = np.array([len(ps) for ps in pred_sets])
        out[f"{a:.2f}"] = {
            "coverage": float(covered.mean()),
            "target": 1 - a,
            "frac_singleton": float((sizes == 1).mean()),
            "frac_both": float((sizes == 2).mean()),
        }
    return out


def alpha_sweep_mondrian(p, sigma, y, chroms, n_bins=5,
                          alphas=(0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)):
    out = {}
    for a in alphas:
        pred_sets = mondrian_class_sigma_conformal(
            p, sigma, y, chroms, a, n_sigma_bins=n_bins)
        covered = np.array([y[i] in ps for i, ps in enumerate(pred_sets)])
        sizes = np.array([len(ps) for ps in pred_sets])
        cov_by_bin = coverage_by_sigma_bin(covered, sigma, n_bins=n_bins)
        sigma_cov_range = (max(b["coverage"] for b in cov_by_bin)
                           - min(b["coverage"] for b in cov_by_bin))
        out[f"{a:.2f}"] = {
            "coverage": float(covered.mean()),
            "target": 1 - a,
            "frac_singleton": float((sizes == 1).mean()),
            "frac_both": float((sizes == 2).mean()),
            "sigma_cov_range": float(sigma_cov_range),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-scores", required=True,
                    help="Output of 13_hetero_head.py: scores_with_sigma.parquet")
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--eps", type=float, default=EPS_DEFAULT,
                    help="Denominator floor ε in s(x,y)=|y-p̂|/(σ̂+ε).")
    ap.add_argument("--sigma-floor-quantile", type=float, default=0.0,
                    help="Clip σ̂ to its q-th quantile from below (e.g. 0.25). "
                    "Prevents pathologically-small σ̂ from dominating scores.")
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    sig = pd.read_parquet(args.sigma_scores).reset_index(drop=True)
    assert len(V) == len(sig), "row mismatch"
    assert (V["chrom"].astype(str).values == sig["chrom"].astype(str).values).all()
    assert (V["label"].astype(int).values == sig["label"].astype(int).values).all()

    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()
    p = sig["p_hat"].to_numpy()
    sigma = sig["sigma"].to_numpy()

    sigma_raw = sigma.copy()
    if args.sigma_floor_quantile > 0:
        floor = float(np.quantile(sigma_raw, args.sigma_floor_quantile))
        sigma = np.maximum(sigma_raw, floor)
        print(f"[σ̂ floor] clipped σ̂ to q{args.sigma_floor_quantile:.2f} = {floor:.4f} "
              f"({(sigma_raw < floor).mean():.1%} variants clipped)")
    eps = float(args.eps)
    hetero_score = make_hetero_score(eps)

    print(f"[load] n={len(V)} pos={int(y.sum())} α={args.alpha}  ε={eps:.4f}")
    print(f"  σ̂ mean={sigma.mean():.4f}  std={sigma.std():.4f}  "
          f"q05/50/95={np.quantile(sigma,0.05):.3f}/{np.quantile(sigma,0.50):.3f}/"
          f"{np.quantile(sigma,0.95):.3f}")

    # 1. Heteroscedastic class-conditional (OUR METHOD)
    ps_het, q_het = class_cond_hetero_conformal(p, sigma, y, chroms, args.alpha,
                                                 score_fn=hetero_score)
    m_het = evaluate(ps_het, y, sigma, p, chroms, args.alpha,
                     label=f"Hetero class-cond (ε={eps:.4f})")

    # 2. Homoscedastic class-conditional (ABLATION = Day 10 class-cond with LAC score)
    ps_hom, q_hom = class_cond_hetero_conformal(p, sigma, y, chroms, args.alpha,
                                                 score_fn=homosc_score)
    m_hom = evaluate(ps_hom, y, sigma, p, chroms, args.alpha,
                     label="Homosc class-cond (s=|y−p̂|, σ̂≡1) — ablation")

    # 3. Mondrian (y × σ̂-bin) — local class-cond conformal (T3 empirical target)
    ps_mon = mondrian_class_sigma_conformal(p, sigma, y, chroms, args.alpha,
                                             n_sigma_bins=5, score_fn=hetero_score)
    m_mon = evaluate(ps_mon, y, sigma, p, chroms, args.alpha,
                     label="Mondrian (y×σ̂-bin) hetero class-cond")

    results = {
        "alpha": args.alpha,
        "n": int(len(V)),
        "eps": eps,
        "hetero_class_cond": m_het,
        "homosc_class_cond": m_hom,
        "mondrian_y_sigma": m_mon,
    }

    # 3. Alpha sweep for hetero class-cond, homosc class-cond, Mondrian (y×σ̂-bin).
    if not args.skip_sweep:
        print("\n--- α sweep (hetero class-cond) ---")
        sweep_het = alpha_sweep(p, sigma, y, chroms, hetero_score)
        print("\n--- α sweep (homosc class-cond) ---")
        sweep_hom = alpha_sweep(p, sigma, y, chroms, homosc_score)
        print("\n--- α sweep (Mondrian y×σ̂-bin, K=5) ---")
        sweep_mon = alpha_sweep_mondrian(p, sigma, y, chroms, n_bins=5)
        for a, v in sweep_het.items():
            vh = sweep_hom[a]
            vm = sweep_mon[a]
            print(f"  α={a}: het cov={v['coverage']:.4f}  hom cov={vh['coverage']:.4f}  "
                  f"mon cov={vm['coverage']:.4f} σ̂-range={vm['sigma_cov_range']:.3f}")
        results["alpha_sweep_hetero"] = sweep_het
        results["alpha_sweep_homosc"] = sweep_hom
        results["alpha_sweep_mondrian"] = sweep_mon

    # Save results
    # Per-variant prediction sets (encode {0,1} → 0b01 etc.)
    def encode(ps):
        return sum(1 << k for k in ps)

    out_df = V[["chrom"]].copy()
    out_df["label"] = y
    out_df["p_hat"] = p
    out_df["sigma"] = sigma
    out_df["hetero_score_0"] = hetero_score(p, sigma, np.zeros_like(y, dtype=float))
    out_df["hetero_score_1"] = hetero_score(p, sigma, np.ones_like(y, dtype=float))
    out_df["hetero_pset_code"] = [encode(ps) for ps in ps_het]
    out_df["hetero_covered"] = [y[i] in ps for i, ps in enumerate(ps_het)]
    out_df["homosc_pset_code"] = [encode(ps) for ps in ps_hom]
    out_df["homosc_covered"] = [y[i] in ps for i, ps in enumerate(ps_hom)]
    out_df["mondrian_pset_code"] = [encode(ps) for ps in ps_mon]
    out_df["mondrian_covered"] = [y[i] in ps for i, ps in enumerate(ps_mon)]
    out_df.to_parquet(out / "conformal_hetero_scores.parquet", index=False)

    (out / "conformal_hetero_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out}/conformal_hetero_scores.parquet, conformal_hetero_results.json")


if __name__ == "__main__":
    main()
