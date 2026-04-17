"""Trait-Leave-One-Out external validation — Day 14 Plan B1.

Companion to Day 10–13 chrom-LOO work. Chrom-LOO shows generalization across
chromosomes; trait-LOO shows generalization across diseases/traits. Together
they span the two axes along which a VEP deployment must generalize.

For each trait t in the dataset:
  1. Train base classifier p̂ on variants with trait != t (all chroms)
  2. Train σ̂ head on residuals of those same variants
  3. Predict p̂, σ̂ for variants with trait == t
  4. Conformal: Mondrian-by-(y × σ̂-bin) using all non-t variants as the
     calibration pool. Thresholds q̂_{k,b} computed per trait-fold.
  5. Record per-trait coverage / AUPRC.

Output feeds Day 14 report. Key success criteria: per-trait coverage gap
comparable to per-chrom gap (~0.05 in Day 10-13), AUPRC drop modest vs. full
chrom-LOO (≤ 0.05).

Usage:
    python scripts/17_trait_loo.py \\
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --features-dir data/raw/traitgym/mendelian_traits_matched_9/features \\
        --feature-set CADD+Borzoi \\
        --trait-col OMIM \\
        --out-dir outputs/trait_loo/CADD+Borzoi_mendelian \\
        --alpha 0.10

For Complex: --trait-col trait.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
mod_11 = __import__("11_aggregator_gbm")
FEATURE_SETS = mod_11.FEATURE_SETS
load_feature_matrix = mod_11.load_feature_matrix


def build_clf(seed=42):
    return Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("hgb", HistGradientBoostingClassifier(
            max_depth=2, max_iter=100, learning_rate=0.1,
            class_weight="balanced", random_state=seed, early_stopping=False)),
    ])


def build_reg(seed=42):
    return Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("hgb", HistGradientBoostingRegressor(
            loss="squared_error", max_depth=2, max_iter=100, learning_rate=0.1,
            random_state=seed, early_stopping=False)),
    ])


def propagate_trait(V: pd.DataFrame, trait_col: str) -> pd.Series:
    """Return trait_prop: trait label for every row, propagated from positive of
    each match_group. NaN/empty-string controls inherit their match_group's trait."""
    def pick(s):
        # Mendelian OMIM is NaN for controls; Complex trait is ''.
        vals = s[s.notna() & (s.astype(str) != "")]
        if len(vals) == 0:
            return ""
        return vals.iloc[0]
    return V.groupby("match_group")[trait_col].transform(pick)


def fit_trait_loo(X: pd.DataFrame, y: np.ndarray, traits: np.ndarray,
                  chroms: np.ndarray, seed: int):
    """Trait-LOO OOF predictions for p̂ and σ̂.

    σ̂ is fit as a second-stage regressor on |y - p̂_{trait-LOO}(x)| using the
    same trait-LOO split, so it sees no residual info from the held-out trait.
    """
    n = len(y)
    p_hat = np.full(n, np.nan, dtype=float)
    sigma = np.full(n, np.nan, dtype=float)
    uniq = sorted(set(traits))

    # Pass 1: fit p̂ trait-LOO
    for t in tqdm(uniq, desc="p̂ trait-LOO"):
        m_te = traits == t
        m_tr = ~m_te
        clf = build_clf(seed=seed)
        clf.fit(X.loc[m_tr].to_numpy(), y[m_tr])
        p_hat[m_te] = clf.predict_proba(X.loc[m_te].to_numpy())[:, 1]
        # Also get in-sample p̂ for training residuals (needed for σ̂ fit below).
        # We do this after loop using a separate pass to avoid storing per-fold.

    # Pass 2: fit σ̂ trait-LOO. For each trait t, train σ̂ on |y - p̂|
    # using the trait-LOO p̂ of non-t variants (i.e., other traits' OOF).
    # This gives σ̂ that generalizes across traits.
    r_abs = np.abs(y - p_hat)
    for t in tqdm(uniq, desc="σ̂ trait-LOO"):
        m_te = traits == t
        m_tr = ~m_te
        reg = build_reg(seed=seed)
        reg.fit(X.loc[m_tr].to_numpy(), r_abs[m_tr])
        pred = reg.predict(X.loc[m_te].to_numpy())
        sigma[m_te] = np.clip(pred, 1e-4, None)
    return p_hat, sigma


def mondrian_trait_loo_conformal(p, sigma, y, traits, alpha, n_sigma_bins=5,
                                 eps=1e-4, min_cell=5):
    """For each trait t, calibrate Mondrian (y × σ̂-bin) on all non-t variants
    and predict sets for t's variants. σ̂-bin edges are computed once globally."""
    n = len(y)
    # Global σ̂ bin edges (label-agnostic).
    edges = np.quantile(sigma, np.linspace(0, 1, n_sigma_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    sigma_bin = np.digitize(sigma, edges[1:-1])  # 0..n_sigma_bins-1

    pred_sets = [set() for _ in range(n)]
    thresholds_per_trait = {}

    def score_fn(p_arr, sig_arr, k_arr):
        return np.abs(k_arr - p_arr) / (sig_arr + eps)

    for t in sorted(set(traits)):
        m_te = traits == t
        m_cal = ~m_te
        q = {}
        for k in (0, 1):
            for b in range(n_sigma_bins):
                cal_kb = m_cal & (y == k) & (sigma_bin == b)
                n_kb = int(cal_kb.sum())
                if n_kb < min_cell:
                    # Fall back to pooled class-cond over all non-t
                    cal_k = m_cal & (y == k)
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
        thresholds_per_trait[str(t)] = {
            f"{k}_{b}": q[(k, b)] for k in (0, 1) for b in range(n_sigma_bins)
        }
        for i in np.where(m_te)[0]:
            b = int(sigma_bin[i])
            for k in (0, 1):
                s_ik = score_fn(np.array([p[i]]), np.array([sigma[i]]), np.array([k]))[0]
                if s_ik <= q[(k, b)]:
                    pred_sets[i].add(k)
    return pred_sets, thresholds_per_trait, sigma_bin, edges


def per_trait_metrics(y, p_hat, pred_sets, traits, min_n_pos=3):
    """For each trait with enough positives, compute AUPRC and coverage."""
    covered = np.array([y[i] in ps for i, ps in enumerate(pred_sets)])
    sizes = np.array([len(ps) for ps in pred_sets])
    rows = []
    for t in sorted(set(traits)):
        m = traits == t
        n_pos = int(y[m].sum())
        n_tot = int(m.sum())
        if n_pos < min_n_pos:
            continue
        try:
            aup = float(average_precision_score(y[m], p_hat[m]))
        except Exception:
            aup = float("nan")
        rows.append({
            "trait": str(t),
            "n_total": n_tot,
            "n_pos": n_pos,
            "AUPRC": aup,
            "coverage": float(covered[m].mean()),
            "cov_pos": float(covered[m & (y == 1)].mean()) if (m & (y == 1)).any() else float("nan"),
            "cov_neg": float(covered[m & (y == 0)].mean()) if (m & (y == 0)).any() else float("nan"),
            "frac_singleton": float((sizes[m] == 1).mean()),
            "frac_both": float((sizes[m] == 2).mean()),
            "frac_empty": float((sizes[m] == 0).mean()),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--feature-set", required=True, choices=list(FEATURE_SETS))
    ap.add_argument("--trait-col", required=True, choices=["OMIM", "trait"],
                    help="Which column holds the trait/disease identifier.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-sigma-bins", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-n-pos", type=int, default=3,
                    help="Report per-trait metrics only for traits with ≥ this many positives.")
    ap.add_argument("--filter-min-pos-per-trait", type=int, default=0,
                    help="Drop all variants belonging to traits with < this many positives "
                    "BEFORE any fitting. Use for Complex (258 traits long tail) to keep only "
                    "big-trait folds. 0 = keep everything (default).")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    X = load_feature_matrix(Path(args.features_dir), FEATURE_SETS[args.feature_set]).reset_index(drop=True)
    assert len(V) == len(X)

    traits_raw = propagate_trait(V, args.trait_col).to_numpy()
    y_raw = V["label"].astype(int).to_numpy()

    # Optionally filter to big-trait subset BEFORE trait-LOO.
    if args.filter_min_pos_per_trait > 0:
        pos_by_trait = pd.Series(y_raw).groupby(traits_raw).sum()
        keep_traits = set(pos_by_trait[pos_by_trait >= args.filter_min_pos_per_trait].index)
        mask_keep = np.array([t in keep_traits for t in traits_raw])
        print(f"[filter] min_pos_per_trait={args.filter_min_pos_per_trait}: "
              f"keep {len(keep_traits)} traits, {int(mask_keep.sum())}/{len(V)} variants")
        V = V.loc[mask_keep].reset_index(drop=True)
        X = X.loc[mask_keep].reset_index(drop=True)

    traits = propagate_trait(V, args.trait_col).to_numpy()
    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()
    uniq_traits = sorted(set(traits))
    n_traits = len(uniq_traits)
    print(f"[load] n={len(V)} dim={X.shape[1]} traits={n_traits} pos={int(y.sum())} "
          f"({y.mean():.1%})")
    print(f"[trait dist] top 5:")
    tc = pd.Series(traits).value_counts().head(5)
    for tr, cnt in tc.items():
        print(f"  {tr!r}: {cnt}")

    # Trait-LOO fits
    p_hat, sigma = fit_trait_loo(X, y, traits, chroms, args.seed)

    # Marginal AUPRC (trait-LOO OOF)
    aupr = float(average_precision_score(y, p_hat))
    auroc = float(roc_auc_score(y, p_hat))
    print(f"\n[trait-LOO p̂] AUPRC={aupr:.4f}  AUROC={auroc:.4f}")
    print(f"[σ̂] mean={sigma.mean():.4f}  std={sigma.std():.4f}  "
          f"q05/50/95={np.quantile(sigma,0.05):.3f}/{np.quantile(sigma,0.50):.3f}/{np.quantile(sigma,0.95):.3f}")

    # Mondrian conformal with trait-LOO calibration
    pred_sets, thr_per_trait, sigma_bin, edges = mondrian_trait_loo_conformal(
        p_hat, sigma, y, traits, args.alpha, n_sigma_bins=args.n_sigma_bins)

    covered = np.array([y[i] in ps for i, ps in enumerate(pred_sets)])
    sizes = np.array([len(ps) for ps in pred_sets])
    print(f"\n[Mondrian trait-LOO] marginal cov={covered.mean():.4f} "
          f"(target {1-args.alpha:.2f})")
    print(f"  Cov|pos={covered[y==1].mean():.4f}  Cov|neg={covered[y==0].mean():.4f}")
    print(f"  sizes: empty={float((sizes==0).mean()):.3f}  "
          f"single={float((sizes==1).mean()):.3f}  "
          f"both={float((sizes==2).mean()):.3f}")

    # σ̂-bin gap
    sigma_bin_cov = []
    for b in range(args.n_sigma_bins):
        m = sigma_bin == b
        if m.sum() < 5:
            continue
        sigma_bin_cov.append({"bin": b, "n": int(m.sum()),
                              "coverage": float(covered[m].mean()),
                              "sigma_mean": float(sigma[m].mean())})
    sigma_bin_gap = (max(x["coverage"] for x in sigma_bin_cov)
                     - min(x["coverage"] for x in sigma_bin_cov))
    print(f"  σ̂-bin coverage gap: {sigma_bin_gap:.3f}")
    for row in sigma_bin_cov:
        print(f"    bin {row['bin']} n={row['n']:>4} σ̂={row['sigma_mean']:.3f} "
              f"cov={row['coverage']:.3f}")

    # Per-trait metrics
    per_trait = per_trait_metrics(y, p_hat, pred_sets, traits, min_n_pos=args.min_n_pos)
    per_trait.to_csv(out / "per_trait_metrics.csv", index=False)
    print(f"\n[per-trait] reporting {len(per_trait)} traits with ≥{args.min_n_pos} positives")
    if len(per_trait) > 0:
        cov_values = per_trait["coverage"].to_numpy()
        print(f"  AUPRC: median={per_trait['AUPRC'].median():.4f} "
              f"iqr=[{per_trait['AUPRC'].quantile(.25):.4f}, {per_trait['AUPRC'].quantile(.75):.4f}]")
        print(f"  per-trait cov: range=[{cov_values.min():.3f}, {cov_values.max():.3f}] "
              f"gap={cov_values.max() - cov_values.min():.3f}  "
              f"std={cov_values.std():.3f}")

    # Save OOF scores and sigma, prediction sets
    def encode(ps): return sum(1 << k for k in ps)
    out_df = V[["chrom", "pos"]].copy()
    out_df["trait"] = traits
    out_df["label"] = y
    out_df["p_hat_trait_loo"] = p_hat
    out_df["sigma_trait_loo"] = sigma
    out_df["sigma_bin"] = sigma_bin
    out_df["covered"] = covered
    out_df["pset_code"] = [encode(ps) for ps in pred_sets]
    out_df.to_parquet(out / "trait_loo_scores.parquet", index=False)

    results = {
        "feature_set": args.feature_set,
        "trait_col": args.trait_col,
        "n": int(len(V)),
        "n_traits": n_traits,
        "alpha": args.alpha,
        "AUPRC_trait_loo": aupr,
        "AUROC_trait_loo": auroc,
        "marginal_coverage": float(covered.mean()),
        "coverage_pos": float(covered[y == 1].mean()),
        "coverage_neg": float(covered[y == 0].mean()),
        "frac_empty": float((sizes == 0).mean()),
        "frac_singleton": float((sizes == 1).mean()),
        "frac_both": float((sizes == 2).mean()),
        "sigma_bin_gap": float(sigma_bin_gap),
        "sigma_bin_coverage": sigma_bin_cov,
        "n_traits_reported": int(len(per_trait)),
        "per_trait_AUPRC_median": float(per_trait["AUPRC"].median()) if len(per_trait) > 0 else None,
        "per_trait_coverage_gap_max_minus_min": float(cov_values.max() - cov_values.min()) if len(per_trait) > 0 else None,
        "per_trait_coverage_std": float(cov_values.std()) if len(per_trait) > 0 else None,
    }
    (out / "trait_loo_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out}/trait_loo_scores.parquet, per_trait_metrics.csv, trait_loo_results.json")


if __name__ == "__main__":
    main()
