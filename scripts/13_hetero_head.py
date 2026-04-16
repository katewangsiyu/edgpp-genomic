"""Heteroscedastic reliability head σ̂(x) — Path A Phase 2 prototype.

Reads a Day 10 GBM OOF base prediction (chrom-LOO p̂_{(-k)}(x)) and fits a
second-stage regressor to predict residual magnitude:

    r(x) = |y - p̂_{(-k)}(x)|
    σ̂(x) = HistGradientBoostingRegressor(r | features)    [mode=abs_residual]
    σ̂(x) = exp(0.5 · HGB(log(r² + ε) | features))         [mode=log_variance]

Uses chrom-LOO again on top: for each held-out chrom c, σ̂ is trained only on
residuals from the other chroms. This is honest but does re-use the Day 10 OOF
scores as target signal, i.e. σ̂ approximates E[|r| | x] where r comes from a
p̂ that never saw the test chrom.

Reference: formulation_v0.md §1.4 (heteroscedastic head), §7 (code mapping).

Usage:
    python scripts/13_hetero_head.py \\
        --base-scores outputs/aggregator_gbm/CADD+GPN-MSA+Borzoi_mendelian/scores.parquet \\
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --features-dir data/raw/traitgym/mendelian_traits_matched_9/features \\
        --feature-set CADD+GPN-MSA+Borzoi \\
        --mode abs_residual \\
        --out-dir outputs/hetero_head/CADD+GPN-MSA+Borzoi_mendelian
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

# Reuse feature-set config from the base aggregator so it stays consistent.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
mod_11 = __import__("11_aggregator_gbm")
FEATURE_SETS = mod_11.FEATURE_SETS
load_feature_matrix = mod_11.load_feature_matrix


def build_sigma_pipeline(seed=42, max_depth=2, max_iter=200, learning_rate=0.05):
    return Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("hgb", HistGradientBoostingRegressor(
            loss="squared_error",
            max_depth=max_depth, max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=seed,
            early_stopping=False,
        )),
    ])


def fit_sigma_chrom_loo(X_df: pd.DataFrame, r: np.ndarray, chroms: np.ndarray,
                       mode: str, seed: int, eps: float = 1e-4):
    """Return σ̂ on every variant via chrom-LOO second-stage regression."""
    sigma = np.full(len(X_df), np.nan, dtype=float)
    raw_pred = np.full(len(X_df), np.nan, dtype=float)  # raw regressor output pre-transform
    uniq = sorted(set(chroms))
    for c in tqdm(uniq, desc="σ̂ chrom-LOO"):
        m_te = chroms == c
        m_tr = ~m_te
        if mode == "abs_residual":
            target_tr = np.abs(r[m_tr])
        elif mode == "log_variance":
            target_tr = np.log(r[m_tr] ** 2 + eps)
        else:
            raise ValueError(mode)
        pipe = build_sigma_pipeline(seed=seed)
        pipe.fit(X_df.loc[m_tr].to_numpy(), target_tr)
        pred = pipe.predict(X_df.loc[m_te].to_numpy())
        raw_pred[m_te] = pred
        if mode == "abs_residual":
            # Regressor can dip below zero on unusual features; clip to a small floor.
            sigma[m_te] = np.clip(pred, eps, None)
        else:
            sigma[m_te] = np.exp(0.5 * pred)
    return sigma, raw_pred


def diagnostics(sigma: np.ndarray, r: np.ndarray, p_hat: np.ndarray, y: np.ndarray,
                chroms: np.ndarray):
    """Quick sanity metrics for the σ̂ head."""
    r_abs = np.abs(r)
    out = {
        "sigma_mean": float(np.mean(sigma)),
        "sigma_std": float(np.std(sigma)),
        "sigma_q05": float(np.quantile(sigma, 0.05)),
        "sigma_q50": float(np.quantile(sigma, 0.50)),
        "sigma_q95": float(np.quantile(sigma, 0.95)),
        "spearman_sigma_vs_absr": float(
            pd.Series(sigma).rank().corr(pd.Series(r_abs).rank())
        ),
        "pearson_sigma_vs_absr": float(np.corrcoef(sigma, r_abs)[0, 1]),
    }
    # Bin by σ̂ decile → check empirical |r|
    deciles = pd.qcut(sigma, q=10, labels=False, duplicates="drop")
    bin_tbl = []
    for b in sorted(set(deciles)):
        m = deciles == b
        bin_tbl.append({
            "bin": int(b),
            "n": int(m.sum()),
            "sigma_mean": float(sigma[m].mean()),
            "abs_r_mean": float(r_abs[m].mean()),
            "phat_mean": float(p_hat[m].mean()),
            "pos_rate": float(y[m].mean()),
        })
    out["sigma_decile_table"] = bin_tbl

    # σ̂ as a function of p̂(x): bin by p̂
    phat_bins = pd.cut(p_hat, bins=[-0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01],
                      labels=["0-10", "10-30", "30-50", "50-70", "70-90", "90-100"])
    phat_tbl = []
    for b in phat_bins.unique().dropna():
        m = phat_bins == b
        if m.sum() < 5:
            continue
        phat_tbl.append({
            "phat_bin": str(b),
            "n": int(m.sum()),
            "sigma_mean": float(sigma[m].mean()),
            "abs_r_mean": float(r_abs[m].mean()),
            "pos_rate": float(y[m].mean()),
        })
    out["phat_bin_table"] = phat_tbl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-scores", required=True,
                    help="Path to Day 10 scores.parquet (chrom/label/score).")
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--feature-set", required=True, choices=list(FEATURE_SETS))
    ap.add_argument("--mode", default="abs_residual",
                    choices=["abs_residual", "log_variance"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    feat_names = FEATURE_SETS[args.feature_set]

    # Load variants, features, base scores
    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    X = load_feature_matrix(Path(args.features_dir), feat_names).reset_index(drop=True)
    base = pd.read_parquet(args.base_scores).reset_index(drop=True)
    assert len(V) == len(X) == len(base), \
        f"length mismatch V={len(V)} X={len(X)} base={len(base)}"
    # sanity: chrom/label must agree
    assert (V["chrom"].astype(str).values == base["chrom"].astype(str).values).all()
    assert (V["label"].astype(int).values == base["label"].astype(int).values).all()

    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()
    p_hat = base["score"].to_numpy()
    r = y - p_hat  # signed residual; we use |r| for target

    print(f"[load] feature_set={args.feature_set} n={len(V)} dim={X.shape[1]} "
          f"mode={args.mode} mean|r|={np.mean(np.abs(r)):.4f}")

    sigma, raw = fit_sigma_chrom_loo(X, r, chroms, mode=args.mode, seed=args.seed)

    # Save per-variant scores
    out_df = V[["chrom"]].copy()
    out_df["label"] = y
    out_df["p_hat"] = p_hat
    out_df["residual"] = r
    out_df["abs_residual"] = np.abs(r)
    out_df["sigma"] = sigma
    out_df["raw_pred"] = raw
    out_df.to_parquet(out / "scores_with_sigma.parquet", index=False)

    # Diagnostics
    diag = diagnostics(sigma, r, p_hat, y, chroms)
    diag["mode"] = args.mode
    diag["feature_set"] = args.feature_set
    diag["n"] = int(len(V))
    diag["mean_abs_r"] = float(np.mean(np.abs(r)))
    (out / "sigma_metrics.json").write_text(json.dumps(diag, indent=2, default=str))

    print(f"\n=== σ̂ head ({args.mode}) on {args.feature_set} ===")
    print(f"σ̂ mean={diag['sigma_mean']:.4f}  std={diag['sigma_std']:.4f}  "
          f"q05/50/95={diag['sigma_q05']:.3f}/{diag['sigma_q50']:.3f}/{diag['sigma_q95']:.3f}")
    print(f"Spearman(σ̂, |r|) = {diag['spearman_sigma_vs_absr']:.4f}")
    print(f"Pearson (σ̂, |r|) = {diag['pearson_sigma_vs_absr']:.4f}")
    print("\nσ̂ decile table (larger bin ⇒ higher σ̂; |r| should track):")
    for row in diag["sigma_decile_table"]:
        print(f"  bin {row['bin']}: n={row['n']:>4}  σ̂={row['sigma_mean']:.4f}  "
              f"|r|={row['abs_r_mean']:.4f}  p̂={row['phat_mean']:.3f}  pos%={row['pos_rate']:.1%}")
    print("\np̂ bin table (σ̂ should peak near boundary p̂≈0.5):")
    for row in diag["phat_bin_table"]:
        print(f"  p̂∈[{row['phat_bin']:>6}]: n={row['n']:>4}  σ̂={row['sigma_mean']:.4f}  "
              f"|r|={row['abs_r_mean']:.4f}  pos%={row['pos_rate']:.1%}")
    print(f"\nsaved: {out}/scores_with_sigma.parquet, sigma_metrics.json")


if __name__ == "__main__":
    main()
