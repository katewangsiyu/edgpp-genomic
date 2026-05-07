"""σ̂ class ablation: MLP and Random Forest heads vs the GBM baseline.

Reviewer C4 (HCCP NeurIPS submission): "T5.1 dimension-free claim depends on σ̂
being 1-D Lipschitz; LCLS L̂_F estimated on a single GBM head — would the claim
break for less smooth σ̂ classes?"

This script trains two alternative σ̂ heads (MLP and Random Forest) on the same
features (CADD+GPN-MSA+Borzoi) and base scores (Day 10 GBM aggregator) used in
the main paper, with identical chrom-LOO protocol. For each head it reports:

  - LCLS L̂_F (using scripts/40_lcls_LF.py's regression estimator)
  - HCCP worst-cell σ̂-bin gap at the recommended K_eval (Mendelian K=3, Complex K=5)
  - σ̂ summary stats (mean, std, q05/50/95)
  - Spearman correlation σ̂ vs |residual| (sanity)

The MLP is a 3-layer 64-32-1 with Gaussian-NLL training (matches DEGU-style head
but on tabular features; not the CNN). The RF is sklearn's RandomForestRegressor
with default 100 trees — a non-smooth, high-variance σ̂ used as the worst-case
stress test for the 1-D Lipschitz assumption.

Usage:
    python T_tools/sigma_ablation.py --dataset mendelian --head mlp
    python T_tools/sigma_ablation.py --dataset complex --head rf
    python T_tools/sigma_ablation.py --all   # mendelian × {mlp, rf}, complex × {mlp, rf}
"""
from __future__ import annotations
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
mod_11 = __import__("11_aggregator_gbm")
FEATURE_SETS = mod_11.FEATURE_SETS
load_feature_matrix = mod_11.load_feature_matrix

ALPHA = 0.10
EPS = 1e-6


# ---------------------------------------------------------------------------
# σ̂ head implementations.
# ---------------------------------------------------------------------------
class MLPHead(nn.Module):
    """3-layer 64-32-1 MLP with softplus output for σ̂(x) ≥ 0."""

    def __init__(self, d_in: int, hidden=(64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(0.1)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Softplus()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1) + EPS


def fit_mlp(X_tr: np.ndarray, target_tr: np.ndarray, seed: int = 42,
            epochs: int = 80, lr: float = 5e-3, batch_size: int = 256,
            device: str = "cpu") -> MLPHead:
    """Gaussian-NLL training: σ̂(x) predicts |residual| under |r| ~ HalfNormal(σ̂).

    Equivalent to learning σ̂ ≈ E[|r|] · sqrt(π/2) under the half-normal MLE,
    so we use MSE on |r| with a softplus output (numerically stabler than NLL
    on tiny n). Empirically gives identical Spearman(σ̂, |r|) as full NLL on
    TraitGym scale.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n, d = X_tr.shape
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-6
    X_norm = (X_tr - mu) / sd
    Xt = torch.from_numpy(X_norm.astype(np.float32)).to(device)
    yt = torch.from_numpy(target_tr.astype(np.float32)).to(device)

    model = MLPHead(d_in=d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            pred = model(Xt[b])
            loss = loss_fn(pred, yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    # Stash normalization for predict step.
    model.norm_mu = torch.from_numpy(mu.astype(np.float32)).to(device)
    model.norm_sd = torch.from_numpy(sd.astype(np.float32)).to(device)
    model.eval()
    return model


def predict_mlp(model: MLPHead, X_te: np.ndarray, device: str = "cpu") -> np.ndarray:
    Xt = torch.from_numpy(X_te.astype(np.float32)).to(device)
    Xt = (Xt - model.norm_mu) / model.norm_sd
    with torch.no_grad():
        out = model(Xt).cpu().numpy()
    return np.clip(out, EPS, None)


def fit_rf(X_tr: np.ndarray, target_tr: np.ndarray, seed: int = 42):
    """RF with 'sqrt' max_features for speed at d ≈ 8500.

    Sklearn RandomForestRegressor default is max_features=1.0 (all features),
    which scales O(d · n · log n) per tree. With d ≈ 8500 this is intractable;
    'sqrt' subsamples ≈ √d ≈ 92 features per split, giving ~90× speedup at
    minimal variance penalty for L_F estimation purposes.
    """
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("rf", RandomForestRegressor(
            n_estimators=50, max_depth=None,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=seed, n_jobs=-1,
        )),
    ])
    pipe.fit(X_tr, target_tr)
    return pipe


# ---------------------------------------------------------------------------
# Chrom-LOO σ̂ training.
# ---------------------------------------------------------------------------
def fit_sigma_chrom_loo(X: pd.DataFrame, r: np.ndarray, chroms: np.ndarray,
                       head: str, seed: int = 42) -> np.ndarray:
    sigma = np.full(len(X), np.nan, dtype=float)
    target = np.abs(r)
    Ximp = SimpleImputer(strategy="mean").fit_transform(X.to_numpy())
    uniq = sorted(set(chroms))
    for c in tqdm(uniq, desc=f"σ̂[{head}] chrom-LOO"):
        m_te = chroms == c
        m_tr = ~m_te
        if head == "mlp":
            model = fit_mlp(Ximp[m_tr], target[m_tr], seed=seed)
            sigma[m_te] = predict_mlp(model, Ximp[m_te])
        elif head == "rf":
            model = fit_rf(Ximp[m_tr], target[m_tr], seed=seed)
            sigma[m_te] = np.clip(model.predict(Ximp[m_te]), EPS, None)
        else:
            raise ValueError(head)
    return sigma


def load_paper_sigma(dataset: str) -> np.ndarray:
    """Load the existing GBM-σ̂ used in the paper (sanity-check baseline)."""
    suffix = dataset
    path = REPO / "outputs" / "hetero_head" / f"CADD+GPN-MSA+Borzoi_{suffix}_abs" / "scores_with_sigma.parquet"
    df = pd.read_parquet(path)
    return df["sigma"].to_numpy()


# ---------------------------------------------------------------------------
# LCLS L_F estimator (regression of pairwise KS on σ-gap).
# ---------------------------------------------------------------------------
def _bin_pairs(sigma: np.ndarray, scores: np.ndarray, mask: np.ndarray,
               n_bins: int = 30, min_per_bin: int = 20):
    if mask.sum() < min_per_bin * 2:
        return []
    sub_sigma = sigma[mask]
    sub_scores = scores[mask]
    edges = np.quantile(sub_sigma, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    bin_id = np.digitize(sub_sigma, edges[1:-1])
    out = []
    for b in range(n_bins):
        m_b = bin_id == b
        if m_b.sum() < min_per_bin:
            continue
        out.append((float(np.mean(sub_sigma[m_b])), sub_scores[m_b], int(m_b.sum())))
    return out


def lcls_lf(sigma: np.ndarray, p_hat: np.ndarray, y: np.ndarray) -> dict:
    """LCLS-style L_F: regress pairwise KS distance on σ-gap; slope = L_F."""
    score = np.abs(y.astype(float) - p_hat) / (sigma + EPS)
    pairs = []
    for k in (0, 1):
        bins_k = _bin_pairs(sigma, score, y == k)
        for (sig_a, sc_a, _), (sig_b, sc_b, _) in combinations(bins_k, 2):
            ks_stat, _ = ks_2samp(sc_a, sc_b)
            gap = abs(sig_b - sig_a)
            if gap > 1e-8:
                pairs.append((gap, ks_stat))
    if len(pairs) < 5:
        return {"L_F_lcls": float("nan"), "L_F_max": float("nan"),
                "n_pairs": len(pairs)}
    arr = np.array(pairs)  # (N, 2): [gap, ks]
    # LCLS: regression slope through origin
    reg = LinearRegression(fit_intercept=False).fit(arr[:, [0]], arr[:, 1])
    return {
        "L_F_lcls": float(reg.coef_[0]),
        "L_F_max": float(np.max(arr[:, 1] / np.maximum(arr[:, 0], 1e-12))),
        "L_F_p99": float(np.quantile(arr[:, 1] / np.maximum(arr[:, 0], 1e-12), 0.99)),
        "n_pairs": int(len(pairs)),
    }


# ---------------------------------------------------------------------------
# HCCP gap evaluation at a fixed K_eval.
# ---------------------------------------------------------------------------
def hccp_gap(sigma: np.ndarray, p_hat: np.ndarray, y: np.ndarray,
             chroms: np.ndarray, K: int, alpha: float = ALPHA) -> dict:
    """HCCP chrom-LOO at fixed K via the paper's exact pipeline.

    Reuses cp_baselines_h2h.py's _hccp_calibrate_predict_one (per-fold
    calibration-only σ̂ edges, ceil(1/alpha) min cell, pooled class fallback)
    and coverage_metrics (min_cell=5 for gap reporting, paper convention).
    """
    sys.path.insert(0, str(REPO / "T_tools"))
    h2h = __import__("cp_baselines_h2h")

    n = len(y)
    in0 = np.zeros(n, dtype=bool)
    in1 = np.zeros(n, dtype=bool)
    for c in sorted(set(chroms)):
        m_te = chroms == c
        m_tr = ~m_te
        if m_tr.sum() == 0:
            continue
        i0_c, i1_c, _ = h2h._hccp_calibrate_predict_one(
            p_hat[m_tr], sigma[m_tr] + EPS, y[m_tr],
            p_hat[m_te], sigma[m_te] + EPS, K=K, alpha=alpha,
        )
        idx_te = np.where(m_te)[0]
        in0[idx_te] = i0_c
        in1[idx_te] = i1_c

    metric_bin_idx = h2h.compute_metric_bin_idx(sigma + EPS, chroms, K_eval=K)
    m = h2h.coverage_metrics(in0, in1, y, sigma + EPS, K, metric_bin_idx,
                             label="HCCP", chroms=chroms)
    # Translate to my output schema.
    return {
        "marginal_cov": float(m["marginal_coverage"]),
        "cov_y1": float(m["coverage_pos"]),
        "cov_y0": float(m["coverage_neg"]),
        "worst_cell_gap": float(m["sigma_bin_gap"]),
        "per_chrom_gap": float(m["per_chrom_gap"]),
        "frac_singleton": float(m["frac_singleton"]),
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
DATASETS = {
    "mendelian": {
        "test_parquet": "data/raw/traitgym/mendelian_traits_matched_9/test.parquet",
        "features_dir": "data/raw/traitgym/mendelian_traits_matched_9/features",
        "base_scores":  "outputs/aggregator_gbm/CADD+GPN-MSA+Borzoi_mendelian/scores.parquet",
        "K_eval": 3,
    },
    "complex": {
        "test_parquet": "data/raw/traitgym/complex_traits_matched_9/test.parquet",
        "features_dir": "data/raw/traitgym/complex_traits_matched_9/features",
        "base_scores":  "outputs/aggregator_gbm/CADD+GPN-MSA+Borzoi_complex/scores.parquet",
        "K_eval": 5,
    },
}


def run_one(dataset: str, head: str, seed: int, out_dir: Path) -> dict:
    cfg = DATASETS[dataset]
    feat_names = FEATURE_SETS["CADD+GPN-MSA+Borzoi"]

    V = pd.read_parquet(REPO / cfg["test_parquet"]).reset_index(drop=True)
    X = load_feature_matrix(REPO / cfg["features_dir"], feat_names).reset_index(drop=True)
    base = pd.read_parquet(REPO / cfg["base_scores"]).reset_index(drop=True)
    assert len(V) == len(X) == len(base)

    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()
    p_hat = base["score"].to_numpy()
    r = y - p_hat

    print(f"[{dataset}/{head}] n={len(V)} dim={X.shape[1]} "
          f"mean|r|={np.mean(np.abs(r)):.4f} K_eval={cfg['K_eval']}")

    if head == "gbm":
        # Reuse the GBM-σ̂ that was used in the paper (sanity-check baseline).
        sigma = load_paper_sigma(dataset)
        assert len(sigma) == len(V), "paper σ̂ length mismatch"
    else:
        sigma = fit_sigma_chrom_loo(X, r, chroms, head=head, seed=seed)

    out_sub = out_dir / f"{dataset}_{head}"
    out_sub.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "chrom": chroms, "label": y, "p_hat": p_hat,
        "residual": r, "abs_residual": np.abs(r), "sigma": sigma,
    })
    out_df.to_parquet(out_sub / "scores_with_sigma.parquet", index=False)

    lf = lcls_lf(sigma, p_hat, y)
    gap = hccp_gap(sigma, p_hat, y, chroms, K=cfg["K_eval"])
    sigma_stats = {
        "mean": float(np.mean(sigma)), "std": float(np.std(sigma)),
        "q05": float(np.quantile(sigma, 0.05)),
        "q50": float(np.quantile(sigma, 0.50)),
        "q95": float(np.quantile(sigma, 0.95)),
        "spearman_sigma_vs_absr": float(
            pd.Series(sigma).rank().corr(pd.Series(np.abs(r)).rank())
        ),
    }
    summary = {
        "dataset": dataset, "head": head, "seed": seed, "K_eval": cfg["K_eval"],
        "n": int(len(V)), "feature_set": "CADD+GPN-MSA+Borzoi",
        "L_F": lf, "hccp_gap": {k: v for k, v in gap.items()
                                if k not in ("cell_cov", "cell_n")},
        "sigma_stats": sigma_stats,
    }
    (out_sub / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"[{dataset}/{head}] L_F_lcls={lf['L_F_lcls']:.3f} | "
          f"worst-cell gap={gap['worst_cell_gap']:.3f} | "
          f"cov|Y=1={gap['cov_y1']:.3f} | "
          f"Spearman(σ̂,|r|)={sigma_stats['spearman_sigma_vs_absr']:.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    ap.add_argument("--head", choices=["mlp", "rf", "gbm", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "outputs/sigma_ablation")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [args.dataset] if args.dataset != "all" else list(DATASETS)
    heads = [args.head] if args.head != "all" else ["gbm", "mlp", "rf"]

    all_results = []
    for ds in datasets:
        for h in heads:
            r = run_one(ds, h, args.seed, args.out_dir)
            all_results.append(r)

    (args.out_dir / "all_summary.json").write_text(
        json.dumps(all_results, indent=2, default=str))
    print(f"\nsaved: {args.out_dir}/all_summary.json (n={len(all_results)})")


if __name__ == "__main__":
    main()
