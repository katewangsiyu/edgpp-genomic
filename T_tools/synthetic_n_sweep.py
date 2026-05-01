"""Synthetic n-sweep validating Theorems 2 (T5.1) and 3 (T5.2) of HCCP.

Generates synthetic binary classification data from a Gaussian-shift family with
a known oracle variance function sigma_star(x). Runs HCCP at K_hat_CV across
n in {500, 1000, 2000, 4000, 8000, 16000, 32000} for two prevalence levels
(pi_min in {0.1, 0.5}). Reports the worst-cell coverage gap G(K_hat_CV) and
fits log G ~ alpha * log n; expected slope alpha = -1/2.

Also includes T3.b plug-in robustness validation (sigma-perturbation eta-sweep).

Outputs:
- R_raw/synthetic_n_sweep/results.json
- papers/neurips2027_pathA/figures/fig_n_sweep.pdf

Usage:
    python T_tools/synthetic_n_sweep.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# --- Unified NeurIPS-classic style ---
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from T_tools.paper_style import apply_paper_style  # noqa: E402
apply_paper_style()
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "R_raw" / "synthetic_n_sweep"
FIG_DIR = REPO / "papers" / "neurips2027_pathA" / "figures"

ALPHA = 0.10
EPS = 1e-6
SEED = 42


def generate(n: int, d: int, pi_min: float, L_F: float, rng: np.random.Generator):
    """Sample (X, Y, sigma_star) from Gaussian-shift family.

    sigma_star(x) = sigma0 + sigma1 * |x_1|  (1-D heteroscedasticity along x_1)
    Score family is implicitly s = |y - p(x)| / sigma_star(x).
    """
    X = rng.standard_normal((n, d)).astype(np.float32)
    sigma_star = (0.1 + 0.4 * np.abs(X[:, 0])).astype(np.float32)
    # logits: small linear signal; sigma_star modulates noise on the logit
    beta = rng.standard_normal(d).astype(np.float32) * 0.3
    intercept = float(np.log(pi_min / (1 - pi_min)))
    eta = X @ beta + intercept + rng.standard_normal(n).astype(np.float32) * sigma_star
    p = 1.0 / (1.0 + np.exp(-eta))
    Y = (rng.random(n) < p).astype(np.int32)
    return X, Y, sigma_star


def fit_aggregator(X_tr, Y_tr, X_cal, Y_cal):
    clf = HistGradientBoostingClassifier(max_depth=2, max_iter=100, class_weight="balanced",
                                          random_state=SEED)
    clf.fit(X_tr, Y_tr)
    p_cal = clf.predict_proba(X_cal)[:, 1]
    return clf, p_cal


def fit_sigma_head(X_tr, Y_tr, p_tr, X_cal):
    """Fit sigma-hat regressor on absolute residual (Gaussian NLL via half-normal)."""
    r_tr = np.abs(Y_tr.astype(np.float32) - p_tr)
    reg = HistGradientBoostingRegressor(max_depth=2, max_iter=100, random_state=SEED)
    reg.fit(X_tr, r_tr)
    sigma_cal = np.maximum(reg.predict(X_cal), EPS)
    return reg, sigma_cal


def hccp_calibrate(p_cal, sigma_cal, Y_cal, K, alpha):
    """Compute Mondrian (y x sigma-bin) per-cell quantile thresholds.

    Returns:
      bin_edges: (K-1,)
      qhat: dict {(k, b): float}
      sigma_to_bin: callable
    """
    bin_edges = np.quantile(sigma_cal, np.linspace(0, 1, K + 1)[1:-1])

    def sigma_to_bin(sigma):
        return np.searchsorted(bin_edges, sigma)

    bin_idx = sigma_to_bin(sigma_cal)
    s_cal = np.abs(Y_cal.astype(np.float32) - p_cal) / (sigma_cal + EPS)
    qhat = {}
    n_kb = {}
    for k in (0, 1):
        for b in range(K):
            mask = (Y_cal == k) & (bin_idx == b)
            n = int(mask.sum())
            n_kb[(k, b)] = n
            if n >= int(np.ceil(1.0 / alpha)):
                idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
                idx = min(idx, n - 1)
                qhat[(k, b)] = float(np.sort(s_cal[mask])[idx])
            else:  # small-cell fallback to pooled class threshold
                mask_class = (Y_cal == k)
                n_k = int(mask_class.sum())
                if n_k >= int(np.ceil(1.0 / alpha)):
                    idx = int(np.ceil((n_k + 1) * (1 - alpha))) - 1
                    idx = min(idx, n_k - 1)
                    qhat[(k, b)] = float(np.sort(s_cal[mask_class])[idx])
                else:
                    qhat[(k, b)] = np.inf
    return bin_edges, qhat, sigma_to_bin, n_kb


def hccp_predict(p_test, sigma_test, qhat, sigma_to_bin):
    bin_idx = sigma_to_bin(sigma_test)
    n = len(p_test)
    in_set_0 = np.abs(0 - p_test) <= np.array([qhat[(0, b)] for b in bin_idx]) * sigma_test
    in_set_1 = np.abs(1 - p_test) <= np.array([qhat[(1, b)] for b in bin_idx]) * sigma_test
    return in_set_0, in_set_1


def coverage_metrics(in_set_0, in_set_1, Y_test, sigma_test, K, sigma_to_bin):
    bin_idx = sigma_to_bin(sigma_test)
    covered = np.where(Y_test == 0, in_set_0, in_set_1)
    marg_cov = float(covered.mean())
    cov_y1 = float(covered[Y_test == 1].mean()) if (Y_test == 1).any() else np.nan

    cell_gaps = []
    for k in (0, 1):
        for b in range(K):
            mask = (Y_test == k) & (bin_idx == b)
            if mask.any():
                cell_cov = float(covered[mask].mean())
                cell_gaps.append(abs(cell_cov - (1 - ALPHA)))
    worst = float(max(cell_gaps)) if cell_gaps else np.nan
    mean_g = float(np.mean(cell_gaps)) if cell_gaps else np.nan
    return dict(marg=marg_cov, cov_y1=cov_y1, worst_gap=worst, mean_gap=mean_g)


def select_K_cv(p_cal, sigma_cal, Y_cal, alpha, K_grid=(2, 3, 5, 8, 10)):
    """LOO-on-calibration K selection: pick K minimising worst-cell gap on held-out
    calibration sub-fold."""
    n = len(p_cal)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    half = n // 2
    cal1, cal2 = perm[:half], perm[half:]

    best_K, best_gap = K_grid[0], np.inf
    for K in K_grid:
        try:
            edges, qhat, s2b, _ = hccp_calibrate(p_cal[cal1], sigma_cal[cal1],
                                                  Y_cal[cal1], K, alpha)
            i0, i1 = hccp_predict(p_cal[cal2], sigma_cal[cal2], qhat, s2b)
            m = coverage_metrics(i0, i1, Y_cal[cal2], sigma_cal[cal2], K, s2b)
            if m["worst_gap"] < best_gap:
                best_gap, best_K = m["worst_gap"], K
        except Exception:
            continue
    return best_K


def run_one(n, d, pi_min, L_F, seed, K_grid=(2, 3, 5, 8, 10)):
    rng = np.random.default_rng(seed)
    n_total = n * 2  # half train, half cal+test
    X, Y, sigma_star = generate(n_total, d, pi_min, L_F, rng)
    X_tr, X_rest, Y_tr, Y_rest, sig_tr, sig_rest = train_test_split(
        X, Y, sigma_star, test_size=0.5, random_state=seed
    )
    X_cal, X_te, Y_cal, Y_te, sig_cal, sig_te = train_test_split(
        X_rest, Y_rest, sig_rest, test_size=0.5, random_state=seed
    )

    clf, p_cal = fit_aggregator(X_tr, Y_tr, X_cal, Y_cal)
    p_tr = clf.predict_proba(X_tr)[:, 1]
    sig_reg, sigma_cal = fit_sigma_head(X_tr, Y_tr, p_tr, X_cal)
    sigma_te = np.maximum(sig_reg.predict(X_te), EPS)
    p_te = clf.predict_proba(X_te)[:, 1]

    K_cv = select_K_cv(p_cal, sigma_cal, Y_cal, ALPHA, K_grid)
    edges, qhat, s2b, _ = hccp_calibrate(p_cal, sigma_cal, Y_cal, K_cv, ALPHA)
    i0, i1 = hccp_predict(p_te, sigma_te, qhat, s2b)
    m = coverage_metrics(i0, i1, Y_te, sigma_te, K_cv, s2b)
    m["K_cv"] = int(K_cv)
    m["n"] = int(n)
    return m


def run_eta_sweep(n, d, pi_min, L_F, eta_grid, seed, K=3):
    """T3.b: inject multiplicative noise into sigma-hat and measure coverage drift."""
    rng = np.random.default_rng(seed)
    X, Y, sigma_star = generate(n * 2, d, pi_min, L_F, rng)
    X_tr, X_rest, Y_tr, Y_rest, sig_tr, sig_rest = train_test_split(
        X, Y, sigma_star, test_size=0.5, random_state=seed
    )
    X_cal, X_te, Y_cal, Y_te, sig_cal, sig_te = train_test_split(
        X_rest, Y_rest, sig_rest, test_size=0.5, random_state=seed
    )

    clf, p_cal = fit_aggregator(X_tr, Y_tr, X_cal, Y_cal)
    p_te = clf.predict_proba(X_te)[:, 1]

    # Oracle sigma calibration (use sigma_star directly)
    edges_o, qhat_o, s2b_o, _ = hccp_calibrate(p_cal, sig_cal, Y_cal, K, ALPHA)
    i0_o, i1_o = hccp_predict(p_te, sig_te, qhat_o, s2b_o)
    m_o = coverage_metrics(i0_o, i1_o, Y_te, sig_te, K, s2b_o)

    results = []
    for eta in eta_grid:
        # Plug-in: sigma_hat = sigma_star * (1 + eta * xi)
        xi_cal = rng.standard_normal(len(sig_cal))
        xi_te = rng.standard_normal(len(sig_te))
        sigma_hat_cal = np.maximum(sig_cal * (1 + eta * xi_cal), EPS)
        sigma_hat_te = np.maximum(sig_te * (1 + eta * xi_te), EPS)
        edges_h, qhat_h, s2b_h, _ = hccp_calibrate(p_cal, sigma_hat_cal, Y_cal, K, ALPHA)
        i0_h, i1_h = hccp_predict(p_te, sigma_hat_te, qhat_h, s2b_h)
        m_h = coverage_metrics(i0_h, i1_h, Y_te, sigma_hat_te, K, s2b_h)
        # Coverage drift between plug-in and oracle on the same test set
        covered_o = np.where(Y_te == 0, i0_o, i1_o)
        covered_h = np.where(Y_te == 0, i0_h, i1_h)
        for k in (0, 1):
            mask = Y_te == k
            if mask.any():
                drift_k = abs(float(covered_h[mask].mean()) - float(covered_o[mask].mean()))
                results.append(dict(eta=float(eta), class_=int(k), drift=drift_k,
                                    n_class=int(mask.sum())))
    return dict(oracle=m_o, eta_results=results)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_grid = [500, 1000, 2000, 4000, 8000, 16000, 32000]
    d_grid = [8, 32, 128]
    pi_grid = [0.10, 0.50]
    L_F = 4.0
    n_replicates = 5

    rate_results = {}
    for d in d_grid:
        for pi_min in pi_grid:
            key = f"d{d}_pi{pi_min:.2f}"
            rate_results[key] = []
            for n in n_grid:
                seed_base = SEED + d * 100 + int(pi_min * 1000)
                ms = []
                for r in range(n_replicates):
                    try:
                        m = run_one(n, d, pi_min, L_F, seed=seed_base + r)
                        ms.append(m)
                    except Exception as e:
                        print(f"  skip d={d} pi={pi_min} n={n} rep={r}: {e}")
                if ms:
                    worst = np.mean([m["worst_gap"] for m in ms])
                    worst_se = np.std([m["worst_gap"] for m in ms]) / np.sqrt(len(ms))
                    K_cv_med = int(np.median([m["K_cv"] for m in ms]))
                    rate_results[key].append(dict(n=n, worst_gap=float(worst),
                                                   worst_gap_se=float(worst_se),
                                                   K_cv=K_cv_med, n_replicates=len(ms)))
                    print(f"d={d} pi={pi_min} n={n}: worst_gap={worst:.4f} +/- {worst_se:.4f}, K_cv={K_cv_med}")

    # T3.b eta-sweep at n=8000, d=32
    print("\n--- T3.b eta-sweep ---")
    eta_grid = [0.05, 0.10, 0.25, 0.50]
    eta_results = run_eta_sweep(n=8000, d=32, pi_min=0.10, L_F=L_F,
                                eta_grid=eta_grid, seed=SEED)
    for r in eta_results["eta_results"]:
        print(f"eta={r['eta']}, class={r['class_']}: drift={r['drift']:.4f}")

    out = dict(
        config=dict(L_F=L_F, ALPHA=ALPHA, n_grid=n_grid, d_grid=d_grid, pi_grid=pi_grid),
        rate_results=rate_results,
        eta_results=eta_results,
    )
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {OUT_DIR / 'results.json'}")

    # Figure: log-log gap vs n for d=32, pi in {0.10, 0.50}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, pi in zip(axes, pi_grid):
        for d in d_grid:
            key = f"d{d}_pi{pi:.2f}"
            data = rate_results[key]
            ns = np.array([r["n"] for r in data])
            ws = np.array([r["worst_gap"] for r in data])
            se = np.array([r["worst_gap_se"] for r in data])
            ax.errorbar(ns, ws, yerr=se, marker="o", label=f"$d={d}$", capsize=3,
                        linewidth=1.5, markersize=5)
        # Theoretical O(n^{-1/2}) overlay (anchored at n=4000, d=32)
        anchor = next(r for r in rate_results[f"d32_pi{pi:.2f}"] if r["n"] == 4000)
        c = anchor["worst_gap"] * np.sqrt(4000)
        n_curve = np.linspace(min(n_grid), max(n_grid), 100)
        ax.plot(n_curve, c / np.sqrt(n_curve), "k--", alpha=0.6,
                label=r"$O(n^{-1/2})$ T5 prediction")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Calibration size $n$")
        ax.set_ylabel(r"Worst-cell gap $G(\hat K_{\mathrm{CV}})$")
        ax.set_title(rf"$\pi_{{\min}}={pi:.2f}$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = FIG_DIR / "fig_n_sweep.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_n_sweep.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
