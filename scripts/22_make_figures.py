"""Generate publication-quality figures for NeurIPS 2027 paper.

Fig 2: K-sweep U-curve (T5 validation) — Mendelian + Complex
Fig 3: σ̂-bin coverage three-axis panel (chrom-LOO / trait-LOO / cross-dataset)
Fig 4: DEGU vs HCCP σ̂ comparison — per-bin coverage bars

Usage:
    conda run -n edgpp_t4 python scripts/22_make_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

OUT = Path("papers/neurips2027_pathA/figures")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Fig 2: K-sweep U-curve (T5 validation)
# ============================================================
def fig2_k_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=False)

    for ax, (dataset, label) in zip(axes, [
        ("CADD+GPN-MSA+Borzoi_mendelian", "Mendelian ($n{=}3{,}380$)"),
        ("CADD+GPN-MSA+Borzoi_complex", "Complex ($n{=}11{,}400$)"),
    ]):
        path = Path(f"outputs/adaptive_K/{dataset}/adaptive_K_results.json")
        if not path.exists():
            ax.set_title(f"{label} — data not found")
            continue

        with open(path) as f:
            d = json.load(f)

        Ks = [r["K"] for r in d["sweep"]]
        worst = [r["worst_cell_gap"] for r in d["sweep"]]
        mean = [r["mean_cell_gap"] for r in d["sweep"]]

        # Theoretical curve
        L_F = d.get("L_F_estimated", 1)
        R = d["R"]
        pi_min = d["pi_min"]
        n = d["n"]
        K_dense = np.linspace(1.5, 35, 200)

        # Use fitted L_F (back-calculated from empirical data)
        # Fit: minimize sum of (mean_gap - (L*R/K + K/(pi*n)))^2
        from scipy.optimize import minimize_scalar
        def fit_loss(log_L):
            L = np.exp(log_L)
            pred = [L * R / K + K / (pi_min * n) for K in Ks]
            return sum((p - m) ** 2 for p, m in zip(pred, mean))
        res = minimize_scalar(fit_loss, bounds=(-5, 5), method="bounded")
        L_fit = np.exp(res.x)
        theory_dense = L_fit * R / K_dense + K_dense / (pi_min * n)
        K_star = np.sqrt(L_fit * R * pi_min * n)

        ax.plot(Ks, worst, "s-", color="#d62728", ms=5, lw=1.2,
                label="Worst-cell gap", zorder=3)
        ax.plot(Ks, mean, "o-", color="#1f77b4", ms=4, lw=1.2,
                label="Mean-cell gap", zorder=3)
        ax.plot(K_dense, theory_dense, "--", color="#aaaaaa", lw=1,
                label=f"T5 bound ($\\hat L_F$={L_fit:.2f})")
        ax.axvline(K_star, color="#2ca02c", ls=":", lw=0.8, alpha=0.7,
                   label=f"$K^\\star$={K_star:.0f}")
        ax.axvline(d["K_cv"], color="#ff7f0e", ls="-.", lw=0.8, alpha=0.7,
                   label=f"$\\hat K_{{\\mathrm{{CV}}}}$={d['K_cv']}")

        ax.set_xlabel("Bin count $K$")
        ax.set_ylabel("Coverage gap $|\\mathrm{cov}_{k,b} - (1{-}\\alpha)|$")
        ax.set_title(label)
        ax.set_xlim(0, 32)
        ax.set_ylim(-0.01, min(0.8, max(worst) * 1.15))
        ax.legend(loc="upper right", framealpha=0.9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / "fig2_k_sweep.pdf")
    fig.savefig(OUT / "fig2_k_sweep.png")
    plt.close(fig)
    print(f"  Saved fig2_k_sweep.pdf")


# ============================================================
# Fig 3: σ̂-bin coverage — three-axis panel
# ============================================================
def fig3_coverage_panel():
    configs = [
        ("Chrom-LOO (Mendelian)",
         "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/conformal_hetero_results.json"),
        ("Chrom-LOO (Complex)",
         "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/conformal_hetero_results.json"),
        ("Trait-LOO (Mendelian)",
         "outputs/trait_loo/CADD+GPN-MSA+Borzoi_mendelian/trait_loo_results.json"),
        ("Trait-LOO (Complex)",
         "outputs/trait_loo/CADD+GPN-MSA+Borzoi_complex/trait_loo_results.json"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.4), sharey=True)
    target = 0.9

    for ax, (title, path) in zip(axes, configs):
        p = Path(path)
        if not p.exists():
            ax.set_title(title + " (N/A)")
            continue

        with open(p) as f:
            d = json.load(f)

        # Extract per-sigma-bin coverage
        if "mondrian_y_sigma" in d:
            bins = d["mondrian_y_sigma"].get("coverage_by_sigma_bin", [])
        elif "hetero_class_cond" in d:
            bins = d["hetero_class_cond"].get("coverage_by_sigma_bin", [])
        elif "per_sigma_bin_coverage" in d:
            bins = d["per_sigma_bin_coverage"]
        else:
            # trait_loo format
            bins = d.get("coverage_by_sigma_bin", [])

        if not bins:
            ax.set_title(title + " (no bin data)")
            continue

        x = range(len(bins))
        covs = [b.get("coverage", b.get("cov", 0)) for b in bins]

        colors = ["#2ca02c" if abs(c - target) < 0.03 else
                  "#ff7f0e" if abs(c - target) < 0.06 else
                  "#d62728" for c in covs]

        ax.bar(x, covs, color=colors, edgecolor="white", lw=0.5)
        ax.axhline(target, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.axhspan(target - 0.03, target + 0.03, color="#2ca02c", alpha=0.07)
        ax.set_xlabel("$\\hat\\sigma$-bin")
        if ax == axes[0]:
            ax.set_ylabel("Coverage")
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0.7, 1.02)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "fig3_coverage_panel.pdf")
    fig.savefig(OUT / "fig3_coverage_panel.png")
    plt.close(fig)
    print(f"  Saved fig3_coverage_panel.pdf")


# ============================================================
# Fig 4: DEGU vs HCCP — per-bin coverage comparison
# ============================================================
def fig4_degu_comparison():
    configs = [
        ("Mendelian",
         "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/conformal_hetero_results.json",
         "outputs/conformal_hetero/DEGU_lite_CADD+GPN-MSA+Borzoi_mendelian_mondrian/conformal_hetero_results.json"),
        ("Complex",
         "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/conformal_hetero_results.json",
         "outputs/conformal_hetero/DEGU_lite_CADD+GPN-MSA+Borzoi_complex_mondrian/conformal_hetero_results.json"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
    target = 0.9

    for ax, (title, hccp_path, degu_path) in zip(axes, configs):
        for path_str, label, color, offset in [
            (hccp_path, "HCCP (learned $\\hat\\sigma$)", "#1f77b4", -0.15),
            (degu_path, "DEGU-lite (ensemble $\\hat\\sigma$)", "#d62728", 0.15),
        ]:
            p = Path(path_str)
            if not p.exists():
                continue
            with open(p) as f:
                d = json.load(f)

            m = d.get("mondrian_y_sigma", d.get("hetero_class_cond", {}))
            bins = m.get("coverage_by_sigma_bin", [])
            if not bins:
                continue

            x = np.arange(len(bins))
            covs = [b["coverage"] for b in bins]
            ax.bar(x + offset, covs, width=0.28, color=color, alpha=0.85,
                   edgecolor="white", lw=0.3, label=label)

        ax.axhline(target, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.axhspan(target - 0.03, target + 0.03, color="gray", alpha=0.05)
        ax.set_xlabel("$\\hat\\sigma$-bin index")
        if ax == axes[0]:
            ax.set_ylabel("Coverage")
        ax.set_title(title)
        ax.set_ylim(0.75, 1.02)
        ax.legend(loc="lower right", fontsize=7)

    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig4_degu_comparison.pdf")
    fig.savefig(OUT / "fig4_degu_comparison.png")
    plt.close(fig)
    print(f"  Saved fig4_degu_comparison.pdf")


# ============================================================
if __name__ == "__main__":
    print("Generating paper figures...")
    fig2_k_sweep()
    fig3_coverage_panel()
    fig4_degu_comparison()
    print("Done.")
