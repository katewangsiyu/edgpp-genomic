"""Publication-quality figures for NeurIPS 2027 paper — v2 redesign.

Design principles (from CQR/RLCP/Barber-2023 NeurIPS style):
  - seaborn whitegrid, muted palette, no chartjunk
  - deviation-centered coverage plots (coverage − target)
  - heatmaps for multi-dimensional comparisons
  - α-sweep calibration curve (CP paper standard)
  - consistent 3.25" single-column or 6.75" double-column width

Usage:
    conda run -n edgpp_t4 python scripts/23_make_figures_v2.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, FuncFormatter

# ── NeurIPS style ──
NEURIPS_COL = 3.25    # single column inches
NEURIPS_DCOL = 6.75   # double column inches
COLORS = {
    "hccp": "#2078B4",     # steel blue
    "degu": "#E45756",     # muted red
    "homosc": "#9E9E9E",   # gray
    "split": "#F58518",    # orange
    "theory": "#54A24B",   # green
    "accent": "#EECA3B",   # gold
}
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

OUT = Path("papers/neurips2027_pathA/figures")
OUT.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ================================================================
# Fig 2: K-sweep bias–variance (T5)
# ================================================================
def fig2_k_sweep():
    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_DCOL, 2.2))

    for ax, (dataset, title, color_accent) in zip(axes, [
        ("CADD+GPN-MSA+Borzoi_mendelian", "Mendelian ($n{=}3{,}380$)", "#2078B4"),
        ("CADD+GPN-MSA+Borzoi_complex", "Complex ($n{=}11{,}400$)", "#54A24B"),
    ]):
        path = Path(f"outputs/adaptive_K/{dataset}/adaptive_K_results.json")
        if not path.exists():
            continue
        d = load_json(path)

        Ks = np.array([r["K"] for r in d["sweep"]])
        mean_gap = np.array([r["mean_cell_gap"] for r in d["sweep"]])
        worst_gap = np.array([r["worst_cell_gap"] for r in d["sweep"]])
        R, pi_min, n = d["R"], d["pi_min"], d["n"]

        # Fit L_F to mean gap
        from scipy.optimize import minimize_scalar
        def fit_loss(log_L):
            L = np.exp(log_L)
            pred = L * R / Ks + Ks / (pi_min * n)
            return np.sum((pred - mean_gap) ** 2)
        res = minimize_scalar(fit_loss, bounds=(-5, 5), method="bounded")
        L_fit = np.exp(res.x)

        K_dense = np.linspace(1.5, 32, 300)
        bias = L_fit * R / K_dense
        variance = K_dense / (pi_min * n)
        total = bias + variance
        K_star = np.sqrt(L_fit * R * pi_min * n)

        # Shaded decomposition
        ax.fill_between(K_dense, 0, bias, alpha=0.12, color=COLORS["theory"],
                        label="Bias $L_F R/K$")
        ax.fill_between(K_dense, bias, total, alpha=0.12, color=COLORS["degu"],
                        label="Variance $K/(\\pi_{\\min} n)$")
        ax.plot(K_dense, total, "-", color="#555555", lw=1.2, alpha=0.7,
                label="T5 bound")

        # Empirical
        ax.plot(Ks, mean_gap, "o-", color=color_accent, ms=4.5, lw=1.3,
                markeredgecolor="white", markeredgewidth=0.5,
                label="Mean-cell gap", zorder=5)
        ax.plot(Ks, worst_gap, "s", color=COLORS["degu"], ms=4, alpha=0.5,
                label="Worst-cell gap", zorder=4)

        # K* and K_CV annotations
        ax.axvline(d["K_cv"], color=color_accent, ls="--", lw=1, alpha=0.8)
        ax.annotate(f'$\\hat{{K}}_{{\\mathrm{{CV}}}}={d["K_cv"]}$',
                    xy=(d["K_cv"], 0), xytext=(d["K_cv"] + 1.5, max(mean_gap) * 0.7),
                    fontsize=7, color=color_accent,
                    arrowprops=dict(arrowstyle="-", color=color_accent, lw=0.8))

        ax.set_xlabel("Bin count $K$")
        ax.set_title(title)
        ax.set_xlim(0, 33)
        ax.set_ylim(-0.005, min(0.25, max(mean_gap) * 2.5))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    axes[0].set_ylabel("Coverage gap")
    axes[0].legend(loc="upper right", framealpha=0.95, edgecolor="none")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig2_k_sweep.pdf")
    fig.savefig(OUT / "fig2_k_sweep.png")
    plt.close(fig)
    print("  fig2_k_sweep ✓")


# ================================================================
# Fig 3: α-sweep calibration curve (CP standard)
# ================================================================
def fig3_alpha_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_DCOL, 2.4))

    for ax, (dataset_label, hccp_path, degu_path) in zip(axes, [
        ("Mendelian",
         "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/conformal_hetero_results.json",
         "outputs/conformal_hetero/DEGU_lite_CADD+GPN-MSA+Borzoi_mendelian_mondrian/conformal_hetero_results.json"),
        ("Complex",
         "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/conformal_hetero_results.json",
         "outputs/conformal_hetero/DEGU_lite_CADD+GPN-MSA+Borzoi_complex_mondrian/conformal_hetero_results.json"),
    ]):
        # Diagonal (perfect calibration)
        ax.plot([0, 1], [0, 1], "k-", lw=0.8, alpha=0.3, label="Ideal")

        for path_str, label, color, marker in [
            (hccp_path, "HCCP (ours)", COLORS["hccp"], "o"),
            (degu_path, "DEGU-lite", COLORS["degu"], "s"),
        ]:
            p = Path(path_str)
            if not p.exists():
                continue
            d = load_json(p)

            # Extract α-sweep from hetero_class_cond or mondrian
            for method_key in ["mondrian_y_sigma", "hetero_class_cond"]:
                if method_key in d and "alpha_sweep" in d[method_key]:
                    sweep = d[method_key]["alpha_sweep"]
                    break
            else:
                # Fallback: use the multiple-α results if stored at top level
                # Try to reconstruct from the α-sweep in the results
                sweep = None
                for key in d:
                    if isinstance(d[key], dict) and "alpha_sweep" in d[key]:
                        sweep = d[key]["alpha_sweep"]
                        break

            if sweep:
                alphas = [s["alpha"] for s in sweep]
                covs = [s["coverage"] for s in sweep]
                ax.plot(1 - np.array(alphas), covs, f"{marker}-", color=color,
                        ms=3.5, lw=1.2, markeredgecolor="white",
                        markeredgewidth=0.4, label=label)
            else:
                # Use the single-alpha data points we have
                # Collect from homosc_class_cond α sweep
                for method_key in ["homosc_class_cond", "hetero_class_cond"]:
                    if method_key not in d:
                        continue
                    m = d[method_key]
                    if "alpha" in m:
                        ax.plot(1 - m["alpha"], m["coverage"], marker,
                                color=color, ms=6, markeredgecolor="white",
                                markeredgewidth=0.5, label=label)

        ax.set_xlabel("Nominal coverage $1{-}\\alpha$")
        ax.set_title(dataset_label)
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.45, 1.02)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", framealpha=0.95, edgecolor="none")

    axes[0].set_ylabel("Empirical coverage")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig3_alpha_calibration.pdf")
    fig.savefig(OUT / "fig3_alpha_calibration.png")
    plt.close(fig)
    print("  fig3_alpha_calibration ✓")


# ================================================================
# Fig 4: Local coverage deviation heatmap
# ================================================================
def fig4_local_coverage_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_DCOL, 2.8))

    for ax, (dataset_label, lc_path) in zip(axes, [
        ("Mendelian",
         "outputs/local_coverage/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/local_coverage_results.json"),
        ("Complex",
         "outputs/local_coverage/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/local_coverage_results.json"),
    ]):
        p = Path(lc_path)
        if not p.exists():
            ax.set_title(f"{dataset_label} (N/A)")
            continue
        d = load_json(p)
        methods = d["per_method"]

        # Collect per-consequence coverage for each method
        method_names = ["homosc", "hetero", "mondrian"]
        method_labels = ["Homo-\nscedastic", "Hetero-\nscedastic", "Mondrian\n(HCCP)"]

        consequence_names = []
        heatmap_data = []

        for method_name in method_names:
            if method_name not in methods:
                continue
            m = methods[method_name]
            pc = m.get("per_consequence", [])
            if not consequence_names:
                consequence_names = [c["bin"] for c in pc if c.get("n", 0) >= 20]
            row = []
            for c in pc:
                if c.get("n", 0) >= 20:
                    row.append(c["coverage"] - 0.9)  # deviation from target
            heatmap_data.append(row)

        if not heatmap_data:
            continue

        # Shorten consequence names
        name_map = {
            "3_prime_UTR_variant": "3'UTR",
            "5_prime_UTR_variant": "5'UTR",
            "downstream_gene_variant": "Downstream",
            "intergenic_variant": "Intergenic",
            "intron_variant": "Intronic",
            "missense_variant": "Missense",
            "non_coding_transcript_exon_variant": "ncRNA exon",
            "regulatory_region_variant": "Regulatory",
            "splice_region_variant": "Splice",
            "synonymous_variant": "Synonymous",
            "TF_binding_site_variant": "TFBS",
            "upstream_gene_variant": "Upstream",
            "dELS": "dELS",
            "pELS": "pELS",
            "PLS": "PLS",
            "CTCF": "CTCF",
        }
        short_names = [name_map.get(n, n[:12]) for n in consequence_names]

        data = np.array(heatmap_data).T  # (consequences × methods)

        # Diverging colormap centered at 0
        vmax = max(0.08, np.abs(data).max())
        cmap = plt.cm.RdBu_r
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm,
                       interpolation="nearest")
        ax.set_xticks(range(len(method_labels)))
        ax.set_xticklabels(method_labels, fontsize=7)
        ax.set_yticks(range(len(short_names)))
        ax.set_yticklabels(short_names, fontsize=6.5)
        ax.set_title(dataset_label)

        # Annotate cells
        for i in range(len(short_names)):
            for j in range(len(method_labels)):
                val = data[i, j]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.02f}", ha="center", va="center",
                        fontsize=5.5, color=color)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Coverage $-$ target", fontsize=8)
    fig.tight_layout(w_pad=1)
    fig.savefig(OUT / "fig4_local_heatmap.pdf")
    fig.savefig(OUT / "fig4_local_heatmap.png")
    plt.close(fig)
    print("  fig4_local_heatmap ✓")


# ================================================================
# Fig 5: DEGU comparison — deviation butterfly chart
# ================================================================
def fig5_degu_deviation():
    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_DCOL, 2.2))

    configs = [
        ("Mendelian",
         "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/conformal_hetero_results.json",
         "outputs/conformal_hetero/DEGU_lite_CADD+GPN-MSA+Borzoi_mendelian_mondrian/conformal_hetero_results.json"),
        ("Complex",
         "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/conformal_hetero_results.json",
         "outputs/conformal_hetero/DEGU_lite_CADD+GPN-MSA+Borzoi_complex_mondrian/conformal_hetero_results.json"),
    ]

    for ax, (title, hccp_path, degu_path) in zip(axes, configs):
        for path_str, label, color in [
            (hccp_path, "HCCP", COLORS["hccp"]),
            (degu_path, "DEGU-lite", COLORS["degu"]),
        ]:
            p = Path(path_str)
            if not p.exists():
                continue
            d = load_json(p)
            m = d.get("mondrian_y_sigma", d.get("hetero_class_cond", {}))
            bins = m.get("coverage_by_sigma_bin", [])
            if not bins:
                continue

            x = np.arange(len(bins))
            devs = [b["coverage"] - 0.9 for b in bins]

            offset = -0.17 if "HCCP" in label else 0.17
            ax.bar(x + offset, devs, width=0.32, color=color, alpha=0.85,
                   edgecolor="white", lw=0.3, label=label)

        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.axhspan(-0.03, 0.03, color="#cccccc", alpha=0.15, zorder=0)
        ax.set_xlabel("$\\hat\\sigma$-bin")
        ax.set_title(title)
        ax.set_ylim(-0.08, 0.08)
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda y, _: f"{y:+.02f}" if y != 0 else "0"))

    axes[0].set_ylabel("Coverage $-$ target (0.90)")
    axes[0].legend(loc="lower left", framealpha=0.95, edgecolor="none")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig5_degu_deviation.pdf")
    fig.savefig(OUT / "fig5_degu_deviation.png")
    plt.close(fig)
    print("  fig5_degu_deviation ✓")


# ================================================================
# Fig 6: σ̂ calibration — learned σ̂ vs actual residual
# ================================================================
def fig6_sigma_calibration():
    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_DCOL, 2.4))

    for ax, (dataset_label, path_str) in zip(axes, [
        ("Mendelian", "outputs/hetero_head/CADD+GPN-MSA+Borzoi_mendelian_abs/scores_with_sigma.parquet"),
        ("Complex", "outputs/hetero_head/CADD+GPN-MSA+Borzoi_complex_abs/scores_with_sigma.parquet"),
    ]):
        import pandas as pd
        p = Path(path_str)
        if not p.exists():
            continue
        df = pd.read_parquet(p)

        sigma = df["sigma"].values
        residual = df["abs_residual"].values

        # Decile binned scatter
        n_bins = 10
        edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
        bin_idx = np.clip(np.digitize(sigma, edges[1:-1]), 0, n_bins - 1)

        bin_sigma_mean = np.array([sigma[bin_idx == b].mean() for b in range(n_bins)])
        bin_resid_mean = np.array([residual[bin_idx == b].mean() for b in range(n_bins)])
        bin_resid_std = np.array([residual[bin_idx == b].std() / np.sqrt((bin_idx == b).sum())
                                  for b in range(n_bins)])

        # Background scatter (subsample)
        rng = np.random.RandomState(42)
        idx = rng.choice(len(sigma), min(800, len(sigma)), replace=False)
        ax.scatter(sigma[idx], residual[idx], s=3, alpha=0.08, color="#888888",
                   rasterized=True, zorder=1)

        # Binned means with error bars
        ax.errorbar(bin_sigma_mean, bin_resid_mean, yerr=bin_resid_std,
                    fmt="o-", color=COLORS["hccp"], ms=5, lw=1.3,
                    markeredgecolor="white", markeredgewidth=0.5,
                    capsize=2, capthick=0.8, zorder=5,
                    label="Decile mean $\\pm$ SE")

        # Perfect calibration line
        lim = max(sigma.max(), residual.max()) * 1.05
        ax.plot([0, lim], [0, lim], "--", color="#999999", lw=0.8, alpha=0.5,
                label="$\\hat\\sigma = |\\mathrm{residual}|$")

        # Spearman correlation
        from scipy.stats import spearmanr
        rho, pval = spearmanr(sigma, residual)
        ax.text(0.05, 0.92, f"$\\rho_S = {rho:.3f}$",
                transform=ax.transAxes, fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.9))

        ax.set_xlabel("Predicted $\\hat\\sigma(x)$")
        ax.set_title(dataset_label)
        ax.set_xlim(-0.005, None)
        ax.set_ylim(-0.005, None)

    axes[0].set_ylabel("Actual $|y - \\hat{p}(x)|$")
    axes[0].legend(loc="lower right", framealpha=0.95, edgecolor="none")
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig6_sigma_calibration.pdf")
    fig.savefig(OUT / "fig6_sigma_calibration.png")
    plt.close(fig)
    print("  fig6_sigma_calibration ✓")


# ================================================================
# Fig 7: Three-axis OOD summary bar (compact)
# ================================================================
def fig7_three_axis_summary():
    fig, ax = plt.subplots(figsize=(NEURIPS_COL, 2.2))

    # Data from experiments
    data = {
        "Mendelian": {"Chrom-LOO": 0.077, "Trait-LOO": 0.004, "Cross-dataset\n(C→M)": 0.035},
        "Complex": {"Chrom-LOO": 0.023, "Trait-LOO": 0.002, "Cross-dataset\n(M→C)": 0.331},
    }

    axes_labels = list(data["Mendelian"].keys())
    x = np.arange(len(axes_labels))
    w = 0.35

    for i, (ds, color) in enumerate([("Mendelian", COLORS["hccp"]),
                                      ("Complex", COLORS["theory"])]):
        vals = list(data[ds].values())
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w, color=color,
                      alpha=0.85, edgecolor="white", lw=0.5, label=ds)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6)

    ax.axhline(0.04, color="black", ls="--", lw=0.8, alpha=0.4)
    ax.text(2.6, 0.045, "target $\\leq 0.04$", fontsize=6.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(axes_labels, fontsize=7.5)
    ax.set_ylabel("$\\hat\\sigma$-bin coverage gap")
    ax.set_ylim(0, 0.38)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="none")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_three_axis.pdf")
    fig.savefig(OUT / "fig7_three_axis.png")
    plt.close(fig)
    print("  fig7_three_axis ✓")


# ================================================================
if __name__ == "__main__":
    print("Generating v2 figures...")
    fig2_k_sweep()
    fig3_alpha_calibration()
    fig4_local_coverage_heatmap()
    fig5_degu_deviation()
    fig6_sigma_calibration()
    fig7_three_axis_summary()
    print("All done.")
