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

# --- Unified NeurIPS-classic style ---
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from T_tools.paper_style import apply_paper_style  # noqa: E402
apply_paper_style()

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
        ("CADD+GPN-MSA+Borzoi_mendelian", "Mendelian ($n{=}3{,}380$)", COLORS["hccp"]),
        ("CADD+GPN-MSA+Borzoi_complex", "Complex ($n{=}11{,}400$)", COLORS["hccp"]),
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

        # K* and K_CV annotations — always place text to the RIGHT with enough margin
        ax.axvline(d["K_cv"], color=color_accent, ls="--", lw=1, alpha=0.8)
        text_x = max(d["K_cv"] + 2, 5)  # ensure text doesn't overlap y-axis
        ax.annotate(f'$\\hat{{K}}_{{\\mathrm{{CV}}}}={d["K_cv"]}$',
                    xy=(d["K_cv"], max(mean_gap) * 0.5),
                    xytext=(text_x, max(mean_gap) * 0.85),
                    fontsize=7, color=color_accent,
                    arrowprops=dict(arrowstyle="->", color=color_accent, lw=0.8))

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
# Fig 3: Why Mondrian? — 3-method σ̂-bin coverage comparison
# ================================================================
def fig3_method_comparison():
    """The 'money figure': shows homosc/hetero/mondrian per-σ̂-bin coverage.
    Mondrian column hugs the target line; others don't."""
    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_DCOL, 2.5), sharey=True)

    method_info = [
        ("homosc", "Split CP", COLORS["homosc"], "//"),
        ("hetero", "Heterosc.", COLORS["split"], ""),
        ("mondrian", "Mondrian (HCCP)", COLORS["hccp"], ""),
    ]

    for ax, (dataset, title) in zip(axes, [
        ("mendelian", "Mendelian"),
        ("complex", "Complex"),
    ]):
        path = Path(f"outputs/local_coverage/CADD+GPN-MSA+Borzoi_{dataset}_abs_mondrian/local_coverage_results.json")
        if not path.exists():
            continue
        d = load_json(path)
        methods = d["per_method"]

        n_methods = len(method_info)
        bar_width = 0.25

        for j, (method_key, method_label, color, hatch) in enumerate(method_info):
            if method_key not in methods:
                continue
            bins = methods[method_key].get("per_sigma_bin", [])
            if not bins:
                continue

            x = np.arange(len(bins))
            covs = [b["coverage"] for b in bins]
            offset = (j - 1) * bar_width

            ax.bar(x + offset, covs, width=bar_width, color=color, alpha=0.8,
                   edgecolor="white", lw=0.3, hatch=hatch, label=method_label)

        # Target line and band
        ax.axhline(0.9, color="black", ls="--", lw=0.8, alpha=0.5)
        ax.axhspan(0.87, 0.93, color=COLORS["hccp"], alpha=0.04, zorder=0)

        # Worst-gap annotations
        for method_key, method_label, color, _ in method_info:
            if method_key not in methods:
                continue
            bins = methods[method_key].get("per_sigma_bin", [])
            covs = [b["coverage"] for b in bins]
            wg = max(abs(c - 0.9) for c in covs) if covs else 0
            # Only annotate mondrian in box
            if method_key == "mondrian":
                ax.text(0.97, 0.03, f"Mondrian gap: {wg:.3f}",
                        transform=ax.transAxes, fontsize=6.5, color=color,
                        ha="right", va="bottom", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor=color, alpha=0.9, lw=0.8))

        ax.set_xlabel("$\\hat\\sigma$-bin (low → high uncertainty)")
        ax.set_title(title)
        ax.set_ylim(0.45, 1.05)

    axes[0].set_ylabel("Coverage")
    axes[0].legend(loc="lower left", framealpha=0.95, edgecolor="none",
                   ncol=1, fontsize=7)
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fig3_method_comparison.pdf")
    fig.savefig(OUT / "fig3_method_comparison.png")
    plt.close(fig)
    print("  fig3_method_comparison ✓")


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

    # Shared colorbar — put on the right with enough space
    fig.subplots_adjust(right=0.88, wspace=0.35)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Coverage $-$ target", fontsize=7.5)
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
        ax.set_ylim(-0.09, 0.09)
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda y, _: f"{y:+.02f}" if y != 0 else "0"))

        # Annotate worst-gap for each method
        for path_str, label, color in [
            (hccp_path, "HCCP", COLORS["hccp"]),
            (degu_path, "DEGU", COLORS["degu"]),
        ]:
            p = Path(path_str)
            if not p.exists():
                continue
            d = load_json(p)
            m = d.get("mondrian_y_sigma", d.get("hetero_class_cond", {}))
            bins = m.get("coverage_by_sigma_bin", [])
            if bins:
                covs = [b["coverage"] for b in bins if b.get("n", 0) >= 5]
                wg = max(abs(c - 0.9) for c in covs) if covs else 0
                x_pos = 0.02 if "HCCP" in label else 0.98
                ha = "left" if "HCCP" in label else "right"
                ax.text(x_pos, 0.95, f"{label} gap: {wg:.3f}",
                        transform=ax.transAxes, fontsize=6, color=color,
                        ha=ha, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor=color, alpha=0.8, lw=0.5))

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
        # Zoom to data range instead of fixed [0,1]
        x_max = np.percentile(sigma, 99) * 1.15
        y_max = np.percentile(residual, 99) * 1.15
        lim = max(x_max, y_max)
        ax.set_xlim(-0.005, lim)
        ax.set_ylim(-0.005, lim)

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
    fig, ax = plt.subplots(figsize=(NEURIPS_COL + 0.3, 2.4))

    # Data from experiments — use log-scale-friendly layout
    labels = ["Chrom\nLOO", "Trait\nLOO", "Cross\n(C→M)"]
    mendelian = [0.077, 0.004, 0.035]
    complex_v = [0.023, 0.002, 0.331]

    x = np.arange(len(labels))
    w = 0.32

    bars_m = ax.bar(x - w/2, mendelian, width=w, color=COLORS["hccp"],
                    alpha=0.85, edgecolor="white", lw=0.5, label="Mendelian")
    bars_c = ax.bar(x + w/2, complex_v, width=w, color=COLORS["theory"],
                    alpha=0.85, edgecolor="white", lw=0.5, label="Complex")

    for bars, vals in [(bars_m, mendelian), (bars_c, complex_v)]:
        for bar, v in zip(bars, vals):
            y_pos = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y_pos + 0.003, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.5, weight="bold")

    ax.axhline(0.04, color="black", ls="--", lw=0.8, alpha=0.4)
    ax.text(0.02, 0.043, "target ≤ 0.04", fontsize=6.5, alpha=0.5,
            transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("$\\hat\\sigma$-bin gap")
    ax.set_yscale("log")
    ax.set_ylim(0.001, 0.5)
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="none", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_three_axis.pdf")
    fig.savefig(OUT / "fig7_three_axis.png")
    plt.close(fig)
    print("  fig7_three_axis ✓")


# ================================================================
# Fig 8: Coverage vs Set-size Pareto (NeurIPS CP standard)
# ================================================================
def fig8_pareto():
    """X = singleton fraction (efficiency), Y = σ̂-bin gap (local coverage).
    Each point = one method × one dataset. Top CP papers always show this."""
    fig, ax = plt.subplots(figsize=(NEURIPS_COL + 0.5, 2.6))

    all_points = []
    configs = [
        # (label, path, marker, dataset_suffix)
        ("HCCP\n(Mendelian)", "CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian", "o", "M"),
        ("DEGU\n(Mendelian)", "DEGU_lite_CADD+GPN-MSA+Borzoi_mendelian_mondrian", "s", "M"),
        ("HCCP\n(Complex)", "CADD+GPN-MSA+Borzoi_complex_abs_mondrian", "o", "C"),
        ("DEGU\n(Complex)", "DEGU_lite_CADD+GPN-MSA+Borzoi_complex_mondrian", "s", "C"),
    ]

    for label, subdir, marker, ds in configs:
        p = Path(f"outputs/conformal_hetero/{subdir}/conformal_hetero_results.json")
        if not p.exists():
            continue
        d = load_json(p)

        # Try mondrian first, then hetero
        for method_key in ["mondrian_y_sigma", "hetero_class_cond", "homosc_class_cond"]:
            if method_key in d:
                m = d[method_key]
                break
        else:
            continue

        bins = m.get("coverage_by_sigma_bin", [])
        if not bins:
            continue
        coverages = [b["coverage"] for b in bins if b.get("n", 0) >= 5]
        worst_gap = max(abs(c - 0.9) for c in coverages) if coverages else np.nan
        singleton = m.get("frac_singleton", np.nan)

        is_hccp = "HCCP" in label
        is_mendelian = ds == "M"
        color = COLORS["hccp"] if is_hccp else (COLORS["degu"] if "DEGU" in label else COLORS["homosc"])
        facecolor = color if is_mendelian else "white"

        flat_label = label.replace("\n", " ")
        ax.scatter(singleton, worst_gap, s=70, marker=marker,
                   facecolors=facecolor, edgecolors=color, linewidths=1.5,
                   label=flat_label, zorder=5)
        all_points.append((singleton, worst_gap, flat_label, color, is_hccp))

    # Iso-cost diagonals: L = gap + λ·(1 − efficiency), with λ = 0.05.
    # Smaller L = better. Lines have positive slope λ in (eff, gap):
    # gap = (L − λ) + λ·eff. Points BELOW a line beat that L.
    #
    # Why λ = 0.05: at the headline α = 0.10 op point, the appendix targets
    # singleton fraction ≈ 0.80 (Mendelian) and ≤ 0.04 worst-bin gap. The
    # ratio 0.04 / 0.80 ≈ 0.05 makes a 1-pp swap of efficiency cost-equal
    # to a 0.0005 swap of gap, the implicit operational tradeoff.
    LAMBDA_ISO = 0.05
    iso_levels = [0.04, 0.06, 0.08, 0.10]
    eff_grid = np.linspace(0.05, 1.0, 50)
    for L in iso_levels:
        gap_iso = (L - LAMBDA_ISO) + LAMBDA_ISO * eff_grid
        ax.plot(eff_grid, gap_iso, ls="--", lw=0.7, color="#888888",
                alpha=0.55, zorder=1)
        # Inline label on the LEFT side (less cluttered than right; matches
        # the diagonal-iso-line convention in operations-research plots).
        x_lbl = 0.13
        y_lbl = (L - LAMBDA_ISO) + LAMBDA_ISO * x_lbl
        if 0 < y_lbl < 0.078:
            ax.text(x_lbl, y_lbl + 0.001, rf"$L{{=}}{L:.2f}$",
                    fontsize=6.0, color="#666", ha="left", va="bottom",
                    rotation=np.degrees(np.arctan(LAMBDA_ISO * 6)))  # ≈ visual slope

    # Per-point iso-cost annotation, so reviewers can rank them numerically.
    for x, y, lbl, color, is_hccp in all_points:
        L_pt = y + LAMBDA_ISO * (1 - x)
        dx, dy = (0.018, 0.0018) if is_hccp else (0.018, -0.0035)
        ax.annotate(rf"$L{{=}}{L_pt:.3f}$",
                    xy=(x, y), xytext=(x + dx, y + dy),
                    fontsize=6.4, color=color, alpha=0.9,
                    ha="left", va="center")

    # Use legend instead of inline labels (avoids overlap)
    ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=True,
              framealpha=0.9, edgecolor="0.7")

    # Ideal region (kept; the iso-lines and the band tell complementary stories).
    ax.axhspan(0, 0.03, color="#2ca02c", alpha=0.08, zorder=0)
    ax.text(0.78, 0.0015, "ideal zone (gap $\\leq$ 0.03)", fontsize=7,
            color="#2ca02c", alpha=0.85, style="italic", ha="right")
    # Iso-cost legend caption — placed in the dead space at lower-right
    # (above the "ideal zone" italics, below the data legend).
    ax.text(0.96, 0.069,
            rf"iso-cost: $L = \mathrm{{gap}} + {LAMBDA_ISO}\,(1-\mathrm{{eff}})$",
            fontsize=6.4, color="#444", ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="#cccccc",
                      boxstyle="round,pad=0.25", linewidth=0.4, alpha=0.9))

    ax.set_xlabel("Singleton fraction (efficiency →)")
    ax.set_ylabel("Worst $\\hat\\sigma$-bin gap (← coverage)")
    ax.set_xlim(0.1, 0.98)
    ax.set_ylim(-0.005, 0.085)

    fig.tight_layout()
    fig.savefig(OUT / "fig8_pareto.pdf")
    fig.savefig(OUT / "fig8_pareto.png")
    plt.close(fig)
    print("  fig8_pareto ✓")


# ================================================================
# Fig 9: Per-chromosome coverage (Barber 2023 style)
# ================================================================
def fig9_per_chrom():
    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_DCOL, 2.2))

    for ax, (title, path_str) in zip(axes, [
        ("Mendelian",
         "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/conformal_hetero_results.json"),
        ("Complex",
         "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/conformal_hetero_results.json"),
    ]):
        p = Path(path_str)
        if not p.exists():
            continue
        d = load_json(p)
        m = d.get("mondrian_y_sigma", d.get("hetero_class_cond", {}))
        pc = m.get("per_chrom_coverage", {})
        if not pc:
            continue

        # Sort chromosomes naturally
        def chrom_sort_key(c):
            try:
                return (0, int(c))
            except ValueError:
                return (1, ord(c[0]))

        chroms = sorted(pc.keys(), key=chrom_sort_key)
        covs = [pc[c] for c in chroms]
        x = range(len(chroms))

        # Color by deviation — continuous RdBu gradient
        cmap_bar = plt.cm.RdYlBu
        norm_bar = mcolors.TwoSlopeNorm(vmin=-0.10, vcenter=0, vmax=0.10)
        colors = [cmap_bar(norm_bar(c - 0.9)) for c in covs]

        bars = ax.bar(x, covs, color=colors, edgecolor="white", lw=0.3, width=0.7)
        ax.axhline(0.9, color="black", ls="--", lw=0.9, alpha=0.6,
                   label=r"target $1{-}\alpha = 0.90$")
        ax.axhspan(0.87, 0.93, color=COLORS["hccp"], alpha=0.08, zorder=0,
                   label=r"$\pm 0.03$ band")
        ax.set_xticks(x)
        ax.set_xticklabels([f"chr{c}" for c in chroms], rotation=45,
                           ha="right", fontsize=6)
        ax.set_title(title)
        ax.set_ylim(0.75, 1.0)
        ax.legend(loc="lower right", fontsize=6.5, framealpha=0.9)

    axes[0].set_ylabel("Mondrian coverage")

    # Shared colorbar for the deviation-from-target encoding
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=norm_bar)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal",
                        shrink=0.4, pad=0.18, aspect=30)
    cbar.set_label(r"coverage $-$ target", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.savefig(OUT / "fig9_per_chrom.pdf")
    fig.savefig(OUT / "fig9_per_chrom.png")
    plt.close(fig)
    print("  fig9_per_chrom ✓")


# ================================================================
if __name__ == "__main__":
    print("Generating v2 figures...")
    fig2_k_sweep()
    fig3_method_comparison()
    fig4_local_coverage_heatmap()
    fig5_degu_deviation()
    fig6_sigma_calibration()
    fig7_three_axis_summary()
    fig8_pareto()
    fig9_per_chrom()
    print("All done.")
