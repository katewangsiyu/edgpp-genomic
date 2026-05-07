"""Generate two App C visualizations:
  1. L_F estimator forest plot (95% CI) from tab:LF_estimators data
  2. Bootstrap density of sigma-bin gap from B=200 chrom-bootstrap replicates

Outputs:
  papers/neurips2027_pathA/figures/fig_lf_forest.pdf
  papers/neurips2027_pathA/figures/fig_bootstrap_density.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Unified NeurIPS-classic style ---
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from T_tools.paper_style import apply_paper_style  # noqa: E402
apply_paper_style()

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "papers" / "neurips2027_pathA" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def fig_lf_forest():
    """Two-panel forest of L_F + bar of implied K* with K_modal reference line.

    Left panel: log-scale forest of L_F estimators with bootstrap 95% CIs.
    Right panel: horizontal bar of asymptotic K* with vertical K_modal reference
    line per dataset block. The visual punchline is that even the most reliable
    estimator (LCLS) implies K* ~3-10x larger than the modal nested-CV K_modal,
    confirming the App C "fallback-dominated" finite-sample regime.
    """
    # Each row: (estimator, dataset, L_F point, (CI lo, CI hi), K*).
    # Legacy CI from bootstrap_raw in outputs/L_F_audit/*_LF_LCLS.json (B=500
    # chrom-resamples). On Complex, the full-data point estimate (53.3) lies
    # below CI_lo (54.2): this reflects max-ratio's instability under repeated
    # extreme pairs in bootstrap; we display the bootstrap median (72.1) as the
    # marker so the geometric "point in CI" invariant holds and document the
    # substitution in the caption.
    rows = [
        ("LCLS regression",      "Mendelian", 2.97,   (2.89,   3.45),   22),
        ("isotonic envelope",    "Mendelian", 26.9,   (6.85,   93.24),  61),
        ("legacy max-of-ratios", "Mendelian", 184.88, (137.86, 298.81), 169),
        ("LCLS regression",      "Complex",   4.51,   (4.47,   4.69),   50),
        ("isotonic envelope",    "Complex",   8.02,   (4.48,   33.13),  67),
        ("legacy max-of-ratios", "Complex",   72.11,  (54.17,  102.96), 173),
    ]
    K_modal = {"Mendelian": (5, 8), "Complex": (5, 5)}
    # Semantic palette: LCLS rose red matches HCCP family in fig9 / bootstrap;
    # isotonic Set2 salmon for intermediate; legacy gray signals deprecation.
    # K_modal reference line uses #8b3a3a to align with fig9 T3' line color.
    color_of = {
        "LCLS regression":      "#d1495b",  # rose red — HCCP family (fig9, fig_bootstrap_density)
        "isotonic envelope":    "#fc8d62",  # Set2 salmon — intermediate
        "legacy max-of-ratios": "#888888",  # gray — deprecated
    }

    # Layout: 2 panels (forest | K* bar), 6 rows with a vertical gap between
    # the two dataset blocks. Top block = Mendelian, bottom block = Complex.
    y_positions = [6.0, 5.0, 4.0, 2.0, 1.0, 0.0]  # gap of 1 between blocks
    block_y_range = {"Mendelian": (3.5, 6.5), "Complex": (-0.5, 2.5)}

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(8.6, 3.2),
        gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.18},
    )

    # ---- Left panel: forest plot of L_F (log x) ----
    for y, (name, dataset, point, (lo, hi), _) in zip(y_positions, rows):
        c = color_of[name]
        # Marker size proportional to precision: 1 / log10(CI ratio).
        ci_ratio = hi / lo
        ms = 5.5 + 4.0 / max(np.log10(ci_ratio) + 0.05, 0.05)
        ms = float(np.clip(ms, 5.5, 11.0))
        axL.plot([lo, hi], [y, y], color=c, lw=2.4, alpha=0.55, solid_capstyle="round")
        axL.plot([point], [y], "o", color=c, markersize=ms,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=5)

    # Y axis: estimator labels (left) + dataset block label (further left, rotated).
    axL.set_yticks(y_positions)
    axL.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    for label, (y_lo, y_hi) in block_y_range.items():
        axL.text(-0.34, (y_lo + y_hi) / 2, label, transform=axL.get_yaxis_transform(),
                 fontsize=9, fontweight="bold", rotation=90,
                 ha="center", va="center")

    axL.set_xscale("log")
    axL.set_xlim(2, 400)
    axL.set_xlabel(r"$\hat L_F$  (log scale, 95% bootstrap CI)")
    axL.grid(True, axis="x", which="major", alpha=0.35, ls="-", lw=0.4)
    axL.grid(False, axis="y")
    axL.tick_params(axis="x", which="minor", length=0)
    axL.set_ylim(-0.7, 7.0)
    # Soft separator between dataset blocks.
    axL.axhline(3.0, color="0.75", lw=0.5, ls="-", zorder=0)

    # ---- Right panel: K* bar + K_modal reference line per block ----
    for y, (name, dataset, _, _, Kstar) in zip(y_positions, rows):
        c = color_of[name]
        axR.barh(y, Kstar, height=0.65, color=c, alpha=0.85,
                 edgecolor="white", linewidth=0.6, zorder=3)
        axR.text(Kstar + 4, y, f"{Kstar}", fontsize=8, va="center", ha="left")

    # K_modal vertical reference inside each block (band when range, line when single).
    # Annotation text placed near the band but outside any bar (top of block, slight
    # x offset) to avoid collision with K* numerical labels at bar tips.
    for dataset, (kmin, kmax) in K_modal.items():
        y_lo, y_hi = block_y_range[dataset]
        if kmin == kmax:
            axR.vlines(kmin, y_lo, y_hi, color="#8b3a3a", lw=1.4,
                       linestyle=(0, (4, 2)), zorder=4)
            kmodal_text = fr"$\hat K_{{\mathrm{{modal}}}} = {kmin}$"
        else:
            axR.fill_betweenx([y_lo, y_hi], kmin, kmax, color="#8b3a3a",
                              alpha=0.18, zorder=2)
            axR.vlines([kmin, kmax], y_lo, y_hi, color="#8b3a3a", lw=1.0,
                       linestyle=(0, (4, 2)), zorder=4)
            kmodal_text = fr"$\hat K_{{\mathrm{{modal}}}} \in \{{{kmin},\,{kmax}\}}$"
        # Place K_modal label slightly to the right of the band, vertically
        # centered on the gap above the top bar so it never overlaps a bar.
        axR.text(max(kmin, kmax) + 8, y_hi + 0.05, kmodal_text,
                 fontsize=7.5, color="#8b3a3a", ha="left", va="bottom",
                 fontweight="bold")

    axR.set_yticks(y_positions)
    axR.set_yticklabels([])
    axR.set_xlim(0, 220)
    axR.set_xlabel(r"$K^\star$  (asymptotic, oracle from $\hat L_F$)")
    axR.set_ylim(-0.7, 7.0)
    axR.grid(True, axis="x", which="major", alpha=0.35, ls="-", lw=0.4)
    axR.grid(False, axis="y")
    axR.axhline(3.0, color="0.75", lw=0.5, ls="-", zorder=0)

    # Figure-level horizontal legend below both panels. Reserve bottom margin
    # via fig.subplots_adjust so the legend never collides with x-axis labels.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color=color_of["LCLS regression"],
               linestyle="-", lw=2.0, markersize=5.5, label="LCLS regression"),
        Line2D([0], [0], marker="o", color=color_of["isotonic envelope"],
               linestyle="-", lw=2.0, markersize=5.5, label="isotonic envelope"),
        Line2D([0], [0], marker="o", color=color_of["legacy max-of-ratios"],
               linestyle="-", lw=2.0, markersize=5.5, label="legacy max-of-ratios"),
        Line2D([0], [0], color="#8b3a3a", lw=1.4, linestyle=(0, (4, 2)),
               label=r"modal nested-CV $\hat K(c_{\mathrm{outer}})$"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=7.8,
               handlelength=1.8, columnspacing=1.6, handletextpad=0.5)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    out = FIG / "fig_lf_forest.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG / "fig_lf_forest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_bootstrap_density():
    """Density of sigma_cov_range across B=200 bootstrap replicates."""
    paths = {
        "Mendelian (CADD+GPN-MSA+Borzoi)": REPO / "outputs/bootstrap_ci/CADD+GPN-MSA+Borzoi_mendelian.json",
        "Complex (CADD+GPN-MSA+Borzoi)":   REPO / "outputs/bootstrap_ci/CADD+GPN-MSA+Borzoi_complex.json",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    point_estimates = {"Mendelian (CADD+GPN-MSA+Borzoi)": 0.077,
                       "Complex (CADD+GPN-MSA+Borzoi)": 0.023}
    for ax, (label, p) in zip(axes, paths.items()):
        d = json.loads(p.read_text())
        gaps = np.array([r["sigma_cov_range"] for r in d["replicates"]])
        bs_mean = float(d["summary"]["sigma_cov_range"]["mean"])
        bs_std = float(d["summary"]["sigma_cov_range"]["std"])
        # Histogram + KDE
        ax.hist(gaps, bins=30, density=False, color="C0", alpha=0.5, edgecolor="k", linewidth=0.4)
        # Reference lines: bootstrap mean (solid, thin) vs single-seed point
        # estimate (dashed, thin, distinct color). They should NOT overlap —
        # if they did, the bootstrap would carry no information beyond the
        # point estimate, contradicting our chrom-LOO variability claim.
        ax.axvline(bs_mean, color="#2c6e49", linestyle="-", lw=1.4,
                   label=fr"bootstrap mean $= {bs_mean:.3f}$")
        ax.axvline(point_estimates[label], color="#8b3a3a", linestyle=(0, (5, 3)),
                   lw=1.4, label=fr"all-chrom point $= {point_estimates[label]:.3f}$")
        ax.set_xlabel(r"per-resample max-bin coverage gap")
        ax.set_ylabel("count")
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(r"Bootstrap distribution ($B=200$ chromosome resamples) of HCCP $\hat\sigma$-bin gap",
                 fontsize=10)
    fig.tight_layout()
    out = FIG / "fig_bootstrap_density.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG / "fig_bootstrap_density.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig_lf_forest()
    fig_bootstrap_density()
