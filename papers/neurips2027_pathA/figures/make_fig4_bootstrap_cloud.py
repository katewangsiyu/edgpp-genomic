"""Figure 4 — bootstrap robustness cloud, redesigned.

Replaces the legacy 1x2 stair-step histogram with a 1x2 joint 2D scatter
cloud of (Cov|Y=1, sigma-bin-gap) per bootstrap replicate (B = 200 chrom
resamples). Visual idiom: corner-plot-style joint + marginals, the gold
standard in Bayesian uncertainty visualization (Vehtari, Gelman, Gabry
"Visualization in Bayesian workflow", arxiv 1709.01449).

Axes deliberately mirror Fig 1(b) Pareto so the reader reads Fig 4 as
"Fig 1(b) HCCP point with B=200 bootstrap uncertainty cloud around it".
The all-chrom point estimate sits in the upper tail because real evaluation
always includes the worst chromosomes (chr6 / chr19 on Mendelian), while
bootstrap chrom resampling sometimes excludes them — a meaningful asymmetry,
not a bug.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = Path(__file__).resolve().parent
OUT_PDF = FIG_DIR / "fig_bootstrap_density.pdf"  # keep filename for LaTeX continuity
OUT_PNG = FIG_DIR / "fig_bootstrap_density.png"

# All-chrom (single-seed) point estimates at K=5, K_eval-fair.
# Computed from scripts/14_conformal_hetero.mondrian_class_sigma_conformal
# on the full test set; matches the Tab 1 max-bin marginal gap metric.
POINT_EST = {
    "Mendelian": {"cov_pos": 0.905, "gap": 0.005},
    "Complex":   {"cov_pos": 0.908, "gap": 0.002},
}

DATASETS = {
    "Mendelian": REPO_ROOT / "outputs/bootstrap_ci/CADD+GPN-MSA+Borzoi_mendelian.json",
    "Complex":   REPO_ROOT / "outputs/bootstrap_ci/CADD+GPN-MSA+Borzoi_complex.json",
}

C_HCCP = "#d1495b"
C_POINT = "#1f4e79"
C_BAND = "#a4c293"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
})


def load_replicates(p: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(p.read_text())
    rep = d["replicates"]
    cov_pos = np.array([r["coverage_pos"] for r in rep])
    gap = np.array([r["sigma_bin_gap"] for r in rep])
    return cov_pos, gap


def kde_contour(ax, x: np.ndarray, y: np.ndarray, levels: list[float],
                color: str, lw_levels: list[float]):
    """Draw KDE contours at given probability mass levels (e.g. 0.50, 0.95)."""
    kde = gaussian_kde(np.vstack([x, y]))
    xx, yy = np.mgrid[
        x.min() - 0.02 : x.max() + 0.02 : 80j,
        y.min() - 0.005 : y.max() + 0.005 : 80j,
    ]
    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    # Convert probability mass levels to density-threshold levels by integrating.
    z_sorted = np.sort(z.ravel())[::-1]
    cum = np.cumsum(z_sorted)
    cum /= cum[-1]
    thresholds = []
    for L in levels:
        i = np.searchsorted(cum, L)
        thresholds.append(z_sorted[min(i, len(z_sorted) - 1)])
    thresholds = sorted(set(thresholds))  # ascending for contourf
    ax.contour(xx, yy, z, levels=thresholds,
               colors=[color] * len(thresholds),
               linewidths=lw_levels[: len(thresholds)],
               linestyles=["-"] * len(thresholds), alpha=0.85, zorder=2)


def make_panel(fig, gs_root, dataset: str, replicates_path: Path):
    cov_pos, gap = load_replicates(replicates_path)
    pe = POINT_EST[dataset]

    # Inner gridspec: 2x2, top-left = marginal x, bottom-right = marginal y, etc.
    inner = gs_root.subgridspec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.06, wspace=0.06,
    )
    ax_main = fig.add_subplot(inner[1, 0])
    ax_top = fig.add_subplot(inner[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)

    # Main scatter + KDE contours.
    ax_main.scatter(cov_pos, gap, s=10, color=C_HCCP, alpha=0.45,
                    edgecolor="none", zorder=2,
                    label=fr"bootstrap replicates ($B={len(cov_pos)}$)")
    kde_contour(ax_main, cov_pos, gap, [0.50, 0.95],
                C_HCCP, [1.4, 0.8])

    # Bootstrap mean (cross marker).
    bs_mean_x = cov_pos.mean()
    bs_mean_y = gap.mean()
    ax_main.scatter([bs_mean_x], [bs_mean_y], marker="P", s=85,
                    facecolor=C_HCCP, edgecolor="white", linewidth=1.0,
                    zorder=4, label=f"bootstrap mean")

    # All-chrom (single-seed) point estimate (star). Bright gold + white halo
    # circle behind it so the marker stays visible inside the dense replicate
    # cloud regardless of how the bootstrap mean lands.
    ax_main.scatter([pe["cov_pos"]], [pe["gap"]], marker="o", s=140,
                    facecolor="white", edgecolor="white", linewidth=0,
                    zorder=5)
    ax_main.scatter([pe["cov_pos"]], [pe["gap"]], marker="*", s=118,
                    facecolor="#f2c14e", edgecolor="#222", linewidth=0.7,
                    zorder=6, label=r"all-chrom point ($K{=}5$)")

    # Reference lines: cov_pos = 0.90 target.
    ax_main.axvline(0.90, color="#222222", linestyle="--", linewidth=0.6,
                    alpha=0.6, zorder=1)

    # Axes formatting.
    ax_main.set_xlabel(r"Cov$|Y{=}1$  ($\to$ better)")
    ax_main.set_ylabel(r"$\hat\sigma$-bin gap  ($\downarrow$ better)")
    # Dataset label sits on the top marginal axes for vertical breathing room.
    ax_top.set_title(f"{dataset}", fontsize=10.5, pad=2, loc="left",
                     fontweight="bold")
    ax_main.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    for s in ["top", "right"]:
        ax_main.spines[s].set_visible(False)

    # Marginal KDEs on top (cov_pos) and right (gap).
    xs = np.linspace(cov_pos.min() - 0.01, cov_pos.max() + 0.01, 200)
    kx = gaussian_kde(cov_pos)
    ax_top.fill_between(xs, kx(xs), color=C_HCCP, alpha=0.30, linewidth=0)
    ax_top.plot(xs, kx(xs), color=C_HCCP, linewidth=1.0)
    ax_top.axvline(pe["cov_pos"], color=C_POINT, linestyle="--", linewidth=0.8)
    ax_top.set_yticks([])
    plt.setp(ax_top.get_xticklabels(), visible=False)
    for s in ["top", "right", "left"]:
        ax_top.spines[s].set_visible(False)
    ax_top.tick_params(axis="x", length=0)

    ys = np.linspace(gap.min() - 0.005, gap.max() + 0.005, 200)
    ky = gaussian_kde(gap)
    ax_right.fill_betweenx(ys, ky(ys), color=C_HCCP, alpha=0.30, linewidth=0)
    ax_right.plot(ky(ys), ys, color=C_HCCP, linewidth=1.0)
    ax_right.axhline(pe["gap"], color=C_POINT, linestyle="--", linewidth=0.8)
    ax_right.set_xticks([])
    plt.setp(ax_right.get_yticklabels(), visible=False)
    for s in ["top", "right", "bottom"]:
        ax_right.spines[s].set_visible(False)
    ax_right.tick_params(axis="y", length=0)

    # CI annotation in main panel — placed lower-left so it never collides
    # with the legend in the upper-right corner of the Mendelian panel.
    ci_xlo, ci_xhi = np.quantile(cov_pos, [0.025, 0.975])
    ci_ylo, ci_yhi = np.quantile(gap, [0.025, 0.975])
    txt = (
        rf"95\% bootstrap CI:" + "\n"
        rf"  Cov$|Y{{=}}1 \in [{ci_xlo:.3f},\ {ci_xhi:.3f}]$" + "\n"
        rf"  gap $\in [{ci_ylo:.3f},\ {ci_yhi:.3f}]$"
    )
    ax_main.text(0.02, 0.04, txt, transform=ax_main.transAxes,
                 ha="left", va="bottom", fontsize=6.8,
                 bbox=dict(facecolor="white", edgecolor="#cccccc",
                           linewidth=0.4, boxstyle="round,pad=0.3", alpha=0.92))

    if dataset == "Mendelian":
        ax_main.legend(loc="upper right", fontsize=6.8, frameon=True,
                       framealpha=0.94, borderpad=0.4, handletextpad=0.4)


def main():
    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.22,
                          top=0.86, bottom=0.10, left=0.06, right=0.985)
    make_panel(fig, gs[0, 0], "Mendelian", DATASETS["Mendelian"])
    make_panel(fig, gs[0, 1], "Complex",   DATASETS["Complex"])

    fig.suptitle(
        r"Bootstrap robustness ($B = 200$ chromosome-level resamples) of HCCP "
        r"on the (Cov$|Y{=}1$, $\hat\sigma$-bin gap) plane",
        fontsize=10.5, y=0.98,
    )

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
