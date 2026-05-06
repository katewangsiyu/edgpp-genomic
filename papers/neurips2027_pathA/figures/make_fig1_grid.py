"""Figure 1 — HCCP at a glance: Mondrian-grid concept + Pareto frontier.

Panel (a): three side-by-side y x sigma-bin grids on Complex CADD+GPN-MSA+Borzoi.
Each cell colored by empirical coverage; numbers annotated inside cells.
The story is: Split CP (no partition) leaks ~0.50 per cell; class-Mondrian
controls class margin but leaves sigma-bin gap ~0.90; HCCP joint partition
holds every cell at 0.90 +/- 0.04.

Panel (b): Pareto frontier on (cov|Y=1, sigma-bin gap) at pi+ = 0.10, using
Tab.~2 bootstrap means (B = 200 chrom-resamples).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

REPO_ROOT = Path(__file__).resolve().parents[3]
PARQUET = REPO_ROOT / "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/conformal_hetero_scores.parquet"

FIG_DIR = Path(__file__).resolve().parent
OUT_PDF = FIG_DIR / "fig1_concept.pdf"
OUT_PNG = FIG_DIR / "fig1_concept.png"

K = 5
TARGET = 0.90

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
})

C_HCCP = "#d1495b"
C_BASE = "#4a6fa5"
C_IDEAL = "#a4c293"

# Diverging colormap centered at TARGET = 0.90.
# Below 0.90: red (under-coverage); at 0.90: pale; above 0.90: blue (over-coverage).
_cmap = LinearSegmentedColormap.from_list(
    "miscov",
    [(0.0, "#a83232"),   # 0.40 -> deep red
     (0.4, "#e89090"),
     (0.5, "#f5f5f5"),   # target -> off-white
     (0.6, "#9fbfdf"),
     (1.0, "#1f4e79")],  # 1.00 -> deep blue
)


# ============================================================================
# Compute per-cell coverage from parquet.
# ============================================================================
def per_cell_coverage(df: pd.DataFrame, col: str, K: int) -> np.ndarray:
    """Return 2 x K array; row 0 = Y=0, row 1 = Y=1; columns are sigma-bin 0..K-1."""
    grid = (
        df.groupby(["label", "sigma_bin"])[col]
        .mean()
        .unstack(fill_value=np.nan)
        .reindex(index=[0, 1], columns=range(K))
    )
    return grid.to_numpy()


def load_data(K: int = K):
    df = pd.read_parquet(PARQUET).copy()
    # Equal-frequency K bins on sigma (TraitGym Mondrian convention).
    df["sigma_bin"] = pd.qcut(df["sigma"].rank(method="first"), q=K, labels=False).astype(int)
    cov_split = per_cell_coverage(df, "homosc_covered", K)
    cov_class = per_cell_coverage(df, "hetero_covered", K)
    cov_hccp  = per_cell_coverage(df, "mondrian_covered", K)
    return cov_split, cov_class, cov_hccp


# ============================================================================
# Panel (a) — three side-by-side y x sigma-bin grids.
# ============================================================================
def draw_grid(ax, cov: np.ndarray, title: str, partition_label: str, K: int):
    """cov: shape (2, K). Row 0 = Y=0, row 1 = Y=1."""
    vmin, vmax = 0.40, 1.00
    # Flip rows so Y=1 is on top (visual priority on the minority class).
    cov_disp = cov[::-1]  # row 0 -> Y=1, row 1 -> Y=0

    im = ax.imshow(cov_disp, cmap=_cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    for i in range(2):
        for j in range(K):
            v = cov_disp[i, j]
            if np.isnan(v):
                continue
            tcol = "white" if (v < 0.65 or v > 0.97) else "#222222"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.0, color=tcol, weight="bold")

    ax.set_xticks(range(K))
    ax.set_xticklabels([f"$b_{{{j+1}}}$" for j in range(K)], fontsize=7.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"$Y{=}1$", r"$Y{=}0$"], fontsize=8.0)
    ax.set_xlabel(r"$\hat\sigma$-bin (low $\to$ high)", fontsize=8.0, labelpad=3)

    # Title block: stacked title + partition spec on two lines.
    ax.set_title(f"{title}\n" + r"$\mathrm{partition:}$ " + partition_label,
                 fontsize=9.0, pad=4)

    valid = cov[~np.isnan(cov)]
    worst = np.max(np.abs(valid - TARGET))
    ax.text(0.5, -0.42, f"worst cell gap = {worst:.2f}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8.0,
            color=("#a83232" if worst > 0.10 else "#3d6a45"),
            weight="bold")

    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)

    return im


def panel_a(fig, gs_left):
    """Panel (a): three grids in a row, sharing a colorbar at the bottom."""
    cov_split, cov_class, cov_hccp = load_data(K=K)

    inner = gs_left.subgridspec(
        2, 3, width_ratios=[1, 1, 1],
        height_ratios=[1.0, 0.06], hspace=1.40, wspace=0.40,
    )
    ax1 = fig.add_subplot(inner[0, 0])
    ax2 = fig.add_subplot(inner[0, 1])
    ax3 = fig.add_subplot(inner[0, 2])
    cax = fig.add_subplot(inner[1, :])

    draw_grid(ax1, cov_split,
              "(i) Split CP",
              r"none ($\emptyset$)", K)
    draw_grid(ax2, cov_class,
              "(ii) class-Mondrian",
              r"$Y$ only", K)
    im = draw_grid(ax3, cov_hccp,
              "(iii) HCCP (ours)",
              r"$Y \times \hat\sigma$-bin", K)

    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7.0, length=2)
    cb.set_label(r"empirical coverage  (target $1-\alpha = 0.90$)",
                 fontsize=7.6, labelpad=2)
    cb.outline.set_linewidth(0.4)
    cb.ax.axvline(TARGET, color="#222222", linestyle="--", linewidth=0.8)

    # Panel-level header (placed via fig.text after layout settles).
    fig.text(
        0.015, 0.965,
        "(a) Per-cell empirical coverage  "
        r"(TraitGym Complex, CADD+GPN-MSA+Borzoi, $\pi_{+}{=}0.10$, $K{=}5$)",
        fontsize=9.5, weight="bold",
    )


# ============================================================================
# Panel (b) — Pareto frontier (unchanged from original concept figure).
# ============================================================================
def panel_b(ax):
    mend = {
        "B1 split CP":              (0.982, 0.265),
        "B2 $\\hat\\sigma$-Mond.":   (0.821, 0.401),
        "B3 class-Mond.":           (0.895, 0.410),
        "HCCP (ours)":              (0.875, 0.173),
    }
    comp = {
        "B1 split CP":              (0.614, 0.822),
        "B2 $\\hat\\sigma$-Mond.":   (0.624, 0.827),
        "B3 class-Mond.":           (0.900, 0.805),
        "HCCP (ours)":              (0.905, 0.060),
    }

    ax.fill_between([0.85, 1.0], 0, 0.20, color=C_IDEAL, alpha=0.30,
                    linewidth=0, zorder=0)
    ax.text(0.992, 0.013, "ideal corner",
            ha="right", va="bottom", fontsize=7.0, color="#3d6a45",
            style="italic", zorder=1)

    markers = {
        "B1 split CP":              ("o", C_BASE, 56),
        "B2 $\\hat\\sigma$-Mond.":   ("s", C_BASE, 56),
        "B3 class-Mond.":           ("^", C_BASE, 60),
        "HCCP (ours)":              ("*", C_HCCP, 240),
    }

    plotted = set()
    for is_complex, ds_data in [(False, mend), (True, comp)]:
        for label, (cov, gap) in ds_data.items():
            mk, color, sz = markers[label]
            face = color if is_complex else "white"
            edge = color
            lw = 1.6 if label == "HCCP (ours)" else 1.0
            ax.scatter(cov, gap, marker=mk, s=sz, facecolor=face,
                       edgecolor=edge, linewidth=lw, zorder=3,
                       label=label if label not in plotted else None)
            plotted.add(label)

    for label in mend:
        if label == "HCCP (ours)":
            continue
        m = mend[label]; c = comp[label]
        ax.plot([m[0], c[0]], [m[1], c[1]],
                color=C_BASE, alpha=0.18, linewidth=0.8, zorder=2)
    ax.plot([mend["HCCP (ours)"][0], comp["HCCP (ours)"][0]],
            [mend["HCCP (ours)"][1], comp["HCCP (ours)"][1]],
            color=C_HCCP, alpha=0.40, linewidth=1.0, zorder=2)

    ax.annotate("Mendelian", xy=mend["HCCP (ours)"],
                xytext=(0.74, 0.30), fontsize=8.0, color=C_HCCP,
                weight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_HCCP, alpha=0.7))
    ax.annotate("Complex", xy=comp["HCCP (ours)"],
                xytext=(0.74, 0.13), fontsize=8.0, color=C_HCCP,
                weight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_HCCP, alpha=0.7))

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    leg = ax.legend([h for h, _ in uniq], [l for _, l in uniq],
                    loc="upper right", frameon=True, framealpha=0.94,
                    fontsize=7.4, handletextpad=0.4, borderpad=0.4)
    leg.get_frame().set_linewidth(0.4)
    leg.set_title("open=Mend., filled=Comp.", prop={"size": 6.6})

    ax.set_xlim(0.50, 1.00)
    ax.set_ylim(-0.02, 0.92)
    ax.set_xlabel(r"Cov$|Y{=}1$  ($\rightarrow$ better)")
    ax.set_ylabel(r"$\hat{\sigma}$-bin gap  ($\downarrow$ better)")
    ax.set_title("(b) Pareto frontier (Tab.~2)", loc="left", pad=4)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axhline(0.20, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0.85, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def main():
    # Match the NeurIPS \linewidth (~6.5 in) so text point sizes survive scaling.
    fig = plt.figure(figsize=(11.0, 3.5))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.85, 1.00], wspace=0.16,
        top=0.84, bottom=0.13, left=0.045, right=0.985,
    )
    panel_a(fig, gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel_b(ax_b)

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
