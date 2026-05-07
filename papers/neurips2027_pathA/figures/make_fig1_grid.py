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
C_BASE = "#4477aa"  # aligned with figD / fig_asl_audit baseline blue
C_IDEAL = "#a4c293"

# Diverging colormap for coverage deviation centered at 0 (cov - 0.90).
# Negative (under-coverage) -> red; 0 (target) -> pale; positive (over) -> blue.
_cmap = LinearSegmentedColormap.from_list(
    "miscov_dev",
    [(0.0, "#a83232"),   # -0.50 (or below) -> deep red
     (0.4, "#e89090"),
     (0.5, "#f5f5f5"),   # 0 deviation -> off-white
     (0.6, "#9fbfdf"),
     (1.0, "#1f4e79")],  # +0.10 (or above) -> deep blue
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
    """cov: shape (2, K). Row 0 = Y=0, row 1 = Y=1.

    Cell values display **deviation from target** (cov - 0.90), centered
    diverging colormap, so reader gets direct miscoverage reading without
    mentally subtracting 0.90.
    """
    # Asymmetric range: most deviations are negative (under-coverage); a small
    # positive overshoot (~+0.10 = perfect coverage) is OK.
    vmin, vmax = -0.50, 0.10
    cov_disp = cov[::-1]  # row 0 -> Y=1 (visual priority on minority class)
    dev = cov_disp - TARGET

    im = ax.imshow(dev, cmap=_cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")

    for i in range(2):
        for j in range(K):
            v = dev[i, j]
            if np.isnan(v):
                continue
            sign = "+" if v > 0.005 else ("" if v < -0.005 else "±")
            tcol = "white" if abs(v) > 0.30 else "#222222"
            ax.text(j, i, f"{sign}{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color=tcol)

    ax.set_xticks(range(K))
    ax.set_xticklabels([f"$b_{{{j+1}}}$" for j in range(K)], fontsize=7.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"$Y{=}1$", r"$Y{=}0$"], fontsize=8.0)
    ax.set_xlabel(r"$\hat\sigma$-bin (low $\to$ high)", fontsize=8.0, labelpad=3)

    # Compact 2-line title: index + method on line 1, partition + worst-gap
    # on line 2. partition_label is a raw string that may include both math
    # ($..$) and text segments (e.g., "$Y \times \hat\sigma$-bin").
    valid = cov[~np.isnan(cov)]
    worst = np.max(np.abs(valid - TARGET))
    worst_color = "#a83232" if worst > 0.10 else "#3d6a45"
    title_line = f"{title}"
    sub_line = (
        fr"part. {partition_label}"
        fr"  ·  worst $|\Delta|={worst:.2f}$"
    )
    ax.set_title(f"{title_line}\n" + sub_line, fontsize=8.5, pad=4,
                 loc="center")
    # The bottom-right inline `worst gap` annotation is now redundant.

    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    # Cleaner cells: thinner white grid lines, lighter colour to reduce
    # "boxy/ugly" feel. Cells now read as continuous heatmap with subtle
    # separators rather than chunky tile blocks.
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
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
              r"$\emptyset$", K)
    draw_grid(ax2, cov_class,
              "(ii) class-Mondrian",
              r"$Y$", K)
    im = draw_grid(ax3, cov_hccp,
              "(iii) HCCP",
              r"$Y \times \hat\sigma$-bin", K)

    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7.0, length=2)
    cb.set_label(r"coverage deviation $\mathrm{cov} - 0.90$",
                 fontsize=7.6, labelpad=2)
    cb.outline.set_linewidth(0.4)
    cb.ax.axvline(0.0, color="#222222", linestyle="--", linewidth=0.8)
    # Endpoint annotations: explicit "under-coverage" / "over-coverage" text
    # at the two colorbar extremes, so direction is parseable without reading
    # the caption (Whitesides "self-contained figure" rule).
    cax.annotate("under-coverage", xy=(0.0, 1.05), xycoords="axes fraction",
                 ha="left", va="bottom", fontsize=7.0, color="#a83232",
                 weight="bold")
    cax.annotate("over-coverage", xy=(1.0, 1.05), xycoords="axes fraction",
                 ha="right", va="bottom", fontsize=7.0, color="#1f4e79",
                 weight="bold")


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
        "HCCP (ours)":              ("*", C_HCCP, 80),  # 3x smaller (was 240)
    }

    plotted = set()
    for is_complex, ds_data in [(False, mend), (True, comp)]:
        for label, (cov, gap) in ds_data.items():
            mk, color, sz = markers[label]
            face = color if is_complex else "white"
            edge = color
            lw = 1.6 if label == "HCCP (ours)" else 1.0
            # White halo behind HCCP star so it stays visible (matches
            # fig_bootstrap_density's all-chrom-point treatment).
            if label == "HCCP (ours)":
                ax.scatter(cov, gap, marker="o", s=sz * 1.3,
                           facecolor="white", edgecolor="white",
                           linewidth=0, zorder=2)
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
        top=0.86, bottom=0.13, left=0.045, right=0.985,
    )
    panel_a(fig, gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel_b(ax_b)

    # Panel (a) header via fig.suptitle (replaces the previous fig.text
    # absolute-positioning hack which was fragile under tight_layout).
    fig.suptitle(
        "(a) Per-cell empirical coverage on TraitGym Complex "
        r"(CADD+GPN-MSA+Borzoi, $\pi_{+}{=}0.10$, $K{=}5$). "
        r"Cells: deviation from target.",
        x=0.045, y=0.985, ha="left", fontsize=9.0, weight="bold",
    )

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
