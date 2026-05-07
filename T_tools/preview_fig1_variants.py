"""Generate X/Y/Z polish variants of fig1_concept for user selection.

Outputs to R_raw/figpolish_previews/:
  fig1_X.png  --- A only: cosmetic polish (1-line titles, color align, ★ halo)
  fig1_Y.png  --- A + B: deviation-from-target display (cells show cov-0.90)
  fig1_Z.png  --- A + B + C: A+B + width rebalance (a:b 1.85→1.50)
"""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
PARQUET = REPO / "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/conformal_hetero_scores.parquet"
PREVIEW = REPO / "R_raw" / "figpolish_previews"
PREVIEW.mkdir(parents=True, exist_ok=True)

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
C_BASE = "#4477aa"  # X polish: align with figD / asl_audit
C_IDEAL = "#a4c293"

# Diverging cmap centered at TARGET = 0.90 (variant X)
_cmap_raw = LinearSegmentedColormap.from_list(
    "miscov_raw",
    [(0.0, "#a83232"), (0.4, "#e89090"),
     (0.5, "#f5f5f5"),
     (0.6, "#9fbfdf"), (1.0, "#1f4e79")],
)

# Diverging cmap centered at 0 (variant Y/Z, for deviation cov - 0.90)
_cmap_dev = LinearSegmentedColormap.from_list(
    "miscov_dev",
    [(0.0, "#a83232"), (0.4, "#e89090"),
     (0.5, "#f5f5f5"),
     (0.6, "#9fbfdf"), (1.0, "#1f4e79")],
)


def per_cell_coverage(df, col, K):
    grid = (df.groupby(["label", "sigma_bin"])[col].mean()
            .unstack(fill_value=np.nan)
            .reindex(index=[0, 1], columns=range(K)))
    return grid.to_numpy()


def load_data():
    df = pd.read_parquet(PARQUET).copy()
    df["sigma_bin"] = pd.qcut(df["sigma"].rank(method="first"),
                              q=K, labels=False).astype(int)
    return (per_cell_coverage(df, "homosc_covered", K),
            per_cell_coverage(df, "hetero_covered", K),
            per_cell_coverage(df, "mondrian_covered", K))


def draw_grid_X(ax, cov, title_text, K):
    """Variant X: cosmetic polish — 1-line title, raw coverage values."""
    vmin, vmax = 0.40, 1.00
    cov_disp = cov[::-1]
    im = ax.imshow(cov_disp, cmap=_cmap_raw, vmin=vmin, vmax=vmax,
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

    # Single-line title with worst-cell gap inline
    valid = cov[~np.isnan(cov)]
    worst = np.max(np.abs(valid - TARGET))
    worst_color = "#a83232" if worst > 0.10 else "#3d6a45"
    ax.set_title(f"{title_text}    "
                 r"$\Delta_{\max}{=}$" + f"{worst:.2f}",
                 fontsize=9.0, pad=4, loc="left",
                 color="#222222")
    # Add worst-cell color emphasis via annotation in lower-right
    ax.text(0.99, -0.30, f"worst gap = {worst:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=worst_color, weight="bold")

    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)
    return im


def draw_grid_dev(ax, cov, title_text, K):
    """Variant Y/Z: deviation display — cells show cov - 0.90."""
    cov_disp = cov[::-1]
    dev = cov_disp - TARGET
    vmin, vmax = -0.50, 0.10  # asymmetric: most deviations are negative
    im = ax.imshow(dev, cmap=_cmap_dev, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    for i in range(2):
        for j in range(K):
            v = dev[i, j]
            if np.isnan(v):
                continue
            sign = "+" if v > 0 else ("" if v < 0 else "±")
            tcol = "white" if abs(v) > 0.25 else "#222222"
            ax.text(j, i, f"{sign}{v:.2f}", ha="center", va="center",
                    fontsize=8.0, color=tcol, weight="bold")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"$b_{{{j+1}}}$" for j in range(K)], fontsize=7.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"$Y{=}1$", r"$Y{=}0$"], fontsize=8.0)
    ax.set_xlabel(r"$\hat\sigma$-bin (low $\to$ high)", fontsize=8.0, labelpad=3)

    valid = cov[~np.isnan(cov)]
    worst = np.max(np.abs(valid - TARGET))
    worst_color = "#a83232" if worst > 0.10 else "#3d6a45"
    ax.set_title(f"{title_text}    "
                 r"$\Delta_{\max}{=}$" + f"{worst:.2f}",
                 fontsize=9.0, pad=4, loc="left",
                 color="#222222")
    ax.text(0.99, -0.30, f"worst |Δ| = {worst:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=worst_color, weight="bold")

    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="both", length=0)
    return im


def panel_a(fig, gs_left, variant: str):
    cov_split, cov_class, cov_hccp = load_data()
    inner = gs_left.subgridspec(
        2, 3, width_ratios=[1, 1, 1],
        height_ratios=[1.0, 0.06], hspace=1.05, wspace=0.40,
    )
    ax1 = fig.add_subplot(inner[0, 0])
    ax2 = fig.add_subplot(inner[0, 1])
    ax3 = fig.add_subplot(inner[0, 2])
    cax = fig.add_subplot(inner[1, :])

    if variant == "X":
        draw = draw_grid_X
        cb_label = r"empirical coverage  (target $1-\alpha = 0.90$)"
        cb_axline = TARGET
    else:
        draw = draw_grid_dev
        cb_label = r"coverage deviation  $\mathrm{cov} - 0.90$  (red: under, blue: over)"
        cb_axline = 0.0

    titles = [
        r"(i) Split CP, partition: $\emptyset$",
        r"(ii) class-Mondrian, partition: $Y$",
        r"(iii) HCCP (ours), partition: $Y \times \hat\sigma$-bin",
    ]
    draw(ax1, cov_split, titles[0], K)
    draw(ax2, cov_class, titles[1], K)
    im = draw(ax3, cov_hccp, titles[2], K)

    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7.0, length=2)
    cb.set_label(cb_label, fontsize=7.6, labelpad=2)
    cb.outline.set_linewidth(0.4)
    cb.ax.axvline(cb_axline, color="#222", linestyle="--", linewidth=0.8)


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
    ax.text(0.992, 0.013, "ideal corner", ha="right", va="bottom",
            fontsize=7.0, color="#3d6a45", style="italic", zorder=1)

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
            # X polish: HCCP star with white halo
            if label == "HCCP (ours)":
                ax.scatter(cov, gap, marker="o", s=sz * 1.2,
                           facecolor="white", edgecolor="white",
                           linewidth=0, zorder=2)
            ax.scatter(cov, gap, marker=mk, s=sz, facecolor=face,
                       edgecolor=edge, linewidth=lw, zorder=3,
                       label=label if label not in plotted else None)
            plotted.add(label)

    for label in mend:
        if label == "HCCP (ours)":
            continue
        m, c = mend[label], comp[label]
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


def make_fig(variant: str):
    if variant == "Z":
        ratios = [1.50, 1.10]  # rebalanced
        figsize = (10.5, 3.6)
    else:
        ratios = [1.85, 1.00]
        figsize = (11.0, 3.5)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1, 2, width_ratios=ratios, wspace=0.16,
        top=0.86, bottom=0.13, left=0.045, right=0.985,
    )
    panel_a(fig, gs[0, 0], variant=variant)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_b(ax_b)

    # Panel (a) header — moved to GridSpec axes title via dummy axis instead
    # of fig.text absolute positioning (X polish: more layout-robust).
    sub_label = "raw coverage" if variant == "X" else "deviation cov $-$ 0.90"
    fig.suptitle(
        f"(a) Per-cell empirical coverage on TraitGym Complex (CADD+GPN-MSA+Borzoi, "
        fr"$\pi_{{+}}{{=}}0.10$, $K{{=}}5$).  Cell values: {sub_label}.",
        x=0.045, y=0.985, ha="left", fontsize=9.0, weight="bold",
    )

    out = PREVIEW / f"fig1_{variant}.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(out)


def main():
    make_fig("X")
    make_fig("Y")
    make_fig("Z")


if __name__ == "__main__":
    main()
