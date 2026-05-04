"""Figure 1 — HCCP at a glance: pipeline schematic + Pareto frontier.

Two panels (no overlap, single-pass readable):
  (a) Pipeline schematic: 4 stages (Input -> Aggregator -> Mondrian -> Per-cell)
      with the score formula and two coverage targets stacked beneath.
  (b) Pareto plot on (Cov|Y=1, sigma-bin gap): Tab.~2 head-to-head numbers.

Layout invariants we enforce:
  - boxes fit on a single horizontal row with arrows that don't touch labels;
  - the score callout sits in its own band below the boxes;
  - the targets band sits in its own band below the callout (no overlap);
  - in panel (b), HCCP Mendelian / Complex labels are placed away from any
    other marker and outside the ideal-corner shading.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent
OUT_PDF = FIG_DIR / "fig1_concept.pdf"
OUT_PNG = FIG_DIR / "fig1_concept.png"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_BOX = "#e8eef5"
C_BOX_EDGE = "#2c3e50"
C_HCCP = "#d1495b"
C_BASE = "#4a6fa5"
C_ARROW = "#2c3e50"
C_IDEAL = "#a4c293"


# ============================================================================
# Panel (a) — Pipeline schematic
#
# Vertical bands (top to bottom):
#   y = 4.50 : panel title (placed via ax.set_title outside the data area)
#   y = 3.30 - 4.20 : 4 stage boxes (h = 0.90)
#   y = 2.10 - 2.80 : score callout (h = 0.70)
#   y = 1.20 - 1.80 : targets (T2 + T3) on two lines, well separated
# ============================================================================
def panel_a(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0.6, 5.0)
    ax.axis("off")
    ax.set_title("(a) HCCP pipeline", loc="left", pad=4)

    box_y = 3.75
    box_h = 0.90
    boxes = [
        (1.10, box_y, 1.85, box_h, "Input\n$(X_i,\\, Y_i,\\, C_i)$",
         "chrom $C_i$ for LOO split"),
        (3.65, box_y, 2.10, box_h, "Aggregator\n$\\hat{p}(x),\\ \\hat{\\sigma}(x)$",
         "GBM + Gaussian-NLL\non chrom-LOO OOF"),
        (6.30, box_y, 2.10, box_h, "Mondrian\n$(y \\times \\hat{\\sigma}\\text{-bin})$",
         "$2K$ cells; $K = \\hat K(c_{\\mathrm{outer}})$"),
        (8.85, box_y, 1.65, box_h, "Per-cell\n$\\hat{q}_{k,b}$",
         "$\\mathcal{C}_\\alpha(x)$"),
    ]

    for cx, cy, w, h, title, sub in boxes:
        box = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.10",
            facecolor=C_BOX, edgecolor=C_BOX_EDGE, linewidth=1.0,
        )
        ax.add_patch(box)
        # Title sits in upper portion of the box (avoids the subtitle below).
        ax.text(cx, cy + 0.14, title, ha="center", va="center", fontsize=8.6)
        ax.text(cx, cy - 0.27, sub, ha="center", va="center",
                fontsize=6.8, color="#444444", style="italic")

    # Arrows between boxes — sit at the box vertical centre.
    for i in range(3):
        sx = boxes[i][0] + boxes[i][2] / 2
        ex = boxes[i + 1][0] - boxes[i + 1][2] / 2
        arrow = FancyArrowPatch(
            (sx + 0.04, box_y), (ex - 0.04, box_y),
            arrowstyle="-|>", mutation_scale=10,
            linewidth=1.0, color=C_ARROW,
        )
        ax.add_patch(arrow)

    # Score callout band — yellow box centred on the panel.
    callout = FancyBboxPatch(
        (1.40, 2.20), 7.20, 0.60,
        boxstyle="round,pad=0.05,rounding_size=0.06",
        facecolor="#fff8e7", edgecolor="#caa455", linewidth=0.6,
    )
    ax.add_patch(callout)
    ax.text(5.0, 2.50,
            r"score:  $s(x, y) = |y - \hat{p}(x)| / (\hat{\sigma}(x) + \epsilon)$"
            r"  $\Longrightarrow$  per-cell quantile  "
            r"$\hat{q}_{k,b} = \mathrm{Quantile}(s : Y{=}k,\, \mathrm{bin}{=}b)$",
            ha="center", va="center", fontsize=8.0)

    # Targets band — two lines with shared "Targets:" gutter, no overlap.
    ax.text(0.30, 1.55, "Targets:", ha="left", va="center",
            fontsize=8.0, color="#222222", weight="bold")
    ax.text(1.55, 1.55,
            r"(i) $P(Y \in \mathcal{C}_\alpha \mid Y{=}k) \geq 1 - \alpha$"
            r"     [T2 class-cond., Barber 2020-compatible]",
            ha="left", va="center", fontsize=7.5, color="#222222")
    ax.text(1.55, 1.05,
            r"(ii) $P(Y \in \mathcal{C}_\alpha \mid Y{=}k,\, b(X){=}b) \geq 1 - \alpha - 1/(n_{k,b}{+}1)$"
            r"     [T3 bin-cond.]",
            ha="left", va="center", fontsize=7.5, color="#222222")


# ============================================================================
# Panel (b) — Pareto frontier on (Cov|Y=1, sigma-bin gap).
# Numbers from sections/C_tables.tex tab:h2h, B = 200 chrom-bootstrap means.
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

    # Ideal zone shading (cov >= 0.85, gap <= 0.20).
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

    # Faint Mendelian-Complex connectors per method (excluding HCCP, which
    # gets its own labelled red connector).
    for label in mend:
        if label == "HCCP (ours)":
            continue
        m = mend[label]; c = comp[label]
        ax.plot([m[0], c[0]], [m[1], c[1]],
                color=C_BASE, alpha=0.18, linewidth=0.8, zorder=2)
    ax.plot([mend["HCCP (ours)"][0], comp["HCCP (ours)"][0]],
            [mend["HCCP (ours)"][1], comp["HCCP (ours)"][1]],
            color=C_HCCP, alpha=0.40, linewidth=1.0, zorder=2)

    # HCCP labels — placed OFF the ideal-corner shading and away from other
    # markers. Mendelian (cov 0.875, gap 0.173) sits inside the corner; we put
    # its label up-and-left so the arrow exits the shading. Complex
    # (cov 0.905, gap 0.060) sits at the bottom; its label goes below-left.
    ax.annotate("Mendelian", xy=mend["HCCP (ours)"],
                xytext=(0.74, 0.30), fontsize=8.0, color=C_HCCP,
                weight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_HCCP, alpha=0.7))
    ax.annotate("Complex", xy=comp["HCCP (ours)"],
                xytext=(0.74, 0.13), fontsize=8.0, color=C_HCCP,
                weight="bold",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=C_HCCP, alpha=0.7))

    # Legend: dedup. Marker key only; open/filled is explained in caption.
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


def main():
    fig = plt.figure(figsize=(11.6, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.00], wspace=0.20)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel_a(ax_a)
    panel_b(ax_b)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
