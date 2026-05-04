"""Figure 1 — BLANK template (no text labels) for user to add text manually.

Two panels:
  (a) Pipeline schematic: 4 empty boxes + 3 arrows
  (b) Pareto plot frame with axes, ideal-zone shading, and 8 markers
      (positions match Tab 8 real numbers but no text labels)

Annotations are printed to stdout describing where each text element should go.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent
OUT_PDF = FIG_DIR / "fig1_blank.pdf"
OUT_PNG = FIG_DIR / "fig1_blank.png"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_BOX = "#e8eef5"
C_BOX_EDGE = "#2c3e50"
C_HCCP = "#d1495b"
C_BASE = "#4a6fa5"
C_ARROW = "#2c3e50"
C_IDEAL = "#a4c293"


def panel_a_blank(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (1.10, 3.4, 1.85, 1.30),
        (3.65, 3.4, 2.10, 1.30),
        (6.30, 3.4, 2.10, 1.30),
        (8.85, 3.4, 1.65, 1.30),
    ]
    for cx, cy, w, h in boxes:
        box = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.10",
            facecolor=C_BOX, edgecolor=C_BOX_EDGE, linewidth=1.0,
        )
        ax.add_patch(box)

    for i in range(3):
        sx = boxes[i][0] + boxes[i][2] / 2
        ex = boxes[i + 1][0] - boxes[i + 1][2] / 2
        arrow = FancyArrowPatch(
            (sx + 0.04, boxes[i][1]), (ex - 0.04, boxes[i + 1][1]),
            arrowstyle="-|>", mutation_scale=10,
            linewidth=1.0, color=C_ARROW,
        )
        ax.add_patch(arrow)

    # Score callout box (empty, light yellow)
    score_box = FancyBboxPatch(
        (1.40, 1.30), 7.20, 0.70,
        boxstyle="round,pad=0.10,rounding_size=0.06",
        facecolor="#fff8e7", edgecolor="#caa455", linewidth=0.6,
    )
    ax.add_patch(score_box)


def panel_b_blank(ax):
    # Real Tab-8 numbers (Mendelian + Complex; cov|Y=1, sigma-bin gap)
    points = [
        # (cov, gap, dataset_is_complex, is_hccp)
        (0.982, 0.265, False, False),  # M-B1
        (0.821, 0.401, False, False),  # M-B2
        (0.895, 0.410, False, False),  # M-B3
        (0.875, 0.173, False, True),   # M-HCCP
        (0.614, 0.822, True,  False),  # C-B1
        (0.624, 0.827, True,  False),  # C-B2
        (0.900, 0.805, True,  False),  # C-B3
        (0.905, 0.060, True,  True),   # C-HCCP
    ]

    # ideal zone shading
    ax.fill_between([0.85, 1.0], 0, 0.20, color=C_IDEAL, alpha=0.30,
                    linewidth=0, zorder=0)
    ax.axhline(0.20, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0.85, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    marker_map = {0: "o", 1: "s", 2: "^", 3: "*"}
    for cov, gap, is_complex, is_hccp in points:
        # Determine marker by row index (B1=o, B2=s, B3=^, HCCP=*)
        if is_hccp:
            mk, color, sz = "*", C_HCCP, 230
        else:
            # Find which baseline (sniff by cov+gap signature)
            if (cov, gap) in [(0.982, 0.265), (0.614, 0.822)]:
                mk, color, sz = "o", C_BASE, 56
            elif (cov, gap) in [(0.821, 0.401), (0.624, 0.827)]:
                mk, color, sz = "s", C_BASE, 56
            else:
                mk, color, sz = "^", C_BASE, 60
        face = color if is_complex else "white"
        lw = 1.6 if is_hccp else 1.0
        ax.scatter(cov, gap, marker=mk, s=sz, facecolor=face,
                   edgecolor=color, linewidth=lw, zorder=3)

    ax.set_xlim(0.50, 1.00)
    ax.set_ylim(-0.02, 0.90)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])


def main():
    fig = plt.figure(figsize=(11.6, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.00], wspace=0.18)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel_a_blank(ax_a)
    panel_b_blank(ax_b)
    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
