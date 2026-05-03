"""Figure 1 — HCCP at a glance: pipeline schematic + Pareto frontier.

Two panels:
  (a) Pipeline schematic: data -> (p_hat, sigma_hat) aggregator -> Mondrian
      (y x sigma-bin) partition -> per-cell threshold q_{k,b} -> prediction
      set C_alpha(x). Score s(x,y) = |y - p_hat(x)| / (sigma_hat(x) + eps)
      shown beneath the pipeline.
  (b) Pareto plot on (Cov|Y=1, sigma-bin gap): real numbers from Tab. 8
      head-to-head (4 methods x 2 datasets, B = 200 chrom-bootstrap means).
      HCCP sits alone in the bottom-right (high coverage + low gap) corner.

Numbers in (b) come from sections/C_tables.tex tab:h2h. The schematic in (a)
is a structural diagram with no synthetic data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent
OUT_PDF = FIG_DIR / "fig1_concept.pdf"
OUT_PNG = FIG_DIR / "fig1_concept.png"

# ----- Style -----------------------------------------------------------------
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
# ============================================================================
def panel_a(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title("(a) HCCP pipeline", loc="left", pad=4)

    # Box positions: x_center, y_center, width, height
    boxes = [
        (1.10, 3.4, 1.85, 1.30, "Input\n$(X_i, Y_i, C_i)$",
         "chrom $C_i$\nfor LOO split"),
        (3.65, 3.4, 2.10, 1.30, "Aggregator\n$\\hat{p}(x),\\ \\hat{\\sigma}(x)$",
         "GBM + Gaussian-NLL\non chrom-LOO OOF"),
        (6.30, 3.4, 2.10, 1.30, "Mondrian\n$(y \\times \\hat{\\sigma}\\text{-bin})$",
         "$2K$ cells; $K = \\hat K(c_{\\mathrm{outer}})$\nvia nested chrom-LOO"),
        (8.85, 3.4, 1.65, 1.30,
         "Per-cell\n$\\hat{q}_{k,b}$",
         "$\\mathcal{C}_\\alpha(x)$\n(T3 / T3$'$)"),
    ]

    for cx, cy, w, h, title, sub in boxes:
        box = FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.10",
            facecolor=C_BOX, edgecolor=C_BOX_EDGE, linewidth=1.0,
        )
        ax.add_patch(box)
        ax.text(cx, cy + 0.20, title, ha="center", va="center",
                fontsize=9.0)
        ax.text(cx, cy - 0.36, sub, ha="center", va="center",
                fontsize=7.2, color="#444444", style="italic")

    # Arrows between boxes.
    arrow_pairs = [
        (boxes[0], boxes[1]),
        (boxes[1], boxes[2]),
        (boxes[2], boxes[3]),
    ]
    for src, dst in arrow_pairs:
        x_src = src[0] + src[2] / 2
        x_dst = dst[0] - dst[2] / 2
        arrow = FancyArrowPatch(
            (x_src + 0.04, src[1]), (x_dst - 0.04, dst[1]),
            arrowstyle="-|>", mutation_scale=10,
            linewidth=1.0, color=C_ARROW,
        )
        ax.add_patch(arrow)

    # Score bar beneath, anchored under the aggregator + Mondrian.
    ax.annotate(
        r"nonconformity score:  $s(x, y) \;=\; |y - \hat{p}(x)| \,/\, "
        r"(\hat{\sigma}(x) + \epsilon)$  "
        r"$\Longrightarrow$  per-cell quantile  "
        r"$\hat{q}_{k,b} = \mathrm{Quantile}(s : Y{=}k, \mathrm{bin}{=}b)$",
        xy=(5.0, 1.65), xycoords="data",
        ha="center", va="center", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff8e7",
                  edgecolor="#caa455", linewidth=0.6),
    )

    # Two coverage targets summarized below.
    ax.text(2.5, 0.55,
            r"Targets:  (i) $P(Y \in \mathcal{C}_\alpha \mid Y{=}k) \geq 1 - \alpha$  "
            r"[T2 class-cond.]",
            ha="left", va="center", fontsize=7.6, color="#333333")
    ax.text(3.18, 0.10,
            r"(ii) $P(Y \in \mathcal{C}_\alpha \mid Y{=}k,\, b(X){=}b) \geq 1 - \alpha - 1/(n_{k,b}+1)$  "
            r"[T3 bin-cond.]",
            ha="left", va="center", fontsize=7.6, color="#333333")


# ============================================================================
# Panel (b) — Pareto frontier on (Cov|Y=1, sigma-bin gap)
# Numbers from sections/C_tables.tex tab:h2h, B = 200 chrom-bootstrap means.
# ============================================================================
def panel_b(ax):
    # Mendelian, pi+ = 0.10
    mend = {
        "B1 split CP":      (0.982, 0.265),
        "B2 $\\hat\\sigma$-Mond.":  (0.821, 0.401),
        "B3 class-Mond.":   (0.895, 0.410),
        "HCCP (ours)":      (0.875, 0.173),
    }
    # Complex, pi+ = 0.10
    comp = {
        "B1 split CP":      (0.614, 0.822),
        "B2 $\\hat\\sigma$-Mond.":  (0.624, 0.827),
        "B3 class-Mond.":   (0.900, 0.805),
        "HCCP (ours)":      (0.905, 0.060),
    }

    # Ideal zone: Cov|Y=1 >= 0.85 and sigma-bin gap <= 0.20 (recommended corner).
    ax.fill_between([0.85, 1.0], 0, 0.20, color=C_IDEAL, alpha=0.30,
                    linewidth=0, zorder=0)
    ax.text(0.992, 0.011, "ideal corner\n(cov $\\geq$ 0.85, gap $\\leq$ 0.20)",
            ha="right", va="bottom", fontsize=7.2, color="#3d6a45",
            style="italic", zorder=1)

    markers = {
        "B1 split CP":     ("o", C_BASE, 56),
        "B2 $\\hat\\sigma$-Mond.": ("s", C_BASE, 56),
        "B3 class-Mond.":  ("^", C_BASE, 60),
        "HCCP (ours)":     ("*", C_HCCP, 230),
    }

    plotted = set()
    for ds_name, ds_data, is_filled in [("Mendelian", mend, False),
                                         ("Complex", comp, True)]:
        for label, (cov, gap) in ds_data.items():
            mk, color, sz = markers[label]
            face = color if is_filled else "white"
            edge = color
            lw = 1.6 if label == "HCCP (ours)" else 1.0
            ax.scatter(cov, gap, marker=mk, s=sz, facecolor=face,
                       edgecolor=edge, linewidth=lw, zorder=3,
                       label=label if label not in plotted else None)
            plotted.add(label)

    # Connect each method's Mendelian-Complex pair with a faint line to show
    # cross-dataset consistency.
    for label in mend:
        if label == "HCCP (ours)":
            continue  # don't clutter HCCP's star markers
        m_cov, m_gap = mend[label]
        c_cov, c_gap = comp[label]
        ax.plot([m_cov, c_cov], [m_gap, c_gap],
                color=C_BASE, alpha=0.18, linewidth=0.8, zorder=2)
    # HCCP connector in red.
    ax.plot([mend["HCCP (ours)"][0], comp["HCCP (ours)"][0]],
            [mend["HCCP (ours)"][1], comp["HCCP (ours)"][1]],
            color=C_HCCP, alpha=0.35, linewidth=1.0, zorder=2)

    # Annotate each HCCP point with dataset name.
    ax.annotate("Mendelian", xy=mend["HCCP (ours)"],
                xytext=(0.835, 0.245), fontsize=7.5, color=C_HCCP,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C_HCCP, alpha=0.6))
    ax.annotate("Complex", xy=comp["HCCP (ours)"],
                xytext=(0.815, 0.105), fontsize=7.5, color=C_HCCP,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C_HCCP, alpha=0.6))

    # Legend: deduplicated, with marker key plus open/filled note.
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    leg = ax.legend([h for h, _ in uniq], [l for _, l in uniq],
                    loc="upper left", frameon=True, framealpha=0.92,
                    fontsize=7.5, handletextpad=0.4, borderpad=0.4)
    leg.set_title("open = Mendelian,  filled = Complex", prop={"size": 6.8})
    leg.get_frame().set_linewidth(0.4)

    ax.set_xlim(0.50, 1.00)
    ax.set_ylim(-0.02, 0.90)
    ax.set_xlabel(r"Cov$|Y{=}1$ (minority-class coverage, $\rightarrow$ better)")
    ax.set_ylabel(r"$\hat{\sigma}$-bin gap (cell-level worst, $\downarrow$ better)")
    ax.set_title("(b) Pareto frontier (Tab.~2)", loc="left", pad=4)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axhline(0.20, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0.85, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)


def main():
    fig = plt.figure(figsize=(11.6, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.00], wspace=0.18)

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
