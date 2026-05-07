"""Generate 2 color-scheme variants of fig_lf_forest for user selection.

Outputs to R_raw/figpolish_previews/:
  lf_forest_A.png  --- semantic: LCLS=rose red, isotonic=salmon, legacy=gray
  lf_forest_B.png  --- declared Set2: LCLS=teal, isotonic=salmon, legacy=blue-purple

Both share K_modal reference = #8b3a3a (matches fig9 T3' line).

After user picks, the chosen palette is ported to T_tools/lf_and_bootstrap_figs.py.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys as _sys
REPO = Path(__file__).resolve().parents[1]
PREVIEW = REPO / "R_raw" / "figpolish_previews"
PREVIEW.mkdir(parents=True, exist_ok=True)

_sys.path.insert(0, str(REPO))
from T_tools.paper_style import apply_paper_style  # noqa: E402
apply_paper_style()


# Static data (same as in production lf_and_bootstrap_figs.fig_lf_forest)
ROWS = [
    ("LCLS regression",      "Mendelian", 2.97,   (2.89,   3.45),   22),
    ("isotonic envelope",    "Mendelian", 26.9,   (6.85,   93.24),  61),
    ("legacy max-of-ratios", "Mendelian", 184.88, (137.86, 298.81), 169),
    ("LCLS regression",      "Complex",   4.51,   (4.47,   4.69),   50),
    ("isotonic envelope",    "Complex",   8.02,   (4.48,   33.13),  67),
    ("legacy max-of-ratios", "Complex",   72.11,  (54.17,  102.96), 173),
]
K_MODAL = {"Mendelian": (5, 8), "Complex": (5, 5)}
Y_POSITIONS = [6.0, 5.0, 4.0, 2.0, 1.0, 0.0]
BLOCK_Y = {"Mendelian": (3.5, 6.5), "Complex": (-0.5, 2.5)}

# K_modal reference line: align with fig9 T3' #8b3a3a in both variants
K_MODAL_COLOR = "#8b3a3a"


PALETTES = {
    "A": {
        "title": "Semantic (LCLS=HCCP red, legacy=gray)",
        "LCLS regression":      "#d1495b",  # rose red, matches HCCP in fig9/bootstrap
        "isotonic envelope":    "#fc8d62",  # Set2 salmon, intermediate
        "legacy max-of-ratios": "#888888",  # gray, deprecated
    },
    "B": {
        "title": "Declared Set2 (matches paper_style.py default cycle)",
        "LCLS regression":      "#66c2a5",  # Set2-1 teal
        "isotonic envelope":    "#fc8d62",  # Set2-2 salmon
        "legacy max-of-ratios": "#8da0cb",  # Set2-3 blue-purple
    },
}


def draw(palette_key: str):
    palette = PALETTES[palette_key]
    color_of = {k: v for k, v in palette.items() if k not in ("title",)}

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(8.6, 3.2),
        gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.18},
    )

    # Left: forest
    for y, (name, dataset, point, (lo, hi), _) in zip(Y_POSITIONS, ROWS):
        c = color_of[name]
        ci_ratio = hi / lo
        ms = 5.5 + 4.0 / max(np.log10(ci_ratio) + 0.05, 0.05)
        ms = float(np.clip(ms, 5.5, 11.0))
        axL.plot([lo, hi], [y, y], color=c, lw=2.4, alpha=0.55, solid_capstyle="round")
        axL.plot([point], [y], "o", color=c, markersize=ms,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=5)

    axL.set_yticks(Y_POSITIONS)
    axL.set_yticklabels([r[0] for r in ROWS], fontsize=8.5)
    for label, (y_lo, y_hi) in BLOCK_Y.items():
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
    axL.axhline(3.0, color="0.75", lw=0.5, ls="-", zorder=0)

    # Right: K* bar + K_modal reference
    for y, (name, dataset, _, _, Kstar) in zip(Y_POSITIONS, ROWS):
        c = color_of[name]
        axR.barh(y, Kstar, height=0.65, color=c, alpha=0.85,
                 edgecolor="white", linewidth=0.6, zorder=3)
        axR.text(Kstar + 4, y, f"{Kstar}", fontsize=8, va="center", ha="left")

    for dataset, (kmin, kmax) in K_MODAL.items():
        y_lo, y_hi = BLOCK_Y[dataset]
        if kmin == kmax:
            axR.vlines(kmin, y_lo, y_hi, color=K_MODAL_COLOR, lw=1.4,
                       linestyle=(0, (4, 2)), zorder=4)
            kmodal_text = fr"$\hat K_{{\mathrm{{modal}}}} = {kmin}$"
        else:
            axR.fill_betweenx([y_lo, y_hi], kmin, kmax, color=K_MODAL_COLOR,
                              alpha=0.18, zorder=2)
            axR.vlines([kmin, kmax], y_lo, y_hi, color=K_MODAL_COLOR, lw=1.0,
                       linestyle=(0, (4, 2)), zorder=4)
            kmodal_text = fr"$\hat K_{{\mathrm{{modal}}}} \in \{{{kmin},\,{kmax}\}}$"
        axR.text(max(kmin, kmax) + 8, y_hi + 0.05, kmodal_text,
                 fontsize=7.5, color=K_MODAL_COLOR, ha="left", va="bottom",
                 fontweight="bold")

    axR.set_yticks(Y_POSITIONS)
    axR.set_yticklabels([])
    axR.set_xlim(0, 220)
    axR.set_xlabel(r"$K^\star$  (asymptotic, oracle from $\hat L_F$)")
    axR.set_ylim(-0.7, 7.0)
    axR.grid(True, axis="x", which="major", alpha=0.35, ls="-", lw=0.4)
    axR.grid(False, axis="y")
    axR.axhline(3.0, color="0.75", lw=0.5, ls="-", zorder=0)

    # Bottom legend
    handles = [
        Line2D([0], [0], marker="o", color=color_of["LCLS regression"],
               linestyle="-", lw=2.0, markersize=5.5, label="LCLS regression"),
        Line2D([0], [0], marker="o", color=color_of["isotonic envelope"],
               linestyle="-", lw=2.0, markersize=5.5, label="isotonic envelope"),
        Line2D([0], [0], marker="o", color=color_of["legacy max-of-ratios"],
               linestyle="-", lw=2.0, markersize=5.5, label="legacy max-of-ratios"),
        Line2D([0], [0], color=K_MODAL_COLOR, lw=1.4, linestyle=(0, (4, 2)),
               label=r"modal nested-CV $\hat K(c_{\mathrm{outer}})$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=7.8,
               handlelength=1.8, columnspacing=1.6, handletextpad=0.5)

    fig.suptitle(f"Variant {palette_key}: {palette['title']}",
                 fontsize=9.5, y=0.995)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    out = PREVIEW / f"lf_forest_{palette_key}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(out)


def main():
    draw("A")
    draw("B")


if __name__ == "__main__":
    main()
