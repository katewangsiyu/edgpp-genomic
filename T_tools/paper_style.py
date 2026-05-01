"""Unified NeurIPS-classic figure style for the HCCP paper.

Usage at the top of any figure-generating script:

    from T_tools.paper_style import apply_paper_style, PALETTE
    apply_paper_style()

Then create plots normally. ColorBrewer Set2 (colorblind-friendly) is the default cycle.
Method-specific colors via PALETTE dict (HCCP / DEGU / class-Mondrian / ...).
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ColorBrewer Set2 (colorblind-friendly, NeurIPS-grade)
SET2 = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"]

# Stable per-method palette — used when method identity must persist across figures
PALETTE = {
    "HCCP":           "#1f77b4",  # primary blue
    "DEGU":           "#ff7f0e",  # orange — direct competitor
    "class-Mondrian": "#2ca02c",  # green
    "sigma-Mondrian": "#d62728",  # red
    "split-CP":       "#9467bd",  # purple
    "RLCP":           "#8c564b",  # brown
    "weighted-CP":    "#e377c2",  # pink
    "SC-CP":          "#7f7f7f",  # grey
    "oracle":         "#000000",  # black
    "target":         "#bcbd22",  # olive — for 1-α reference lines
}


def apply_paper_style() -> None:
    """Apply NeurIPS-classic matplotlib style. Call once before plotting."""
    mpl.rcParams.update(
        {
            # --- Fonts ---
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 10,
            "mathtext.fontset": "stix",  # serif math, matches Times
            # --- Lines / markers ---
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "axes.linewidth": 0.7,
            # --- Axes / grid ---
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.4,
            "grid.color": "0.7",
            "axes.axisbelow": True,
            # --- Legend ---
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            "legend.borderpad": 0.3,
            "legend.columnspacing": 1.0,
            # --- Ticks ---
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            # --- Colors ---
            "axes.prop_cycle": mpl.cycler(color=SET2),
            # --- Figure / saving ---
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,  # TrueType — embed real fonts in PDF
            "ps.fonttype": 42,
        }
    )


def figure_size(width: str = "single", height_ratio: float = 0.65) -> tuple[float, float]:
    """NeurIPS column-aware figure sizing.

    width: 'single' = 3.25" (column), 'wide' = 6.75" (full text width), 'double' = same as wide.
    height_ratio: aspect ratio (height/width). 0.65 is a common landscape default.
    """
    w = {"single": 3.25, "wide": 6.75, "double": 6.75}[width]
    return (w, w * height_ratio)


def style_axis(ax: plt.Axes, *, integer_ticks: bool = False, log_y: bool = False) -> None:
    """Apply common per-axis touches after plotting."""
    if integer_ticks:
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    if log_y:
        ax.set_yscale("log")
    ax.tick_params(which="both", top=False, right=False)


if __name__ == "__main__":
    # Sanity: render a small test figure
    apply_paper_style()
    fig, ax = plt.subplots(figsize=figure_size("single"))
    import numpy as np
    x = np.linspace(0, 10, 50)
    for i, label in enumerate(["HCCP", "DEGU", "class-Mondrian"]):
        ax.plot(x, np.sin(x + i * 0.5), label=label, color=PALETTE[label])
    ax.set_xlabel("Calibration size $n$")
    ax.set_ylabel("Worst-cell coverage gap")
    ax.legend()
    fig.savefig("/tmp/paper_style_smoke.pdf")
    print("Smoke test → /tmp/paper_style_smoke.pdf")
