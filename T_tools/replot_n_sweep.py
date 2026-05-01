"""Re-plot fig_n_sweep.pdf from cached R_raw/synthetic_n_sweep/results.json.

Skips the expensive simulation. Style via T_tools.paper_style.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from T_tools.paper_style import apply_paper_style, figure_size  # noqa: E402

apply_paper_style()

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "R_raw" / "synthetic_n_sweep" / "results.json"
FIG_DIR = REPO / "papers" / "neurips2027_pathA" / "figures"

data = json.loads(RESULTS.read_text())
rate_results = data["rate_results"]
pi_grid = data["config"]["pi_grid"]
d_grid = data["config"]["d_grid"]
n_grid = data["config"]["n_grid"]

fig, axes = plt.subplots(1, 2, figsize=figure_size("wide", 0.42))
for ax, pi in zip(axes, pi_grid):
    for d in d_grid:
        key = f"d{d}_pi{pi:.2f}"
        rows = rate_results[key]
        ns = np.array([r["n"] for r in rows])
        ws = np.array([r["worst_gap"] for r in rows])
        se = np.array([r["worst_gap_se"] for r in rows])
        ax.errorbar(ns, ws, yerr=se, marker="o", label=f"$d={d}$", capsize=2.5)
    anchor = next(r for r in rate_results[f"d32_pi{pi:.2f}"] if r["n"] == 4000)
    c = anchor["worst_gap"] * np.sqrt(4000)
    n_curve = np.linspace(min(n_grid), max(n_grid), 100)
    ax.plot(n_curve, c / np.sqrt(n_curve), "k--", alpha=0.7,
            label=r"$O(n^{-1/2})$ T5 prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Calibration size $n$")
    ax.set_ylabel(r"Worst-cell gap $G(\hat K_{\mathrm{CV}})$")
    ax.set_title(rf"$\pi_{{\min}}={pi:.2f}$")
    ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_n_sweep.pdf")
fig.savefig(FIG_DIR / "fig_n_sweep.png", dpi=200)
plt.close(fig)
print(f"Saved {FIG_DIR / 'fig_n_sweep.pdf'}")
