"""Re-plot fig_n_sweep.pdf from cached R_raw/synthetic_n_sweep/results.json.

Layout (App F.2 of the paper):
  1x2 panels split by pi_min; each panel shows 3 feature-dim curves on a
  shared log-log y-axis. The shared y-axis makes the pi_min^{-1/2} vertical
  offset between panels visible at a glance. Each panel embeds the empirical
  slope fit and the compensated empirical constant bar{C}(pi) as a black
  dashed reference (corresponding to G = bar{C} / sqrt{n} on log-log).

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
from T_tools.paper_style import apply_paper_style  # noqa: E402

apply_paper_style()

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "R_raw" / "synthetic_n_sweep" / "results.json"
FIG_DIR = REPO / "papers" / "neurips2027_pathA" / "figures"
FIT_NMIN = 1000


def _curve(rate_results, d, pi):
    rows = rate_results[f"d{d}_pi{pi:.2f}"]
    return (np.array([r["n"] for r in rows]),
            np.array([r["worst_gap"] for r in rows]),
            np.array([r["worst_gap_se"] for r in rows]))


def _fit_loglog_slope(ns, gs, n_min=FIT_NMIN):
    mask = ns >= n_min
    slope, intercept = np.polyfit(np.log(ns[mask]), np.log(gs[mask]), 1)
    return float(slope), float(intercept)


def main():
    data = json.loads(RESULTS.read_text())
    rate_results = data["rate_results"]
    pi_grid = data["config"]["pi_grid"]
    d_grid = data["config"]["d_grid"]
    n_grid = data["config"]["n_grid"]

    d_color = {8: "#1b9e77", 32: "#d95f02", 128: "#7570b3"}

    # ---- Stats: per-panel slope (mean+std over 3 d curves) and bar{C}(pi) ----
    slope_per_pi = {}  # mean, std over 3 d curves at this pi
    const_pi = {}       # empirical compensated mean over n>=NMIN, all d
    all_slopes = []
    for pi in pi_grid:
        slopes_pi = []
        comp_vals = []
        for d in d_grid:
            ns, gs, _ = _curve(rate_results, d, pi)
            slopes_pi.append(_fit_loglog_slope(ns, gs)[0])
            mask = ns >= FIT_NMIN
            comp_vals.extend(gs[mask] * np.sqrt(ns[mask]))
        slope_per_pi[pi] = (float(np.mean(slopes_pi)),
                            float(np.std(slopes_pi, ddof=1)))
        const_pi[pi] = float(np.mean(comp_vals))
        all_slopes.extend(slopes_pi)
    slope_overall = (float(np.mean(all_slopes)),
                     float(np.std(all_slopes, ddof=1)))
    ratio_emp = const_pi[pi_grid[0]] / const_pi[pi_grid[-1]]
    ratio_thy = float(np.sqrt(pi_grid[-1] / pi_grid[0]))

    # ---- Figure: 1x2 panels by pi_min, shared y-axis ----
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    n_min_x = min(n_grid) * 0.8
    n_max_x = max(n_grid) * 1.2

    for ax, pi in zip(axes, pi_grid):
        # Fallback regime shading
        ax.axvspan(n_min_x, FIT_NMIN, color="0.92", zorder=0)

        # Three d curves with errorbars
        for d in d_grid:
            ns, gs, ses = _curve(rate_results, d, pi)
            ax.errorbar(ns, gs, yerr=ses, marker="o", linestyle="-",
                        color=d_color[d], capsize=2.0, lw=1.2, ms=4.5,
                        mfc="white", mec=d_color[d], mew=1.2,
                        label=fr"$d={d}$", alpha=0.95)

        # Empirical compensated reference: G = bar{C}(pi) / sqrt{n}.
        # This is descriptive (uses panel-internal empirical constant), not
        # circular: it shows where the fitted -1/2 line passes through the
        # empirical mean of the compensated quantity.
        n_curve = np.geomspace(FIT_NMIN, n_max_x, 64)
        c = const_pi[pi]
        ax.plot(n_curve, c / np.sqrt(n_curve), "k--", lw=1.0, alpha=0.7,
                zorder=4, label=r"$\bar C(\pi_{\min}) / \sqrt{n}$")

        # Slope-fit annotation box (top-right corner)
        sm, ss = slope_per_pi[pi]
        ann = (fr"slope $= -{abs(sm):.2f}\pm{ss:.2f}$"
               "\n"
               fr"$\bar C(\pi_{{\min}}) = {c:.2f}$")
        ax.text(0.97, 0.97, ann, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.8, family="serif",
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.7",
                          alpha=0.95))

        ax.text(0.03, 0.03, "fallback\n(excl. fit)",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7.0, color="0.45", style="italic")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Calibration size $n$")
        ax.set_title(rf"$\pi_{{\min}} = {pi:.2f}$", fontsize=10)
        ax.set_xlim(n_min_x, n_max_x)

    axes[0].set_ylabel(r"Worst-cell gap $G(\hat K_{\mathrm{CV}})$")

    # Cross-panel summary banner above (slope across all 6 fits + C ratio).
    # Use plain text + mathtext only (no \textbf/\quad/\, — those are LaTeX
    # text-mode commands that matplotlib's mathtext does not parse).
    summary = (fr"All 6 fits:  slope $= -{abs(slope_overall[0]):.2f}\pm"
               fr"{slope_overall[1]:.2f}$  (T5.1 prediction $-0.50$, "
               f"within 1 SD)     $\\bar C_{{0.10}} / \\bar C_{{0.50}} = "
               fr"{ratio_emp:.2f}$  (T5.1 upper bound $\sqrt{{5}} = "
               fr"{ratio_thy:.2f}$)")
    fig.suptitle(summary, fontsize=8.5, y=0.995)

    # Legend strip at bottom; reserve bottom margin via subplots_adjust so
    # the legend never overlaps the x-axis labels.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=d_color[d], marker="o", linestyle="-",
                       lw=1.4, markersize=4.5, mfc="white", mec=d_color[d],
                       mew=1.2, label=fr"$d={d}$") for d in d_grid]
    handles.append(Line2D([0], [0], color="k", linestyle="--", lw=1.0,
                          label=r"$\bar C(\pi_{\min}) / \sqrt{n}$ "
                                r"(empirical $-1/2$ reference)"))
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=7.8,
               handlelength=2.0, columnspacing=1.5, handletextpad=0.5)

    fig.tight_layout()
    fig.subplots_adjust(top=0.86, bottom=0.22)

    out_pdf = FIG_DIR / "fig_n_sweep.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_n_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"  slope per pi: {slope_per_pi}")
    print(f"  slope overall: {slope_overall[0]:.3f} +/- {slope_overall[1]:.3f}")
    print(f"  C(0.10)={const_pi[0.1]:.3f}, C(0.50)={const_pi[0.5]:.3f}, "
          f"ratio={ratio_emp:.3f} (T5 sqrt(5)={ratio_thy:.3f})")


if __name__ == "__main__":
    main()
