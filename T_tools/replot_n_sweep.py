"""Re-plot fig_n_sweep.pdf from cached R_raw/synthetic_n_sweep/results.json.

Two-panel design (App F.2 of the paper):
  Left  --- log-log G(n) vs n with empirical slope fit + T5 prediction line
  Right --- compensated G(n) * sqrt(n) vs n with empirical pi-stratified means
            (a horizontal line on the compensated panel directly verifies the
            -1/2 exponent and the pi_min^{-1/2} constant; ratio between the
            two pi levels should match sqrt(pi_max / pi_min) = sqrt(5) ~ 2.24).

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
FIT_NMIN = 1000  # exclude small-n fallback regime, per F_synthetic.tex caption


def _curve(rate_results, d, pi):
    rows = rate_results[f"d{d}_pi{pi:.2f}"]
    return (np.array([r["n"] for r in rows]),
            np.array([r["worst_gap"] for r in rows]),
            np.array([r["worst_gap_se"] for r in rows]))


def _fit_loglog_slope(ns, gs, n_min=FIT_NMIN):
    mask = ns >= n_min
    log_n = np.log(ns[mask])
    log_g = np.log(gs[mask])
    slope, intercept = np.polyfit(log_n, log_g, 1)
    return float(slope), float(intercept)


def main():
    data = json.loads(RESULTS.read_text())
    rate_results = data["rate_results"]
    pi_grid = data["config"]["pi_grid"]
    d_grid = data["config"]["d_grid"]
    n_grid = data["config"]["n_grid"]

    # Style: color encodes feature dim d, linestyle/marker encodes pi_min.
    # Three Set2 colors (chosen to match other figures in the paper).
    d_color = {8: "#1b9e77", 32: "#d95f02", 128: "#7570b3"}
    pi_style = {0.1: dict(marker="o", linestyle="-", mfc="none"),
                0.5: dict(marker="s", linestyle="--", mfc="none")}

    # ---- Compute slope fits and compensated empirical constants ----
    slopes = []
    for d in d_grid:
        for pi in pi_grid:
            ns, gs, _ = _curve(rate_results, d, pi)
            slopes.append(_fit_loglog_slope(ns, gs)[0])
    slope_mean = float(np.mean(slopes))
    slope_std = float(np.std(slopes, ddof=1))

    # Compensated constant per pi: average of G * sqrt(n) over n >= n_min, all d.
    const_pi = {}
    for pi in pi_grid:
        vals = []
        for d in d_grid:
            ns, gs, _ = _curve(rate_results, d, pi)
            mask = ns >= FIT_NMIN
            vals.extend(gs[mask] * np.sqrt(ns[mask]))
        const_pi[pi] = float(np.mean(vals))
    ratio_emp = const_pi[pi_grid[0]] / const_pi[pi_grid[-1]]
    ratio_thy = float(np.sqrt(pi_grid[-1] / pi_grid[0]))

    # ---- Figure ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.0, 3.4),
                                   gridspec_kw={"wspace": 0.28})

    # ---- Left panel: log-log G(n) vs n ----
    n_min_x = min(n_grid) * 0.8
    n_max_x = max(n_grid) * 1.2

    # Fallback regime shading (n < FIT_NMIN). zorder=0 puts it under the grid.
    for ax in (axL, axR):
        ax.axvspan(n_min_x, FIT_NMIN, color="0.92", zorder=0)

    for d in d_grid:
        for pi in pi_grid:
            ns, gs, ses = _curve(rate_results, d, pi)
            label = fr"$d={d},\;\pi_{{\min}}={pi:.2f}$"
            axL.errorbar(ns, gs, yerr=ses, color=d_color[d],
                         capsize=2.0, lw=1.1, ms=4.5, alpha=0.92,
                         label=label, **pi_style[pi])

    # T5.1 prediction line: slope -1/2 anchored not at any datapoint but at
    # the *theoretical compensated constant* extracted from the right panel
    # (so the two panels are mutually consistent; the line is `predict' in the
    # weak sense that it uses no per-curve fit).
    n_curve = np.geomspace(FIT_NMIN, n_max_x, 64)
    for pi in pi_grid:
        c = const_pi[pi]
        axL.plot(n_curve, c / np.sqrt(n_curve), "k--", lw=1.0, alpha=0.55,
                 zorder=4)
    # Annotate the upper line as the slope reference; lower line is implicit.
    axL.text(2200, const_pi[pi_grid[0]] / np.sqrt(2200) * 1.15,
             r"slope $-1/2$", fontsize=8, color="0.25",
             rotation=-21, ha="left", va="bottom")

    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xlabel(r"Calibration size $n$")
    axL.set_ylabel(r"Worst-cell gap $G(\hat K_{\mathrm{CV}})$")
    axL.set_title("(a) log-log empirical rate", fontsize=9.5, loc="left")
    axL.set_xlim(n_min_x, n_max_x)

    # Slope-fit summary box (top-right corner where there is whitespace).
    fit_text = (fr"empirical slope: $-{abs(slope_mean):.2f}\pm{slope_std:.2f}$"
                "\n"
                fr"T5.1 prediction:  $-0.50$"
                "\n"
                fr"(fit on $n \geq {FIT_NMIN}$, mean$\pm$std over 6 curves)")
    axL.text(0.97, 0.97, fit_text, transform=axL.transAxes,
             ha="right", va="top", fontsize=7.5, family="serif",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7",
                       alpha=0.92))
    axL.text(0.02, 0.02, "fallback regime\n(excluded from fit)",
             transform=axL.transAxes, ha="left", va="bottom",
             fontsize=7.0, color="0.45", style="italic")

    # ---- Right panel: compensated G(n) * sqrt(n) vs n ----
    for d in d_grid:
        for pi in pi_grid:
            ns, gs, ses = _curve(rate_results, d, pi)
            comp = gs * np.sqrt(ns)
            comp_se = ses * np.sqrt(ns)
            axR.errorbar(ns, comp, yerr=comp_se, color=d_color[d],
                         capsize=2.0, lw=1.1, ms=4.5, alpha=0.92,
                         **pi_style[pi])

    # Empirical constants per pi as horizontal dashed lines (n >= FIT_NMIN).
    # Annotations placed at the left edge of the post-fallback region so they
    # never collide with the data curves at large n.
    for pi in pi_grid:
        c = const_pi[pi]
        axR.axhline(c, color="#c53b3b", linestyle=(0, (5, 3)), lw=1.2,
                    alpha=0.85, zorder=3)
        axR.text(FIT_NMIN * 1.05, c + 0.18,
                 fr"$\bar C(\pi_{{\min}}{{=}}{pi:.2f}) = {c:.2f}$",
                 fontsize=7.5, color="#c53b3b",
                 ha="left", va="bottom")

    # Ratio annotation comparing empirical vs theoretical sqrt(5).
    axR.text(0.97, 0.97,
             fr"empirical $\bar C_{{0.10}}/\bar C_{{0.50}} = {ratio_emp:.2f}$"
             "\n"
             fr"T5.1 prediction $\sqrt{{5}} = {ratio_thy:.2f}$",
             transform=axR.transAxes, ha="right", va="top",
             fontsize=7.5, family="serif",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7",
                       alpha=0.92))

    axR.set_xscale("log")
    axR.set_xlabel(r"Calibration size $n$")
    axR.set_ylabel(r"$G(\hat K_{\mathrm{CV}}) \cdot \sqrt{n}$ (compensated)")
    axR.set_title(r"(b) compensated: $\sqrt{n}$ rate scrubbed",
                  fontsize=9.5, loc="left")
    axR.set_xlim(n_min_x, n_max_x)
    # Linear y; auto y-limits but keep some headroom for annotation box.
    ymin, ymax = axR.get_ylim()
    axR.set_ylim(0, ymax * 1.05)

    # ---- Shared legend below both panels ----
    from matplotlib.lines import Line2D
    handles = []
    for d in d_grid:
        handles.append(Line2D([0], [0], color=d_color[d], marker="o",
                              linestyle="-", lw=1.4, markersize=4.5,
                              label=fr"$d={d}$"))
    handles.append(Line2D([0], [0], color="0.25", marker="o",
                          linestyle="-", lw=1.4, markersize=4.5, mfc="none",
                          label=r"$\pi_{\min}=0.10$"))
    handles.append(Line2D([0], [0], color="0.25", marker="s",
                          linestyle="--", lw=1.4, markersize=4.5, mfc="none",
                          label=r"$\pi_{\min}=0.50$"))
    handles.append(Line2D([0], [0], color="#c53b3b", linestyle=(0, (5, 3)),
                          lw=1.2, label=r"empirical $\bar C(\pi_{\min})$"))
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=7.6,
               handlelength=1.8, columnspacing=1.4, handletextpad=0.5)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)

    out_pdf = FIG_DIR / "fig_n_sweep.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_n_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"  empirical slope: {slope_mean:.3f} +/- {slope_std:.3f}")
    print(f"  empirical C(0.10)={const_pi[0.1]:.3f}, C(0.50)={const_pi[0.5]:.3f}")
    print(f"  ratio emp={ratio_emp:.3f} vs theoretical sqrt(5)={ratio_thy:.3f}")


if __name__ == "__main__":
    main()
