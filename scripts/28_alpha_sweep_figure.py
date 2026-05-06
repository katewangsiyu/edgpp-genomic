"""Figure D1: α sweep for HCCP — 2×3 compact layout.

Layout: rows = {Mendelian, Complex}; cols = {(a) calibration residual for
3 methods, (b) σ̂-bin gap (Mondrian) vs the 0.04 headline threshold,
(c) singleton fraction vs α (Mondrian) — the operational knob the appendix
text discusses but the previous 1×3 layout did not visualize}.

Compact in width (figsize 9.5×5.4 vs the legacy 13.5×3.6) so the figure
respects \\linewidth without text shrinkage, and exposes the operating-point
panels (third column) that previously lived only in the table.

Data: outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_{ds}_abs_mondrian/
      conformal_hetero_results.json under alpha_sweep_{homosc, hetero, mondrian}.
"""
from __future__ import annotations

import argparse
import json
import sys as _sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from T_tools.paper_style import apply_paper_style  # noqa: E402

apply_paper_style()


METHODS = [
    ("Split CP (homoscedastic)",                          "homosc",   "o", "#888888"),
    (r"HCCP class-cond (Eq.~3 score)",                    "hetero",   "s", "#4477aa"),
    (r"HCCP Mondrian ($y{\times}\hat\sigma$-bin), $K{=}5$", "mondrian", "D", "#cc3311"),
]
DS_COLOR = {"Mendelian": "#cc3311", "Complex": "#4477aa"}
HEADLINE_GAP = 0.04


def load_sweeps(path: Path) -> dict:
    d = json.load(open(path))
    return {
        "hetero":   d.get("alpha_sweep_hetero", {}),
        "homosc":   d.get("alpha_sweep_homosc", {}),
        "mondrian": d.get("alpha_sweep_mondrian", {}),
    }


def alphas_of(sweeps: dict) -> tuple[list[str], list[float]]:
    a_str = sorted(sweeps["mondrian"].keys(), key=float)
    return a_str, [float(a) for a in a_str]


def plot_residual(ax, sweeps: dict, ds_name: str, *, show_legend: bool, show_xlabel: bool):
    a_str, alphas = alphas_of(sweeps)
    ax.axhspan(-0.015, 0.015, color="#dde6f0", alpha=0.6, zorder=0,
               label=r"finite-sample slack $\pm 1/(n_{\min}{+}1)$")
    ax.axhline(0.0, color="#222222", ls="--", lw=0.7, zorder=1)

    for label, key, marker, color in METHODS:
        sweep = sweeps[key]
        if not sweep:
            continue
        residual = [sweep[a]["coverage"] - (1 - float(a)) for a in a_str]
        ax.plot(alphas, residual, marker=marker, color=color, label=label,
                lw=1.4, markersize=5, markerfacecolor="white",
                markeredgewidth=1.4, zorder=3)

    ax.set_ylabel(r"cov $-\,(1-\alpha)$")
    if show_xlabel:
        ax.set_xlabel(r"miscoverage $\alpha$")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(-0.04, 0.04)
    ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")
    if show_legend:
        ax.legend(fontsize=6.6, loc="lower right", frameon=True, framealpha=0.92,
                  borderpad=0.3, handletextpad=0.4)


def plot_sigma_gap(ax, sweeps: dict, ds_name: str, *, show_xlabel: bool):
    a_str, alphas = alphas_of(sweeps)
    rng = np.array([sweeps["mondrian"][a]["sigma_cov_range"] for a in a_str])
    color = DS_COLOR[ds_name]
    ax.bar(alphas, rng, width=0.025, color=color, alpha=0.85,
           edgecolor="white", linewidth=0.4)
    ax.axhline(HEADLINE_GAP, color="#222", ls="--", lw=0.7,
               label=rf"$\alpha{{=}}0.10$ headline ($\leq {HEADLINE_GAP}$)")
    if show_xlabel:
        ax.set_xlabel(r"miscoverage $\alpha$")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel(r"$\hat\sigma$-bin gap")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0, max(0.10, rng.max() * 1.25))
    ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")
    if ds_name == "Mendelian":
        ax.legend(fontsize=6.6, loc="upper right", frameon=True, framealpha=0.92,
                  borderpad=0.3, handletextpad=0.4)


def plot_singleton(ax, sweeps: dict, ds_name: str, *, show_xlabel: bool):
    a_str, alphas = alphas_of(sweeps)
    sf = np.array([sweeps["mondrian"][a]["frac_singleton"] for a in a_str])
    color = DS_COLOR[ds_name]
    ax.plot(alphas, sf, marker="D", color=color, lw=1.4, markersize=5,
            markerfacecolor="white", markeredgewidth=1.4,
            label=f"{ds_name} (Mondrian)")
    # Mark the three benchmarked operating points (vertical lines only;
    # labels live in the figure caption to avoid clutter).
    for ai in (0.10, 0.20, 0.30):
        if any(abs(a - ai) < 1e-6 for a in alphas):
            ax.axvline(ai, color="#888", ls=":", lw=0.5, zorder=0)
    if show_xlabel:
        ax.set_xlabel(r"miscoverage $\alpha$")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_ylabel("singleton fraction")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0, 1.0)
    ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mendelian-results",
                    default="outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/conformal_hetero_results.json")
    ap.add_argument("--complex-results",
                    default="outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/conformal_hetero_results.json")
    ap.add_argument("--out",
                    default="papers/neurips2027_pathA/figures/figD_alpha_sweep.pdf")
    args = ap.parse_args()

    mend = load_sweeps(Path(args.mendelian_results))
    comp = load_sweeps(Path(args.complex_results))

    fig = plt.figure(figsize=(9.8, 5.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.0],
                          hspace=0.22, wspace=0.38,
                          top=0.88, bottom=0.10, left=0.085, right=0.985)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)]
                     for i in range(2)])
    # Row 1: Mendelian
    plot_residual(axes[0, 0], mend, "Mendelian", show_legend=True,  show_xlabel=False)
    plot_sigma_gap(axes[0, 1], mend, "Mendelian", show_xlabel=False)
    plot_singleton(axes[0, 2], mend, "Mendelian", show_xlabel=False)
    # Row 2: Complex
    plot_residual(axes[1, 0], comp, "Complex", show_legend=False, show_xlabel=True)
    plot_sigma_gap(axes[1, 1], comp, "Complex", show_xlabel=True)
    plot_singleton(axes[1, 2], comp, "Complex", show_xlabel=True)

    # Column titles (above row 1) and row labels (left of col 0).
    col_titles = [
        r"(a) calibration residual",
        r"(b) $\hat\sigma$-bin gap (Mondrian)",
        r"(c) singleton fraction (Mondrian)",
    ]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=9.5, pad=4)
    # Row labels: dataset name vertically left of each row's first axis.
    for i, ds in enumerate(("Mendelian", "Complex")):
        bb = axes[i, 0].get_position()
        fig.text(0.018, (bb.y0 + bb.y1) / 2, ds, rotation=90,
                 ha="center", va="center", fontsize=10.5, fontweight="bold",
                 color=DS_COLOR[ds])

    fig.suptitle(
        r"HCCP $\alpha$ sweep on TraitGym — calibration, $\hat\sigma$-bin gap, and operational knob",
        fontsize=10.5, y=0.965,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"saved: {out}, {out.with_suffix('.png')}")

    # Also emit the LaTeX-ready table (unchanged from previous version).
    tbl_path = out.parent / "figD_alpha_sweep_table.tex"
    with open(tbl_path, "w") as f:
        f.write("% Auto-generated by scripts/28_alpha_sweep_figure.py\n")
        f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n")
        f.write("$\\alpha$ & M cov & M $\\hat\\sigma$-gap & C cov & C $\\hat\\sigma$-gap & singleton (M / C)\\\\\n")
        f.write("\\midrule\n")
        a_str = sorted(mend["mondrian"].keys(), key=float)
        for a in a_str:
            mm = mend["mondrian"][a]
            cc = comp["mondrian"][a]
            f.write(f"{float(a):.2f} & {mm['coverage']:.3f} & {mm['sigma_cov_range']:.3f} & "
                    f"{cc['coverage']:.3f} & {cc['sigma_cov_range']:.3f} & "
                    f"{mm['frac_singleton']*100:.0f}\\% / {cc['frac_singleton']*100:.0f}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved: {tbl_path}")


if __name__ == "__main__":
    main()
