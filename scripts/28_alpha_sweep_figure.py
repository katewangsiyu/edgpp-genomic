"""Figure D1: alpha sweep for HCCP — 1x3 compact layout (datasets via color).

Layout: single row, 3 columns:
  (a) calibration residual cov - (1-alpha) for 3 methods x 2 datasets;
      Mendelian = solid + filled markers, Complex = dashed + hollow markers.
  (b) sigma-hat-bin gap (Mondrian) as grouped bars (Mendelian | Complex per
      alpha), with the alpha=0.10 headline 0.04 dashed reference line. Bar
      values are annotated to keep small Complex bars readable.
  (c) singleton fraction (Mondrian, two datasets), with 3 operating points
      labeled at top: high-stakes (alpha=0.10), triage (0.20), hypothesis (0.30).

Compact 1x3 (figsize 10.5x3.4) replaces the legacy 2x3 (10x5.4) so the
figure respects \\linewidth without text shrinkage and exposes operating-point
context inline. Companion table at scripts/figD_alpha_sweep_table.tex.

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
from matplotlib.lines import Line2D

_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from T_tools.paper_style import apply_paper_style  # noqa: E402

apply_paper_style()


METHODS = [
    ("Split CP (homoscedastic)",                 "homosc",   "o", "#888888"),
    ("HCCP class-cond",                          "hetero",   "s", "#4477aa"),
    (r"HCCP Mondrian ($K{=}5$)",                 "mondrian", "D", "#cc3311"),
]
DS_COLOR = {"Mendelian": "#cc3311", "Complex": "#4477aa"}
HEADLINE_GAP = 0.04

OP_POINTS = [
    (0.10, "high-stakes",  "#5e2750"),
    (0.20, "triage",       "#7e6e3a"),
    (0.30, "hypothesis",   "#2c6e49"),
]


def load_sweeps(path: Path) -> dict:
    d = json.load(open(path))
    return {
        "hetero":   d.get("alpha_sweep_hetero", {}),
        "homosc":   d.get("alpha_sweep_homosc", {}),
        "mondrian": d.get("alpha_sweep_mondrian", {}),
    }


def alphas_of(sweep: dict) -> tuple[list[str], list[float]]:
    a_str = sorted(sweep["mondrian"].keys(), key=float)
    return a_str, [float(a) for a in a_str]


def plot_calibration(ax, mend, comp):
    ax.axhspan(-0.015, 0.015, color="#dde6f0", alpha=0.6, zorder=0)
    ax.axhline(0.0, color="#222", ls="--", lw=0.7, zorder=1)
    for label, key, marker, color in METHODS:
        for ds_name, sw in [("Mendelian", mend[key]), ("Complex", comp[key])]:
            if not sw:
                continue
            a_str_ds = sorted(sw.keys(), key=float)
            alphas_ds = [float(a) for a in a_str_ds]
            res = [sw[a]["coverage"] - (1 - float(a)) for a in a_str_ds]
            ls = "-" if ds_name == "Mendelian" else "--"
            mfc = color if ds_name == "Mendelian" else "white"
            ax.plot(alphas_ds, res, marker=marker, ms=4.5, lw=1.1,
                    color=color, ls=ls, mfc=mfc, mec=color, mew=1.0,
                    alpha=0.88)
    ax.set_xlabel(r"miscoverage $\alpha$")
    ax.set_ylabel(r"cov $-\,(1-\alpha)$")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(-0.04, 0.04)
    ax.set_title("(a) calibration residual", fontsize=9.5)
    ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")


def plot_sigma_gap(ax, mend, comp):
    a_str, alphas = alphas_of(mend)
    rng_m = np.array([mend["mondrian"][a]["sigma_cov_range"] for a in a_str])
    rng_c = np.array([comp["mondrian"][a]["sigma_cov_range"] for a in a_str])
    bw = 0.018
    bars_m = ax.bar(np.array(alphas) - bw / 2, rng_m, width=bw,
                    color=DS_COLOR["Mendelian"], alpha=0.85,
                    edgecolor="white", linewidth=0.4, label="Mendelian")
    bars_c = ax.bar(np.array(alphas) + bw / 2, rng_c, width=bw,
                    color=DS_COLOR["Complex"], alpha=0.85,
                    edgecolor="white", linewidth=0.4, label="Complex")
    ax.axhline(HEADLINE_GAP, color="#222", ls="--", lw=0.7,
               label=rf"$\leq 0.04$ headline")

    # Annotate small Complex bars (< 0.005) with their numeric value, since
    # at this scale they would otherwise be visually invisible.
    for x, v in zip(alphas, rng_c):
        if v < 0.005:
            ax.text(x + bw / 2, v + 0.001, f"{v:.3f}",
                    fontsize=5.5, color=DS_COLOR["Complex"],
                    ha="center", va="bottom", rotation=90)

    ax.set_xlabel(r"miscoverage $\alpha$")
    ax.set_ylabel(r"$\hat\sigma$-bin gap")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0, max(0.10, max(rng_m.max(), rng_c.max()) * 1.25))
    ax.set_title(r"(b) $\hat\sigma$-bin gap (Mondrian)", fontsize=9.5)
    ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")
    ax.legend(fontsize=6.6, loc="upper left", frameon=True, framealpha=0.92,
              borderpad=0.3, handletextpad=0.4)


def plot_singleton(ax, mend, comp):
    a_str, alphas = alphas_of(mend)
    sf_m = np.array([mend["mondrian"][a]["frac_singleton"] for a in a_str])
    sf_c = np.array([comp["mondrian"][a]["frac_singleton"] for a in a_str])
    ax.plot(alphas, sf_m, marker="D", color=DS_COLOR["Mendelian"], lw=1.4,
            ms=5, mfc="white", mew=1.4, label="Mendelian")
    ax.plot(alphas, sf_c, marker="D", color=DS_COLOR["Complex"], lw=1.4,
            ms=5, mfc="white", mew=1.4, label="Complex")

    # Operating-point vertical guides + top labels (single row, low font).
    for op_a, op_label, op_color in OP_POINTS:
        ax.axvline(op_a, color=op_color, ls=":", lw=0.7, zorder=0, alpha=0.7)
        ax.text(op_a, 1.04, op_label, color=op_color, fontsize=6.4,
                ha="center", va="bottom", style="italic", fontweight="bold",
                transform=ax.get_xaxis_transform())

    # Mendelian non-monotonicity callout: peak then drop is the operationally
    # important observation; explicitly mark it so reviewers see honest reporting.
    peak_idx = int(np.argmax(sf_m))
    ax.annotate(
        fr"max $\approx {sf_m[peak_idx]:.2f}$ at $\alpha={alphas[peak_idx]:.2f}$",
        xy=(alphas[peak_idx], sf_m[peak_idx]),
        xytext=(alphas[peak_idx] + 0.13, sf_m[peak_idx] - 0.16),
        fontsize=6.4, color="#333",
        arrowprops=dict(arrowstyle="-", lw=0.4, color="#777"))

    ax.set_xlabel(r"miscoverage $\alpha$")
    ax.set_ylabel("singleton fraction")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0, 1.0)
    ax.set_title("(c) singleton fraction (Mondrian)", fontsize=9.5, pad=18)
    ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")
    ax.legend(fontsize=6.6, loc="lower right", frameon=True, framealpha=0.92,
              borderpad=0.3, handletextpad=0.4)


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

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4),
                             gridspec_kw={"width_ratios": [1.2, 1.0, 1.0],
                                          "wspace": 0.30})
    plot_calibration(axes[0], mend, comp)
    plot_sigma_gap(axes[1], mend, comp)
    plot_singleton(axes[2], mend, comp)

    # Method legend for (a) at figure bottom.
    method_handles = [
        Line2D([0], [0], marker=m, color=c, lw=1.4, markersize=5,
               mfc="white", mew=1.4, label=lab)
        for lab, _, m, c in METHODS
    ]
    method_handles.append(Line2D([0], [0], color="0.4", ls="-", lw=1.2,
                                  marker="o", mfc="0.4", mec="0.4",
                                  label="Mendelian (filled, solid)"))
    method_handles.append(Line2D([0], [0], color="0.4", ls="--", lw=1.2,
                                  marker="o", mfc="white", mec="0.4",
                                  label="Complex (hollow, dashed)"))
    fig.legend(handles=method_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=6.8,
               handletextpad=0.4, columnspacing=1.4)

    fig.suptitle(
        r"HCCP $\alpha$ sweep on TraitGym  ---  calibration, $\hat\sigma$-bin gap, operational knob",
        fontsize=10.5, y=1.04,
    )
    fig.tight_layout()

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
