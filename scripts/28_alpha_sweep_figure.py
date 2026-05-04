"""Figure D1: α sweep for HCCP Mondrian vs class-cond vs homoscedastic.

Produces a 2-panel figure (Mendelian, Complex) showing empirical coverage vs α
and σ̂-bin range vs α. Data from outputs/conformal_hetero/*_mondrian/.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Unified NeurIPS-classic style ---
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from T_tools.paper_style import apply_paper_style  # noqa: E402
apply_paper_style()


def load_sweeps(path: Path) -> dict:
    d = json.load(open(path))
    return {
        "hetero": d.get("alpha_sweep_hetero", {}),
        "homosc": d.get("alpha_sweep_homosc", {}),
        "mondrian": d.get("alpha_sweep_mondrian", {}),
    }


def plot(ax, sweeps: dict, ds_name: str) -> None:
    """Residual plot: cov - (1 - α) per method, ± 1/(n_min+1) finite-sample band.

    Three lines collapse onto y=1-α in absolute coverage; the residual view
    magnifies the actual differences so reviewers can see calibration quality.
    """
    alphas_str = sorted(sweeps["mondrian"].keys(), key=float)
    alphas = [float(a) for a in alphas_str]

    # Finite-sample slack band: ±0.015 corresponds to 1/(n_min + 1) at n ≥ 65
    # per cell, comfortably met for all reported α.
    ax.axhspan(-0.015, 0.015, color="#dde6f0", alpha=0.6, zorder=0,
               label=r"finite-sample slack $\pm 1/(n_{\min}+1)$")
    ax.axhline(0.0, color="#222222", ls="--", lw=0.8, zorder=1)

    for label, key, marker, color in [
        ("Split CP (homoscedastic)",          "homosc",   "o", "#888888"),
        ("HCCP class-cond (Eq.~3 score)",     "hetero",   "s", "#4477aa"),
        ("HCCP Mondrian ($y{\\times}\\hat\\sigma$-bin), K=5", "mondrian", "D", "#cc3311"),
    ]:
        sweep = sweeps[key]
        if not sweep:
            continue
        residual = [sweep[a]["coverage"] - (1 - float(a)) for a in alphas_str]
        ax.plot(alphas, residual, marker=marker, color=color, label=label,
                lw=1.6, markersize=6, markerfacecolor="white",
                markeredgewidth=1.6, zorder=3)

    ax.set_xlabel(r"miscoverage $\alpha$")
    ax.set_ylabel(r"coverage residual:  cov $-\,(1-\alpha)$")
    ax.set_title(f"{ds_name}: calibration residual (zoomed)")
    ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(-0.04, 0.04)
    ax.legend(fontsize=7.5, loc="lower right", frameon=True, framealpha=0.9)


def plot_sigma_range_combined(ax, mend_sweeps: dict, comp_sweeps: dict) -> None:
    """Single panel: M and C side-by-side bars per α (was 2 sparse panels)."""
    alphas_str = sorted(mend_sweeps["mondrian"].keys(), key=float)
    alphas = np.array([float(a) for a in alphas_str])
    m_rng = [mend_sweeps["mondrian"][a]["sigma_cov_range"] for a in alphas_str]
    c_rng = [comp_sweeps["mondrian"][a]["sigma_cov_range"] for a in alphas_str]
    width = 0.012
    ax.bar(alphas - width/2, m_rng, width=width, color="#cc3311", alpha=0.85,
           label="Mendelian", edgecolor="white", linewidth=0.4)
    ax.bar(alphas + width/2, c_rng, width=width, color="#4477aa", alpha=0.85,
           label="Complex",  edgecolor="white", linewidth=0.4)
    ax.axhline(0.04, color="#222", ls="--", lw=0.8,
               label=r"$\alpha = 0.10$ headline ($\leq 0.04$)")
    ax.set_xlabel(r"miscoverage $\alpha$")
    ax.set_ylabel(r"$\hat\sigma$-bin gap  $\max_b \mathrm{cov}_b - \min_b$")
    ax.set_title(r"$\hat\sigma$-bin gap is $\alpha$-stable on both datasets")
    ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0, max(max(m_rng), max(c_rng), 0.08) * 1.2)
    ax.legend(fontsize=7.5, loc="upper right", frameon=True, framealpha=0.9)


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

    # New layout: 1 row × 3 cols. Left two columns = residual (M / C);
    # right column = combined sigma-bin gap bar (was 2 sparse panels).
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})
    plot(axes[0], mend, "Mendelian")
    plot(axes[1], comp, "Complex")
    plot_sigma_range_combined(axes[2], mend, comp)
    fig.suptitle(r"HCCP $\alpha$ sweep on TraitGym — calibration residual + $\hat\sigma$-bin gap",
                 fontsize=11, y=1.02)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"saved: {out}, {out.with_suffix('.png')}")

    # Also emit a LaTeX-ready table
    tbl_path = out.parent / "figD_alpha_sweep_table.tex"
    with open(tbl_path, "w") as f:
        f.write("% Auto-generated by scripts/28_alpha_sweep_figure.py\n")
        f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n")
        f.write("$\\alpha$ & M cov & M $\\hat\\sigma$-gap & C cov & C $\\hat\\sigma$-gap & singleton (M / C)\\\\\n")
        f.write("\\midrule\n")
        alphas_str = sorted(mend["mondrian"].keys(), key=float)
        for a in alphas_str:
            mm = mend["mondrian"][a]
            cc = comp["mondrian"][a]
            f.write(f"{float(a):.2f} & {mm['coverage']:.3f} & {mm['sigma_cov_range']:.3f} & "
                    f"{cc['coverage']:.3f} & {cc['sigma_cov_range']:.3f} & "
                    f"{mm['frac_singleton']*100:.0f}\\% / {cc['frac_singleton']*100:.0f}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"saved: {tbl_path}")


if __name__ == "__main__":
    main()
