"""Generate two App C visualizations:
  1. L_F estimator forest plot (95% CI) from tab:LF_estimators data
  2. Bootstrap density of sigma-bin gap from B=200 chrom-bootstrap replicates

Outputs:
  papers/neurips2027_pathA/figures/fig_lf_forest.pdf
  papers/neurips2027_pathA/figures/fig_bootstrap_density.pdf
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "papers" / "neurips2027_pathA" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def fig_lf_forest():
    """L_F estimator forest plot from tab:LF_estimators numbers."""
    rows = [
        ("LCLS regression",   "Mendelian", 2.97, (2.89, 3.45),  22),
        ("isotonic envelope", "Mendelian", 26.9, (6.85, 93.2),  61),
        ("legacy max-of-ratios", "Mendelian", 183, (183, 183),  169),  # no CI
        ("LCLS regression",   "Complex",  4.51, (4.47, 4.69),  50),
        ("isotonic envelope", "Complex",  8.02, (4.48, 33.1),  67),
        ("legacy max-of-ratios", "Complex",  53.3, (53.3, 53.3), 173),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    for ax, dataset in zip(axes, ("Mendelian", "Complex")):
        sub = [r for r in rows if r[1] == dataset]
        ys = np.arange(len(sub))[::-1]
        for i, (name, _, point, (lo, hi), Kstar) in enumerate(zip(
                [r[0] for r in sub], [r[1] for r in sub],
                [r[2] for r in sub], [r[3] for r in sub], [r[4] for r in sub])):
            color = "C0" if "LCLS" in name else ("C1" if "isotonic" in name else "C3")
            y = ys[i]
            ax.plot([lo, hi], [y, y], color=color, lw=3, alpha=0.6)
            ax.plot([point], [y], "o", color=color, markersize=8)
            ax.text(max(point, hi) * 1.15, y, f"$K^\\star\\!=\\!{Kstar}$",
                    fontsize=8, va="center")
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in sub], fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\hat L_F$ (log scale)")
        ax.set_title(f"{dataset}")
        ax.grid(True, axis="x", which="both", alpha=0.3)
        # K_CV reference
        K_cv = 3 if dataset == "Mendelian" else 2
        ax.axvline(0.4, color="k", linestyle=":", alpha=0)  # invisible spacer
        ax.text(0.97, 0.05, fr"$\hat K_{{\mathrm{{CV}}}}={K_cv}$",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", alpha=0.8))
    fig.suptitle(r"$L_F$ estimator 95\% CIs and resulting $K^\star$ "
                 r"(LCLS gives tight CIs; legacy max-of-ratios saturates)",
                 fontsize=10)
    fig.tight_layout()
    out = FIG / "fig_lf_forest.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG / "fig_lf_forest.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_bootstrap_density():
    """Density of sigma_cov_range across B=200 bootstrap replicates."""
    paths = {
        "Mendelian (CADD+GPN-MSA+Borzoi)": REPO / "outputs/bootstrap_ci/CADD+GPN-MSA+Borzoi_mendelian.json",
        "Complex (CADD+GPN-MSA+Borzoi)":   REPO / "outputs/bootstrap_ci/CADD+GPN-MSA+Borzoi_complex.json",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    point_estimates = {"Mendelian (CADD+GPN-MSA+Borzoi)": 0.077,
                       "Complex (CADD+GPN-MSA+Borzoi)": 0.023}
    for ax, (label, p) in zip(axes, paths.items()):
        d = json.loads(p.read_text())
        gaps = np.array([r["sigma_cov_range"] for r in d["replicates"]])
        bs_mean = float(d["summary"]["sigma_cov_range"]["mean"])
        bs_std = float(d["summary"]["sigma_cov_range"]["std"])
        # Histogram + KDE
        ax.hist(gaps, bins=30, density=True, color="C0", alpha=0.5, edgecolor="k", linewidth=0.4)
        # Mean / point reference lines
        ax.axvline(bs_mean, color="C0", linestyle="-", lw=2,
                   label=fr"bootstrap mean $= {bs_mean:.3f}$")
        ax.axvline(point_estimates[label], color="C3", linestyle="--", lw=2,
                   label=fr"all-chrom point $= {point_estimates[label]:.3f}$")
        ax.set_xlabel(r"per-resample max-bin coverage gap")
        ax.set_ylabel("density")
        ax.set_title(label, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(r"Bootstrap distribution ($B=200$ chromosome resamples) of HCCP $\hat\sigma$-bin gap",
                 fontsize=10)
    fig.tight_layout()
    out = FIG / "fig_bootstrap_density.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(FIG / "fig_bootstrap_density.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig_lf_forest()
    fig_bootstrap_density()
