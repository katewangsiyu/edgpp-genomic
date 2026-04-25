"""E3 — Joint frontier scatter plot: visualize HCCP's unique Pareto position.

Produces 2-panel figure:
  (a) cov|pos vs sigma-bin gap, scatter colored by method, shape by pi_pos.
      HCCP cluster occupies upper-left (high cov|pos, low gap) alone.
  (b) cov|pos vs pi_pos line plot showing B1/B2 collapse and B3/B4 stability.

Inputs: outputs of scripts/22_imbalance_sweep.py.

Usage:
    conda run -n edgpp_t4 --no-capture-output python scripts/24_frontier_plot.py \
        --summary outputs/imbalance_sweep/CADD+GPN-MSA+Borzoi_complex/imbalance_sweep_summary.csv \
        --out figures/fig_frontier_complex.pdf
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "B1_split": "#9e9e9e",
    "B2_sigma": "#1f77b4",
    "B3_class": "#ff7f0e",
    "B4_HCCP": "#d62728",
}
METHOD_LABELS = {
    "B1_split": r"B1: vanilla normalized split CP",
    "B2_sigma": r"B2: $\hat\sigma$-Mondrian (Boström 2020)",
    "B3_class": r"B3: class-Mondrian (Sadinle 2019)",
    "B4_HCCP": r"B4: HCCP (ours, joint $y \times \hat\sigma$)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-cov-pos", type=float, default=0.90)
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel (a): cov|pos vs sigma-bin gap (Pareto frontier view)
    ax = axes[0]
    pi_unique = sorted(df["pi"].unique())
    sizes = np.linspace(40, 130, len(pi_unique))
    pi_to_size = dict(zip(pi_unique, sizes))

    for method, color in METHOD_COLORS.items():
        sub = df[df["method"] == method]
        for _, r in sub.iterrows():
            ax.scatter(r["sigma_bin_gap"], r["coverage_pos"],
                       s=pi_to_size[r["pi"]], color=color, edgecolor="white",
                       linewidth=0.6, alpha=0.85, zorder=3)
        ax.plot(sub["sigma_bin_gap"], sub["coverage_pos"], color=color,
                linewidth=1.0, alpha=0.4, zorder=2)

    # target lines
    ax.axhline(args.target_cov_pos, linestyle="--", color="black",
               linewidth=0.8, alpha=0.5, zorder=1)
    ax.text(ax.get_xlim()[1] * 0.98, args.target_cov_pos + 0.005,
            r"target $\mathrm{cov}_{|Y=1} = 0.90$",
            ha="right", va="bottom", fontsize=8, color="black", alpha=0.7)

    # legend (method)
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=METHOD_COLORS[m],
                          markersize=8, markeredgecolor='white',
                          label=METHOD_LABELS[m]) for m in METHOD_COLORS]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    ax.set_xlabel(r"$\hat\sigma$-bin coverage gap (lower = better local coverage)")
    ax.set_ylabel(r"$\mathrm{cov}_{|Y=1}$ (higher = better minority coverage)")
    ax.set_title(r"(a) Joint frontier under imbalance sweep")
    ax.invert_xaxis()  # lower x is better, so flip
    ax.grid(True, alpha=0.25)

    # annotate frontier point
    hccp = df[df["method"] == "B4_HCCP"]
    hccp_avg_gap = hccp["sigma_bin_gap"].mean()
    hccp_avg_cov = hccp["coverage_pos"].mean()
    ax.annotate("HCCP cluster:\nhigh cov$|+$ AND low gap\n(unique frontier point)",
                xy=(hccp_avg_gap, hccp_avg_cov),
                xytext=(hccp_avg_gap + 0.05, hccp_avg_cov - 0.06),
                fontsize=8, color=METHOD_COLORS["B4_HCCP"],
                arrowprops=dict(arrowstyle="->", color=METHOD_COLORS["B4_HCCP"], lw=0.8))

    # Panel (b): cov|pos vs pi_pos
    ax = axes[1]
    for method, color in METHOD_COLORS.items():
        sub = df[df["method"] == method].sort_values("pi")
        ax.plot(sub["pi"], sub["coverage_pos"], "-o", color=color,
                label=METHOD_LABELS[method], markersize=6, linewidth=1.6,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(args.target_cov_pos, linestyle="--", color="black",
               linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"Positive-class prevalence $\pi_{+}$")
    ax.set_ylabel(r"$\mathrm{cov}_{|Y=1}$")
    ax.set_title(r"(b) Imbalance scaling: single-axis methods collapse as $\pi_+ \to 0$")
    ax.set_ylim(0.55, 0.96)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
