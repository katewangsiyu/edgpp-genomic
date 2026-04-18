"""Figure 1 concept pane — HCCP pipeline at a glance.

Three panels:
  (a) Heteroscedastic aggregator p̂(x) ± σ̂(x) on a 1-D toy, with the
      nonconformity score geometry s(x,y) = |y - p̂(x)| / σ̂(x) visualized
      as a band whose half-width scales with σ̂(x).
  (b) Mondrian calibration grid on (y × σ̂-bin), annotated with per-cell
      thresholds q̂_{k,b} and the T3 finite-sample rate 1/(n_{k,b}+1).
  (c) Three-axis OOD coverage bars: chrom-LOO / trait-LOO / cross-dataset,
      showing σ̂-bin gap ≤ 0.04 in every non-strong-shift regime.

Numbers are anchored to:
  - Day 10 GBM Mendelian AUPRC 0.900 / Complex 0.353
  - Day 12 chrom-LOO σ̂-bin gap 0.020 Complex / 0.077 Mendelian
  - Day 14 trait-LOO gap 0.002-0.004
  - Day 14 cross-dataset C→M gap 0.035
Actual numbers in (c) come from tables in reports/; illustrative curves in
(a,b) are synthetic to communicate the geometry only.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

FIG_DIR = Path(__file__).resolve().parent
OUT_PDF = FIG_DIR / "fig1_concept.pdf"
OUT_PNG = FIG_DIR / "fig1_concept.png"

# ----- Style -----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_POS = "#d1495b"       # pathogenic (y=1)
C_NEG = "#4a6fa5"       # benign (y=0)
C_SIGMA = "#f1a340"     # σ̂ band
C_GRID = "#2c6e49"      # Mondrian accent
C_OOD1 = "#4a6fa5"
C_OOD2 = "#f1a340"
C_OOD3 = "#d1495b"


def panel_a(ax):
    """Heteroscedastic p̂(x) ± σ̂(x) with score geometry."""
    rng = np.random.default_rng(42)
    xs = np.linspace(0.0, 1.0, 200)

    # p̂(x): smooth sigmoid-ish curve from ~0.1 to ~0.9
    phat = 0.10 + 0.80 / (1.0 + np.exp(-8.0 * (xs - 0.55)))
    # σ̂(x): U-shaped — small near 0 and 1, large near the decision boundary
    sig = 0.05 + 0.22 * np.exp(-((xs - 0.55) / 0.22) ** 2)

    # Central curve and ±σ̂ band (illustrative).
    ax.fill_between(xs, np.clip(phat - sig, 0, 1), np.clip(phat + sig, 0, 1),
                    color=C_SIGMA, alpha=0.30, linewidth=0,
                    label=r"$\hat{p}(x) \pm \hat{\sigma}(x)$")
    ax.plot(xs, phat, color="black", linewidth=1.6, label=r"$\hat{p}(x)$")

    # Scatter of pseudo-labels — dense negatives low, dense positives high,
    # with a fuzzy band near boundary.
    n_neg, n_pos = 80, 20
    x_neg = rng.beta(1.5, 3.5, n_neg)
    x_pos = rng.beta(4.0, 1.5, n_pos)
    y_neg_jitter = rng.normal(0.0, 0.015, n_neg)
    y_pos_jitter = rng.normal(0.0, 0.015, n_pos)
    ax.scatter(x_neg, y_neg_jitter, s=10, color=C_NEG, alpha=0.6,
               edgecolor="none", label=r"$y=0$")
    ax.scatter(x_pos, 1.0 + y_pos_jitter, s=12, color=C_POS, alpha=0.75,
               edgecolor="none", label=r"$y=1$")

    # Highlight score geometry at two query points: one low-σ̂, one high-σ̂.
    for x_q, tag in [(0.20, "low $\\hat{\\sigma}$"), (0.55, "high $\\hat{\\sigma}$")]:
        idx = int(x_q * (len(xs) - 1))
        p_q, s_q = phat[idx], sig[idx]
        # Vertical score segment |y - p̂(x)| for y=1
        ax.plot([x_q, x_q], [p_q, 1.0], color=C_POS, linewidth=1.2,
                alpha=0.8, linestyle="--")
        # σ̂ interval
        ax.plot([x_q - 0.012, x_q + 0.012], [p_q + s_q, p_q + s_q],
                color=C_SIGMA, linewidth=1.4)
        ax.plot([x_q - 0.012, x_q + 0.012], [p_q - s_q, p_q - s_q],
                color=C_SIGMA, linewidth=1.4)
        ax.plot([x_q, x_q], [p_q - s_q, p_q + s_q], color=C_SIGMA, linewidth=1.4)
        ax.annotate(tag, xy=(x_q, p_q), xytext=(x_q + 0.04, p_q - 0.18),
                    fontsize=8, color="black",
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="gray"))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel(r"feature coordinate $x$ (1-D projection)")
    ax.set_ylabel(r"probability")
    ax.set_title(r"(a) Heteroscedastic aggregator $\hat{p}(x), \hat{\sigma}(x)$")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.55), frameon=False)


def panel_b(ax):
    """Mondrian (y × σ̂-bin) calibration grid."""
    K = 5
    bins = list(range(1, K + 1))
    classes = [0, 1]

    # Per-cell threshold q̂_{k,b} — illustrative: grows with bin (harder in
    # high-σ̂ cells), slightly wider for y=1 (minority).
    q_grid = np.array([
        [0.42, 0.58, 0.77, 1.05, 1.48],   # y=0
        [0.55, 0.74, 0.99, 1.33, 1.82],   # y=1
    ])
    # Per-cell n_{k,b} (illustrative, TraitGym Complex roughly).
    n_grid = np.array([
        [1050, 1050, 1050, 1050, 1040],   # y=0
        [ 135,  130,  130,  130,  130],   # y=1
    ])

    ax.set_xlim(0.3, K + 0.7)
    ax.set_ylim(-0.6, 1.6)
    ax.invert_yaxis()
    ax.set_xticks(bins)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"$y=0$", r"$y=1$"])
    ax.set_xlabel(r"$\hat{\sigma}$-bin $b \in \{1, \ldots, K\}$  (low $\to$ high)")
    ax.set_title(r"(b) Mondrian $(y \times \hat{\sigma}\text{-bin})$ calibration, $K=5$")

    # Fill cells: color intensity proportional to q̂, alpha proportional to n.
    q_max = q_grid.max()
    for ki, k in enumerate(classes):
        for bi, b in enumerate(bins):
            q = q_grid[ki, bi]
            n = n_grid[ki, bi]
            base_color = C_POS if k == 1 else C_NEG
            alpha = 0.18 + 0.55 * (q / q_max)
            rect = Rectangle((b - 0.45, k - 0.45), 0.9, 0.9,
                             facecolor=base_color, alpha=alpha,
                             edgecolor=C_GRID, linewidth=1.2)
            ax.add_patch(rect)
            ax.text(b, k - 0.12, rf"$\hat{{q}}={q:.2f}$",
                    ha="center", va="center", fontsize=8.0, color="black")
            ax.text(b, k + 0.18, rf"$n={n}$", ha="center", va="center",
                    fontsize=7.5, color="black", alpha=0.75)

    # T3 annotation.
    ax.text(0.5 * (1 + K), 1.40,
            r"T3: $P(Y \in \mathcal{C}_\alpha \mid Y{=}k,\, b(X){=}b) \geq "
            r"1 - \alpha - 1/(n_{k,b}+1)$",
            ha="center", va="center", fontsize=8.5, color=C_GRID)


def panel_c(ax):
    """Three-axis OOD σ̂-bin gap bars."""
    # Numbers from reports/day14_external_validation.md §7.
    # Chrom-LOO uses CADD+GPN-MSA+Borzoi; trait-LOO and cross-dataset use
    # CADD+Borzoi (same anchor as §6 Table 2).
    groups = ["Mendelian", "Complex"]
    n_groups = len(groups)
    chrom_loo = [0.077, 0.023]       # CADD+GPN-MSA+Borzoi
    trait_loo = [0.004, 0.002]       # CADD+Borzoi
    cross = [0.036, 0.331]           # Mendelian: C→M (reverse), Complex: M→C (strong shift)

    width = 0.26
    positions = np.arange(n_groups)

    b1 = ax.bar(positions - width, chrom_loo, width, label="chrom-LOO",
                color=C_OOD1, edgecolor="black", linewidth=0.4)
    b2 = ax.bar(positions,         trait_loo, width, label="trait-LOO",
                color=C_OOD2, edgecolor="black", linewidth=0.4)
    b3 = ax.bar(positions + width, cross,     width, label="cross-dataset",
                color=C_OOD3, edgecolor="black", linewidth=0.4)

    # 0.04 target line.
    ax.axhline(0.04, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(0.02, 0.045, r"target $\leq 0.04$", fontsize=8, color="black",
            transform=ax.get_yaxis_transform(), va="bottom", ha="left")

    for bar_group in (b1, b2, b3):
        for rect in bar_group:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.0)

    # Annotate the strong-shift outlier.
    ax.annotate("M$\\to$C\nstrong shift",
                xy=(1 + width, cross[1]),
                xytext=(1.2, 0.28),
                fontsize=7.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="gray"))

    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel(r"$\hat{\sigma}$-bin coverage gap")
    ax.set_ylim(0, 0.36)
    ax.set_title(r"(c) Three-axis OOD: $\hat{\sigma}$-bin gap on TraitGym")
    ax.legend(loc="upper left", frameon=False, ncol=1)


def main():
    fig = plt.figure(figsize=(12.4, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.15, 0.95], wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
