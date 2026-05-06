"""Generate 4 candidate redesigns of fig9_per_chrom for user selection.

Outputs to /tmp/fig9_variant_{A,B,C,D}.png (NOT the canonical figures dir).
After the user picks, the chosen layout is ported to make_fig2_perchrom.py.

A: forest only (drop bottom scatter), 1x2
B: scatter/funnel only (drop top forest), 1x2
C: polished 2x2 in place (fix split-CP overlay, K=5 label, T3' line, chr6 annotation)
D: hybrid 1x2 with marker size encoding log(n_c)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
DATASETS = {
    "Mendelian": REPO / "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/conformal_hetero_scores.parquet",
    "Complex":   REPO / "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/conformal_hetero_scores.parquet",
}
TARGET = 0.90
ALPHA = 0.10
B = 1000
SEED = 42
T3PRIME_LB = 0.59  # 1 - alpha - delta_TV ≈ 0.90 - 0.31 = 0.59 (Mendelian worst-cell)

C_HCCP = "#d1495b"
C_BASE = "#4a6fa5"
C_BAND = "#a4c293"
C_ENV  = "#777777"
C_T3P  = "#8b3a3a"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def chrom_sort_key(c: str):
    try:
        return (0, int(c))
    except (TypeError, ValueError):
        return (1, str(c))


def per_chrom_stats(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for chrom, g in df.groupby("chrom"):
        n = len(g)
        hccp = g["mondrian_covered"].astype(float).to_numpy()
        split = g["homosc_covered"].astype(float).to_numpy()
        idx = rng.integers(0, n, size=(B, n))
        boot = hccp[idx].mean(axis=1)
        ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
        rows.append({"chrom": str(chrom), "n": n,
                     "cov_hccp": hccp.mean(), "cov_split": split.mean(),
                     "ci_lo": ci_lo, "ci_hi": ci_hi})
    return pd.DataFrame(rows).sort_values(
        "chrom", key=lambda s: s.map(chrom_sort_key)).reset_index(drop=True)


def load_all_stats():
    rng = np.random.default_rng(SEED)
    return {name: per_chrom_stats(pd.read_parquet(p), rng)
            for name, p in DATASETS.items()}


# -----------------------------------------------------------------------------
def draw_forest(ax, stats, dataset, *, show_split=True, polish=False,
                annotate_chr6_inline=True):
    s = stats.sort_values("cov_hccp", ascending=True).reset_index(drop=True)
    y = np.arange(len(s))
    ax.axvspan(TARGET - 0.03, TARGET + 0.03, color=C_BAND, alpha=0.18,
               linewidth=0, zorder=0)
    ax.axvline(TARGET, color="#222", linestyle="--", linewidth=0.8, zorder=1)
    if polish:
        ax.axvline(T3PRIME_LB, color=C_T3P, linestyle=(0, (4, 2)),
                   linewidth=0.9, zorder=1, alpha=0.7)
        ax.text(T3PRIME_LB + 0.005, len(s) - 0.5,
                fr"T3$'$ LB $\geq {T3PRIME_LB:.2f}$",
                color=C_T3P, fontsize=6.4, va="top", style="italic")

    worst_idx = (s.cov_hccp - TARGET).abs().idxmax()
    for i, row in s.iterrows():
        is_worst = i == worst_idx
        lw = 2.2 if is_worst else 1.4
        ax.plot([row.ci_lo, row.ci_hi], [i, i], color=C_HCCP, lw=lw,
                alpha=1.0 if is_worst else 0.85, zorder=2)
    sizes = np.where(s.index == worst_idx, 50, 22)
    ax.scatter(s.cov_hccp, y, marker="o", s=sizes, facecolor=C_HCCP,
               edgecolor="white", lw=0.6, zorder=3, label="HCCP (95% CI)")
    if show_split:
        ax.scatter(s.cov_split, y, marker="s", s=10, facecolor="white",
                   edgecolor=C_BASE, lw=0.8, alpha=0.75, zorder=2,
                   label="split CP (control)")

    if annotate_chr6_inline:
        worst = s.loc[worst_idx]
        if abs(worst.cov_hccp - TARGET) >= 0.05:
            ax.annotate(
                f"chr{worst.chrom}: cov $=$ {worst.cov_hccp:.2f},\n"
                f"95% CI $=$ [{worst.ci_lo:.2f}, {worst.ci_hi:.2f}]\n"
                f"crosses target ($n=${int(worst.n)})",
                xy=(worst.cov_hccp, worst_idx),
                xytext=(0.46, worst_idx + 3.5),
                fontsize=6.4, color="#333",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#666"))

    ax.set_yticks(y)
    ax.set_yticklabels([f"chr{c}" for c in s.chrom], fontsize=6.5)
    ax.set_xlim(0.40, 1.05)
    ax.set_xlabel("empirical coverage")
    title_letter = "a" if dataset == "Mendelian" else "b"
    ax.set_title(f"({title_letter}) {dataset} — per-chrom forest")
    ax.tick_params(axis="y", length=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.5)
    if dataset == "Mendelian":
        ax.legend(loc="lower right", frameon=True, framealpha=0.94,
                  borderpad=0.4, handletextpad=0.4, fontsize=6.6)


def draw_scatter(ax, stats, dataset, *, label_chrom=False, polish=False,
                 panel_letter="c"):
    n_grid = np.logspace(np.log10(stats.n.min() * 0.7),
                         np.log10(stats.n.max() * 1.4), 200)
    se = np.sqrt(TARGET * (1 - TARGET) / n_grid)
    ax.fill_between(n_grid, -1.96 * se, 1.96 * se, color=C_ENV, alpha=0.12,
                    linewidth=0,
                    label=r"$\pm 1.96\sqrt{p(1{-}p)/n}$ binomial envelope")
    ax.plot(n_grid, 1.96 * se, color=C_ENV, ls=":", lw=0.6)
    ax.plot(n_grid, -1.96 * se, color=C_ENV, ls=":", lw=0.6)

    if polish:
        ax.axhline(T3PRIME_LB - TARGET, color=C_T3P, ls=(0, (4, 2)), lw=0.9,
                   alpha=0.7)
        ax.text(stats.n.max() * 0.9, T3PRIME_LB - TARGET + 0.005,
                fr"T3$'$ LB $\geq -0.31$",
                color=C_T3P, fontsize=6.4, ha="right", va="bottom",
                style="italic")

    dev = stats.cov_hccp - TARGET
    abs_dev = dev.abs()
    sizes = 30 + 80 * (abs_dev / max(abs_dev.max(), 0.05))
    ax.scatter(stats.n, dev, s=sizes, facecolor=C_HCCP, edgecolor="white",
               lw=0.6, zorder=3)
    ax.axhline(0.0, color="#222", ls="--", lw=0.6, alpha=0.6)

    top3 = abs_dev.sort_values(ascending=False).head(3 if not label_chrom else 5).index
    for idx in top3:
        row = stats.loc[idx]
        ax.annotate(f"chr{row.chrom}", (row.n, dev.loc[idx]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=6.8, color="#333", weight="bold")

    if label_chrom:
        # Label every chromosome (used in funnel-only variant B)
        rest = abs_dev.sort_values(ascending=False).iloc[3:].index
        for idx in rest:
            row = stats.loc[idx]
            ax.annotate(f"chr{row.chrom}", (row.n, dev.loc[idx]),
                        xytext=(4, -8), textcoords="offset points",
                        fontsize=5.5, color="#888")

    ax.set_xscale("log")
    ax.set_xlabel(r"variants per chromosome $n_c$")
    ax.set_ylabel(r"$\mathrm{cov}_c - 0.90$")
    ax.set_ylim(-0.18, 0.18)
    ax.set_title(f"({panel_letter}) {dataset} — coverage vs $n_c$")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    if dataset == "Mendelian":
        ax.legend(loc="lower right", frameon=True, framealpha=0.94,
                  borderpad=0.4, handletextpad=0.4, fontsize=6.4)


def draw_hybrid(ax, stats, dataset):
    """Single panel: forest-style with marker size = log(n_c)."""
    s = stats.sort_values("cov_hccp", ascending=True).reset_index(drop=True)
    y = np.arange(len(s))
    ax.axvspan(TARGET - 0.03, TARGET + 0.03, color=C_BAND, alpha=0.18,
               linewidth=0, zorder=0)
    ax.axvline(TARGET, color="#222", ls="--", lw=0.8, zorder=1)

    worst_idx = (s.cov_hccp - TARGET).abs().idxmax()
    for i, row in s.iterrows():
        is_worst = i == worst_idx
        lw = 2.0 if is_worst else 1.2
        ax.plot([row.ci_lo, row.ci_hi], [i, i], color=C_HCCP, lw=lw,
                alpha=0.85, zorder=2)

    # Marker size proportional to log(n_c)
    log_n = np.log10(s.n.values)
    s_norm = (log_n - log_n.min()) / max(log_n.max() - log_n.min(), 1e-6)
    msizes = 12 + 80 * s_norm  # range 12-92
    ax.scatter(s.cov_hccp, y, s=msizes, facecolor=C_HCCP, edgecolor="white",
               lw=0.7, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([f"chr{c}" for c in s.chrom], fontsize=6.5)
    ax.set_xlim(0.40, 1.05)
    ax.set_xlabel("empirical coverage  (marker size $\\propto \\log n_c$)")
    title_letter = "a" if dataset == "Mendelian" else "b"
    ax.set_title(f"({title_letter}) {dataset}")
    ax.tick_params(axis="y", length=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", ls=":", lw=0.4, alpha=0.5)

    # Marker-size legend
    if dataset == "Mendelian":
        n_examples = [50, 200, 500]
        handles = []
        for ne in n_examples:
            log_ne = np.log10(ne)
            s_e = (log_ne - log_n.min()) / max(log_n.max() - log_n.min(), 1e-6)
            ms = 12 + 80 * s_e
            h = Line2D([0], [0], marker="o", color=C_HCCP, ls="",
                       markersize=np.sqrt(ms), mfc=C_HCCP, mec="white",
                       label=f"$n_c={ne}$")
            handles.append(h)
        ax.legend(handles=handles, loc="lower right", frameon=True,
                  framealpha=0.94, fontsize=6.2, borderpad=0.4,
                  handletextpad=0.4, labelspacing=0.6)


# -----------------------------------------------------------------------------
def variant_A(stats):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    draw_forest(axes[0], stats["Mendelian"], "Mendelian", polish=True)
    draw_forest(axes[1], stats["Complex"],   "Complex",   polish=True,
                annotate_chr6_inline=False)
    fig.suptitle(r"HCCP per-chromosome coverage  —  $K{=}5$, $\alpha{=}0.10$, target $0.90$",
                 fontsize=9.5, y=0.995)
    fig.tight_layout()
    fig.savefig("/tmp/fig9_variant_A.png", dpi=150)
    plt.close(fig)


def variant_B(stats):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.5))
    draw_scatter(axes[0], stats["Mendelian"], "Mendelian",
                 label_chrom=True, polish=True, panel_letter="a")
    draw_scatter(axes[1], stats["Complex"],   "Complex",
                 label_chrom=True, polish=True, panel_letter="b")
    fig.suptitle(r"HCCP per-chromosome coverage vs $n_c$  —  $K{=}5$, $\alpha{=}0.10$, target $0.90$",
                 fontsize=9.5, y=0.995)
    fig.tight_layout()
    fig.savefig("/tmp/fig9_variant_B.png", dpi=150)
    plt.close(fig)


def variant_C(stats):
    fig = plt.figure(figsize=(10.0, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.65],
                          hspace=0.50, wspace=0.30,
                          top=0.94, bottom=0.075, left=0.075, right=0.985)
    ax_fa = fig.add_subplot(gs[0, 0])
    ax_fb = fig.add_subplot(gs[0, 1])
    ax_nc = fig.add_subplot(gs[1, 0])
    ax_nd = fig.add_subplot(gs[1, 1])
    draw_forest(ax_fa, stats["Mendelian"], "Mendelian", polish=True)
    draw_forest(ax_fb, stats["Complex"],   "Complex",   polish=True,
                annotate_chr6_inline=False)
    draw_scatter(ax_nc, stats["Mendelian"], "Mendelian", polish=True)
    draw_scatter(ax_nd, stats["Complex"],   "Complex",   polish=True,
                 panel_letter="d")
    fig.suptitle(r"HCCP per-chromosome coverage  —  $K{=}5$, $\alpha{=}0.10$, target $0.90$  "
                 r"(T3$'$ certificate visualised in red)",
                 fontsize=9.5, y=0.985)
    fig.savefig("/tmp/fig9_variant_C.png", dpi=150)
    plt.close(fig)


def variant_D(stats):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    draw_hybrid(axes[0], stats["Mendelian"], "Mendelian")
    draw_hybrid(axes[1], stats["Complex"],   "Complex")
    fig.suptitle(r"HCCP per-chromosome coverage  —  $K{=}5$, marker size $\propto \log n_c$  (compact 1$\times$2)",
                 fontsize=9.5, y=0.995)
    fig.tight_layout()
    fig.savefig("/tmp/fig9_variant_D.png", dpi=150)
    plt.close(fig)


def main():
    stats = load_all_stats()
    for name, s in stats.items():
        worst = s.iloc[(s.cov_hccp - TARGET).abs().argmax()]
        print(f"{name}: worst chr{worst.chrom} cov={worst.cov_hccp:.3f} "
              f"CI=[{worst.ci_lo:.3f}, {worst.ci_hi:.3f}] n={int(worst.n)}")
    variant_A(stats); print("/tmp/fig9_variant_A.png")
    variant_B(stats); print("/tmp/fig9_variant_B.png")
    variant_C(stats); print("/tmp/fig9_variant_C.png")
    variant_D(stats); print("/tmp/fig9_variant_D.png")


if __name__ == "__main__":
    main()
