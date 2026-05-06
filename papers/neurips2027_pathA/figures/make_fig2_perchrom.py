"""Figure 2 — per-chromosome coverage honesty check, redesigned.

Two-row, two-column layout:
  Top row: (a) Mendelian, (b) Complex — sorted forest plot.
    Each row = one chromosome (sorted by HCCP coverage). HCCP point + 95%
    bootstrap CI (B = 1000 over within-chrom variant resamples). Open marker
    on the same row = baseline split-CP coverage. Vertical reference line at
    target = 0.90, light shaded +/- 0.03 band.

  Bottom row: (c) Mendelian, (d) Complex — coverage vs log(n) scatter.
    x = log10(n_per_chrom), y = cov - 0.90 (HCCP), marker = chromosome label.
    Overlaid: 95% binomial-finite-sample envelope sqrt(p(1-p)/n) * 1.96.
    chr6 etc. should sit inside the envelope -> heterogeneity is sample-size,
    motivates T3' robust certificate.

Idiom precedents:
  - Forest + CI: Romano et al. "With Malice Toward None" (arxiv 2007.07365)
                 + Foygel-Barber jackknife+ (1905.02928 Fig 5).
  - Coverage vs n scatter: ProteinGym (Notin et al. NeurIPS 2023).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = Path(__file__).resolve().parent
OUT_PDF = FIG_DIR / "fig9_per_chrom.pdf"  # keep filename for LaTeX continuity
OUT_PNG = FIG_DIR / "fig9_per_chrom.png"

DATASETS = {
    "Mendelian": REPO_ROOT / "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/conformal_hetero_scores.parquet",
    "Complex":   REPO_ROOT / "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/conformal_hetero_scores.parquet",
}

TARGET = 0.90
ALPHA = 0.10
B = 1000
SEED = 42

C_HCCP = "#d1495b"
C_BASE = "#4a6fa5"
C_BAND = "#a4c293"
C_ENV  = "#777777"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8,
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
        cov_h = hccp.mean()
        cov_s = split.mean()
        # Within-chrom bootstrap of HCCP coverage.
        idx = rng.integers(0, n, size=(B, n))
        boot = hccp[idx].mean(axis=1)
        ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
        rows.append({
            "chrom": str(chrom),
            "n": n,
            "cov_hccp": cov_h,
            "cov_split": cov_s,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })
    return pd.DataFrame(rows).sort_values(
        "chrom", key=lambda s: s.map(chrom_sort_key)
    ).reset_index(drop=True)


def draw_forest(ax, stats: pd.DataFrame, dataset: str):
    # Sort by HCCP coverage so worst-to-best reads top-down.
    s = stats.sort_values("cov_hccp", ascending=True).reset_index(drop=True)
    y = np.arange(len(s))

    # Target band shading.
    ax.axvspan(TARGET - 0.03, TARGET + 0.03, color=C_BAND, alpha=0.18,
               linewidth=0, zorder=0)
    ax.axvline(TARGET, color="#222222", linestyle="--", linewidth=0.8, zorder=1)

    # Identify the worst chromosome (largest |cov - target|) for emphasis.
    worst_idx = (s.cov_hccp - TARGET).abs().idxmax()

    # CI lines + HCCP points.
    for i, row in s.iterrows():
        is_worst = i == worst_idx
        lw = 2.2 if is_worst else 1.4
        alpha = 1.0 if is_worst else 0.85
        ax.plot([row.ci_lo, row.ci_hi], [i, i],
                color=C_HCCP, linewidth=lw, alpha=alpha, zorder=2)
    sizes = np.where(s.index == worst_idx, 50, 22)
    ax.scatter(s.cov_hccp, y, marker="o", s=sizes,
               facecolor=C_HCCP, edgecolor="white", linewidth=0.6,
               zorder=3, label="HCCP (95% CI)")

    # Baseline split-CP marker — small ghost square so it doesn't compete.
    ax.scatter(s.cov_split, y, marker="s", s=10,
               facecolor="white", edgecolor=C_BASE, linewidth=0.8,
               alpha=0.75, zorder=2, label="split CP (per-chrom)")

    # Annotate the worst chromosome's CI to make T3' narrative visible.
    worst = s.loc[worst_idx]
    if abs(worst.cov_hccp - TARGET) >= 0.05:
        ax.annotate(
            f"chr{worst.chrom}: cov $=$ {worst.cov_hccp:.2f},\n"
            f"95\\% CI $=$ [{worst.ci_lo:.2f}, {worst.ci_hi:.2f}]\n"
            f"crosses target  (n $=$ {int(worst.n)})",
            xy=(worst.cov_hccp, worst_idx),
            xytext=(0.50, worst_idx + 3.0),
            fontsize=6.4, color="#333333",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666"),
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"chr{c}" for c in s.chrom], fontsize=6.5)
    ax.set_xlim(0.40, 1.05)
    ax.set_xlabel("empirical coverage")
    ax.set_title(f"({'a' if dataset == 'Mendelian' else 'b'}) {dataset} — per-chrom forest")
    ax.tick_params(axis="y", length=0)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.5)
    if dataset == "Mendelian":
        ax.legend(loc="lower right", frameon=True, framealpha=0.94,
                  borderpad=0.4, handletextpad=0.4)


def draw_nscatter(ax, stats: pd.DataFrame, dataset: str):
    # Theoretical 95% binomial envelope: cov_hat ~ Bin(n, target)/n
    # so SE = sqrt(p(1-p)/n), p = TARGET.
    n_grid = np.logspace(np.log10(stats.n.min() * 0.7),
                         np.log10(stats.n.max() * 1.4), 200)
    se = np.sqrt(TARGET * (1 - TARGET) / n_grid)
    ax.fill_between(n_grid, -1.96 * se, 1.96 * se,
                    color=C_ENV, alpha=0.12, linewidth=0,
                    label=r"$\pm 1.96 \sqrt{p(1-p)/n}$ envelope")
    ax.plot(n_grid, 1.96 * se, color=C_ENV, linestyle=":", linewidth=0.6)
    ax.plot(n_grid, -1.96 * se, color=C_ENV, linestyle=":", linewidth=0.6)

    # Scatter points; size emphasizes worst-deviation chromosomes.
    dev = stats.cov_hccp - TARGET
    abs_dev = dev.abs()
    sizes = 30 + 80 * (abs_dev / max(abs_dev.max(), 0.05))
    ax.scatter(stats.n, dev, s=sizes, facecolor=C_HCCP, edgecolor="white",
               linewidth=0.6, zorder=3)
    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=0.6, alpha=0.6)

    # Annotate only the top 3 outliers (so labels don't crowd the panel).
    top3 = abs_dev.sort_values(ascending=False).head(3).index
    for idx in top3:
        row = stats.loc[idx]
        ax.annotate(f"chr{row.chrom}", (row.n, dev.loc[idx]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=6.8, color="#333333", weight="bold")

    ax.set_xscale("log")
    ax.set_xlabel(r"variants per chromosome  $n_c$  (log)")
    ax.set_ylabel(r"$\mathrm{cov}_c - 0.90$")
    ax.set_title(f"({'c' if dataset == 'Mendelian' else 'd'}) {dataset} — coverage vs $n_c$")
    ax.set_ylim(-0.18, 0.18)
    for s_ in ["top", "right"]:
        ax.spines[s_].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    if dataset == "Mendelian":
        ax.legend(loc="lower right", frameon=True, framealpha=0.94,
                  borderpad=0.4, handletextpad=0.4, fontsize=6.4)


def main():
    rng = np.random.default_rng(SEED)
    stats_per_dataset = {}
    for name, path in DATASETS.items():
        df = pd.read_parquet(path)
        stats_per_dataset[name] = per_chrom_stats(df, rng)
        # Quick sanity print.
        s = stats_per_dataset[name]
        worst = s.iloc[(s.cov_hccp - TARGET).abs().argmax()]
        print(f"{name}: n_chrom={len(s)} | worst chr{worst.chrom} cov={worst.cov_hccp:.3f} "
              f"(CI [{worst.ci_lo:.3f}, {worst.ci_hi:.3f}], n={int(worst.n)})")

    fig = plt.figure(figsize=(10.0, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.65],
                          hspace=0.45, wspace=0.30,
                          top=0.96, bottom=0.075, left=0.075, right=0.985)
    ax_fa = fig.add_subplot(gs[0, 0])
    ax_fb = fig.add_subplot(gs[0, 1])
    ax_nc = fig.add_subplot(gs[1, 0])
    ax_nd = fig.add_subplot(gs[1, 1])

    draw_forest(ax_fa, stats_per_dataset["Mendelian"], "Mendelian")
    draw_forest(ax_fb, stats_per_dataset["Complex"],   "Complex")
    draw_nscatter(ax_nc, stats_per_dataset["Mendelian"], "Mendelian")
    draw_nscatter(ax_nd, stats_per_dataset["Complex"],   "Complex")

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
