"""Figure 2 — per-chromosome coverage honesty check (hybrid layout).

Single-row 1x2 layout, one panel per dataset (Mendelian, Complex). Each panel
shows a forest of per-chromosome HCCP coverage with 95% within-chrom bootstrap
CIs (B = 1000), but the marker size encodes log(n_c) — small chromosomes get
small markers, large get large markers. This compresses three pieces of
information into a single visual unit:

  * forest distribution (each chromosome's coverage point + CI bar)
  * sample-size precision (marker size ∝ log n_c)
  * binomial-noise narrative (small markers + wide CIs co-locate at low cov)

A red dashed vertical line marks the T3' robust lower bound (cov >= 0.59,
derived from worst-cell delta_TV <= 0.31 on Mendelian; see Appendix A.6).
The +/- 0.03 target band shades the operational ideal zone.

Idiom precedents: forest + per-study marker size in Cochrane meta-analysis
plots; the simplification w.r.t. an explicit funnel/scatter panel preserves
the binomial-noise mechanism via marker size while saving vertical space.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = Path(__file__).resolve().parent
OUT_PDF = FIG_DIR / "fig9_per_chrom.pdf"
OUT_PNG = FIG_DIR / "fig9_per_chrom.png"

DATASETS = {
    "Mendelian": REPO_ROOT / "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/conformal_hetero_scores.parquet",
    "Complex":   REPO_ROOT / "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/conformal_hetero_scores.parquet",
}

TARGET = 0.90
ALPHA = 0.10
B = 1000
SEED = 42
T3PRIME_LB = 0.59  # 1 - alpha - bar_delta_TV; Mendelian worst cell delta_TV <= 0.31

C_HCCP = "#d1495b"
C_BAND = "#a4c293"
C_T3P  = "#8b3a3a"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
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
        idx = rng.integers(0, n, size=(B, n))
        boot = hccp[idx].mean(axis=1)
        ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
        rows.append({"chrom": str(chrom), "n": n,
                     "cov_hccp": hccp.mean(),
                     "ci_lo": ci_lo, "ci_hi": ci_hi})
    return pd.DataFrame(rows).sort_values(
        "chrom", key=lambda s: s.map(chrom_sort_key)).reset_index(drop=True)


def draw_hybrid(ax, stats: pd.DataFrame, dataset: str, *, panel_letter: str):
    s = stats.sort_values("cov_hccp", ascending=True).reset_index(drop=True)
    y = np.arange(len(s))

    ax.axvspan(TARGET - 0.03, TARGET + 0.03, color=C_BAND, alpha=0.18,
               linewidth=0, zorder=0)
    ax.axvline(TARGET, color="#222", linestyle="--", linewidth=0.8, zorder=1)

    ax.axvline(T3PRIME_LB, color=C_T3P, linestyle=(0, (4, 2)),
               linewidth=1.0, zorder=1, alpha=0.75)
    ax.text(T3PRIME_LB - 0.005, len(s) - 0.4,
            fr"T3$'$ LB $\geq {T3PRIME_LB:.2f}$",
            color=C_T3P, fontsize=6.6, ha="right", va="top", style="italic")

    worst_idx = (s.cov_hccp - TARGET).abs().idxmax()
    for i, row in s.iterrows():
        is_worst = i == worst_idx
        lw = 2.0 if is_worst else 1.2
        ax.plot([row.ci_lo, row.ci_hi], [i, i],
                color=C_HCCP, linewidth=lw, alpha=0.85, zorder=2)

    log_n = np.log10(s.n.values.astype(float))
    s_norm = (log_n - log_n.min()) / max(log_n.max() - log_n.min(), 1e-6)
    msizes = 12 + 80 * s_norm
    ax.scatter(s.cov_hccp, y, s=msizes,
               facecolor=C_HCCP, edgecolor="white",
               linewidth=0.7, zorder=3)

    worst = s.loc[worst_idx]
    if abs(worst.cov_hccp - TARGET) >= 0.05:
        ax.annotate(f"chr{worst.chrom}: cov $=$ {worst.cov_hccp:.2f}, $n=${int(worst.n)}",
                    xy=(worst.cov_hccp, worst_idx),
                    xytext=(0.43, worst_idx + 0.35),
                    fontsize=6.4, color="#333",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#777"))

    ax.set_yticks(y)
    ax.set_yticklabels([f"chr{c}" for c in s.chrom], fontsize=6.5)
    ax.set_xlim(0.40, 1.05)
    ax.set_xlabel(r"empirical coverage  (marker size $\propto \log n_c$)")
    ax.set_title(f"({panel_letter}) {dataset}")
    ax.tick_params(axis="y", length=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.4, alpha=0.5)

    if dataset == "Mendelian":
        n_examples = [50, 200, 500]
        handles = []
        for ne in n_examples:
            log_ne = np.log10(ne)
            s_e = (log_ne - log_n.min()) / max(log_n.max() - log_n.min(), 1e-6)
            ms = 12 + 80 * s_e
            handles.append(Line2D([0], [0], marker="o", color=C_HCCP,
                                   linestyle="", markersize=np.sqrt(ms),
                                   mfc=C_HCCP, mec="white",
                                   label=f"$n_c={ne}$"))
        ax.legend(handles=handles, loc="lower right", frameon=True,
                  framealpha=0.94, fontsize=6.4, borderpad=0.4,
                  handletextpad=0.4, labelspacing=0.7)


def main():
    rng = np.random.default_rng(SEED)
    stats_per_dataset = {}
    for name, path in DATASETS.items():
        df = pd.read_parquet(path)
        stats_per_dataset[name] = per_chrom_stats(df, rng)
        s = stats_per_dataset[name]
        worst = s.iloc[(s.cov_hccp - TARGET).abs().argmax()]
        print(f"{name}: n_chrom={len(s)} | worst chr{worst.chrom} cov={worst.cov_hccp:.3f} "
              f"(CI [{worst.ci_lo:.3f}, {worst.ci_hi:.3f}], n={int(worst.n)})")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    draw_hybrid(axes[0], stats_per_dataset["Mendelian"], "Mendelian",
                panel_letter="a")
    draw_hybrid(axes[1], stats_per_dataset["Complex"],   "Complex",
                panel_letter="b")

    fig.suptitle(r"HCCP per-chromosome coverage  ---  $K{=}5$, $\alpha{=}0.10$, target $0.90$.  "
                 r"Marker size $\propto \log n_c$; red dashed = T3$'$ robust lower bound.",
                 fontsize=9.0, y=0.995)
    fig.tight_layout()

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=200)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
