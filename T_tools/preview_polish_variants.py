"""Generate polish variants for fig_asl_audit and fig_bootstrap_density.

Outputs to R_raw/figpolish_previews/. After user picks, the chosen polish
level is ported to the production scripts (T_tools/asl_audit.py and
papers/.../make_fig4_bootstrap_cloud.py).

Naming:
  asl_audit_X.png  --- minimal: y-axis clip + slope inset on (b)
  asl_audit_Y.png  --- X + statistics box reorganization on (a)
  bootstrap_X.png  --- minimal: Mendelian CI text repositioned
  bootstrap_Y.png  --- X + bigger star marker + shared y-axis between panels
"""
from __future__ import annotations

from pathlib import Path
import json
import sys as _sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parents[1]
PREVIEW = REPO / "R_raw" / "figpolish_previews"
PREVIEW.mkdir(parents=True, exist_ok=True)

_sys.path.insert(0, str(REPO))
from T_tools.paper_style import apply_paper_style  # noqa: E402
apply_paper_style()

# ----------------------------------------------------------------------------
# fig_asl_audit polish: import data extraction from asl_audit module
# ----------------------------------------------------------------------------
from T_tools.asl_audit import (
    lcls_pairwise_ks,
    _ks_noise_floor,
    _noise_adjusted_ratio,
)

DATASET = "mendelian"
SCORES_PARQUET = REPO / f"outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_{DATASET}_abs_mondrian/conformal_hetero_scores.parquet"
ASL_JSON = REPO / f"R_raw/asl_audit/{DATASET}.json"
BETA_LCLS = 2.97  # Mendelian
N_BINS = 30
MIN_PER_BIN = 20


def _asl_pairs():
    df = pd.read_parquet(SCORES_PARQUET)
    sigma = df["sigma"].astype(float).to_numpy()
    y = df["label"].astype(int).to_numpy()
    p = df["p_hat"].astype(float).to_numpy()
    pairs = lcls_pairwise_ks(p, sigma, y, n_bins=N_BINS, min_per_bin=MIN_PER_BIN)
    return pairs


def _draw_asl(ax_h, ax_s, pairs, *, polish_level):
    g = pairs["sigma_gaps"]
    k = pairs["ks_stats"]
    floor = _ks_noise_floor(pairs["n_a"], pairs["n_b"], alpha=0.05)
    ratios = k / np.maximum(g, 1e-12)
    ratios_clean = _noise_adjusted_ratio(pairs, alpha_noise=0.05)
    median = float(np.median(ratios))
    p95 = float(np.quantile(ratios, 0.95))
    p99 = float(np.quantile(ratios, 0.99))
    rmax = float(ratios.max())
    p95_clean = float(np.quantile(ratios_clean, 0.95))

    # ---- (a) Histogram of per-pair ratios ----
    ax_h.hist(ratios, bins=40, color="#4477AA", alpha=0.85,
              edgecolor="white", linewidth=0.4)
    x_cap = p99 * 1.20
    ax_h.set_xlim(0, x_cap)

    if polish_level == "Y":
        # 2-column compact statistics box (left = legend lines, right = values)
        # No vertical lines for stats; just markers at top.
        for x, c, ls in [
            (median,     "#222222", "--"),
            (p95,        "#DDAA33", "--"),
            (p99,        "#EE6677", "-."),
            (BETA_LCLS,  "#117733", ":"),
        ]:
            ax_h.axvline(x, color=c, linestyle=ls, linewidth=1.2)
        # Compact statistics box top-right: 4 lines, single column, smaller
        stats_text = (
            fr"$\hat L_F^\mathrm{{LCLS}}$ = {BETA_LCLS:.2f}" + "\n"
            fr"median = {median:.2f}" + "\n"
            fr"p95 = {p95:.2f}" + "\n"
            fr"p99 = {p99:.2f}" + "\n"
            fr"max = {rmax:.0f} (out of range)"
        )
        ax_h.text(0.98, 0.98, stats_text, transform=ax_h.transAxes,
                  ha="right", va="top", fontsize=6.8, family="serif",
                  bbox=dict(boxstyle="round,pad=0.32", fc="white",
                            ec="0.7", alpha=0.94))
    else:  # X (minimal): keep original layout
        for x, lab, c, ls in [
            (median,    f"median = {median:.2f}",                 "#222222", "--"),
            (p95,       f"p95 = {p95:.2f}",                       "#DDAA33", "--"),
            (p99,       f"p99 = {p99:.2f}",                       "#EE6677", "-."),
            (BETA_LCLS, fr"LCLS $\hat L_F$ = {BETA_LCLS:.2f}",    "#117733", ":"),
        ]:
            ax_h.axvline(x, color=c, linestyle=ls, linewidth=1.2, label=lab)
        ax_h.text(0.97, 0.97, f"max = {rmax:.1f}\n(KS-saturation;\nout of plotted range)",
                  transform=ax_h.transAxes, ha="right", va="top",
                  fontsize=7, color="#CC3311",
                  bbox=dict(boxstyle="round,pad=0.25", fc="white",
                            ec="#CC3311", lw=0.5, alpha=0.9))
        ax_h.legend(loc="upper left", fontsize=7, frameon=False,
                    bbox_to_anchor=(0.30, 0.98))

    ax_h.set_xlabel(r"per-pair ratio $\rho = \mathrm{KS} / |\bar\sigma_a - \bar\sigma_b|$")
    ax_h.set_ylabel("count")
    ax_h.set_title(f"(a) ratio distribution: {DATASET}", fontsize=10)
    ax_h.grid(axis="y", alpha=0.3)

    # ---- (b) KS vs sigma-gap scatter with slope reference ----
    clean_kk = np.maximum(k - floor, 0.0)
    is_viol = clean_kk > BETA_LCLS * g
    ax_s.scatter(g[~is_viol], k[~is_viol], s=8, c="#4477AA", alpha=0.55,
                 label=f"within LCLS bound (n={int((~is_viol).sum())})", edgecolor="none")
    ax_s.scatter(g[is_viol], k[is_viol], s=14, c="#CC3311", alpha=0.85,
                 label=f"above LCLS bound (n={int(is_viol.sum())})", edgecolor="none")
    grid_g = np.linspace(g.min(), g.max(), 200)
    ax_s.plot(grid_g, BETA_LCLS * grid_g, color="#117733", linewidth=1.5,
              label=fr"LCLS slope $\hat L_F$={BETA_LCLS:.2f}")
    ax_s.plot(grid_g, p95_clean * grid_g, color="#AA3377", linewidth=1.5, linestyle="--",
              label=fr"$L_F^{{(95\%)}}$={p95_clean:.2f} (noise-adj. p95)")

    # CRITICAL FIX (both X and Y): clip y-axis to [0, 1.05] since KS is
    # mathematically bounded in [0, 1]. The slope reference lines exit the
    # panel — annotate their off-axis behaviour with arrows + text.
    ax_s.set_ylim(0, 1.05)

    # Slope reference annotation: the LCLS slope hits y=1 at sigma-gap = 1/L_F
    # = 0.337; the p95 line hits y=1 at sigma-gap = 1/13.78 = 0.073. Mark the
    # exit points with small triangles and an annotation in the upper margin.
    x_lcls_exit = 1.0 / BETA_LCLS  # 0.337
    x_p95_exit = 1.0 / p95_clean   # 0.073
    if x_p95_exit < ax_s.get_xlim()[1]:
        ax_s.annotate("", xy=(x_p95_exit, 1.0), xytext=(x_p95_exit, 0.95),
                      arrowprops=dict(arrowstyle="->", color="#AA3377", lw=1.0))
        ax_s.text(x_p95_exit, 1.0, "exits y=1\nhere",
                  ha="center", va="bottom", fontsize=6.0, color="#AA3377")

    ax_s.set_xlabel(r"$\sigma$-gap $|\bar\sigma_a - \bar\sigma_b|$")
    ax_s.set_ylabel(r"KS distance $\hat F_{k,a}$ vs $\hat F_{k,b}$")
    ax_s.set_title(f"(b) KS vs $\\sigma$-gap: {DATASET}", fontsize=10)
    ax_s.legend(loc="upper right", fontsize=6.6, frameon=True, framealpha=0.92,
                borderpad=0.3, handletextpad=0.3)
    ax_s.grid(alpha=0.3)


def variant_asl(level):
    pairs = _asl_pairs()
    fig, (ax_h, ax_s) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    _draw_asl(ax_h, ax_s, pairs, polish_level=level)
    fig.tight_layout()
    out = PREVIEW / f"asl_audit_{level}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(out)


# ----------------------------------------------------------------------------
# fig_bootstrap_density polish
# ----------------------------------------------------------------------------
POINT_EST = {
    "Mendelian": {"cov_pos": 0.905, "gap": 0.005},
    "Complex":   {"cov_pos": 0.908, "gap": 0.002},
}
BOOTSTRAP_PATHS = {
    "Mendelian": REPO / "outputs/bootstrap_ci/CADD+GPN-MSA+Borzoi_mendelian.json",
    "Complex":   REPO / "outputs/bootstrap_ci/CADD+GPN-MSA+Borzoi_complex.json",
}
C_HCCP = "#d1495b"
C_POINT_X = "#1f4e79"
C_POINT_Y = "#f2c14e"  # bright yellow for high-contrast star


def _load_replicates(p):
    d = json.loads(Path(p).read_text())
    rep = d["replicates"]
    cov_pos = np.array([r["coverage_pos"] for r in rep])
    gap = np.array([r["sigma_bin_gap"] for r in rep])
    return cov_pos, gap


def _kde_contour(ax, x, y, levels, color, lw_levels):
    kde = gaussian_kde(np.vstack([x, y]))
    xx, yy = np.mgrid[
        x.min() - 0.02:x.max() + 0.02:80j,
        y.min() - 0.005:y.max() + 0.005:80j,
    ]
    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    z_sorted = np.sort(z.ravel())[::-1]
    cum = np.cumsum(z_sorted) / z_sorted.sum()
    thresholds = sorted(set(z_sorted[min(np.searchsorted(cum, L), len(z_sorted) - 1)]
                            for L in levels))
    ax.contour(xx, yy, z, levels=thresholds, colors=[color] * len(thresholds),
               linewidths=lw_levels[: len(thresholds)],
               linestyles=["-"] * len(thresholds), alpha=0.85, zorder=2)


def _bootstrap_panel(fig, gs_root, dataset, replicates_path, *,
                     polish_level, shared_ymax=None):
    cov_pos, gap = _load_replicates(replicates_path)
    pe = POINT_EST[dataset]

    inner = gs_root.subgridspec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.06, wspace=0.06,
    )
    ax_main = fig.add_subplot(inner[1, 0])
    ax_top = fig.add_subplot(inner[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)

    # Main scatter + KDE contours.
    ax_main.scatter(cov_pos, gap, s=10, color=C_HCCP, alpha=0.45,
                    edgecolor="none", zorder=2,
                    label=fr"bootstrap reps ($B={len(cov_pos)}$)")
    _kde_contour(ax_main, cov_pos, gap, [0.50, 0.95], C_HCCP, [1.4, 0.8])

    bs_mean_x = cov_pos.mean()
    bs_mean_y = gap.mean()
    ax_main.scatter([bs_mean_x], [bs_mean_y], marker="P", s=85,
                    facecolor=C_HCCP, edgecolor="white", linewidth=1.0,
                    zorder=4, label="bootstrap mean")

    # Y polish: bigger gold star with thicker outline + slightly different colour
    if polish_level == "Y":
        # Halo (white circle) + bright gold star
        ax_main.scatter([pe["cov_pos"]], [pe["gap"]], marker="o", s=560,
                        facecolor="white", edgecolor="white",
                        linewidth=0, zorder=5)
        ax_main.scatter([pe["cov_pos"]], [pe["gap"]], marker="*", s=470,
                        facecolor=C_POINT_Y, edgecolor="#222", linewidth=1.4,
                        zorder=6, label=r"all-chrom point ($K{=}5$)")
    else:
        ax_main.scatter([pe["cov_pos"]], [pe["gap"]], marker="*", s=320,
                        facecolor=C_POINT_X, edgecolor="white", linewidth=1.6,
                        zorder=6, label=r"all-chrom point ($K{=}5$)")

    ax_main.axvline(0.90, color="#222", ls="--", lw=0.6, alpha=0.6, zorder=1)

    ax_main.set_xlabel(r"Cov$|Y{=}1$  ($\to$ better)")
    ax_main.set_ylabel(r"$\hat\sigma$-bin gap  ($\downarrow$ better)")
    ax_top.set_title(f"{dataset}", fontsize=10.5, pad=2, loc="left",
                     fontweight="bold")
    ax_main.grid(True, ls=":", lw=0.4, alpha=0.5)
    for s in ["top", "right"]:
        ax_main.spines[s].set_visible(False)

    # Marginal KDEs
    xs = np.linspace(cov_pos.min() - 0.01, cov_pos.max() + 0.01, 200)
    kx = gaussian_kde(cov_pos)
    ax_top.fill_between(xs, kx(xs), color=C_HCCP, alpha=0.30, linewidth=0)
    ax_top.plot(xs, kx(xs), color=C_HCCP, linewidth=1.0)
    ax_top.axvline(pe["cov_pos"], color=C_POINT_X, linestyle="--", linewidth=0.8)
    ax_top.set_yticks([])
    plt.setp(ax_top.get_xticklabels(), visible=False)
    for s in ["top", "right", "left"]:
        ax_top.spines[s].set_visible(False)
    ax_top.tick_params(axis="x", length=0)

    ys = np.linspace(gap.min() - 0.005, gap.max() + 0.005, 200)
    ky = gaussian_kde(gap)
    ax_right.fill_betweenx(ys, ky(ys), color=C_HCCP, alpha=0.30, linewidth=0)
    ax_right.plot(ky(ys), ys, color=C_HCCP, linewidth=1.0)
    ax_right.axhline(pe["gap"], color=C_POINT_X, linestyle="--", linewidth=0.8)
    ax_right.set_xticks([])
    plt.setp(ax_right.get_yticklabels(), visible=False)
    for s in ["top", "right", "bottom"]:
        ax_right.spines[s].set_visible(False)
    ax_right.tick_params(axis="y", length=0)

    # Shared y-axis (Y polish): same y range for both panels
    if polish_level == "Y" and shared_ymax is not None:
        ax_main.set_ylim(-0.001, shared_ymax)

    # CI annotation: X polish moves to bottom-left (away from legend);
    # Y polish moves below the panel (caption-style strip).
    ci_xlo, ci_xhi = np.quantile(cov_pos, [0.025, 0.975])
    ci_ylo, ci_yhi = np.quantile(gap, [0.025, 0.975])
    txt = (
        rf"95% CI: " + "\n"
        rf"  Cov$|Y{{=}}1 \in [{ci_xlo:.3f},\,{ci_xhi:.3f}]$" + "\n"
        rf"  gap $\in [{ci_ylo:.3f},\,{ci_yhi:.3f}]$"
    )
    if polish_level == "X":
        # Bottom-left, away from the upper-right legend.
        ax_main.text(0.02, 0.40, txt, transform=ax_main.transAxes,
                     ha="left", va="top", fontsize=6.8,
                     bbox=dict(facecolor="white", edgecolor="0.75",
                               linewidth=0.4, boxstyle="round,pad=0.3",
                               alpha=0.92))
    else:  # Y: even more compact, bottom-left
        ax_main.text(0.02, 0.30, txt, transform=ax_main.transAxes,
                     ha="left", va="top", fontsize=6.5,
                     bbox=dict(facecolor="white", edgecolor="0.75",
                               linewidth=0.4, boxstyle="round,pad=0.25",
                               alpha=0.94))

    if dataset == "Mendelian":
        ax_main.legend(loc="upper right", fontsize=6.6, frameon=True,
                       framealpha=0.94, borderpad=0.4, handletextpad=0.4)


def variant_bootstrap(level):
    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.22,
                          top=0.86, bottom=0.10, left=0.06, right=0.985)

    shared_ymax = None
    if level == "Y":
        # Compute shared y-axis upper bound = max bootstrap-gap across both datasets, with a small margin
        max_gap = 0
        for path in BOOTSTRAP_PATHS.values():
            _, gap = _load_replicates(path)
            max_gap = max(max_gap, gap.max())
        shared_ymax = max_gap * 1.10

    _bootstrap_panel(fig, gs[0, 0], "Mendelian", BOOTSTRAP_PATHS["Mendelian"],
                     polish_level=level, shared_ymax=shared_ymax)
    _bootstrap_panel(fig, gs[0, 1], "Complex",   BOOTSTRAP_PATHS["Complex"],
                     polish_level=level, shared_ymax=shared_ymax)

    fig.suptitle(
        r"Bootstrap robustness ($B = 200$ chromosome-level resamples) of HCCP "
        r"on the (Cov$|Y{=}1$, $\hat\sigma$-bin gap) plane",
        fontsize=10.5, y=0.98,
    )

    out = PREVIEW / f"bootstrap_{level}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(out)


def main():
    print("--- ASL audit Mendelian ---")
    variant_asl("X")
    variant_asl("Y")
    print("--- Bootstrap density ---")
    variant_bootstrap("X")
    variant_bootstrap("Y")


if __name__ == "__main__":
    main()
