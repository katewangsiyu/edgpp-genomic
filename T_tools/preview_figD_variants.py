"""Generate 3 candidate redesigns of figD_alpha_sweep for user selection.

Outputs to /tmp/figD_variant_{A,B,C}.png + copy into R_raw/figD_previews/.
After the user picks, the chosen layout is ported to scripts/28_alpha_sweep_figure.py.

A: 2x3 polish in place (op-point labels, Mendelian peak callout, shared legend, wider bars)
B: 1x3 compact (datasets via color, single row)
C: Operating-point centric (hero singleton panel + side-by-side calibration/sigma-gap)
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from T_tools.paper_style import apply_paper_style  # noqa: E402

apply_paper_style()

REPO = Path(__file__).resolve().parents[1]
PREVIEW = REPO / "R_raw" / "figD_previews"
PREVIEW.mkdir(parents=True, exist_ok=True)

PATHS = {
    "Mendelian": REPO / "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/conformal_hetero_results.json",
    "Complex":   REPO / "outputs/conformal_hetero/CADD+GPN-MSA+Borzoi_complex_abs_mondrian/conformal_hetero_results.json",
}

METHODS = [
    ("Split CP (homoscedastic)",                          "homosc",   "o", "#888888"),
    (r"HCCP class-cond",                                  "hetero",   "s", "#4477aa"),
    (r"HCCP Mondrian ($K{=}5$)",                          "mondrian", "D", "#cc3311"),
]
DS_COLOR = {"Mendelian": "#cc3311", "Complex": "#4477aa"}
HEADLINE_GAP = 0.04

OP_POINTS = [
    (0.10, "high-stakes\nrare-disease",  "#5e2750"),
    (0.20, "triage\nscreening",          "#7e6e3a"),
    (0.30, "hypothesis\ngeneration",     "#2c6e49"),
]


def load_sweep(path):
    d = json.loads(Path(path).read_text())
    return {
        "hetero":   d.get("alpha_sweep_hetero", {}),
        "homosc":   d.get("alpha_sweep_homosc", {}),
        "mondrian": d.get("alpha_sweep_mondrian", {}),
    }


def alphas_of(sweep):
    a_str = sorted(sweep["mondrian"].keys(), key=float)
    return a_str, [float(a) for a in a_str]


# ============================================================================
# Variant A: 2x3 polish in place
# ============================================================================
def variant_A(mend, comp):
    fig = plt.figure(figsize=(10.0, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.0],
                          hspace=0.30, wspace=0.36,
                          top=0.84, bottom=0.10, left=0.085, right=0.985)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)])

    for i, (ds_name, sweeps) in enumerate([("Mendelian", mend), ("Complex", comp)]):
        a_str, alphas = alphas_of(sweeps)

        # (a) calibration residual
        ax = axes[i, 0]
        ax.axhspan(-0.015, 0.015, color="#dde6f0", alpha=0.6, zorder=0)
        ax.axhline(0.0, color="#222", ls="--", lw=0.7, zorder=1)
        for label, key, marker, color in METHODS:
            sw = sweeps[key]
            if not sw:
                continue
            res = [sw[a]["coverage"] - (1 - float(a)) for a in a_str]
            ax.plot(alphas, res, marker=marker, color=color, lw=1.4,
                    markersize=5, mfc="white", mew=1.4)
        ax.set_ylabel(r"cov $-\,(1-\alpha)$")
        ax.set_xlim(-0.02, 0.55)
        ax.set_ylim(-0.04, 0.04)
        ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")

        # (b) sigma-bin gap (Mondrian only) — wider bars
        ax = axes[i, 1]
        rng = np.array([sweeps["mondrian"][a]["sigma_cov_range"] for a in a_str])
        ax.bar(alphas, rng, width=0.04, color=DS_COLOR[ds_name], alpha=0.85,
               edgecolor="white", linewidth=0.4)
        ax.axhline(HEADLINE_GAP, color="#222", ls="--", lw=0.7,
                   label=rf"$\leq 0.04$ headline")
        ax.set_ylabel(r"$\hat\sigma$-bin gap")
        ax.set_xlim(-0.02, 0.55)
        ax.set_ylim(0, max(0.10, rng.max() * 1.25))
        ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")
        if i == 0:
            ax.legend(fontsize=6.6, loc="upper right", frameon=True,
                      framealpha=0.92, borderpad=0.3, handletextpad=0.4)

        # (c) singleton fraction with op-point labels
        ax = axes[i, 2]
        sf = np.array([sweeps["mondrian"][a]["frac_singleton"] for a in a_str])
        ax.plot(alphas, sf, marker="D", color=DS_COLOR[ds_name], lw=1.4,
                markersize=5, mfc="white", mew=1.4)

        # Operating-point vertical guides + top labels
        for op_a, op_label, op_color in OP_POINTS:
            ax.axvline(op_a, color=op_color, ls=":", lw=0.7, zorder=0, alpha=0.7)
            if i == 0:
                ax.text(op_a, 1.05, op_label, color=op_color, fontsize=6.4,
                        ha="center", va="bottom", style="italic",
                        transform=ax.get_xaxis_transform())

        # Mendelian non-monotonic peak callout
        if ds_name == "Mendelian":
            peak_idx = int(np.argmax(sf))
            ax.annotate(
                fr"max $\approx${sf[peak_idx]:.2f}"
                "\n"
                fr"at $\alpha={alphas[peak_idx]:.2f}$",
                xy=(alphas[peak_idx], sf[peak_idx]),
                xytext=(alphas[peak_idx] + 0.15, sf[peak_idx] - 0.15),
                fontsize=6.4, color="#333",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="#777"))

        ax.set_ylabel("singleton fraction")
        ax.set_xlim(-0.02, 0.55)
        ax.set_ylim(0, 1.05)
        ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")

        # x-axis only on bottom row
        for j in range(3):
            if i == 0:
                axes[i, j].tick_params(axis="x", labelbottom=False)
            else:
                axes[i, j].set_xlabel(r"miscoverage $\alpha$")

    # Column titles
    for j, t in enumerate([
        r"(a) calibration residual",
        r"(b) $\hat\sigma$-bin gap (Mondrian)",
        r"(c) singleton fraction (Mondrian)",
    ]):
        axes[0, j].set_title(t, fontsize=9.0, pad=24)  # extra pad for op labels

    # Row labels
    for i, ds in enumerate(("Mendelian", "Complex")):
        bb = axes[i, 0].get_position()
        fig.text(0.025, (bb.y0 + bb.y1) / 2, ds, rotation=90,
                 ha="center", va="center", fontsize=10.5, fontweight="bold",
                 color=DS_COLOR[ds])

    # Shared method legend at top (above column titles)
    method_handles = [
        Line2D([0], [0], marker=m, color=c, lw=1.4, markersize=5,
               mfc="white", mew=1.4, label=lab)
        for lab, _, m, c in METHODS
    ]
    fig.legend(handles=method_handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.94), frameon=False, fontsize=7.5,
               handletextpad=0.4, columnspacing=1.6)

    fig.suptitle(
        r"HCCP $\alpha$ sweep on TraitGym  ---  calibration, $\hat\sigma$-bin gap, operational knob",
        fontsize=10.5, y=0.99,
    )
    fig.savefig(PREVIEW / "figD_variant_A.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Variant B: 1x3 compact (datasets via color)
# ============================================================================
def variant_B(mend, comp):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4),
                             gridspec_kw={"width_ratios": [1.2, 1.0, 1.0],
                                          "wspace": 0.30})
    a_str_m, alphas = alphas_of(mend)

    # (a) calibration residual: 3 methods × 2 datasets — use line style for dataset
    ax = axes[0]
    ax.axhspan(-0.015, 0.015, color="#dde6f0", alpha=0.6, zorder=0)
    ax.axhline(0.0, color="#222", ls="--", lw=0.7, zorder=1)
    for label, key, marker, color in METHODS:
        for ds_name, sw in [("Mendelian", mend[key]), ("Complex", comp[key])]:
            if not sw:
                continue
            ls = "-" if ds_name == "Mendelian" else "--"
            mfc = color if ds_name == "Mendelian" else "white"
            a_str_ds = sorted(sw.keys(), key=float)
            alphas_ds = [float(a) for a in a_str_ds]
            res = [sw[a]["coverage"] - (1 - float(a)) for a in a_str_ds]
            ax.plot(alphas_ds, res, marker=marker, ms=4.5, lw=1.1,
                    color=color, ls=ls, mfc=mfc, mec=color, mew=1.0,
                    alpha=0.85)
    ax.set_xlabel(r"miscoverage $\alpha$")
    ax.set_ylabel(r"cov $-\,(1-\alpha)$")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(-0.04, 0.04)
    ax.set_title("(a) calibration residual", fontsize=9.0)
    ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")

    # (b) σ̂-bin gap grouped bars (Mendelian / Complex side by side)
    ax = axes[1]
    rng_m = np.array([mend["mondrian"][a]["sigma_cov_range"] for a in a_str_m])
    rng_c = np.array([comp["mondrian"][a]["sigma_cov_range"] for a in a_str_m])
    bw = 0.018
    ax.bar(np.array(alphas) - bw / 2, rng_m, width=bw,
           color=DS_COLOR["Mendelian"], alpha=0.85, edgecolor="white",
           linewidth=0.4, label="Mendelian")
    ax.bar(np.array(alphas) + bw / 2, rng_c, width=bw,
           color=DS_COLOR["Complex"], alpha=0.85, edgecolor="white",
           linewidth=0.4, label="Complex")
    ax.axhline(HEADLINE_GAP, color="#222", ls="--", lw=0.7,
               label=rf"$\leq 0.04$ headline")
    ax.set_xlabel(r"miscoverage $\alpha$")
    ax.set_ylabel(r"$\hat\sigma$-bin gap")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0, max(0.10, max(rng_m.max(), rng_c.max()) * 1.25))
    ax.set_title(r"(b) $\hat\sigma$-bin gap (Mondrian)", fontsize=9.0)
    ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")
    ax.legend(fontsize=6.6, loc="upper left", frameon=True, framealpha=0.92,
              borderpad=0.3, handletextpad=0.4)

    # (c) singleton fraction 2 lines + op-point regions
    ax = axes[2]
    for ds_name, sw, color in [("Mendelian", mend, DS_COLOR["Mendelian"]),
                                ("Complex",   comp, DS_COLOR["Complex"])]:
        sf = np.array([sw["mondrian"][a]["frac_singleton"] for a in a_str_m])
        ax.plot(alphas, sf, marker="D", color=color, lw=1.4, ms=5,
                mfc="white", mew=1.4, label=ds_name)
    for op_a, op_label, op_color in OP_POINTS:
        ax.axvline(op_a, color=op_color, ls=":", lw=0.7, zorder=0, alpha=0.7)
        ax.text(op_a, 1.04, op_label, color=op_color, fontsize=6.0,
                ha="center", va="bottom", style="italic",
                transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"miscoverage $\alpha$")
    ax.set_ylabel("singleton fraction")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0, 1.0)
    ax.set_title("(c) singleton fraction (Mondrian)", fontsize=9.0, pad=18)
    ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")
    ax.legend(fontsize=6.6, loc="lower right", frameon=True, framealpha=0.92,
              borderpad=0.3, handletextpad=0.4)

    # Method legend for (a) at figure level
    method_handles = [
        Line2D([0], [0], marker=m, color=c, lw=1.4, markersize=5,
               mfc="white", mew=1.4, label=lab) for lab, _, m, c in METHODS
    ]
    method_handles.append(Line2D([0], [0], color="0.4", ls="-", lw=1.2,
                                  label="Mendelian (filled, solid)"))
    method_handles.append(Line2D([0], [0], color="0.4", ls="--", lw=1.2,
                                  label="Complex (hollow, dashed)"))
    fig.legend(handles=method_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=6.8,
               handletextpad=0.4, columnspacing=1.4)

    fig.suptitle(
        r"HCCP $\alpha$ sweep on TraitGym  (1$\times$3 compact, datasets via color)",
        fontsize=10.0, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(PREVIEW / "figD_variant_B.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Variant C: Operating-point centric (hero singleton panel + side-by-side)
# ============================================================================
def variant_C(mend, comp):
    fig = plt.figure(figsize=(10.0, 4.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.4],
                          hspace=0.50, wspace=0.40,
                          top=0.86, bottom=0.13, left=0.07, right=0.985)
    ax_cal_m = fig.add_subplot(gs[0, 0])
    ax_cal_c = fig.add_subplot(gs[1, 0])
    ax_gap_m = fig.add_subplot(gs[0, 1])
    ax_gap_c = fig.add_subplot(gs[1, 1])
    ax_hero  = fig.add_subplot(gs[:, 2])

    a_str_m, alphas = alphas_of(mend)

    # Small calibration residual panels
    for ax, ds_name, sweeps in [(ax_cal_m, "Mendelian", mend),
                                  (ax_cal_c, "Complex",   comp)]:
        ax.axhspan(-0.015, 0.015, color="#dde6f0", alpha=0.6, zorder=0)
        ax.axhline(0.0, color="#222", ls="--", lw=0.7)
        for label, key, marker, color in METHODS:
            sw = sweeps[key]
            if not sw:
                continue
            a_str_ds = sorted(sw.keys(), key=float)
            alphas_ds = [float(a) for a in a_str_ds]
            res = [sw[a]["coverage"] - (1 - float(a)) for a in a_str_ds]
            ax.plot(alphas_ds, res, marker=marker, color=color, lw=1.2,
                    markersize=4, mfc="white", mew=1.2, alpha=0.9)
        ax.set_ylabel(r"cov $-\,(1{-}\alpha)$", fontsize=7.5)
        ax.set_xlim(-0.02, 0.55)
        ax.set_ylim(-0.03, 0.03)
        ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")
        ax.set_title(f"{ds_name}: calibration", fontsize=8.5)
        ax.tick_params(labelsize=6.5)
        if ds_name == "Complex":
            ax.set_xlabel(r"$\alpha$", fontsize=7.5)

    # Small sigma-bin gap panels
    for ax, ds_name, sweeps in [(ax_gap_m, "Mendelian", mend),
                                  (ax_gap_c, "Complex",   comp)]:
        rng = np.array([sweeps["mondrian"][a]["sigma_cov_range"] for a in a_str_m])
        ax.bar(alphas, rng, width=0.035, color=DS_COLOR[ds_name], alpha=0.85,
               edgecolor="white", linewidth=0.4)
        ax.axhline(HEADLINE_GAP, color="#222", ls="--", lw=0.7)
        ax.set_ylabel(r"$\hat\sigma$-bin gap", fontsize=7.5)
        ax.set_xlim(-0.02, 0.55)
        ax.set_ylim(0, max(0.10, rng.max() * 1.25))
        ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")
        ax.set_title(f"{ds_name}: $\\hat\\sigma$-bin gap", fontsize=8.5)
        ax.tick_params(labelsize=6.5)
        if ds_name == "Complex":
            ax.set_xlabel(r"$\alpha$", fontsize=7.5)

    # HERO singleton panel with operating-point shaded regions
    ax = ax_hero
    # Draw shaded regions per OP (slightly extending to mid-points between OPs)
    op_alphas = [0.0] + [op[0] for op in OP_POINTS] + [0.55]
    op_mids = [(op_alphas[i] + op_alphas[i + 1]) / 2 for i in range(len(op_alphas) - 1)]
    for k, (op_a, op_label, op_color) in enumerate(OP_POINTS):
        # Shaded region centered around op_a between adjacent midpoints
        x_lo = op_mids[k]
        x_hi = op_mids[k + 1]
        ax.axvspan(x_lo, x_hi, color=op_color, alpha=0.10, zorder=0)
        ax.axvline(op_a, color=op_color, ls=":", lw=0.9, alpha=0.85, zorder=1)
        ax.text(op_a, 1.04, op_label, color=op_color, fontsize=7.5,
                ha="center", va="bottom", style="italic", fontweight="bold",
                transform=ax.get_xaxis_transform())

    # Two singleton lines
    for ds_name, sw, color in [("Mendelian", mend, DS_COLOR["Mendelian"]),
                                ("Complex",   comp, DS_COLOR["Complex"])]:
        sf = np.array([sw["mondrian"][a]["frac_singleton"] for a in a_str_m])
        ax.plot(alphas, sf, marker="D", color=color, lw=2.0, ms=7,
                mfc="white", mew=1.6, label=ds_name, zorder=4)
        # Annotate values at the 3 OP points
        for op_a, _, op_color in OP_POINTS:
            for a, val in zip(alphas, sf):
                if abs(a - op_a) < 1e-6:
                    ax.annotate(f"{val:.0%}",
                                xy=(a, val),
                                xytext=(0, 9 if ds_name == "Mendelian" else -16),
                                textcoords="offset points",
                                fontsize=7.0, color=color, fontweight="bold",
                                ha="center", zorder=5)

    ax.set_xlabel(r"miscoverage $\alpha$")
    ax.set_ylabel("singleton fraction")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0, 1.05)
    ax.set_title(r"Operating knob: singleton fraction vs $\alpha$  (HCCP Mondrian, $K{=}5$)",
                 fontsize=9.5, pad=22)
    ax.grid(ls=":", lw=0.4, alpha=0.5, axis="y")
    ax.legend(fontsize=7.5, loc="lower right", frameon=True, framealpha=0.94,
              borderpad=0.4, handletextpad=0.4)

    # Method legend below small panels
    method_handles = [
        Line2D([0], [0], marker=m, color=c, lw=1.2, markersize=4,
               mfc="white", mew=1.2, label=lab) for lab, _, m, c in METHODS
    ]
    fig.legend(handles=method_handles, loc="lower left", ncol=3,
               bbox_to_anchor=(0.07, -0.02), frameon=False, fontsize=6.8,
               handletextpad=0.4, columnspacing=1.4)

    fig.suptitle(r"HCCP $\alpha$ sweep on TraitGym  ---  operating-point centric",
                 fontsize=10.5, y=0.97)
    fig.savefig(PREVIEW / "figD_variant_C.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    mend = load_sweep(PATHS["Mendelian"])
    comp = load_sweep(PATHS["Complex"])
    variant_A(mend, comp); print(PREVIEW / "figD_variant_A.png")
    variant_B(mend, comp); print(PREVIEW / "figD_variant_B.png")
    variant_C(mend, comp); print(PREVIEW / "figD_variant_C.png")


if __name__ == "__main__":
    main()
