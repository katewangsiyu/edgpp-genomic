"""Scatter figure for ProteinGym assay-LOO coverage distribution."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Unified NeurIPS-classic style ---
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from T_tools.paper_style import apply_paper_style  # noqa: E402
apply_paper_style()


def main() -> None:
    results = json.load(open("outputs/proteingym_hccp_n50/per_assay_results.json"))
    valid = [r for r in results if "marginal_coverage" in r]

    auprcs = np.array([r["AUPRC_test"] for r in valid])
    covs = np.array([r["marginal_coverage"] for r in valid])
    gaps = np.array([r["sigma_bin_range"] for r in valid])
    names = [r["target_assay"] for r in valid]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: AUPRC vs coverage
    ax = axes[0]
    ax.scatter(auprcs, covs, s=60, color="#cc3311", alpha=0.75,
               edgecolor="black", linewidth=0.5)
    ax.axhline(0.90, color="#444", ls="--", lw=1, label=r"target $1-\alpha = 0.90$")
    ax.axhspan(0.88, 0.92, color="#9a9a9a", alpha=0.15,
               label=r"$\pm 0.02$ band")
    ax.set_xlabel("AUPRC on target assay")
    ax.set_ylabel("Marginal coverage on target assay")
    ax.set_title(f"ProteinGym assay-LOO (n={len(valid)} held-out assays)")
    ax.grid(ls="--", lw=0.3, alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(0.75, 0.97)

    # Highlight outliers (cov < 0.85, the §6.5 6-outlier set) — alternate up/down offsets
    outlier_idxs = [i for i, c in enumerate(covs) if c < 0.85]
    # Sort by AUPRC so we can alternate placement
    outlier_idxs.sort(key=lambda i: auprcs[i])
    for rank, i in enumerate(outlier_idxs):
        short = names[i].split("_")[0][:8]
        # Alternate offsets: even=upper-right, odd=lower-left
        if rank % 2 == 0:
            xytext, ha, va = (10, 8), "left", "bottom"
        else:
            xytext, ha, va = (-10, -8), "right", "top"
        ax.annotate(short,
                    (auprcs[i], covs[i]),
                    xytext=xytext, textcoords="offset points",
                    fontsize=6.5, color="#882200", ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", color="#882200",
                                    lw=0.5, alpha=0.6))

    # Panel B: distribution of σ̂-bin gap
    ax2 = axes[1]
    ax2.hist(gaps, bins=12, color="#4477aa", alpha=0.8,
             edgecolor="black", linewidth=0.5)
    ax2.axvline(gaps.mean(), color="#cc3311", ls="--", lw=1.5,
                label=f"mean = {gaps.mean():.3f}")
    ax2.axvline(np.median(gaps), color="#006000", ls="--", lw=1.5,
                label=f"median = {np.median(gaps):.3f}")
    # TraitGym benchmarks (thicker dotted lines so they're visible)
    ax2.axvline(0.004, color="#aa7700", ls=":", lw=1.8,
                label="TraitGym trait-LOO floor (0.004)")
    ax2.axvline(0.077, color="#550055", ls=":", lw=1.8,
                label="TraitGym chrom-LOO (0.077)")
    ax2.set_xlabel(r"$\hat\sigma$-bin coverage gap")
    ax2.set_ylabel("number of assays")
    ax2.set_title(rf"HCCP $\hat\sigma$-bin gap across {len(valid)} assays")
    ax2.grid(ls="--", lw=0.3, alpha=0.4)
    ax2.legend(fontsize=7, loc="upper right")

    fig.suptitle("HCCP cross-domain stress test on ProteinGym (App.~E)", y=1.02)
    fig.tight_layout()
    out = Path("papers/neurips2027_pathA/figures/figE_proteingym.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
