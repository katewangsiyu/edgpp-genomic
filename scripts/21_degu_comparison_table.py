"""Generate HCCP vs DEGU-lite comparison table for paper.

Reads conformal_hetero results for matched feature sets and produces
a LaTeX-ready comparison table.

Usage:
    conda run -n edgpp_t4 python scripts/21_degu_comparison_table.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np


RESULTS_DIR = Path("outputs/conformal_hetero")

# (label, path, feature_set, dataset)
CONFIGS = [
    # Same features (CADD+GPN-MSA+Borzoi) — fair comparison
    ("HCCP", "CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian", "CADD+GPN-MSA+Borzoi", "Mendelian"),
    ("DEGU-lite", "DEGU_lite_CADD+GPN-MSA+Borzoi_mendelian_mondrian", "CADD+GPN-MSA+Borzoi", "Mendelian"),
    ("HCCP", "CADD+GPN-MSA+Borzoi_complex_abs_mondrian", "CADD+GPN-MSA+Borzoi", "Complex"),
    ("DEGU-lite", "DEGU_lite_CADD+GPN-MSA+Borzoi_complex_mondrian", "CADD+GPN-MSA+Borzoi", "Complex"),
    # Ablation: CADD+Borzoi only
    ("HCCP", "CADD+Borzoi_mendelian_abs_mondrian", "CADD+Borzoi", "Mendelian"),
    ("DEGU-lite", "DEGU_lite_CADD+Borzoi_mendelian_mondrian", "CADD+Borzoi", "Mendelian"),
    ("HCCP", "CADD+Borzoi_complex_abs_mondrian", "CADD+Borzoi", "Complex"),
    ("DEGU-lite", "DEGU_lite_CADD+Borzoi_complex_mondrian", "CADD+Borzoi", "Complex"),
]


def extract_metrics(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)

    mondrian = d.get("mondrian_y_sigma", {})
    hetero = d.get("hetero_class_cond", {})

    # Use mondrian if available, else hetero
    m = mondrian if mondrian else hetero

    bins = m.get("coverage_by_sigma_bin", [])
    coverages = [b["coverage"] for b in bins if b.get("n", 0) >= 5]

    return {
        "coverage": m.get("coverage", np.nan),
        "coverage_pos": m.get("coverage_pos", np.nan),
        "worst_gap": max(abs(c - 0.9) for c in coverages) if coverages else np.nan,
        "mean_gap": np.mean([abs(c - 0.9) for c in coverages]) if coverages else np.nan,
        "frac_singleton": m.get("frac_singleton", np.nan),
        "frac_empty": m.get("frac_empty", np.nan),
        "n_bins": len(coverages),
        "worst_bin_cov": min(coverages) if coverages else np.nan,
    }


def main():
    print("=" * 90)
    print("HCCP vs DEGU-lite Head-to-Head Comparison (Mondrian y×σ̂-bin)")
    print("=" * 90)

    results = []
    for label, subdir, feat, dataset in CONFIGS:
        path = RESULTS_DIR / subdir / "conformal_hetero_results.json"
        if not path.exists():
            print(f"  SKIP (not found): {subdir}")
            continue
        m = extract_metrics(path)
        m["label"] = label
        m["features"] = feat
        m["dataset"] = dataset
        results.append(m)

    # Print grouped by feature set
    for feat in ["CADD+GPN-MSA+Borzoi", "CADD+Borzoi"]:
        group = [r for r in results if r["features"] == feat]
        if not group:
            continue
        print(f"\n--- Feature set: {feat} ---")
        print(f"{'Method':12s} {'Dataset':10s} | {'Cov':>6s} {'Cov|+':>6s} "
              f"{'W-Gap':>6s} {'M-Gap':>6s} {'Worst':>6s} {'Sgl%':>5s}")
        print("-" * 72)
        for r in group:
            print(f"{r['label']:12s} {r['dataset']:10s} | "
                  f"{r['coverage']:.4f} {r['coverage_pos']:.4f} "
                  f"{r['worst_gap']:.4f} {r['mean_gap']:.4f} "
                  f"{r['worst_bin_cov']:.4f} {r['frac_singleton']:.3f}")

    # Print key takeaways
    print("\n" + "=" * 90)
    print("KEY TAKEAWAYS:")

    # Find pairs for comparison
    for feat in ["CADD+GPN-MSA+Borzoi", "CADD+Borzoi"]:
        for ds in ["Mendelian", "Complex"]:
            hccp = [r for r in results if r["label"]=="HCCP" and r["features"]==feat and r["dataset"]==ds]
            degu = [r for r in results if r["label"]=="DEGU-lite" and r["features"]==feat and r["dataset"]==ds]
            if hccp and degu:
                h, d = hccp[0], degu[0]
                ratio = d["worst_gap"] / h["worst_gap"] if h["worst_gap"] > 0 else float("inf")
                winner = "HCCP" if h["worst_gap"] < d["worst_gap"] else "DEGU"
                print(f"  {feat} / {ds}: {winner} wins "
                      f"(HCCP {h['worst_gap']:.4f} vs DEGU {d['worst_gap']:.4f}, "
                      f"ratio {ratio:.1f}x)")

    # Save as JSON for programmatic use
    out = Path("outputs/degu_comparison_summary.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
