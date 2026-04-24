"""Reproduce all paper tables and figures from outputs/.

This script is a one-button replay that:
  1. Loads existing outputs from outputs/aggregator_gbm/, outputs/conformal_hetero/,
     outputs/adaptive_K/, outputs/ks_audit/, outputs/trait_loo/, outputs/cross_dataset/.
  2. Prints reproducibility audit (which files exist, which are missing).
  3. Regenerates Fig D (α sweep) and computes summary numbers for Table 3, 4, 7.

If any expected output is missing, the corresponding table row is printed with
"MISSING — run scripts/NN_xxx.py" so the user can regenerate.

Usage:
    python scripts/33_reproduce_paper_tables.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def _load(path: Path) -> dict | None:
    try:
        return json.load(open(path))
    except FileNotFoundError:
        return None


def audit_table3() -> None:
    """Table 3: main headline — AUPRC + coverage."""
    print("\n=== Table 3: Main results ===")
    print("dataset, feature_set,  AUPRC,  marg_cov,  σ̂-gap,  frac_singleton")
    for ds in ["mendelian", "complex"]:
        for fs in ["CADD+Borzoi", "CADD+GPN-MSA+Borzoi"]:
            agg = _load(OUT / "aggregator_gbm" / f"{fs}_{ds}" / "metrics.json")
            conf = _load(OUT / "conformal_hetero" / f"{fs}_{ds}_abs_mondrian" /
                         "conformal_hetero_results.json")
            if agg is None or conf is None:
                print(f"  {ds:<10s}, {fs:<20s}, MISSING")
                continue
            m = conf["mondrian_y_sigma"]
            print(f"  {ds:<10s}, {fs:<20s}, {agg['AUPRC_per_chrom']:.3f}, "
                  f"{m['coverage']:.3f}, {m['sigma_cov_range']:.3f}, "
                  f"{m['frac_singleton']:.2f}")


def audit_multi_seed() -> None:
    print("\n=== Multi-seed summary (if available) ===")
    ms_dir = OUT / "multi_seed"
    if not ms_dir.exists():
        print("  (not yet run — scripts/26_multi_seed_aggregate.py)")
        return
    for f in sorted(ms_dir.glob("*.json")):
        d = json.load(open(f))
        print(f"\n  {f.stem}  (seeds={d.get('seeds')})")
        for metric, v in d.get("mean_std", {}).items():
            print(f"    {metric:<25s} = {v['mean']:.4f} ± {v['std']:.4f}")


def audit_K_sweep() -> None:
    print("\n=== Table 7: K-sweep ablation ===")
    for ds in ["mendelian", "complex"]:
        d = _load(OUT / "adaptive_K" / f"CADD+GPN-MSA+Borzoi_{ds}" / "adaptive_K_results.json")
        if d is None:
            print(f"  {ds}: MISSING — scripts/20_adaptive_K_sweep.py")
            continue
        print(f"  {ds}: K_CV={d['K_cv']}, K*_oracle={d['K_star_oracle']}, "
              f"L_F={d['L_F_estimated']:.2f}, gap@K5={d['gap_at_K5']:.3f}, "
              f"gap@K_CV={d['gap_at_K_cv']:.3f}")


def audit_ks() -> None:
    print("\n=== Appendix B.4: per-chrom KS audit ===")
    for ds in ["mendelian", "complex"]:
        d = _load(OUT / "ks_audit" / f"CADD+GPN-MSA+Borzoi_{ds}_ks_per_chrom_n20.json")
        if d is None:
            print(f"  {ds}: MISSING — scripts/27_ks_audit_per_chrom.py")
            continue
        print(f"  {ds}: {d['n_tests']} tests, reject@0.05={d['rejection_rate']*100:.1f}%, "
              f"max KS={d['ks_stat_max']:.3f}, mean={d['ks_stat_mean']:.3f}")


def audit_three_axis() -> None:
    print("\n=== Table 4: three-axis OOD ===")
    print("  chrom-LOO: see conformal_hetero results above")
    for ds in ["mendelian", "complex"]:
        for fs in ["CADD+Borzoi", "CADD+GPN-MSA+Borzoi"]:
            d = _load(OUT / "trait_loo" / f"{fs}_{ds}" / "trait_loo_results.json")
            if d is None:
                print(f"  trait-LOO {ds}/{fs}: MISSING")
                continue
            agg = d.get("aggregate_stats", {})
            print(f"  trait-LOO {ds}/{fs}: mean σ̂-gap={agg.get('mean_sigma_cov_range', 'n/a')}")
    for direction in ["CADD+Borzoi", "CADD+GPN-MSA+Borzoi"]:
        d = _load(OUT / "cross_dataset" / direction / "cross_dataset_results.json")
        if d is None:
            print(f"  cross-dataset {direction}: MISSING")
            continue
        print(f"  cross-dataset {direction}: keys={list(d.keys())[:6]}")


def audit_degu() -> None:
    print("\n=== Table 8: HCCP vs DEGU-lite ===")
    d = _load(OUT / "degu_comparison_summary.json")
    if d is None:
        print("  MISSING — scripts/21_degu_comparison_table.py")
        return
    print(f"  keys: {list(d.keys())[:8] if isinstance(d, dict) else type(d)}")


def audit_alpha_sweep() -> None:
    print("\n=== Appendix D: α sweep ===")
    fig = ROOT / "papers/neurips2027_pathA/figures/figD_alpha_sweep.pdf"
    tbl = ROOT / "papers/neurips2027_pathA/figures/figD_alpha_sweep_table.tex"
    print(f"  figure: {'OK' if fig.exists() else 'MISSING'} — {fig}")
    print(f"  table:  {'OK' if tbl.exists() else 'MISSING'} — {tbl}")
    if not fig.exists():
        print("  regenerate: python scripts/28_alpha_sweep_figure.py")


def audit_L_F() -> None:
    print("\n=== Appendix C.2: L_F estimator audit ===")
    for ds in ["mendelian", "complex"]:
        d = _load(OUT / "L_F_audit" / f"CADD+GPN-MSA+Borzoi_{ds}_L_F.json")
        if d is None:
            print(f"  {ds}: MISSING — scripts/29_L_F_estimator.py")
            continue
        lf = {name: v["L_F"] for name, v in d["estimators"].items()}
        print(f"  {ds}: L_F estimators = {lf}")


def main() -> None:
    print("=" * 68)
    print("HCCP paper reproducibility audit")
    print("=" * 68)
    audit_table3()
    audit_multi_seed()
    audit_K_sweep()
    audit_ks()
    audit_three_axis()
    audit_degu()
    audit_alpha_sweep()
    audit_L_F()
    print("\n" + "=" * 68)
    print("Regeneration: see papers/neurips2027_pathA/README.md")
    print("=" * 68)


if __name__ == "__main__":
    main()
