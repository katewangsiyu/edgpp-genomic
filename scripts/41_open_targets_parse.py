"""Parse Open Targets credible_set parquet shards into a TraitGym-style binary
classification dataset for HCCP polygenic-regime evaluation.

For each studyLocus (row), the `locus` column is an array of
    {variantId, posteriorProbability, logBF, beta, standardError, r2Overall, ...}
We explode it to one row per (studyLocus, variantId), inheriting the
credible-set-level features (purity, sampleSize, finemappingMethod, etc.).

Binary task: predict whether a variant is the causal candidate
    label = 1  if posteriorProbability >= --pip-pos-threshold
    label = 0  if posteriorProbability <= --pip-neg-threshold

Filtering for the polygenic regime that complements TraitGym Complex:
  - studyType == 'gwas' (drop eQTL / sQTL / pQTL — those aren't disease VEP)
  - locus length >= --min-set-size (default 5; drops the 21% single-variant
    sets so we get genuine imbalanced classification within each set)
  - Optional --max-set-size to drop pathologically wide LD blocks (default 200)

Output:
    parquet at --out, one row per variant, columns:
        chromosome, position, variantId, ref, alt,
        label,                           # binary, per --pip thresholds
        pip,                             # the underlying posteriorProbability
        beta, standardError, log10BF,    # variant-level summary stats
        r2Overall,                       # LD with the credible-set lead
        credibleSetlog10BF, purityMeanR2, purityMinR2, sampleSize,
        studyId, studyLocusId, studyType, finemappingMethod, locus_size

Downstream: T_tools/cp_baselines_h2h.py adapter (TODO scripts/42_open_targets_h2h.py)
runs HCCP + 6 baselines on this parquet, matching the TraitGym chrom-LOO
protocol (test = chr {17, 18, 19, 20, 21, 22, X}).

Usage:
    python scripts/41_open_targets_parse.py \\
        --in-dir data/raw/open_targets/credible_set \\
        --out data/processed/open_targets/gwas_polygenic.parquet \\
        --pip-pos-threshold 0.5 --pip-neg-threshold 0.05 \\
        --min-set-size 5 --max-set-size 200
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


CREDIBLE_LEVEL_COLS = [
    "studyLocusId", "studyId", "studyType", "chromosome",
    "credibleSetlog10BF", "purityMeanR2", "purityMinR2", "sampleSize",
    "finemappingMethod", "confidence",
]
LOCUS_VARIANT_COLS = [
    "variantId", "posteriorProbability", "logBF",
    "beta", "standardError", "pValueMantissa", "pValueExponent", "r2Overall",
]


def parse_variant_id(vid: str) -> tuple[str, int, str, str] | tuple[None, None, None, None]:
    """Open Targets variantId format: 'CHROM_POS_REF_ALT' (hg38)."""
    if not isinstance(vid, str):
        return (None, None, None, None)
    m = re.fullmatch(r"([0-9XYM]+)_(\d+)_([ACGTN-]+)_([ACGTN-]+)", vid)
    if m is None:
        return (None, None, None, None)
    chrom, pos, ref, alt = m.groups()
    return (chrom, int(pos), ref, alt)


def explode_one_part(path: Path,
                     min_set_size: int, max_set_size: int,
                     studytype: str = "gwas") -> pd.DataFrame:
    """Read one credible_set parquet part, filter, explode locus → variant rows."""
    df = pd.read_parquet(path)

    # Drop non-target studyType rows
    df = df[df["studyType"] == studytype].copy()
    if df.empty:
        return df

    # Filter on locus size
    df["locus_size"] = df["locus"].apply(
        lambda a: 0 if a is None else len(a))
    df = df[(df["locus_size"] >= min_set_size)
            & (df["locus_size"] <= max_set_size)]
    if df.empty:
        return df

    # Explode locus to one row per (studyLocus, variant)
    df_exp = df.explode("locus", ignore_index=True)

    # Pull per-variant fields out of the struct dict
    locus_struct = df_exp["locus"]
    for col in LOCUS_VARIANT_COLS:
        df_exp[col] = locus_struct.apply(
            lambda d: d.get(col) if isinstance(d, dict) else None)

    # Parse variantId → chrom + pos + ref + alt (hg38)
    parsed = df_exp["variantId"].apply(parse_variant_id)
    df_exp["chromosome_v"] = parsed.apply(lambda t: t[0])
    df_exp["position"] = parsed.apply(lambda t: t[1])
    df_exp["ref"] = parsed.apply(lambda t: t[2])
    df_exp["alt"] = parsed.apply(lambda t: t[3])

    # Keep only successfully parsed variants
    df_exp = df_exp[df_exp["chromosome_v"].notna()].copy()

    # Use variant chrom (overrides credible-set-level chromosome which can be
    # blank for some entries)
    df_exp["chromosome"] = df_exp["chromosome_v"]
    df_exp = df_exp.drop(columns=["chromosome_v", "locus"])

    keep = (CREDIBLE_LEVEL_COLS
            + ["position", "ref", "alt", "locus_size"]
            + LOCUS_VARIANT_COLS)
    return df_exp[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="Directory containing credible_set part-*.parquet")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output parquet path (variant-level).")
    ap.add_argument("--pip-pos-threshold", type=float, default=0.5,
                    help="Variants with posteriorProbability >= this are positive (causal)")
    ap.add_argument("--pip-neg-threshold", type=float, default=0.05,
                    help="Variants with posteriorProbability <= this are negative")
    ap.add_argument("--min-set-size", type=int, default=5)
    ap.add_argument("--max-set-size", type=int, default=200)
    ap.add_argument("--studytype", default="gwas",
                    choices=["gwas", "eqtl", "tuqtl", "sqtl", "sceqtl", "pqtl"])
    ap.add_argument("--summary-out", type=Path, default=None,
                    help="Optional JSON path for per-chromosome n / pi_+ summary")
    args = ap.parse_args()

    parts = sorted(args.in_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No part-*.parquet under {args.in_dir}")
    print(f"[plan] {len(parts)} parts under {args.in_dir}")

    frames: list[pd.DataFrame] = []
    for p in tqdm(parts, desc="parse"):
        try:
            sub = explode_one_part(p, args.min_set_size, args.max_set_size,
                                   args.studytype)
        except Exception as e:
            print(f"[warn] {p.name}: {e}")
            continue
        if not sub.empty:
            frames.append(sub)

    if not frames:
        raise RuntimeError("No rows survived filtering.")

    big = pd.concat(frames, ignore_index=True)
    print(f"[parsed] {len(big):,} variant-rows across "
          f"{big['studyLocusId'].nunique():,} credible sets, "
          f"{big['chromosome'].nunique()} chromosomes")

    pip = big["posteriorProbability"].astype(float)
    pos_mask = pip >= args.pip_pos_threshold
    neg_mask = pip <= args.pip_neg_threshold
    big["label"] = -1
    big.loc[pos_mask, "label"] = 1
    big.loc[neg_mask, "label"] = 0
    labelled = big[big["label"] >= 0].copy()
    print(f"[labelled] {len(labelled):,} (drop "
          f"{len(big) - len(labelled):,} with PIP in the gray zone "
          f"({args.pip_neg_threshold}, {args.pip_pos_threshold}))")
    pos_rate = labelled["label"].mean()
    print(f"[balance] π_+ = {pos_rate:.4f}  "
          f"({int(labelled['label'].sum()):,} / {len(labelled):,})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    labelled.to_parquet(args.out, index=False)
    print(f"saved: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")

    if args.summary_out is not None:
        summary = {
            "n_total": int(len(labelled)),
            "n_credible_sets": int(labelled["studyLocusId"].nunique()),
            "n_studies": int(labelled["studyId"].nunique()),
            "pi_pos_overall": float(pos_rate),
            "per_chromosome": {
                str(c): {
                    "n": int((labelled["chromosome"] == c).sum()),
                    "pi_pos": float(labelled.loc[labelled["chromosome"] == c,
                                                  "label"].mean()),
                }
                for c in sorted(labelled["chromosome"].unique(),
                                key=lambda x: (x.isdigit() == False, x))
            },
            "filters": {
                "studytype": args.studytype,
                "pip_pos_threshold": args.pip_pos_threshold,
                "pip_neg_threshold": args.pip_neg_threshold,
                "min_set_size": args.min_set_size,
                "max_set_size": args.max_set_size,
            },
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2))
        print(f"saved: {args.summary_out}")


if __name__ == "__main__":
    main()
