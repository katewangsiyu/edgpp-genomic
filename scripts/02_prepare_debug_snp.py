"""Prepare a small debug SNP subset for Phase 0 smoke.

If Borzoi QTL VCFs exist in data/raw/qtl_vcf/, sample N from them.
Otherwise generate synthetic SNPs so Phase 0 smoke still runs (pipeline
validation only — not meaningful metrics).

Usage:
    python scripts/02_prepare_debug_snp.py --n 100
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from edgpp_genomic.data.vcf import load_vcf_as_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_vcf", default="data/raw/qtl_vcf/Whole_Blood_pos.vcf")
    ap.add_argument("--neg_vcf", default="data/raw/qtl_vcf/Whole_Blood_neg.vcf")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="data/debug/snp_100.parquet")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    pos_p, neg_p = Path(args.pos_vcf), Path(args.neg_vcf)
    if pos_p.exists() and neg_p.exists():
        print(f"[real] loading {pos_p} + {neg_p}")
        pos = load_vcf_as_df(pos_p); pos["label"] = 1
        neg = load_vcf_as_df(neg_p); neg["label"] = 0
        df = pd.concat([pos, neg], ignore_index=True)
        if "coef" not in df.columns:
            df["coef"] = rng.normal(0, 0.5, size=len(df))
    else:
        print("[synthetic] VCFs not found, generating synthetic SNPs for pipeline smoke.")
        N = args.n * 4
        chroms = ["chr1", "chr7", "chr22"]
        ref = rng.choice(list("ACGT"), size=N)
        # choose alt ≠ ref
        alt = np.array([rng.choice([b for b in "ACGT" if b != r]) for r in ref])
        df = pd.DataFrame({
            "chrom": rng.choice(chroms, size=N),
            "pos": rng.integers(1_000_000, 10_000_000, size=N),
            "snp_id": [f"syn_{i}" for i in range(N)],
            "ref": ref,
            "alt": alt,
            "label": rng.integers(0, 2, size=N),
            "coef": rng.normal(0, 0.5, size=N),
        })

    df["teacher_var"] = rng.uniform(0, 1, size=len(df))
    df["maf"] = rng.uniform(0.001, 0.5, size=len(df))
    df["phylop"] = rng.normal(0, 2, size=len(df))

    df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
    df.to_parquet(out, index=False)
    print(f"[done] wrote {len(df)} SNPs -> {out}")


if __name__ == "__main__":
    main()
