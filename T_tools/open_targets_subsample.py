"""Subsample Open Targets parsed credible-set parquet to a TraitGym-Complex-
comparable benchmark dataset for HCCP polygenic-regime evaluation.

Design constraints (matching TraitGym Complex matched_9):
  - Total n ≈ 11,400 variants (--n-target, default 11400)
  - π_+ ≈ 0.10 (--target-pi-pos)
  - chrom-LOO test split: test_chroms = {17, 18, 19, 20, 21, 22, X}
    matches papers/.../06_experiments.tex setup verbatim.

Sampling unit = credible set, not variant. Each credible set contributes:
  - All positives in the set (PIP >= pos_threshold; usually 1)
  - up to neg_per_pos negatives (PIP <= neg_threshold) sampled uniformly
This preserves within-set independence (positives and negatives share GWAS
study, sample size, finemapping method, region) — essential for HCCP's
calibration-fold exchangeability assumption A1' on TraitGym chrom-LOO.

Output:
  parquet with columns aligned for downstream cp_baselines_h2h.py adapter:
    chromosome, position, ref, alt, variantId, label, split,
    pip, beta, standardError, log10BF, r2Overall,
    credibleSetlog10BF, purityMeanR2, purityMinR2, sampleSize,
    studyId, studyLocusId

Usage:
    python T_tools/open_targets_subsample.py \\
        --in data/processed/open_targets/gwas_polygenic.parquet \\
        --out data/processed/open_targets/gwas_complex_aligned.parquet \\
        --n-target 11400 --target-pi-pos 0.10 \\
        --pos-pip-threshold 0.5 --neg-pip-threshold 0.05 \\
        --neg-per-pos 9 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TEST_CHROMS = {"17", "18", "19", "20", "21", "22", "X"}


def subsample_credible_set_aware(
    df: pd.DataFrame,
    n_target: int,
    target_pi_pos: float,
    pos_threshold: float,
    neg_threshold: float,
    neg_per_pos: int,
    seed: int,
) -> pd.DataFrame:
    """Pick credible sets, take all positives + up to neg_per_pos negatives each.

    Stops when accumulated row count exceeds n_target.  pi_+ is enforced via
    neg_per_pos = round((1 - target_pi_pos) / target_pi_pos).
    """
    rng = np.random.default_rng(seed)

    sets = (df.groupby("studyLocusId", as_index=False)
              .agg(n_pos=("label", lambda s: int((s == 1).sum())),
                   n_neg=("label", lambda s: int((s == 0).sum())),
                   chromosome=("chromosome", "first"),
                   studyId=("studyId", "first")))
    sets = sets[(sets["n_pos"] >= 1) & (sets["n_neg"] >= 1)]
    print(f"[sets] {len(sets):,} credible sets with >=1 pos and >=1 neg")

    # Shuffle credible sets, walk until we hit n_target
    sets = sets.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    selected_locus_ids: list[str] = []
    pos_seen, neg_seen = 0, 0
    for row in sets.itertuples(index=False):
        sl_id = row.studyLocusId
        n_pos_take = int(row.n_pos)
        n_neg_take = min(int(row.n_neg), neg_per_pos * n_pos_take)
        pos_seen += n_pos_take
        neg_seen += n_neg_take
        selected_locus_ids.append(sl_id)
        if (pos_seen + neg_seen) >= n_target:
            break

    print(f"[picked] {len(selected_locus_ids):,} credible sets "
          f"⇒ ~{pos_seen + neg_seen:,} variant rows planned "
          f"(π_+ planned ≈ {pos_seen / max(pos_seen + neg_seen, 1):.4f})")

    # Take all positives, sample neg_per_pos × n_pos negatives per set
    chosen = df[df["studyLocusId"].isin(selected_locus_ids)].copy()
    pos = chosen[chosen["label"] == 1]
    neg_pool = chosen[chosen["label"] == 0]

    per_set_pos = pos.groupby("studyLocusId").size()
    neg_quota = (per_set_pos * neg_per_pos).to_dict()

    neg_kept_idx: list[int] = []
    for sl_id, group in neg_pool.groupby("studyLocusId"):
        q = neg_quota.get(sl_id, 0)
        if q <= 0:
            continue
        if len(group) <= q:
            neg_kept_idx.extend(group.index.tolist())
        else:
            neg_kept_idx.extend(rng.choice(group.index.to_numpy(),
                                            size=q, replace=False).tolist())
    neg = neg_pool.loc[neg_kept_idx]
    out = pd.concat([pos, neg], ignore_index=True).copy()
    print(f"[final] {len(out):,} rows  "
          f"({int(out['label'].sum()):,} pos, "
          f"π_+ = {out['label'].mean():.4f})")
    return out.reset_index(drop=True)


def assign_chrom_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `split` column = 'train' or 'test' per TraitGym chrom-LOO."""
    df = df.copy()
    df["chromosome"] = df["chromosome"].astype(str)
    df["split"] = np.where(df["chromosome"].isin(TEST_CHROMS), "test", "train")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="Output of scripts/41_open_targets_parse.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    ap.add_argument("--n-target", type=int, default=11400,
                    help="Total variant count target (TraitGym Complex matched_9 = 11400)")
    ap.add_argument("--target-pi-pos", type=float, default=0.10)
    ap.add_argument("--pos-pip-threshold", type=float, default=0.5)
    ap.add_argument("--neg-pip-threshold", type=float, default=0.05)
    ap.add_argument("--neg-per-pos", type=int, default=9,
                    help="Negatives per positive (9 ⇒ π_+=0.10).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[load] {args.inp}")
    df = pd.read_parquet(args.inp)
    print(f"  full pool: {len(df):,} rows, "
          f"{df['studyLocusId'].nunique():,} credible sets")
    print(f"  pi_+ overall = {df['label'].mean():.4f}")

    out = subsample_credible_set_aware(
        df=df,
        n_target=args.n_target,
        target_pi_pos=args.target_pi_pos,
        pos_threshold=args.pos_pip_threshold,
        neg_threshold=args.neg_pip_threshold,
        neg_per_pos=args.neg_per_pos,
        seed=args.seed,
    )
    out = assign_chrom_split(out)

    print()
    print(f"=== chrom split: test={sorted(TEST_CHROMS)} ===")
    by_chrom = (out.groupby(["split", "chromosome"], as_index=False)
                  .agg(n=("label", "size"), pi_pos=("label", "mean")))
    print(by_chrom.to_string(index=False))
    print()
    print(f"  train n = {(out.split=='train').sum():,}, "
          f"π_+ = {out.loc[out.split=='train','label'].mean():.4f}")
    print(f"  test  n = {(out.split=='test').sum():,}, "
          f"π_+ = {out.loc[out.split=='test','label'].mean():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\nsaved: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")

    if args.summary_out is not None:
        summary = {
            "n_total": int(len(out)),
            "n_credible_sets": int(out["studyLocusId"].nunique()),
            "n_train": int((out.split == "train").sum()),
            "n_test": int((out.split == "test").sum()),
            "pi_pos_train": float(out.loc[out.split == "train", "label"].mean()),
            "pi_pos_test": float(out.loc[out.split == "test", "label"].mean()),
            "test_chroms": sorted(TEST_CHROMS),
            "filters": {
                "n_target": args.n_target,
                "target_pi_pos": args.target_pi_pos,
                "pos_pip_threshold": args.pos_pip_threshold,
                "neg_pip_threshold": args.neg_pip_threshold,
                "neg_per_pos": args.neg_per_pos,
                "seed": args.seed,
            },
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2))
        print(f"saved: {args.summary_out}")


if __name__ == "__main__":
    main()
