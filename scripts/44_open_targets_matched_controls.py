"""Build a TraitGym-Complex-style imbalanced classification dataset on Open
Targets credible sets:

  positives = OT credible-set variants with PIP >= --pos-pip-threshold
              (deduplicated by variantId; each is a high-confidence GWAS
              causal candidate).
  negatives = AF-matched random hg38 variants drawn from variant.parquet
              that are NOT in any OT credible set (no GWAS signal).

For each positive, we sample --neg-per-pos negatives from the variant pool
whose gnomAD AF (af_overall preferred, falling back to af_max) is within
--af-window of the positive's AF. Match groups are recorded so the H2H
chrom-LOO calibration can stratify or weight by group.

Output schema mirrors TraitGym Complex test.parquet for downstream reuse:
    chrom, pos, ref, alt, label, match_group,
    af_overall, af_nfe, af_max, gerp, alphamissense, sift, foldx,
    is_lof_loftee, is_lof_vep, consequence_severity,
    has_alphamissense, has_sift, has_foldx,
    pip (PIP from credible_set; NaN for negatives), source ('positive'|'control')

Usage:
    python scripts/44_open_targets_matched_controls.py \\
        --positives data/processed/open_targets/gwas_polygenic.parquet \\
        --variant-features data/processed/open_targets/variant_features.parquet \\
        --credible-set-dir data/raw/open_targets/credible_set \\
        --out data/processed/open_targets/gwas_matched9.parquet \\
        --n-pos 1140 --neg-per-pos 9 --af-window 0.05 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def collect_credible_variants(credible_set_dir: Path) -> set[str]:
    """variantId set across all credible_set parts (any studyType)."""
    parts = sorted(credible_set_dir.glob("part-*.parquet"))
    seen: set[str] = set()
    for p in tqdm(parts, desc="credible-set ids"):
        df = pd.read_parquet(p, columns=["locus"])
        for arr in df["locus"]:
            if arr is None:
                continue
            for entry in arr:
                if entry is None:
                    continue
                vid = entry.get("variantId")
                if vid is not None:
                    seen.add(vid)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", type=Path, required=True,
                    help="Output of scripts/41_open_targets_parse.py")
    ap.add_argument("--variant-features", type=Path, required=True,
                    help="Output of scripts/43_open_targets_features.py")
    ap.add_argument("--credible-set-dir", type=Path, required=True,
                    help="To exclude credible-set variants from the negative pool")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    ap.add_argument("--n-pos", type=int, default=1140,
                    help="Number of positives to keep (TraitGym Complex = 1140).")
    ap.add_argument("--neg-per-pos", type=int, default=9,
                    help="Matched negatives per positive (9 ⇒ π_+=0.10).")
    ap.add_argument("--af-window", type=float, default=0.05)
    ap.add_argument("--pos-pip-threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cached-credible-ids", type=Path, default=None,
                    help="Optional path to cache the credible-set variantId set "
                         "as JSON (saves ~1 min on repeated runs).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---------- positives ----------
    print(f"[load] positives: {args.positives}")
    pos_df = pd.read_parquet(args.positives)
    pos_df = pos_df[pos_df["label"] == 1].copy()
    print(f"[positives] {len(pos_df):,} OT high-PIP rows")

    pos_df = (pos_df.sort_values("posteriorProbability", ascending=False)
                     .drop_duplicates(subset="variantId", keep="first"))
    print(f"  dedup: {len(pos_df):,} unique variants")

    # ---------- features ----------
    print(f"[load] variant features: {args.variant_features}")
    feat = pd.read_parquet(args.variant_features)
    feat["chromosome"] = feat["chromosome"].astype(str)
    print(f"  {len(feat):,} variant rows; {feat['chromosome'].nunique()} chroms")

    feat = feat.drop_duplicates(subset="variantId")
    pos_with_feat = feat.merge(pos_df[["variantId", "posteriorProbability"]],
                                on="variantId", how="inner")
    print(f"[join] pos with features: {len(pos_with_feat):,} of "
          f"{len(pos_df):,} positives")

    pos_with_feat = pos_with_feat[pos_with_feat["af_overall"].notna()
                                    & (pos_with_feat["af_overall"] > 0)]
    print(f"  drop AF NA: {len(pos_with_feat):,} positives remain")

    if len(pos_with_feat) > args.n_pos:
        pos_with_feat = (pos_with_feat
                          .sample(n=args.n_pos, random_state=args.seed)
                          .reset_index(drop=True))
    print(f"[positives subsampled] {len(pos_with_feat):,}")

    # ---------- credible-set exclusion set ----------
    if args.cached_credible_ids is not None and args.cached_credible_ids.exists():
        credible_ids = set(json.loads(args.cached_credible_ids.read_text()))
        print(f"[cache] credible-set ids: {len(credible_ids):,} "
              f"(from {args.cached_credible_ids})")
    else:
        credible_ids = collect_credible_variants(args.credible_set_dir)
        print(f"[scanned] credible-set ids: {len(credible_ids):,}")
        if args.cached_credible_ids is not None:
            args.cached_credible_ids.parent.mkdir(parents=True, exist_ok=True)
            args.cached_credible_ids.write_text(json.dumps(sorted(credible_ids)))
            print(f"  cached: {args.cached_credible_ids}")

    # ---------- negative pool ----------
    neg_pool = feat[~feat["variantId"].isin(credible_ids)
                     & feat["af_overall"].notna()
                     & (feat["af_overall"] > 0)].copy()
    print(f"[neg pool] {len(neg_pool):,} variants outside any OT credible set "
          f"(of {len(feat):,})")

    # Pre-bin negatives by AF (sorted) for fast windowed lookup
    neg_pool = neg_pool.sort_values("af_overall").reset_index(drop=True)
    neg_af = neg_pool["af_overall"].to_numpy()

    # ---------- AF-matched sampling ----------
    print(f"[match] sampling {args.neg_per_pos} negs / pos within "
          f"af_window=±{args.af_window}")
    neg_rows: list[pd.DataFrame] = []
    match_groups: list[int] = []
    pos_rows = pos_with_feat.copy()
    pos_rows["match_group"] = np.arange(len(pos_rows))

    used_neg_idx: set[int] = set()
    for grp_id, row in tqdm(pos_rows.iterrows(),
                              total=len(pos_rows), desc="AF-match"):
        af = float(row["af_overall"])
        lo = af - args.af_window
        hi = af + args.af_window
        i_lo = int(np.searchsorted(neg_af, lo, side="left"))
        i_hi = int(np.searchsorted(neg_af, hi, side="right"))
        candidate_idx = np.arange(i_lo, i_hi)
        candidate_idx = np.array([i for i in candidate_idx
                                    if i not in used_neg_idx])
        if len(candidate_idx) < args.neg_per_pos:
            # widen until we get enough
            for w_mult in (2, 4, 8):
                lo_w = af - args.af_window * w_mult
                hi_w = af + args.af_window * w_mult
                i_lo = int(np.searchsorted(neg_af, lo_w, side="left"))
                i_hi = int(np.searchsorted(neg_af, hi_w, side="right"))
                candidate_idx = np.array([i for i in range(i_lo, i_hi)
                                            if i not in used_neg_idx])
                if len(candidate_idx) >= args.neg_per_pos:
                    break
        if len(candidate_idx) == 0:
            continue
        take = rng.choice(candidate_idx,
                           size=min(args.neg_per_pos, len(candidate_idx)),
                           replace=False)
        used_neg_idx.update(take.tolist())
        chunk = neg_pool.iloc[take].copy()
        chunk["match_group"] = grp_id
        neg_rows.append(chunk)
    if not neg_rows:
        raise RuntimeError("No matched negatives sampled. Check AF column.")
    neg_full = pd.concat(neg_rows, ignore_index=True)

    # ---------- assemble final dataset ----------
    pos_rows["label"] = 1
    pos_rows["pip"] = pos_rows["posteriorProbability"].astype(float)
    pos_rows["source"] = "positive"

    neg_full["label"] = 0
    neg_full["pip"] = float("nan")
    neg_full["source"] = "control"

    out = pd.concat([pos_rows.drop(columns=["posteriorProbability"]), neg_full],
                     ignore_index=True)

    # rename to TraitGym-aligned schema
    out = out.rename(columns={"chromosome": "chrom", "position": "pos"})

    cols = ["chrom", "pos", "ref", "alt", "variantId",
            "label", "match_group", "source", "pip",
            "af_overall", "af_nfe", "af_max",
            "gerp", "alphamissense", "sift", "foldx",
            "is_lof_loftee", "is_lof_vep", "consequence_severity",
            "has_alphamissense", "has_sift", "has_foldx"]
    out = out[[c for c in cols if c in out.columns]]
    print(f"\n[final] {len(out):,} rows  "
          f"({int(out['label'].sum()):,} pos, "
          f"π_+ = {out['label'].mean():.4f})")

    # split for downstream chrom-LOO
    test_chroms = {"17", "18", "19", "20", "21", "22", "X"}
    out["split"] = np.where(out["chrom"].astype(str).isin(test_chroms),
                              "test", "train")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"saved: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")

    if args.summary_out is not None:
        per_chrom = (out.groupby("chrom", as_index=False)
                        .agg(n=("label", "size"), pi_pos=("label", "mean")))
        summary = {
            "n_total": int(len(out)),
            "n_pos": int(out["label"].sum()),
            "pi_pos": float(out["label"].mean()),
            "n_train": int((out.split == "train").sum()),
            "n_test": int((out.split == "test").sum()),
            "n_match_groups": int(out["match_group"].nunique()),
            "af_window": args.af_window,
            "neg_per_pos": args.neg_per_pos,
            "per_chrom": per_chrom.to_dict("records"),
            "source_counts": out["source"].value_counts().to_dict(),
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2))
        print(f"saved: {args.summary_out}")


if __name__ == "__main__":
    main()
