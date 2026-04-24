"""AlphaGenome feature extraction for TraitGym.

Uses the official AlphaGenome API (Avsec et al., Nature 2026) to extract
per-variant embeddings as additional features for the HCCP aggregator.

API key required (export ALPHAGENOME_API_KEY). If the key is not set or the
package is not installed, the script emits a zero-length feature parquet with
a sentinel column so downstream pipelines can branch on presence.

Output: <out-dir>/AlphaGenome.parquet with one row per variant and columns
  AlphaGenome_emb_{0..D-1} where D depends on the selected output.

Usage (when API key available):
    export ALPHAGENOME_API_KEY=<key>
    python scripts/32_alphagenome_features.py \\
        --traitgym-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --out-dir data/raw/traitgym/mendelian_traits_matched_9/features \\
        --output-type VARIANT_EFFECT
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SENTINEL_COLUMN = "_alphagenome_unavailable"


def try_import_alphagenome():
    try:
        import alphagenome
        from alphagenome.data import genome
        from alphagenome.models import dna_client
        return dna_client, genome, alphagenome.__version__
    except ImportError:
        return None, None, None


def check_api_key() -> str | None:
    return os.environ.get("ALPHAGENOME_API_KEY")


def write_sentinel(out: Path, reason: str, variants: pd.DataFrame) -> None:
    """Emit an empty parquet with a sentinel marker so the pipeline can detect it."""
    df = variants[["chrom", "pos", "ref", "alt"]].copy() \
        if all(c in variants.columns for c in ["chrom", "pos", "ref", "alt"]) else pd.DataFrame()
    df[SENTINEL_COLUMN] = reason
    df.to_parquet(out / "AlphaGenome.parquet", index=False)
    (out / "AlphaGenome.skipped").write_text(reason + "\n")
    print(f"[skip] {reason}")
    print(f"       wrote sentinel to {out}/AlphaGenome.parquet")


def extract_features(variants: pd.DataFrame, dna_client, genome,
                      output_type: str, sequence_length: int,
                      batch_sleep: float) -> pd.DataFrame:
    """Call AlphaGenome for each variant and stack embeddings."""
    client = dna_client.create()
    feats: list[np.ndarray] = []
    failures: list[int] = []
    start = time.time()
    for i, row in variants.iterrows():
        try:
            interval = genome.Interval(
                chromosome=f"chr{row['chrom']}",
                start=int(row["pos"]) - sequence_length // 2,
                end=int(row["pos"]) + sequence_length // 2)
            variant = genome.Variant(
                chromosome=f"chr{row['chrom']}",
                position=int(row["pos"]),
                reference_bases=row["ref"],
                alternate_bases=row["alt"])
            out = client.predict_variant(
                interval=interval, variant=variant,
                requested_outputs=[getattr(dna_client.OutputType, output_type)])
            # Expected: out.reference / out.alternate with shape (L, C)
            ref_mean = np.asarray(out.reference.values).mean(axis=0)
            alt_mean = np.asarray(out.alternate.values).mean(axis=0)
            feats.append(np.concatenate([ref_mean, alt_mean, alt_mean - ref_mean]))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] variant {i}: {e}")
            failures.append(i)
            feats.append(None)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(variants) - i - 1) / rate if rate > 0 else float("inf")
            print(f"  [{i+1}/{len(variants)}] rate={rate:.1f}/s eta={eta/60:.1f}min")
        if batch_sleep > 0:
            time.sleep(batch_sleep)
    # Align to common dimension (first successful)
    dim = next((len(f) for f in feats if f is not None), 0)
    rows = []
    for v, f in zip(variants.itertuples(), feats):
        if f is None:
            rows.append(np.full(dim, np.nan))
        else:
            rows.append(f)
    arr = np.stack(rows)
    cols = [f"AlphaGenome_emb_{j}" for j in range(dim)]
    return pd.DataFrame(arr, columns=cols), failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traitgym-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--output-type", default="VARIANT_EFFECT",
                    help="AlphaGenome OutputType (VARIANT_EFFECT, DNASE, RNA_SEQ, etc.)")
    ap.add_argument("--sequence-length", type=int, default=16384,
                    help="Input window size (AlphaGenome supports up to 1 Mb)")
    ap.add_argument("--batch-sleep", type=float, default=0.0,
                    help="Seconds to sleep between API calls (rate limit)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: cap number of variants (0 = all)")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    V = pd.read_parquet(args.traitgym_parquet).reset_index(drop=True)
    if args.limit > 0:
        V = V.iloc[:args.limit].reset_index(drop=True)

    # Pre-flight checks
    api_key = check_api_key()
    if not api_key:
        write_sentinel(out, "ALPHAGENOME_API_KEY not set in environment", V)
        return

    dna_client, genome, version = try_import_alphagenome()
    if dna_client is None:
        write_sentinel(out,
                       "alphagenome Python package not installed "
                       "(pip install alphagenome)",
                       V)
        return

    print(f"[alphagenome v{version}] extracting {args.output_type} for {len(V)} variants")
    feats_df, failures = extract_features(
        V, dna_client, genome, args.output_type,
        args.sequence_length, args.batch_sleep)

    # Join chrom/pos metadata
    meta_cols = [c for c in ["chrom", "pos", "ref", "alt"] if c in V.columns]
    full = pd.concat([V[meta_cols].reset_index(drop=True),
                       feats_df.reset_index(drop=True)], axis=1)
    full.to_parquet(out / "AlphaGenome.parquet", index=False)

    (out / "AlphaGenome.meta.json").write_text(json.dumps({
        "output_type": args.output_type,
        "sequence_length": args.sequence_length,
        "n_variants": int(len(V)),
        "n_failures": len(failures),
        "dim": feats_df.shape[1],
        "alphagenome_version": version,
    }, indent=2))
    print(f"saved: {out}/AlphaGenome.parquet (dim={feats_df.shape[1]}, "
          f"failures={len(failures)}/{len(V)})")


if __name__ == "__main__":
    main()
