"""AlphaGenome feature extraction for TraitGym variants.

Uses the official AlphaGenome API (google-deepmind/alphagenome, free for
non-commercial research) to extract per-variant predictions across a panel of
tissues and compile summary features.

Output: ``<out-dir>/AlphaGenome.parquet`` with one row per variant and columns
``AlphaGenome_f{i}`` where the feature vector concatenates:
  - ref-mean and alt-mean of each (OutputType × tissue) track, log-scaled
  - alt-minus-ref delta at the variant center position
  - |delta| magnitude

API key must be set via environment variable ``ALPHAGENOME_API_KEY``.

Usage:
    export ALPHAGENOME_API_KEY=...
    python scripts/32_alphagenome_features.py \\
        --traitgym-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --out-dir data/raw/traitgym/mendelian_traits_matched_9/features \\
        --n-workers 8
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


SENTINEL_COLUMN = "_alphagenome_unavailable"

# Representative tissue panel covering common disease-relevant cell types.
# UBERON / CL ontology terms.
DEFAULT_TISSUES = (
    "UBERON:0000955",  # brain
    "UBERON:0002107",  # liver
    "UBERON:0000948",  # heart
    "UBERON:0002048",  # lung
    "UBERON:0002113",  # kidney
    "UBERON:0001157",  # transverse colon
    "UBERON:0002371",  # bone marrow
    "UBERON:0001264",  # pancreas
)

SEQ_LEN = 16384  # AlphaGenome smallest supported context
CENTER = SEQ_LEN // 2


def try_import_alphagenome():
    try:
        from alphagenome.data import genome
        from alphagenome.models import dna_client
        return dna_client, genome
    except ImportError:
        return None, None


def check_api_key() -> str | None:
    return os.environ.get("ALPHAGENOME_API_KEY")


def write_sentinel(out: Path, reason: str, variants: pd.DataFrame) -> None:
    df = variants[["chrom", "pos", "ref", "alt"]].copy() \
        if all(c in variants.columns for c in ["chrom", "pos", "ref", "alt"]) else pd.DataFrame()
    df[SENTINEL_COLUMN] = reason
    df.to_parquet(out / "AlphaGenome.parquet", index=False)
    (out / "AlphaGenome.skipped").write_text(reason + "\n")
    print(f"[skip] {reason}")
    print(f"       wrote sentinel to {out}/AlphaGenome.parquet")


@dataclass(frozen=True)
class ExtractionConfig:
    tissues: tuple[str, ...]
    output_types: tuple[str, ...]
    seq_len: int


def variant_to_feature(client, genome, var_row, cfg: ExtractionConfig,
                        dna_client) -> np.ndarray | None:
    """Call the API for one variant and return a flat feature vector."""
    chrom = str(var_row["chrom"])
    chrom_key = chrom if chrom.startswith("chr") else f"chr{chrom}"
    pos = int(var_row["pos"])
    start = max(0, pos - cfg.seq_len // 2)
    end = start + cfg.seq_len
    iv = genome.Interval(chromosome=chrom_key, start=start, end=end)
    var = genome.Variant(
        chromosome=chrom_key, position=pos,
        reference_bases=str(var_row["ref"]),
        alternate_bases=str(var_row["alt"]),
    )
    try:
        out = client.predict_variant(
            interval=iv, variant=var,
            ontology_terms=list(cfg.tissues),
            requested_outputs=[getattr(dna_client.OutputType, t)
                               for t in cfg.output_types],
        )
    except Exception:  # noqa: BLE001
        return None

    # Concatenate features across (output_type, tissue) tracks.
    feats: list[float] = []
    for output_type in cfg.output_types:
        attr = output_type.lower()
        ref_track = getattr(out.reference, attr, None)
        alt_track = getattr(out.alternate, attr, None)
        if ref_track is None or alt_track is None:
            return None
        # Values shape: (L, T) where T = len(tissues)
        ref_vals = np.asarray(ref_track.values, dtype=np.float32)
        alt_vals = np.asarray(alt_track.values, dtype=np.float32)
        # Safe log transform (predictions are non-negative)
        ref_log = np.log1p(np.maximum(ref_vals, 0.0))
        alt_log = np.log1p(np.maximum(alt_vals, 0.0))
        for t in range(ref_vals.shape[1]):
            # Summary statistics per track
            feats.append(float(ref_log[:, t].mean()))
            feats.append(float(alt_log[:, t].mean()))
            # Center-position delta (most variant-specific signal)
            feats.append(float(alt_log[CENTER, t] - ref_log[CENTER, t]))
            # Peak delta across the window
            d = alt_log[:, t] - ref_log[:, t]
            feats.append(float(np.max(np.abs(d))))
    return np.array(feats, dtype=np.float32)


def worker(args_pack):
    i, row, cfg, dna_client, genome, api_key = args_pack
    client = dna_client.create(api_key)
    f = variant_to_feature(client, genome, row, cfg, dna_client)
    return i, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traitgym-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tissues", default=",".join(DEFAULT_TISSUES),
                    help="Comma-separated UBERON/CL ontology terms")
    ap.add_argument("--output-types", default="RNA_SEQ,DNASE",
                    help="Comma-separated OutputType enum values")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--n-workers", type=int, default=4,
                    help="Parallel API threads (respect rate limits)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Debug: cap number of variants (0 = all)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip variants already written to checkpoint")
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

    dna_client, genome = try_import_alphagenome()
    if dna_client is None:
        write_sentinel(out,
                       "alphagenome Python package not installed "
                       "(pip install alphagenome)",
                       V)
        return

    cfg = ExtractionConfig(
        tissues=tuple(args.tissues.split(",")),
        output_types=tuple(args.output_types.split(",")),
        seq_len=args.seq_len,
    )
    print(f"[alphagenome] extracting {cfg.output_types} × {len(cfg.tissues)} tissues "
          f"for {len(V)} variants, seq_len={cfg.seq_len}, workers={args.n_workers}")

    checkpoint_path = out / "AlphaGenome.checkpoint.npz"
    feats: dict[int, np.ndarray | None] = {}
    if args.resume and checkpoint_path.exists():
        z = np.load(checkpoint_path, allow_pickle=True)
        for i, f in zip(z["indices"], z["feats"]):
            feats[int(i)] = f
        print(f"[resume] loaded {len(feats)} cached features")

    pending_idx = [i for i in range(len(V)) if i not in feats]
    print(f"[plan] {len(pending_idx)} variants remaining")

    t0 = time.time()
    last_ckpt = t0
    with ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        futures = {ex.submit(worker, (i, V.iloc[i], cfg, dna_client,
                                        genome, api_key)): i
                   for i in pending_idx}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="API calls"):
            i, f = fut.result()
            feats[i] = f
            now = time.time()
            # Checkpoint every ~5 min or every 500 new features
            if now - last_ckpt > 300 or len(feats) % 500 == 0:
                _save_checkpoint(checkpoint_path, feats)
                last_ckpt = now

    _save_checkpoint(checkpoint_path, feats)

    # Build final parquet aligned to V
    dim = next((len(f) for f in feats.values() if f is not None), 0)
    mat = np.full((len(V), dim), np.nan, dtype=np.float32)
    n_fail = 0
    for i in range(len(V)):
        f = feats.get(i)
        if f is None:
            n_fail += 1
        else:
            mat[i] = f

    cols = [f"AlphaGenome_f{j:04d}" for j in range(dim)]
    feat_df = pd.DataFrame(mat, columns=cols)
    full = pd.concat([
        V[["chrom", "pos", "ref", "alt"]].reset_index(drop=True),
        feat_df.reset_index(drop=True),
    ], axis=1)
    full.to_parquet(out / "AlphaGenome.parquet", index=False)

    elapsed = time.time() - t0
    (out / "AlphaGenome.meta.json").write_text(json.dumps({
        "n_variants": int(len(V)),
        "n_success": int(len(V) - n_fail),
        "n_failures": int(n_fail),
        "dim": int(dim),
        "tissues": list(cfg.tissues),
        "output_types": list(cfg.output_types),
        "seq_len": cfg.seq_len,
        "elapsed_seconds": float(elapsed),
        "throughput_variants_per_sec": float(len(V) / max(elapsed, 1e-6)),
    }, indent=2))
    print(f"[done] {len(V)-n_fail}/{len(V)} variants succeeded "
          f"({elapsed/60:.1f} min, {len(V)/elapsed:.2f} var/sec)")
    print(f"saved: {out}/AlphaGenome.parquet (dim={dim})")


def _save_checkpoint(path: Path, feats: dict[int, np.ndarray | None]) -> None:
    """Persist partial results so we can --resume after network failures."""
    indices = np.array(sorted(feats.keys()), dtype=np.int64)
    arrs = np.array([feats[i] if feats[i] is not None else None
                     for i in indices], dtype=object)
    np.savez(path, indices=indices, feats=arrs)


if __name__ == "__main__":
    main()
