"""Extract one-hot DNA windows for DEGU-actual training.

For each variant (chrom, pos, ref, alt) in a TraitGym parquet, extract ±W/2 bp
from hg38 and produce a (L, 8) one-hot tensor:
    channels 0-3: ref sequence (A, C, G, T)
    channels 4-7: alt sequence (A, C, G, T)

Output: <out-dir>/<variant_id>.npy with shape (L, 8), plus a manifest CSV
mapping variant_id → parquet row index.

variant_id = f"chr{chrom}_{pos}_{ref}_{alt}"

Usage:
    python scripts/36_extract_dna_windows.py \\
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --hg38 data/raw/hg38.fa \\
        --window 200 \\
        --out-dir data/raw/traitgym/mendelian_traits_matched_9/dna_windows
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyfaidx import Fasta
from tqdm.auto import tqdm


NUC_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}


def one_hot(seq: str) -> np.ndarray:
    L = len(seq)
    arr = np.zeros((L, 4), dtype=np.float32)
    for i, c in enumerate(seq):
        idx = NUC_MAP.get(c, -1)
        if idx >= 0:
            arr[i, idx] = 1.0
        # Else (N, softmask, indel char): leave zeros — model treats as ambiguous
    return arr


def extract_window(fa: Fasta, chrom: str, pos: int, window: int) -> str:
    """Extract window bp centered on pos (1-based inclusive)."""
    half = window // 2
    chrom_key = f"chr{chrom}" if not chrom.startswith("chr") else chrom
    # pyfaidx uses 1-based, closed interval via fa[chrom][start:end]
    start = max(1, pos - half)
    end = pos + (window - half) - 1
    seq = str(fa[chrom_key][start - 1:end])
    # Pad if near chrom boundary
    expected_len = end - start + 1
    if len(seq) < window:
        seq = seq + "N" * (window - len(seq))
    return seq[:window]


def apply_variant(seq: str, window: int, ref: str, alt: str) -> str:
    """Return the ALT-allele sequence by replacing the center position."""
    center = window // 2
    # Confirm ref matches (pyfaidx returns uppercase)
    ref_upper = ref.upper()
    if not seq[center:center + len(ref_upper)].upper() == ref_upper:
        # Some TraitGym rows may have ref mismatch due to hg19 vs hg38 or indel
        # handling. Skip cleanly.
        return None
    return seq[:center] + alt.upper() + seq[center + len(ref_upper):]


def variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    chrom_c = chrom if chrom.startswith("chr") else f"chr{chrom}"
    return f"{chrom_c}_{pos}_{ref}_{alt}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--hg38", required=True)
    ap.add_argument("--window", type=int, default=200,
                    help="Total window length (must be even)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip variants whose .npy already exists")
    args = ap.parse_args()

    assert args.window % 2 == 0, "window must be even"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    print(f"[load] n={len(V)} variants from {args.test_parquet}")

    fa = Fasta(args.hg38, as_raw=True)

    manifest = []
    n_written = 0
    n_skipped = 0
    n_failed = 0
    for i, row in tqdm(V.iterrows(), total=len(V), desc="extract"):
        vid = variant_id(row["chrom"], row["pos"], row["ref"], row["alt"])
        out_file = out / f"{vid}.npy"
        if args.skip_existing and out_file.exists():
            n_skipped += 1
            manifest.append({"row": i, "variant_id": vid, "status": "skipped_existing"})
            continue
        try:
            ref_seq = extract_window(fa, row["chrom"], int(row["pos"]), args.window)
            alt_seq = apply_variant(ref_seq, args.window, row["ref"], row["alt"])
            if alt_seq is None:
                n_failed += 1
                manifest.append({"row": i, "variant_id": vid, "status": "ref_mismatch"})
                continue
            ref_oh = one_hot(ref_seq)
            alt_oh = one_hot(alt_seq)
            stacked = np.concatenate([ref_oh, alt_oh], axis=1)  # (L, 8)
            np.save(out_file, stacked)
            n_written += 1
            manifest.append({"row": i, "variant_id": vid, "status": "ok"})
        except Exception as e:  # noqa: BLE001
            n_failed += 1
            manifest.append({"row": i, "variant_id": vid, "status": f"error: {e}"})

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(out / "manifest.csv", index=False)
    (out / "meta.json").write_text(json.dumps({
        "n_variants": int(len(V)),
        "n_written": n_written,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "window": args.window,
        "hg38": args.hg38,
        "source_parquet": args.test_parquet,
    }, indent=2))

    print(f"\n[done] written={n_written}  skipped={n_skipped}  failed={n_failed}")
    print(f"saved: {out}/ ({n_written} .npy files, manifest.csv, meta.json)")


if __name__ == "__main__":
    main()
