"""VCF parsing (plain text or gzipped)."""
from __future__ import annotations
import gzip
from pathlib import Path
import pandas as pd


def load_vcf_as_df(path: str | Path) -> pd.DataFrame:
    """Load VCF to DataFrame with cols: chrom, pos, snp_id, ref, alt, + INFO numeric fields."""
    path = str(path)
    rows = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            chrom, pos, snp_id, ref, alt = parts[:5]
            rec = {
                "chrom": chrom if chrom.startswith("chr") else f"chr{chrom}",
                "pos": int(pos),
                "snp_id": snp_id,
                "ref": ref,
                "alt": alt,
            }
            if len(parts) >= 8:
                for kv in parts[7].split(";"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        try:
                            rec[k.lower()] = float(v)
                        except ValueError:
                            rec[k.lower()] = v
            rows.append(rec)
    return pd.DataFrame(rows)
