"""Mutation-level features for ProteinGym without GPU / neural baselines.

For each (wild_type_seq, mutant_spec like 'A16C') we compute:
  - One-hot encodings of WT and MT amino acids (40-dim)
  - Position normalized by sequence length
  - BLOSUM62 substitution score
  - Physicochemical deltas (hydrophobicity, volume, charge, polarity)
  - Local context: AA composition in ±5 neighborhood (20-dim)

Total ~65 features per mutation, all CPU-computable. Output is a parquet with
one row per ProteinGym substitution row.

Usage:
    python scripts/37_proteingym_features.py \\
        --in-parquet data/raw/proteingym/all_substitutions.parquet \\
        --out-parquet data/raw/proteingym/features.parquet
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# Standard 20 AAs in alphabetical order
AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AAS)}

# Kyte-Doolittle hydrophobicity
HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
    "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
    "W": -0.9, "Y": -1.3,
}

# Volume (van der Waals, Å^3) — Chothia 1975
VOLUME = {
    "A": 67, "C": 86, "D": 91, "E": 109, "F": 135, "G": 48,
    "H": 118, "I": 124, "K": 135, "L": 124, "M": 124, "N": 96,
    "P": 90, "Q": 114, "R": 148, "S": 73, "T": 93, "V": 105,
    "W": 163, "Y": 141,
}

# Charge at pH 7
CHARGE = {
    "A": 0, "C": 0, "D": -1, "E": -1, "F": 0, "G": 0,
    "H": 0.1, "I": 0, "K": 1, "L": 0, "M": 0, "N": 0,
    "P": 0, "Q": 0, "R": 1, "S": 0, "T": 0, "V": 0,
    "W": 0, "Y": 0,
}

# Polarity (Grantham)
POLARITY = {
    "A": 8.1, "C": 5.5, "D": 13.0, "E": 12.3, "F": 5.2, "G": 9.0,
    "H": 10.4, "I": 5.2, "K": 11.3, "L": 4.9, "M": 5.7, "N": 11.6,
    "P": 8.0, "Q": 10.5, "R": 10.5, "S": 9.2, "T": 8.6, "V": 5.9,
    "W": 5.4, "Y": 6.2,
}

# BLOSUM62 from NCBI — only the diagonal and symmetric upper triangle encoded inline
BLOSUM62_STR = """
A R N D C Q E G H I L K M F P S T W Y V
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""


def parse_blosum62() -> dict[tuple[str, str], int]:
    lines = [ln for ln in BLOSUM62_STR.strip().split("\n") if ln.strip()]
    cols = lines[0].split()
    matrix: dict[tuple[str, str], int] = {}
    for line in lines[1:]:
        toks = line.split()
        row_aa = toks[0]
        for col_aa, val in zip(cols, toks[1:]):
            matrix[(row_aa, col_aa)] = int(val)
    return matrix


BLOSUM62 = parse_blosum62()


MUT_REGEX = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def parse_mutant(mut: str) -> tuple[str, int, str] | None:
    """'A16C' → ('A', 16, 'C'). Multi-mutants (e.g. 'A1C:D2E') not supported."""
    m = MUT_REGEX.match(mut.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def local_composition(seq: str, pos_0based: int, window: int = 5) -> np.ndarray:
    lo = max(0, pos_0based - window)
    hi = min(len(seq), pos_0based + window + 1)
    sub = seq[lo:hi]
    counts = np.zeros(20, dtype=np.float32)
    for c in sub:
        idx = AA_IDX.get(c)
        if idx is not None:
            counts[idx] += 1
    if len(sub) > 0:
        counts /= len(sub)
    return counts


def featurize_row(wt_seq: str, mutant_str: str) -> np.ndarray | None:
    parsed = parse_mutant(mutant_str)
    if parsed is None:
        return None
    wt, pos1, mt = parsed
    if wt not in AA_IDX or mt not in AA_IDX:
        return None
    if pos1 < 1 or pos1 > len(wt_seq):
        return None
    pos0 = pos1 - 1
    if wt_seq[pos0] != wt:
        # ProteinGym occasionally has mismatches due to isoforms; fall through
        # but mark with a flag feature.
        mismatch = 1.0
    else:
        mismatch = 0.0

    wt_1hot = np.zeros(20, dtype=np.float32); wt_1hot[AA_IDX[wt]] = 1
    mt_1hot = np.zeros(20, dtype=np.float32); mt_1hot[AA_IDX[mt]] = 1
    position_norm = pos1 / max(1, len(wt_seq))
    blosum = BLOSUM62.get((wt, mt), 0)
    delta_hyd = HYDROPHOBICITY[mt] - HYDROPHOBICITY[wt]
    delta_vol = VOLUME[mt] - VOLUME[wt]
    delta_chg = CHARGE[mt] - CHARGE[wt]
    delta_pol = POLARITY[mt] - POLARITY[wt]
    local = local_composition(wt_seq, pos0, window=5)

    return np.concatenate([
        wt_1hot, mt_1hot,
        np.array([position_norm, blosum, delta_hyd, delta_vol,
                  delta_chg, delta_pol, mismatch], dtype=np.float32),
        local,
    ])  # total 20 + 20 + 7 + 20 = 67


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True)
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--max-rows-per-assay", type=int, default=5000,
                    help="Cap per-assay mutants to keep feature extraction tractable.")
    args = ap.parse_args()

    df = pd.read_parquet(args.in_parquet)
    print(f"[load] {len(df):,} rows across {df.DMS_id.nunique()} assays")

    # Cap per-assay via row index selection (preserves all columns).
    if args.max_rows_per_assay > 0:
        rng = np.random.default_rng(42)
        keep_idx = []
        for _, grp in df.groupby("DMS_id"):
            if len(grp) <= args.max_rows_per_assay:
                keep_idx.extend(grp.index.tolist())
            else:
                keep_idx.extend(rng.choice(grp.index, size=args.max_rows_per_assay, replace=False).tolist())
        df = df.loc[keep_idx].reset_index(drop=True)
        print(f"[cap] after per-assay cap ≤{args.max_rows_per_assay}: {len(df):,} rows")

    feats: list[np.ndarray | None] = []
    n_fail = 0
    for wt_seq, mutant in tqdm(zip(df["target_seq"].to_numpy(),
                                    df["mutant"].to_numpy()),
                                total=len(df), desc="featurize"):
        f = featurize_row(wt_seq, mutant)
        if f is None:
            n_fail += 1
            feats.append(None)
        else:
            feats.append(f)

    dim = next((len(f) for f in feats if f is not None), 67)
    mat = np.full((len(feats), dim), np.nan, dtype=np.float32)
    for i, f in enumerate(feats):
        if f is not None:
            mat[i] = f

    feat_df = pd.DataFrame(mat, columns=[f"f{i:02d}" for i in range(dim)])
    out_df = pd.concat([df[["DMS_id", "DMS_score", "DMS_score_bin", "mutant"]].reset_index(drop=True),
                        feat_df], axis=1)
    out_df.dropna(subset=feat_df.columns.tolist(), how="all").to_parquet(args.out_parquet, index=False)
    print(f"[done] parsed {len(df)-n_fail}/{len(df)} mutants ({n_fail} failed)")
    print(f"saved: {args.out_parquet}")


if __name__ == "__main__":
    main()
