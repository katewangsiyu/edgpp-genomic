"""Genome window extraction + one-hot encoding."""
from __future__ import annotations
import numpy as np
from pyfaidx import Fasta


BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "a": 0, "c": 1, "g": 2, "t": 3}


def one_hot_encode(seq: str) -> np.ndarray:
    """String → (4, L) float32. Unknown bases → uniform 0.25."""
    L = len(seq)
    arr = np.full((4, L), 0.25, dtype=np.float32)
    for i, b in enumerate(seq):
        idx = BASE_TO_IDX.get(b)
        if idx is not None:
            arr[:, i] = 0.0
            arr[idx, i] = 1.0
    return arr


class WindowExtractor:
    def __init__(self, fasta_path: str, seq_len: int):
        self.fa = Fasta(fasta_path)
        self.seq_len = seq_len
        assert seq_len % 2 == 0, "seq_len must be even"

    def _resolve_chrom(self, chrom: str) -> str:
        if chrom in self.fa:
            return chrom
        alt = chrom.lstrip("chr") if chrom.startswith("chr") else f"chr{chrom}"
        if alt in self.fa:
            return alt
        raise KeyError(f"{chrom} not found in fasta")

    def get_ref_alt(self, chrom: str, pos: int, ref: str, alt: str):
        """Extract ±seq_len/2 centered on pos (1-based). Returns (ref_oh, alt_oh) each (4, L)."""
        chrom = self._resolve_chrom(chrom)
        half = self.seq_len // 2
        # pyfaidx is 1-based inclusive like VCF
        start = max(1, pos - half + 1)
        end = pos + half
        chrom_len = len(self.fa[chrom])
        end = min(chrom_len, end)

        seq = str(self.fa[chrom][start - 1:end]).upper()
        # Pad to exact seq_len if clipped at chromosome ends
        pad_left = half - (pos - start + 1)
        pad_right = self.seq_len - len(seq) - pad_left
        pad_left = max(0, pad_left)
        pad_right = max(0, pad_right)
        seq = "N" * pad_left + seq + "N" * pad_right
        if len(seq) != self.seq_len:
            seq = seq[:self.seq_len] if len(seq) > self.seq_len else seq + "N" * (self.seq_len - len(seq))

        ref_oh = one_hot_encode(seq)
        center = half
        alt_seq = seq[:center] + alt[0] + seq[center + 1:]
        alt_oh = one_hot_encode(alt_seq)
        return ref_oh, alt_oh
