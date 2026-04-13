"""TraitGym dataset adapter — uses precomputed Borzoi teacher scores.

Pairs test.parquet (variants + labels) with Borzoi_L2_L2.parquet (6-D
precomputed teacher variant-effect scores). Teacher is NOT forwarded;
it is simply indexed by row.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
from .fasta import WindowExtractor


BORZOI_L2L2_COLS = ["CAGE", "DNASE", "ATAC", "CHIP", "RNA", "all"]
_VALID_CHROMS = {str(i) for i in range(1, 23)} | {"X", "Y", "M",
                                                   *[f"chr{i}" for i in range(1, 23)],
                                                   "chrX", "chrY", "chrM"}


class TraitGymDataset(Dataset):
    def __init__(
        self,
        test_parquet: str,
        teacher_parquet: str,
        fasta_path: str,
        seq_len: int,
        max_rows: int | None = None,
        seed: int = 42,
        normalize_teacher: bool = True,
    ):
        test = pd.read_parquet(test_parquet)
        teacher = pd.read_parquet(teacher_parquet)
        assert len(test) == len(teacher), (
            f"row mismatch: test={len(test)} teacher={len(teacher)}"
        )
        teacher = teacher[BORZOI_L2L2_COLS].reset_index(drop=True)
        test = test.reset_index(drop=True)
        df = pd.concat([test, teacher], axis=1)

        df = df[df["chrom"].astype(str).isin(_VALID_CHROMS)].reset_index(drop=True)

        if max_rows is not None and max_rows < len(df):
            df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

        self.df = df
        self.extractor = WindowExtractor(fasta_path, seq_len)

        tss = df["tss_dist"].astype(np.float32).to_numpy()
        teacher_raw = df[BORZOI_L2L2_COLS].astype(np.float32).to_numpy().copy()

        # Side features use RAW teacher magnitude (has physical meaning).
        self._side = np.stack([
            np.log1p(np.abs(tss)),
            np.linalg.norm(teacher_raw, axis=1),
            teacher_raw.std(axis=1),
        ], axis=1).astype(np.float32).copy()

        # Teacher target is normalized (prevents fp16 overflow in distill MSE).
        if normalize_teacher:
            self.teacher_mean = teacher_raw.mean(0, keepdims=True).astype(np.float32)
            self.teacher_std = (teacher_raw.std(0, keepdims=True) + 1e-6).astype(np.float32)
            self._teacher = ((teacher_raw - self.teacher_mean) / self.teacher_std).astype(np.float32).copy()
        else:
            self.teacher_mean = np.zeros((1, teacher_raw.shape[1]), dtype=np.float32)
            self.teacher_std = np.ones((1, teacher_raw.shape[1]), dtype=np.float32)
            self._teacher = teacher_raw

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        ref_oh, alt_oh = self.extractor.get_ref_alt(
            str(row["chrom"]), int(row["pos"]), row["ref"], row["alt"]
        )
        return {
            "ref": torch.from_numpy(ref_oh),
            "alt": torch.from_numpy(alt_oh),
            "teacher_score": torch.from_numpy(self._teacher[idx]),
            "side_features": torch.from_numpy(self._side[idx]),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "snp_id": f"{row['chrom']}:{row['pos']}:{row['ref']}>{row['alt']}",
        }


def build_traitgym(cfg: DictConfig) -> TraitGymDataset:
    return TraitGymDataset(
        test_parquet=cfg.test_parquet,
        teacher_parquet=cfg.teacher_parquet,
        fasta_path=cfg.fasta_path,
        seq_len=cfg.seq_len,
        max_rows=cfg.get("max_rows", None),
        seed=cfg.get("seed", 42),
        normalize_teacher=cfg.get("normalize_teacher", True),
    )
