"""SNP Dataset returning ref/alt one-hot + side features + optional label."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
from .fasta import WindowExtractor


class SNPDataset(Dataset):
    def __init__(self, snp_df: pd.DataFrame, fasta_path: str, seq_len: int,
                 side_feature_cols: list[str] | None = None,
                 label_col: str | None = None):
        self.df = snp_df.reset_index(drop=True)
        self.extractor = WindowExtractor(fasta_path, seq_len)
        self.side_feature_cols = side_feature_cols or []
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        ref_oh, alt_oh = self.extractor.get_ref_alt(
            row["chrom"], int(row["pos"]), row["ref"], row["alt"]
        )
        item = {
            "ref": torch.from_numpy(ref_oh),
            "alt": torch.from_numpy(alt_oh),
            "snp_id": str(row.get("snp_id", f"snp{idx}")),
        }
        if self.side_feature_cols:
            feats = np.array([float(row.get(c, 0.0)) for c in self.side_feature_cols],
                             dtype=np.float32)
            item["side_features"] = torch.from_numpy(feats)
        if self.label_col and self.label_col in row:
            item["label"] = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        return item


def build_dataset(cfg: DictConfig) -> SNPDataset:
    snp_path = Path(cfg.snp_subset_path)
    if snp_path.suffix == ".parquet":
        df = pd.read_parquet(snp_path)
    else:
        df = pd.read_csv(snp_path)
    return SNPDataset(
        df, cfg.fasta_path, cfg.seq_len,
        side_feature_cols=list(cfg.get("side_feature_cols", [])),
        label_col=cfg.get("label_col", None),
    )
