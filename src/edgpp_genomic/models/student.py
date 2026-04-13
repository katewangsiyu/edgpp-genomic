"""Compact student model for distillation."""
from __future__ import annotations
import torch
import torch.nn as nn
from omegaconf import DictConfig


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int = 5, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel, stride=stride, padding=kernel // 2)
        self.norm = nn.GroupNorm(min(8, out_c), out_c)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CompactStudent(nn.Module):
    """Small CNN student, output track count matches teacher."""
    def __init__(self, n_tracks: int, hidden: int = 128, n_layers: int = 4, **_):
        super().__init__()
        self.stem = nn.Conv1d(4, hidden, 15, padding=7)
        chs = [hidden] + [hidden * (2 ** min(i, 2)) for i in range(1, n_layers + 1)]
        self.blocks = nn.Sequential(*[ConvBlock(chs[i], chs[i + 1]) for i in range(n_layers)])
        self.head = nn.Conv1d(chs[-1], n_tracks, 1)
        self.n_tracks = n_tracks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))  # (B, C, T)

    def sed(self, ref: torch.Tensor, alt: torch.Tensor, center_frac: float = 0.25) -> torch.Tensor:
        """SED = sum(alt - ref) over center region, per track."""
        r = self.forward(ref)
        a = self.forward(alt)
        diff = a - r
        T = diff.shape[-1]
        cw = max(1, int(T * center_frac))
        cs = (T - cw) // 2
        return diff[..., cs:cs + cw].sum(dim=-1)  # (B, C)


def build_student(cfg: DictConfig) -> nn.Module:
    if cfg.kind == "compact_cnn":
        return CompactStudent(n_tracks=cfg.n_tracks, hidden=cfg.hidden, n_layers=cfg.n_layers)
    raise ValueError(f"Unknown student kind: {cfg.kind}")
