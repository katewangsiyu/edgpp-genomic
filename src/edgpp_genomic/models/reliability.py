"""Reliability estimator: predicts per-SNP teacher-reliability weight w ∈ [0,1].

Input: side features [teacher_replicate_var, MAF, phyloP, dist_to_TSS, ...]
Output: scalar w per SNP.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from omegaconf import DictConfig


class ReliabilityEstimator(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int] | tuple[int, ...] = (128, 64)):
        super().__init__()
        dims = [input_dim, *hidden, 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)  # (B,)


def build_reliability(cfg: DictConfig) -> nn.Module:
    return ReliabilityEstimator(input_dim=cfg.input_dim, hidden=tuple(cfg.hidden))
