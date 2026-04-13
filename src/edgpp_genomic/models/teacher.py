"""Teacher wrappers.

FakeTeacher: small random CNN for T4 Phase 0 smoke (no flash-attn needed).
FlashzoiTeacher: wraps johahi/borzoi-pytorch for 5090 (requires flash-attn).

Both expose:
    forward(seq_onehot) -> (B, n_tracks, T)       # track × length output
"""
from __future__ import annotations
import torch
import torch.nn as nn
from omegaconf import DictConfig


class FakeTeacher(nn.Module):
    """Random-init CNN. Only used on T4 to validate pipeline / loss / gate logic."""
    def __init__(self, n_tracks: int = 16, n_filters: int = 64, **_):
        super().__init__()
        self.n_tracks = n_tracks
        self.stem = nn.Conv1d(4, n_filters, kernel_size=15, padding=7)
        self.blocks = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(n_filters, n_filters * 2, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(n_filters * 2, n_filters * 2, 5, stride=2, padding=2), nn.GELU(),
        )
        self.head = nn.Conv1d(n_filters * 2, n_tracks, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class FlashzoiTeacher(nn.Module):
    """Wraps johahi/borzoi-pytorch replicates. Requires flash-attn (5090 only)."""
    def __init__(self, replicate: int | list[int] = 0, **_):
        super().__init__()
        try:
            from borzoi_pytorch import Borzoi
        except ImportError as e:
            raise ImportError(
                "borzoi-pytorch not installed. On 5090: "
                "git clone https://github.com/johahi/borzoi-pytorch baselines/borzoi-pytorch && "
                "pip install -e baselines/borzoi-pytorch/"
            ) from e

        reps = [replicate] if isinstance(replicate, int) else list(replicate)
        self.models = nn.ModuleList([
            Borzoi.from_pretrained(f"johahi/borzoi-replicate-{r}") for r in reps
        ])
        self.n_replicates = len(reps)
        for m in self.models:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [m(x) for m in self.models]   # each: (B, C, T)
        return torch.stack(outs, dim=0).mean(dim=0)  # (B, C, T)

    @torch.no_grad()
    def predict_mean_var(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outs = [m(x) for m in self.models]
        stacked = torch.stack(outs, dim=0)
        mean = stacked.mean(dim=0)
        var = stacked.var(dim=0) if self.n_replicates > 1 else torch.zeros_like(mean)
        return mean, var


def build_teacher(cfg: DictConfig) -> nn.Module:
    kind = cfg.kind
    if kind == "fake":
        return FakeTeacher(n_tracks=cfg.n_tracks, n_filters=cfg.get("n_filters", 64))
    if kind == "flashzoi_single":
        return FlashzoiTeacher(replicate=cfg.get("replicate", 0))
    if kind == "flashzoi_4rep":
        return FlashzoiTeacher(replicate=[0, 1, 2, 3])
    raise ValueError(f"Unknown teacher kind: {kind}")
