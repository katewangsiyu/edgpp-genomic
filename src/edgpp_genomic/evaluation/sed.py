"""SED (span-expression-difference) statistic."""
from __future__ import annotations
import torch


@torch.no_grad()
def compute_sed(ref_pred: torch.Tensor, alt_pred: torch.Tensor,
                center_frac: float = 0.25) -> torch.Tensor:
    """Per-track SED = sum(alt - ref) over center_frac of the output window.

    ref_pred / alt_pred: (B, n_tracks, T)
    Returns: (B, n_tracks)
    """
    diff = alt_pred - ref_pred
    T = diff.shape[-1]
    cw = max(1, int(T * center_frac))
    cs = (T - cw) // 2
    return diff[..., cs:cs + cw].sum(dim=-1)
