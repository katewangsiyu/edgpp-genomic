"""SelectiveDistillLoss — the EDG++ core.

L = [w * distill_loss(student, teacher)]_(gated)
  + lambda_task * [task_loss(student, label)]_(inv-gated)

Adaptive threshold = alpha * EMA(global median of w) + (1 - alpha) * local median of w.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveDistillLoss(nn.Module):
    def __init__(self, gate_threshold: float = 0.5, lambda_task: float = 0.3,
                 adaptive: bool = False, adaptive_alpha: float = 0.5,
                 ema_beta: float = 0.99,
                 w_reg_lambda: float = 0.05,
                 w_reg_target: float = 0.5):
        super().__init__()
        self.threshold = gate_threshold
        self.lambda_task = lambda_task
        self.adaptive = adaptive
        self.alpha = adaptive_alpha
        self._ema_beta = ema_beta
        self._global_median = 0.5  # EMA state
        # Prevent reliability collapse: L2 penalty pulling w toward target (0.5).
        self.w_reg_lambda = w_reg_lambda
        self.w_reg_target = w_reg_target

    def _get_threshold(self, w: torch.Tensor) -> float:
        if not self.adaptive:
            return self.threshold
        local_median = float(w.detach().median())
        self._global_median = self._ema_beta * self._global_median + (1 - self._ema_beta) * local_median
        return self.alpha * self._global_median + (1 - self.alpha) * local_median

    def forward(self, student_out: torch.Tensor, teacher_out: torch.Tensor,
                w: torch.Tensor, label: torch.Tensor | None = None) -> dict:
        """
        student_out, teacher_out: (B, C)  per-SNP per-track SED
        w: (B,)   reliability weight
        label: (B,) optional ground-truth effect coef
        """
        per_snp_distill = F.mse_loss(student_out, teacher_out, reduction="none").mean(dim=-1)
        thresh = self._get_threshold(w)
        gate = (w >= thresh).float()

        # Selective distill: only reliable samples contribute, weighted by w
        soft_w = w * gate
        distill_loss = (soft_w * per_snp_distill).sum() / (soft_w.sum() + 1e-8)

        total = distill_loss
        comp = {"distill": distill_loss.detach()}

        if label is not None and self.lambda_task > 0:
            student_scalar = student_out.mean(dim=-1)
            per_snp_task = F.mse_loss(student_scalar, label, reduction="none")
            inv = (1.0 - gate)
            task_loss = (inv * per_snp_task).sum() / (inv.sum() + 1e-8)
            total = distill_loss + self.lambda_task * task_loss
            comp["task"] = task_loss.detach()

        if self.w_reg_lambda > 0:
            w_reg = (w - self.w_reg_target).pow(2).mean()
            total = total + self.w_reg_lambda * w_reg
            comp["w_reg"] = w_reg.detach()

        comp["total"] = total.detach()
        comp["threshold"] = torch.tensor(float(thresh))
        comp["mean_w"] = w.detach().mean()
        comp["frac_gated_in"] = gate.mean().detach()
        return {"loss": total, "components": comp}
