"""T4 smoke with REAL TraitGym data and precomputed Borzoi teacher scores.

Diff from 03_smoke.py:
  * No teacher forward — teacher_score indexed from parquet (Borzoi_L2_L2, 6-D)
  * Uses TraitGymDataset (mendelian_traits_matched_9 by default)
  * Adaptive reliability gate ON (so selective behavior is observable)

Usage:
    python scripts/03b_smoke_real.py --config configs/t4_real.yaml
"""
from __future__ import annotations
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from edgpp_genomic.config import load_config
from edgpp_genomic.models import build_student, build_reliability
from edgpp_genomic.data import build_traitgym
from edgpp_genomic.training import SelectiveDistillLoss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[cfg.precision]

    print(f"[cfg] {cfg.name}  device={device}  dtype={cfg.precision}")

    student = build_student(cfg.student).to(device)
    reliability = build_reliability(cfg.reliability).to(device)
    loss_fn = SelectiveDistillLoss(
        gate_threshold=cfg.training.gate_threshold,
        lambda_task=cfg.training.lambda_task,
        adaptive=cfg.training.get("gate_adaptive", False),
        adaptive_alpha=cfg.training.get("gate_adaptive_alpha", 0.5),
    )
    optim = AdamW(
        list(student.parameters()) + list(reliability.parameters()),
        lr=cfg.training.lr,
    )

    dataset = build_traitgym(cfg.data)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )
    pos_frac = float(dataset.df["label"].mean())
    print(f"[data] {len(dataset)} SNPs (pos {pos_frac:.2%}) | "
          f"batch={cfg.data.batch_size} | seq_len={cfg.data.seq_len}")

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    student.train(); reliability.train()
    step = 0
    max_steps = cfg.training.max_steps or 10**9
    log_every = cfg.logging.log_every_n_steps

    for _ in range(cfg.training.n_epochs):
        for batch in loader:
            ref = batch["ref"].to(device)
            alt = batch["alt"].to(device)
            sidef = batch["side_features"].to(device)
            t_sed = batch["teacher_score"].to(device)
            label = batch["label"].to(device)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                s_sed = student.sed(ref, alt)
                w = reliability(sidef)
                out = loss_fn(s_sed, t_sed, w, label=label)

            optim.zero_grad(set_to_none=True)
            if dtype == torch.float16:
                scaler.scale(out["loss"]).backward()
                scaler.step(optim)
                scaler.update()
            else:
                out["loss"].backward()
                optim.step()

            if step % log_every == 0:
                c = out["components"]
                task_str = f" task {c['task'].item():.4f}" if "task" in c else ""
                print(f"step {step:4d} | total {c['total'].item():.4f} | "
                      f"distill {c['distill'].item():.4f}{task_str} | "
                      f"w̄ {c['mean_w'].item():.3f} | "
                      f"gated_in {c['frac_gated_in'].item():.2f} | "
                      f"thresh {c['threshold'].item():.3f}")
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    print("\n[smoke_real] PASSED. Pipeline runs on real TraitGym data with precomputed Borzoi teacher.")


if __name__ == "__main__":
    main()
