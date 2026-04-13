"""Phase 0 (T4) / Phase 1 (5090) smoke test — config-driven.

Validates:
  * data pipeline (VCF → one-hot ref/alt)
  * SelectiveDistillLoss (gate + adaptive threshold)
  * ReliabilityEstimator training
  * gradient flow through teacher(frozen) → student + reliability

Usage:
    # Phase 0 (T4)
    python scripts/03_smoke.py --config configs/t4_debug.yaml
    # Phase 1 (5090)
    python scripts/03_smoke.py --config configs/gpu5090_smoke.yaml
"""
from __future__ import annotations
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from edgpp_genomic.config import load_config
from edgpp_genomic.models import build_teacher, build_student, build_reliability
from edgpp_genomic.data import build_dataset
from edgpp_genomic.training import SelectiveDistillLoss
from edgpp_genomic.evaluation import compute_sed


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

    # Optional: on 5090, assert flash-attn kernel is available
    if cfg.hardware == "5090":
        try:
            from torch.backends.cuda import sdp_kernel
            ok = sdp_kernel.is_flash_attention_available()
            print(f"[flash] flash kernel available: {ok}")
        except Exception as e:
            print(f"[flash] skipped: {e}")

    # Build modules
    teacher = build_teacher(cfg.teacher).to(device).eval()
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

    # Data
    dataset = build_dataset(cfg.data)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=True,
    )
    print(f"[data] {len(dataset)} SNPs | batch={cfg.data.batch_size} | seq_len={cfg.data.seq_len}")

    # Train
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    student.train(); reliability.train()
    step = 0
    max_steps = cfg.training.max_steps or 10**9
    log_every = cfg.logging.log_every_n_steps

    for epoch in range(cfg.training.n_epochs):
        for batch in loader:
            ref = batch["ref"].to(device)                    # (B, 4, L)
            alt = batch["alt"].to(device)
            sidef = batch["side_features"].to(device)        # (B, F)
            label = batch.get("label")
            if label is not None:
                label = label.to(device)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                with torch.no_grad():
                    t_ref = teacher(ref)
                    t_alt = teacher(alt)
                    t_sed = compute_sed(t_ref, t_alt)

                s_sed = student.sed(ref, alt)
                w = reliability(sidef)

                out = loss_fn(s_sed, t_sed.detach(), w, label=label)

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

    print("\n[smoke] PASSED. Modules wired correctly.")


if __name__ == "__main__":
    main()
