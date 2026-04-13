"""Train student + reliability under one of {baseline, degu, edgpp}.

All three methods share: student architecture, data, optimizer, max_steps.
They differ in:
  * baseline — pure MSE distillation (no gate, no reliability, no uncertainty head)
  * degu    — heteroscedastic NLL (student doubles its output: mean + log_var)
  * edgpp   — SelectiveDistillLoss with reliability gate (ours)

Chrom-based train/val split (val = chr17-22, X) follows TraitGym convention.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW

from edgpp_genomic.config import load_config
from edgpp_genomic.models.student import CompactStudent
from edgpp_genomic.models import build_reliability
from edgpp_genomic.data import build_traitgym
from edgpp_genomic.training import SelectiveDistillLoss


VAL_CHROMS = {"17", "18", "19", "20", "21", "22", "X"}


def build_method_student(cfg, method: str, device):
    n = cfg.student.n_tracks
    # DEGU doubles output channels: [0:n]=mean, [n:2n]=log_var
    if method == "degu":
        n = n * 2
    return CompactStudent(
        n_tracks=n,
        hidden=cfg.student.hidden,
        n_layers=cfg.student.n_layers,
    ).to(device)


def method_loss(method: str, student_out, teacher, side_feat, label, cfg,
                reliability=None, edgpp_loss=None):
    """Returns (loss, meta_dict)."""
    n = cfg.student.n_tracks
    if method == "baseline":
        distill = F.mse_loss(student_out, teacher)
        task = F.mse_loss(student_out.mean(dim=-1), label)
        loss = distill + cfg.training.lambda_task * task
        return loss, {"distill": distill.item(), "task": task.item()}

    if method == "degu":
        mean = student_out[:, :n]
        log_var = student_out[:, n:].clamp(-6.0, 6.0)
        nll = (0.5 * (teacher - mean).pow(2) * torch.exp(-log_var) + 0.5 * log_var).mean()
        task = F.mse_loss(mean.mean(dim=-1), label)
        loss = nll + cfg.training.lambda_task * task
        return loss, {
            "nll": nll.item(),
            "task": task.item(),
            "sigma": torch.exp(0.5 * log_var).mean().item(),
        }

    if method == "edgpp":
        w = reliability(side_feat)
        out = edgpp_loss(student_out, teacher, w, label=label)
        c = out["components"]
        meta = {
            "distill": c["distill"].item(),
            "thresh": c["threshold"].item(),
            "gate_in": c["frac_gated_in"].item(),
            "w̄": c["mean_w"].item(),
        }
        if "task" in c:
            meta["task"] = c["task"].item()
        return out["loss"], meta

    raise ValueError(f"unknown method: {method}")


def infer_scores(student, reliability, loader, method: str, cfg, device):
    n = cfg.student.n_tracks
    student.eval()
    if reliability is not None:
        reliability.eval()
    records = []
    with torch.no_grad():
        for batch in loader:
            ref = batch["ref"].to(device)
            alt = batch["alt"].to(device)
            sidef = batch["side_features"].to(device)
            s = student.sed(ref, alt)
            if method == "degu":
                mean = s[:, :n]
                log_var = s[:, n:].clamp(-6.0, 6.0)
                mag = torch.linalg.norm(mean, dim=-1)
                sigma = torch.exp(0.5 * log_var).mean(dim=-1)
            else:
                mag = torch.linalg.norm(s, dim=-1)
                sigma = torch.zeros_like(mag)

            w = reliability(sidef) if reliability is not None else None

            for i in range(s.size(0)):
                rec = {
                    "snp_id": batch["snp_id"][i],
                    "label": int(batch["label"][i].item()),
                    "score_mag": mag[i].item(),
                }
                if method == "degu":
                    rec["sigma"] = sigma[i].item()
                    rec["score_uncert"] = (mag[i] / (sigma[i] + 1e-6)).item()
                if w is not None:
                    rec["reliability"] = w[i].item()
                records.append(rec)
    df = pd.DataFrame(records)

    # Post-hoc score combinations for edgpp.
    # Key insight: "reliability" is high where teacher-student fit is easy.
    # Causal variants tend to live where fit is hard, so the ranking
    # signal is (1 - reliability), i.e. distillation difficulty.
    if method == "edgpp" and "reliability" in df.columns:
        mag = df["score_mag"].to_numpy()
        diff = 1.0 - df["reliability"].to_numpy()
        mag_z = (mag - mag.mean()) / (mag.std() + 1e-6)
        diff_z = (diff - diff.mean()) / (diff.std() + 1e-6)
        df["score_difficulty"] = diff
        df["score_combined"] = mag_z + diff_z
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", choices=["baseline", "degu", "edgpp"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[cfg.precision]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    student = build_method_student(cfg, args.method, device)
    reliability = build_reliability(cfg.reliability).to(device) if args.method == "edgpp" else None
    edgpp_loss = None
    if args.method == "edgpp":
        edgpp_loss = SelectiveDistillLoss(
            gate_threshold=cfg.training.gate_threshold,
            lambda_task=cfg.training.lambda_task,
            adaptive=cfg.training.get("gate_adaptive", True),
            adaptive_alpha=cfg.training.get("gate_adaptive_alpha", 0.5),
            w_reg_lambda=cfg.training.get("w_reg_lambda", 0.05),
            w_reg_target=cfg.training.get("w_reg_target", 0.5),
        )

    params = list(student.parameters())
    if reliability is not None:
        params += list(reliability.parameters())
    optim = AdamW(params, lr=cfg.training.lr)

    full_ds = build_traitgym(cfg.data)
    chroms = full_ds.df["chrom"].astype(str).values
    train_idx = [i for i, c in enumerate(chroms) if c not in VAL_CHROMS]
    val_idx = [i for i, c in enumerate(chroms) if c in VAL_CHROMS]

    train_loader = DataLoader(Subset(full_ds, train_idx),
                              batch_size=cfg.data.batch_size, shuffle=True,
                              num_workers=cfg.data.num_workers, drop_last=True)
    val_loader = DataLoader(Subset(full_ds, val_idx),
                            batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=cfg.data.num_workers)

    pos_frac = float(full_ds.df["label"].mean())
    print(f"[cfg] method={args.method} device={device} dtype={cfg.precision}")
    print(f"[data] total={len(full_ds)} train={len(train_idx)} val={len(val_idx)} pos_frac={pos_frac:.2%}")

    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    student.train()
    if reliability is not None:
        reliability.train()
    max_steps = args.max_steps or cfg.training.max_steps or 10**9

    step = 0
    for _ in range(cfg.training.n_epochs):
        for batch in train_loader:
            ref = batch["ref"].to(device)
            alt = batch["alt"].to(device)
            t = batch["teacher_score"].to(device)
            sidef = batch["side_features"].to(device)
            label = batch["label"].to(device)

            with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                s = student.sed(ref, alt)
                loss, meta = method_loss(args.method, s, t, sidef, label, cfg,
                                         reliability, edgpp_loss)

            optim.zero_grad(set_to_none=True)
            if dtype == torch.float16:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            if step % cfg.logging.log_every_n_steps == 0:
                meta_str = " | ".join(f"{k} {v:.3f}" for k, v in meta.items())
                print(f"[{args.method}] step {step:4d} | loss {loss.item():.4f} | {meta_str}")
            step += 1
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    scores = infer_scores(student, reliability, val_loader, args.method, cfg, device)
    val_df = full_ds.df.iloc[val_idx].reset_index(drop=True)
    scores["chrom"] = val_df["chrom"].astype(str).values
    scores.to_parquet(out / "scores.parquet", index=False)
    torch.save(student.state_dict(), out / "student.pt")
    if reliability is not None:
        torch.save(reliability.state_dict(), out / "reliability.pt")
    print(f"\n[done] {args.method}: {len(scores)} val scores -> {out}/scores.parquet")


if __name__ == "__main__":
    main()
