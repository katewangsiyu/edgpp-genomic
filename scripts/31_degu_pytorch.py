"""DEGU Deep Ensemble with Gaussian Uncertainty — PyTorch port.

Ported from baselines/ensemble_distillation/degu.py (TF/Keras). Produces the
same three-stage pipeline:
    1. Train an ensemble of M models with different random seeds
    2. Compute ensemble mean + uncertainty (log-variance)
    3. Distill into a single student that reproduces (mean, uncertainty)

This script provides the training harness; GPU is required for non-trivial
runs. On TraitGym the base predictor is a 1-D CNN over one-hot encoded genomic
windows (not pre-computed features), matching the original DEGU's MPRA regime.

For head-to-head with HCCP, we adapt:
  - Regression head → binary classification head (sigmoid output)
  - Target MPRA activity → target pathogenicity label
  - Gaussian NLL loss → binary focal+BCE + heteroscedastic variance head

Usage (deferred until GPU available):
    python scripts/31_degu_pytorch.py \\
        --traitgym-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --seq-window-dir data/raw/traitgym/mendelian_traits_matched_9/sequences \\
        --ensemble-size 10 --epochs 30 \\
        --out-dir outputs/degu_pytorch/mendelian_M10
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    raise SystemExit(f"PyTorch required: {e}. Install via `pip install torch`.")


# -------------------- Model --------------------
class DEGUCNN(nn.Module):
    """1-D CNN for binary variant effect prediction with heteroscedastic head.

    Matches DEGU's DeepSTARR backbone but shrunk for TraitGym scale. Outputs
    two logits: mean (before sigmoid) and log-variance. Input expects 8-channel
    one-hot (ref 4-hot + alt 4-hot concatenated).
    """

    def __init__(self, seq_len: int = 200, n_channels: int = 8,
                 hidden: int = 64, out_heads: int = 2) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=11, padding=5),
            nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(hidden, hidden * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden * 2), nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden * 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mean_head = nn.Linear(hidden * 2, 1)
        self.logvar_head = nn.Linear(hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).squeeze(-1)
        return self.mean_head(h).squeeze(-1), self.logvar_head(h).squeeze(-1)


# -------------------- Losses --------------------
def heteroscedastic_bce(logit_mean: torch.Tensor, logvar: torch.Tensor,
                       y: torch.Tensor, logvar_min: float = -4.0,
                       logvar_max: float = 4.0) -> torch.Tensor:
    """Binary classification analog of DEGU's Gaussian NLL.

    We model P(y | x) = σ(μ(x)) and uncertainty via var = exp(logvar).
    Loss: BCE + 0.5 * logvar scaled by BCE → high logvar suppresses gradient on
    uncertain points. Logvar is clamped to prevent exp(-logvar) blowup which
    otherwise destabilizes training in the first few epochs.
    """
    logvar_clamped = logvar.clamp(logvar_min, logvar_max)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logit_mean, y.float(), reduction="none")
    return (bce * torch.exp(-logvar_clamped) + 0.5 * logvar_clamped).mean()


# -------------------- Dataset --------------------
class OneHotVariantSet(Dataset):
    """Loads (L, 8) one-hot windows produced by scripts/36_extract_dna_windows.py.

    Channel layout: ``[ref_A, ref_C, ref_G, ref_T, alt_A, alt_C, alt_G, alt_T]``.
    Variant ID is reconstructed from the parquet columns as
    ``chr{chrom}_{pos}_{ref}_{alt}`` to match the extractor's naming.
    """

    def __init__(self, parquet_path: Path, seq_dir: Path, seq_len: int = 200) -> None:
        import pandas as pd
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.seq_dir = Path(seq_dir)
        self.seq_len = seq_len
        if not self.seq_dir.exists():
            raise FileNotFoundError(
                f"Sequence windows not found at {self.seq_dir}. "
                "Run scripts/36_extract_dna_windows.py first.")

    def __len__(self) -> int:
        return len(self.df)

    def _variant_id(self, row) -> str:
        c = str(row["chrom"])
        prefix = c if c.startswith("chr") else f"chr{c}"
        return f"{prefix}_{int(row['pos'])}_{row['ref']}_{row['alt']}"

    def __getitem__(self, i: int) -> tuple[np.ndarray, int, str]:
        row = self.df.iloc[i]
        vid = self._variant_id(row)
        path = self.seq_dir / f"{vid}.npy"
        x = np.load(path).astype(np.float32)  # (L, 8)
        assert x.shape == (self.seq_len, 8), f"{path}: {x.shape}"
        return x.T, int(row["label"]), str(row["chrom"])  # -> (8, L)


# -------------------- Ensemble training loop --------------------
def train_single(model: DEGUCNN, train_loader: DataLoader, val_loader: DataLoader,
                 epochs: int, lr: float, device: torch.device,
                 grad_clip: float = 1.0, pos_weight: float = 9.0) -> dict:
    """Train one teacher with BCE. DEGU's σ̂ is the cross-ensemble std, not a
    learned logvar head, so we use plain BCE (matching baselines/degu.py
    uncertainty_std). The logvar head is trained as a secondary signal but
    σ̂ = std(ensemble) is the reported uncertainty.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Balance ~10% positive class via pos_weight ≈ n_neg / n_pos
    pos_w = torch.tensor([pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            mu, _ = model(x)
            loss = loss_fn(mu, y.float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(float(loss))
        val_loss = eval_loss(model, val_loader, device, pos_weight)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val_loss)
        print(f"  ep {ep+1}/{epochs}: train {np.mean(losses):.4f}  val {val_loss:.4f}")
    return history


@torch.no_grad()
def eval_loss(model: DEGUCNN, loader: DataLoader, device: torch.device,
              pos_weight: float = 9.0) -> float:
    model.eval()
    pos_w = torch.tensor([pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    losses = []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        mu, _ = model(x)
        losses.append(float(loss_fn(mu, y.float())))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model: DEGUCNN, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    mus, logvars, ys, chroms_out = [], [], [], []
    for x, y, c in loader:
        x = x.to(device)
        mu, logvar = model(x)
        mus.append(torch.sigmoid(mu).cpu().numpy())
        logvars.append(logvar.cpu().numpy())
        ys.append(y.numpy()); chroms_out.extend(c)
    return {
        "p_hat": np.concatenate(mus),
        "logvar": np.concatenate(logvars),
        "label": np.concatenate(ys),
        "chrom": np.array(chroms_out),
    }


def train_one_member(member_idx: int, args, ds, idx_cal: list[int],
                      idx_test: list[int], device: torch.device,
                      out: Path) -> dict:
    """Train one ensemble member with OOM retry (halve batch size on OOM)."""
    torch.manual_seed(42 + member_idx)

    batch_size = args.batch_size
    for retry in range(3):
        try:
            train_loader = DataLoader(torch.utils.data.Subset(ds, idx_cal),
                                      batch_size=batch_size, shuffle=True,
                                      num_workers=2, drop_last=False,
                                      pin_memory=True)
            test_loader = DataLoader(torch.utils.data.Subset(ds, idx_test),
                                     batch_size=batch_size, shuffle=False,
                                     num_workers=2, drop_last=False,
                                     pin_memory=True)
            model = DEGUCNN(seq_len=args.seq_len).to(device)
            history = train_single(model, train_loader, test_loader,
                                   args.epochs, args.lr, device)
            preds = predict(model, test_loader, device)
            torch.save(model.state_dict(), out / f"member_{member_idx:02d}.pt")
            (out / f"member_{member_idx:02d}_history.json").write_text(
                json.dumps(history, indent=2))
            # Save predictions immediately — survives crashes of later members.
            np.savez(out / f"member_{member_idx:02d}_preds.npz",
                     p_hat=preds["p_hat"], logvar=preds["logvar"],
                     label=preds["label"], chrom=preds["chrom"])
            del model
            torch.cuda.empty_cache()
            return preds
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = max(batch_size // 2, 4)
            print(f"  [OOM] retry {retry+1}/3 with batch_size={batch_size}")
    raise RuntimeError(f"member {member_idx} failed after 3 OOM retries")


def load_cached_member(out: Path, member_idx: int) -> dict | None:
    path = out / f"member_{member_idx:02d}_preds.npz"
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return {"p_hat": z["p_hat"], "logvar": z["logvar"],
            "label": z["label"], "chrom": z["chrom"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traitgym-parquet", required=True)
    ap.add_argument("--seq-window-dir", required=True)
    ap.add_argument("--ensemble-size", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq-len", type=int, default=200)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.28,
                    help="Upper bound on fraction of GPU memory this process may allocate "
                         "(shared-GPU safety bound). Set to 0 to disable.")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and args.gpu_memory_fraction > 0:
        torch.cuda.set_per_process_memory_fraction(
            args.gpu_memory_fraction, device.index or 0)
        print(f"[memory-guard] capped to {args.gpu_memory_fraction*100:.0f}% "
              f"of GPU {device.index or 0} (≈{15360*args.gpu_memory_fraction:.0f} MiB)")
    print(f"[DEGU-PyTorch] device={device} ensemble={args.ensemble_size} "
          f"batch={args.batch_size} epochs={args.epochs}")

    ds = OneHotVariantSet(args.traitgym_parquet, args.seq_window_dir, args.seq_len)
    # Fixed train/test split aligned with TraitGym test chroms. DEGU's original
    # pipeline uses one split (not per-chrom LOO); we retain that to isolate the
    # σ̂ contribution for downstream HCCP consumption.
    chroms = ds.df["chrom"].astype(str).tolist()
    test_chroms = {"17", "18", "19", "20", "21", "22", "X"}
    idx_cal = [i for i, c in enumerate(chroms) if c not in test_chroms]
    idx_test = [i for i, c in enumerate(chroms) if c in test_chroms]
    y_cal = ds.df["label"].iloc[idx_cal].astype(int).sum()
    y_test = ds.df["label"].iloc[idx_test].astype(int).sum()
    print(f"  cal n={len(idx_cal)} (pos {y_cal})  test n={len(idx_test)} (pos {y_test})")

    ensemble_preds: list[dict] = []
    for m in range(args.ensemble_size):
        cached = load_cached_member(out, m)
        if cached is not None:
            print(f"\n--- member {m+1}/{args.ensemble_size} [resume from cache] ---")
            ensemble_preds.append(cached)
            continue
        print(f"\n--- member {m+1}/{args.ensemble_size} ---")
        preds = train_one_member(m, args, ds, idx_cal, idx_test, device, out)
        ensemble_preds.append(preds)

    # Ensemble mean + uncertainty
    p_stack = np.stack([e["p_hat"] for e in ensemble_preds])
    p_mean = p_stack.mean(axis=0)
    p_var = p_stack.var(axis=0)
    sigma_ensemble = np.sqrt(p_var + 1e-8)

    summary = {
        "ensemble_size": args.ensemble_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_cal": int(len(idx_cal)),
        "n_test": int(len(idx_test)),
        "mean_p_hat": float(p_mean.mean()),
        "mean_sigma_ensemble": float(sigma_ensemble.mean()),
    }
    (out / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))

    # Save per-variant predictions matching the format 14_conformal_hetero expects.
    import pandas as pd
    y = ensemble_preds[0]["label"]
    chroms_out = ensemble_preds[0]["chrom"]
    pd.DataFrame({
        "chrom": chroms_out, "label": y,
        "p_hat": p_mean, "sigma": sigma_ensemble,
        "residual": y - p_mean, "abs_residual": np.abs(y - p_mean),
        "raw_pred": sigma_ensemble,
    }).to_parquet(out / "scores_with_sigma.parquet", index=False)

    print(f"\nsaved: {out}/scores_with_sigma.parquet + {args.ensemble_size} members")


if __name__ == "__main__":
    main()
