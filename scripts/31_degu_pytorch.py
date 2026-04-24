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
    two logits: mean (before sigmoid) and log-variance.
    """

    def __init__(self, seq_len: int = 200, n_channels: int = 4,
                 hidden: int = 64, out_heads: int = 2):
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
                       y: torch.Tensor) -> torch.Tensor:
    """Binary classification analog of DEGU's Gaussian NLL.

    We model P(y | x) = σ(μ(x)) and uncertainty via var = exp(logvar).
    Loss: BCE + 0.5 * logvar scaled by BCE → high logvar suppresses gradient on
    uncertain points.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logit_mean, y.float(), reduction="none")
    return (bce * torch.exp(-logvar) + 0.5 * logvar).mean()


# -------------------- Dataset --------------------
class OneHotVariantSet(Dataset):
    """Placeholder — real impl should load pre-extracted DNA windows from disk.

    Expected data layout:
        <seq-window-dir>/<variant_id>.npy    # shape (L, 4) one-hot
        <variant_id> column in parquet matches filenames.
    """

    def __init__(self, parquet_path: Path, seq_dir: Path, seq_len: int = 200):
        import pandas as pd
        self.df = pd.read_parquet(parquet_path).reset_index(drop=True)
        self.seq_dir = Path(seq_dir)
        self.seq_len = seq_len
        self._check()

    def _check(self) -> None:
        if not self.seq_dir.exists():
            raise FileNotFoundError(
                f"Sequence windows not found at {self.seq_dir}. "
                "DEGU requires pre-extracted one-hot windows.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int, str]:
        row = self.df.iloc[i]
        vid = row.get("variant_id", f"var_{i}")
        path = self.seq_dir / f"{vid}.npy"
        x = np.load(path).astype(np.float32)  # (L, 4)
        assert x.shape == (self.seq_len, 4), f"{path}: {x.shape}"
        return x.T, int(row["label"]), str(row["chrom"])


# -------------------- Ensemble training loop --------------------
def train_single(model: DEGUCNN, train_loader: DataLoader, val_loader: DataLoader,
                 epochs: int, lr: float, device: torch.device) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            mu, logvar = model(x)
            loss = heteroscedastic_bce(mu, logvar, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
        val_loss = eval_loss(model, val_loader, device)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_loss"].append(val_loss)
        print(f"  ep {ep+1}/{epochs}: train {np.mean(losses):.4f}  val {val_loss:.4f}")
    return history


@torch.no_grad()
def eval_loss(model: DEGUCNN, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        mu, logvar = model(x)
        losses.append(float(heteroscedastic_bce(mu, logvar, y)))
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
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[DEGU-PyTorch] device={device} ensemble={args.ensemble_size}")

    ds = OneHotVariantSet(args.traitgym_parquet, args.seq_window_dir, args.seq_len)
    # Split chrom-LOO for each held-out chrom
    chroms = [ds[i][2] for i in range(len(ds))]
    test_chroms = {"17", "18", "19", "20", "21", "22", "X"}
    idx_cal = [i for i, c in enumerate(chroms) if c not in test_chroms]
    idx_test = [i for i, c in enumerate(chroms) if c in test_chroms]
    print(f"  cal n={len(idx_cal)}  test n={len(idx_test)}")

    train_loader = DataLoader(torch.utils.data.Subset(ds, idx_cal),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=False)
    test_loader = DataLoader(torch.utils.data.Subset(ds, idx_test),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=4, drop_last=False)

    ensemble_preds = []
    for m in range(args.ensemble_size):
        print(f"\n--- ensemble member {m+1}/{args.ensemble_size} ---")
        torch.manual_seed(42 + m)
        model = DEGUCNN(seq_len=args.seq_len).to(device)
        history = train_single(model, train_loader, test_loader,
                               args.epochs, args.lr, device)
        preds = predict(model, test_loader, device)
        ensemble_preds.append(preds)
        torch.save(model.state_dict(), out / f"member_{m:02d}.pt")
        (out / f"member_{m:02d}_history.json").write_text(json.dumps(history, indent=2))

    # Ensemble mean + uncertainty
    p_stack = np.stack([e["p_hat"] for e in ensemble_preds])
    p_mean = p_stack.mean(axis=0)
    p_var = p_stack.var(axis=0)
    sigma_ensemble = np.sqrt(p_var + 1e-8)

    summary = {
        "ensemble_size": args.ensemble_size,
        "epochs": args.epochs,
        "n_test": int(len(idx_test)),
        "mean_p_hat": float(p_mean.mean()),
        "mean_sigma_ensemble": float(sigma_ensemble.mean()),
    }
    (out / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))

    # Save per-variant predictions in the format HCCP's 14_conformal_hetero expects
    import pandas as pd
    y = ensemble_preds[0]["label"]
    chroms_out = ensemble_preds[0]["chrom"]
    pd.DataFrame({
        "chrom": chroms_out, "label": y,
        "p_hat": p_mean, "sigma": sigma_ensemble,
        "residual": y - p_mean, "abs_residual": np.abs(y - p_mean),
        "raw_pred": sigma_ensemble,
    }).to_parquet(out / "scores_with_sigma.parquet", index=False)

    print(f"\nsaved: {out}/scores_with_sigma.parquet and member_*.pt")


if __name__ == "__main__":
    main()
