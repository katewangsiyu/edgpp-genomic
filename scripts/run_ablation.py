"""Run 3-way ablation: train baseline / degu / edgpp then print comparison table.

Usage:
    python scripts/run_ablation.py --config configs/t4_ablation.yaml
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


METHODS = ["baseline", "degu", "edgpp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-root", default="outputs/ablation")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: train each method ---
    for m in METHODS:
        out_m = out_root / m
        print(f"\n{'='*60}\n>>> TRAIN {m}\n{'='*60}")
        cmd = [sys.executable, "scripts/06_train.py",
               "--config", args.config, "--method", m,
               "--out-dir", str(out_m)]
        if args.max_steps is not None:
            cmd += ["--max-steps", str(args.max_steps)]
        subprocess.check_call(cmd)

    # --- Phase 2: eval each method ---
    for m in METHODS:
        scores = out_root / m / "scores.parquet"
        print(f"\n{'='*60}\n>>> EVAL {m}\n{'='*60}")
        subprocess.check_call([sys.executable, "scripts/07_evaluate.py",
                               "--scores", str(scores)])

    # --- Phase 3: assemble comparison table ---
    rows = []
    for m in METHODS:
        metrics_path = out_root / m / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        for score_col, vals in metrics.items():
            rows.append({"method": m, "score": score_col, **vals})
    table = pd.DataFrame(rows)
    print("\n\n" + "=" * 60)
    print(">>> ABLATION SUMMARY")
    print("=" * 60)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    table_path = out_root / "summary.csv"
    table.to_csv(table_path, index=False)
    print(f"\nsaved: {table_path}")


if __name__ == "__main__":
    main()
