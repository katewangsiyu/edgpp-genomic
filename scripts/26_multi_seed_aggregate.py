"""Run aggregator + sigma head + conformal across seeds {42, 7, 2024} to produce
mean±std for Table 3.

Idempotent: skips existing outputs unless --force.

Usage:
    python scripts/26_multi_seed_aggregate.py --feature-set CADD+GPN-MSA+Borzoi --dataset mendelian
    python scripts/26_multi_seed_aggregate.py --feature-set CADD+GPN-MSA+Borzoi --dataset complex
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path


SEEDS = (42, 7, 2024)

DATASETS = {
    "mendelian": {
        "test": "data/raw/traitgym/mendelian_traits_matched_9/test.parquet",
        "features": "data/raw/traitgym/mendelian_traits_matched_9/features",
    },
    "complex": {
        "test": "data/raw/traitgym/complex_traits_matched_9/test.parquet",
        "features": "data/raw/traitgym/complex_traits_matched_9/features",
    },
}


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"failed: {cmd}")


def agg_dir(feature_set: str, dataset: str, seed: int) -> Path:
    base = f"outputs/aggregator_gbm/{feature_set}_{dataset}"
    return Path(base if seed == 42 else f"{base}_seed{seed}")


def sigma_dir(feature_set: str, dataset: str, seed: int) -> Path:
    base = f"outputs/hetero_head/{feature_set}_{dataset}_abs"
    return Path(base if seed == 42 else f"{base}_seed{seed}")


def conformal_dir(feature_set: str, dataset: str, seed: int) -> Path:
    base = f"outputs/conformal_hetero/{feature_set}_{dataset}_abs_mondrian"
    return Path(base if seed == 42 else f"{base}_seed{seed}")


def stage_aggregator(feature_set: str, dataset: str, seed: int, force: bool) -> None:
    out = agg_dir(feature_set, dataset, seed)
    metrics = out / "metrics.json"
    if metrics.exists() and not force:
        print(f"[skip] {out} (exists)")
        return
    ds = DATASETS[dataset]
    run([
        sys.executable, "scripts/11_aggregator_gbm.py",
        "--test-parquet", ds["test"],
        "--features-dir", ds["features"],
        "--feature-set", feature_set,
        "--out-dir", str(out),
        "--seed", str(seed),
    ])


def stage_sigma(feature_set: str, dataset: str, seed: int, force: bool) -> None:
    out = sigma_dir(feature_set, dataset, seed)
    if (out / "scores_with_sigma.parquet").exists() and not force:
        print(f"[skip] {out} (exists)")
        return
    ds = DATASETS[dataset]
    agg = agg_dir(feature_set, dataset, seed) / "scores.parquet"
    run([
        sys.executable, "scripts/13_hetero_head.py",
        "--base-scores", str(agg),
        "--test-parquet", ds["test"],
        "--features-dir", ds["features"],
        "--feature-set", feature_set,
        "--mode", "abs_residual",
        "--out-dir", str(out),
        "--seed", str(seed),
    ])


def stage_conformal(feature_set: str, dataset: str, seed: int, force: bool) -> None:
    out = conformal_dir(feature_set, dataset, seed)
    if (out / "conformal_hetero_results.json").exists() and not force:
        print(f"[skip] {out} (exists)")
        return
    ds = DATASETS[dataset]
    sig = sigma_dir(feature_set, dataset, seed) / "scores_with_sigma.parquet"
    run([
        sys.executable, "scripts/14_conformal_hetero.py",
        "--sigma-scores", str(sig),
        "--test-parquet", ds["test"],
        "--out-dir", str(out),
        "--alpha", "0.10",
    ])


def aggregate_across_seeds(feature_set: str, dataset: str) -> dict:
    import numpy as np
    rows = []
    for seed in SEEDS:
        agg_metrics = json.load(open(agg_dir(feature_set, dataset, seed) / "metrics.json"))
        conf = json.load(open(conformal_dir(feature_set, dataset, seed) / "conformal_hetero_results.json"))
        mon = conf["mondrian_y_sigma"]
        rows.append({
            "seed": seed,
            "AUPRC_chrom_weighted": agg_metrics["AUPRC_per_chrom"],
            "marginal_coverage": mon["coverage"],
            "coverage_pos": mon["coverage_pos"],
            "sigma_cov_range": mon["sigma_cov_range"],
            "frac_singleton": mon["frac_singleton"],
        })

    def stats(key: str) -> tuple[float, float]:
        vals = np.array([r[key] for r in rows], dtype=float)
        return float(vals.mean()), float(vals.std(ddof=1))

    summary = {
        "feature_set": feature_set,
        "dataset": dataset,
        "seeds": list(SEEDS),
        "per_seed": rows,
        "mean_std": {
            k: {"mean": stats(k)[0], "std": stats(k)[1]}
            for k in ["AUPRC_chrom_weighted", "marginal_coverage",
                      "coverage_pos", "sigma_cov_range", "frac_singleton"]
        },
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=None,
                    help="Summary JSON destination (default outputs/multi_seed/<set>_<ds>.json)")
    args = ap.parse_args()

    for seed in SEEDS:
        stage_aggregator(args.feature_set, args.dataset, seed, args.force)
        stage_sigma(args.feature_set, args.dataset, seed, args.force)
        stage_conformal(args.feature_set, args.dataset, seed, args.force)

    summary = aggregate_across_seeds(args.feature_set, args.dataset)

    out = Path(args.out) if args.out else Path(
        f"outputs/multi_seed/{args.feature_set}_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n=== {args.feature_set} / {args.dataset} (n={len(SEEDS)} seeds) ===")
    for k, v in summary["mean_std"].items():
        print(f"  {k:<25s}: {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
