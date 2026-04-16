"""Local conditional coverage evaluator — Path A Phase 2 T3 empirical probe.

Given the output of `14_conformal_hetero.py` (conformal_hetero_scores.parquet)
and the original test.parquet (for `consequence` and `tss_dist`), compute
empirical coverage of the prediction sets restricted to sub-populations:

    - per chromosome
    - per functional consequence category (PLS, dELS, 5_prime_UTR, intron, ...)
    - per |tss_dist| decile (log-scaled for complex, linear for mendelian)
    - per consequence × |tss_dist|-bin cross (for cells with n ≥ 25)

For each partition *and each method* (homoscedastic / heteroscedastic /
Mondrian-by-(y×σ̂)), we report empirical coverage. The headline number per
partition is `max_gap = max(cov_bin) - min(cov_bin)`: a partition-level
local-coverage gap that is the empirical analogue of Theorem T3.

Theory note: T3 (local coverage) is stated in feature-space neighbourhoods.
Here we use *discrete proxies* for the neighbourhood structure. σ̂-deciles
are already exercised inside 14_conformal_hetero.py. `consequence` and
`tss_dist` are x-space partitions that are semantically meaningful for VEP:
- consequence captures regulatory *element type* (biology)
- tss_dist captures *positional context* (distance to transcription start)
and are not directly seen by the σ̂ head (both derive from the variant
annotation files, not the feature matrices we train on).

Usage:
    python scripts/15_eval_local_coverage.py \\
        --conformal-scores outputs/conformal_hetero/\\
            CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian/conformal_hetero_scores.parquet \\
        --test-parquet data/raw/traitgym/mendelian_traits_matched_9/test.parquet \\
        --out-dir outputs/local_coverage/CADD+GPN-MSA+Borzoi_mendelian_abs_mondrian \\
        --alpha 0.10
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = ("homosc", "hetero", "mondrian")
METHOD_LABEL = {
    "homosc": "Homosc class-cond (s=|y−p̂|, σ̂≡1) — ablation",
    "hetero": "Hetero class-cond (s=|y−p̂|/(σ̂+ε))",
    "mondrian": "Mondrian (y×σ̂-bin) hetero class-cond",
}


def bin_by_quantile(values: np.ndarray, n_bins: int = 10, transform=None):
    """Quantile-bin `values`. Returns a pandas Series of integer bin labels."""
    v = values if transform is None else transform(values)
    try:
        bins = pd.qcut(v, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(v, bins=n_bins, labels=False)
    return pd.Series(bins)


def summarize_partition(covered: np.ndarray, partition: pd.Series,
                        meta_fn=None, min_n: int = 5):
    """Coverage table grouped by `partition`. Returns list of dicts + max gap."""
    rows = []
    for lab in sorted(partition.dropna().unique(), key=lambda x: (str(type(x)), x)):
        m = (partition == lab).to_numpy()
        n = int(m.sum())
        if n < min_n:
            continue
        row = {"bin": str(lab), "n": n, "coverage": float(covered[m].mean())}
        if meta_fn is not None:
            row.update(meta_fn(m))
        rows.append(row)
    if not rows:
        return rows, float("nan")
    covs = [r["coverage"] for r in rows]
    gap = float(max(covs) - min(covs))
    return rows, gap


def eval_method(method: str, df: pd.DataFrame, alpha: float):
    covered = df[f"{method}_covered"].to_numpy()
    pset_code = df[f"{method}_pset_code"].to_numpy()
    n = len(df)
    target = 1 - alpha

    # Prediction-set composition
    sizes = np.array([bin(c).count("1") for c in pset_code])
    frac_empty = float((sizes == 0).mean())
    frac_single = float((sizes == 1).mean())
    frac_both = float((sizes == 2).mean())

    out = {
        "method": method,
        "label": METHOD_LABEL[method],
        "marginal_coverage": float(covered.mean()),
        "target_coverage": target,
        "marginal_gap": float(covered.mean() - target),
        "cov_pos": float(covered[df["label"] == 1].mean()),
        "cov_neg": float(covered[df["label"] == 0].mean()),
        "frac_empty": frac_empty,
        "frac_singleton": frac_single,
        "frac_both": frac_both,
    }

    # Per-chrom
    chrom_tbl, chrom_gap = summarize_partition(covered, df["chrom"].astype(str))
    out["per_chrom"] = chrom_tbl
    out["per_chrom_gap"] = chrom_gap

    # Per consequence
    cons_tbl, cons_gap = summarize_partition(
        covered, df["consequence"].astype("string"), min_n=10
    )
    out["per_consequence"] = cons_tbl
    out["per_consequence_gap"] = cons_gap

    # Per |tss_dist| decile (log1p scale for wide range)
    tss_abs = df["tss_dist"].abs().to_numpy()
    tss_bin = bin_by_quantile(tss_abs, n_bins=10, transform=np.log1p)

    def tss_meta(m):
        return {
            "tss_dist_median": float(np.median(np.asarray(tss_abs)[m])),
            "tss_dist_min": float(np.min(np.asarray(tss_abs)[m])),
            "tss_dist_max": float(np.max(np.asarray(tss_abs)[m])),
        }

    tss_tbl, tss_gap = summarize_partition(
        covered, tss_bin, meta_fn=tss_meta, min_n=10
    )
    out["per_tss_bin"] = tss_tbl
    out["per_tss_bin_gap"] = tss_gap

    # Per σ̂-decile (reference; already in 14's output but recompute for convenience)
    sigma = df["sigma"].to_numpy()
    sig_bin = bin_by_quantile(sigma, n_bins=10)

    def sig_meta(m):
        return {"sigma_mean": float(sigma[m].mean())}

    sig_tbl, sig_gap = summarize_partition(
        covered, sig_bin, meta_fn=sig_meta, min_n=10
    )
    out["per_sigma_bin"] = sig_tbl
    out["per_sigma_bin_gap"] = sig_gap

    # Per p̂-decile
    p = df["p_hat"].to_numpy()
    p_bin = bin_by_quantile(p, n_bins=10)

    def p_meta(m):
        return {"phat_mean": float(p[m].mean())}

    p_tbl, p_gap = summarize_partition(
        covered, p_bin, meta_fn=p_meta, min_n=10
    )
    out["per_phat_bin"] = p_tbl
    out["per_phat_bin_gap"] = p_gap

    # Cross: consequence × tss-quintile (coarser; only cells with n ≥ 25)
    tss_q5 = bin_by_quantile(tss_abs, n_bins=5, transform=np.log1p)
    cross = df["consequence"].astype(str).reset_index(drop=True) + \
        "|tssQ" + tss_q5.reset_index(drop=True).astype(str)
    cross_tbl, cross_gap = summarize_partition(covered, cross, min_n=25)
    out["per_consequence_x_tss"] = cross_tbl
    out["per_consequence_x_tss_gap"] = cross_gap

    return out


def print_summary(m: dict, n_total: int):
    lab = m["label"]
    print(f"\n{'='*72}")
    print(f"  {lab}")
    print(f"{'='*72}")
    print(f"  marginal cov={m['marginal_coverage']:.4f}  "
          f"Cov|pos={m['cov_pos']:.4f}  Cov|neg={m['cov_neg']:.4f}   "
          f"(target={m['target_coverage']:.2f})")
    print(f"  size distribution: empty={m['frac_empty']:.3f}  "
          f"single={m['frac_singleton']:.3f}  both={m['frac_both']:.3f}")
    print(f"\n  Local-coverage gap per partition (max_cov − min_cov):")
    for key, gap_key in [
        ("per_chrom", "per_chrom_gap"),
        ("per_consequence", "per_consequence_gap"),
        ("per_tss_bin", "per_tss_bin_gap"),
        ("per_sigma_bin", "per_sigma_bin_gap"),
        ("per_phat_bin", "per_phat_bin_gap"),
        ("per_consequence_x_tss", "per_consequence_x_tss_gap"),
    ]:
        nb = len(m[key])
        print(f"    {key:28s}: gap={m[gap_key]:.3f}  (bins used={nb})")

    # Detailed per-consequence breakdown
    print(f"\n  cov by consequence (sorted by n desc, top 8):")
    cons = sorted(m["per_consequence"], key=lambda r: -r["n"])[:8]
    for r in cons:
        print(f"    {r['bin']:>40s}: n={r['n']:>5}  cov={r['coverage']:.3f}")

    # Detailed per-tss-bin breakdown
    print(f"\n  cov by |tss_dist|-decile (0=closest to TSS):")
    for r in m["per_tss_bin"]:
        print(f"    bin {r['bin']:>3s}: n={r['n']:>5}  "
              f"|tss| median={r['tss_dist_median']:.0f}  "
              f"[{r['tss_dist_min']:.0f}, {r['tss_dist_max']:.0f}]  "
              f"cov={r['coverage']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conformal-scores", required=True,
                    help="Output of 14_conformal_hetero.py: "
                         "conformal_hetero_scores.parquet")
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cs = pd.read_parquet(args.conformal_scores).reset_index(drop=True)
    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    assert len(cs) == len(V), f"length mismatch cs={len(cs)} V={len(V)}"
    assert (cs["chrom"].astype(str).values == V["chrom"].astype(str).values).all()
    assert (cs["label"].astype(int).values == V["label"].astype(int).values).all()

    # Attach test-parquet metadata (consequence, tss_dist)
    df = cs.copy()
    df["consequence"] = V["consequence"].astype(str).values
    df["tss_dist"] = V["tss_dist"].astype(float).values

    print(f"[load] n={len(df)} pos={int(df['label'].sum())}  α={args.alpha}")
    print(f"  consequence categories: {df['consequence'].nunique()}")
    print(f"  |tss_dist| range: [{df['tss_dist'].abs().min():.0f}, "
          f"{df['tss_dist'].abs().max():.0f}]")

    per_method = {}
    for method in METHODS:
        if f"{method}_covered" not in df.columns:
            print(f"[skip] method {method} not in parquet")
            continue
        m = eval_method(method, df, args.alpha)
        per_method[method] = m
        print_summary(m, n_total=len(df))

    # Cross-method comparison table on selected partitions
    print(f"\n{'='*72}")
    print(f"  LOCAL COVERAGE GAP — cross-method comparison  (target={1-args.alpha:.2f})")
    print(f"{'='*72}")
    partitions = [
        ("chrom", "per_chrom_gap"),
        ("consequence", "per_consequence_gap"),
        ("|tss_dist|-decile", "per_tss_bin_gap"),
        ("σ̂-decile", "per_sigma_bin_gap"),
        ("p̂-decile", "per_phat_bin_gap"),
        ("consequence × tss-quintile", "per_consequence_x_tss_gap"),
    ]
    header = "  partition                     "
    for method in METHODS:
        if method in per_method:
            header += f"{method:>12s}"
    print(header)
    for label, key in partitions:
        row = f"  {label:28s}: "
        for method in METHODS:
            if method in per_method:
                gap = per_method[method][key]
                if np.isnan(gap):
                    row += f"{'—':>12s}"
                else:
                    row += f"{gap:>12.3f}"
        print(row)

    results = {
        "alpha": args.alpha,
        "n": int(len(df)),
        "n_pos": int(df["label"].sum()),
        "consequence_categories": int(df["consequence"].nunique()),
        "per_method": per_method,
    }
    (out / "local_coverage_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out}/local_coverage_results.json")


if __name__ == "__main__":
    main()
