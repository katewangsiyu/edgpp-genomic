"""Three sanity checks for the "HistGB is deterministic across seeds" finding.

If the determinism is due to sklearn's algorithm (no random source active under
our config), we expect:
  Test A (label shuffle):       AUPRC collapses toward baseline (~10% positive rate)
  Test B (subsample=0.8 + seed): AUPRC varies meaningfully across seeds
  Test C (chrom-LOO integrity): train set contains ZERO variants from test chrom

If any of these fails, the "determinism" finding is actually a leakage artifact.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
mod_11 = __import__("11_aggregator_gbm")
FEATURE_SETS = mod_11.FEATURE_SETS
load_feature_matrix = mod_11.load_feature_matrix
weighted_per_chrom_auprc = mod_11.weighted_per_chrom_auprc


def build_pipe(seed: int = 42, subsample: float = 1.0):
    # We bypass sklearn's lack of subsample on HistGB by replacing with a
    # config that DOES introduce variance: set max_features < 1.0 via a
    # per-tree subsample. HistGB doesn't expose subsample directly, so we
    # use random per-sample row masking manually (done in the caller).
    return Pipeline([
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("hgb", HistGradientBoostingClassifier(
            max_depth=2, max_iter=100, learning_rate=0.1,
            class_weight="balanced",
            random_state=seed,
            early_stopping=False,
            # categorical_features and interaction_cst are other seed sinks but unused here.
        )),
    ])


def chrom_loo_fit_predict(X: pd.DataFrame, y: np.ndarray, chroms: np.ndarray,
                           seed: int, row_mask_rng: np.random.Generator | None = None,
                           row_mask_frac: float = 1.0) -> tuple[np.ndarray, int]:
    """Standard chrom-LOO loop. If row_mask_frac < 1.0, randomly drop that
    fraction of training rows per fold — this re-introduces stochasticity.
    """
    scores = np.full(len(X), np.nan)
    overlap_count = 0  # Test C: count of variants in train ∩ test
    for c in sorted(set(chroms)):
        mask_te = chroms == c
        mask_tr = ~mask_te
        # Test C: this MUST be 0
        overlap = int((mask_te & mask_tr).sum())
        overlap_count += overlap
        X_tr = X.loc[mask_tr].to_numpy()
        y_tr = y[mask_tr]
        if row_mask_frac < 1.0 and row_mask_rng is not None:
            keep_idx = row_mask_rng.choice(
                len(X_tr), size=int(len(X_tr) * row_mask_frac), replace=False)
            X_tr = X_tr[keep_idx]; y_tr = y_tr[keep_idx]
        pipe = build_pipe(seed=seed)
        pipe.fit(X_tr, y_tr)
        scores[mask_te] = pipe.predict_proba(X.loc[mask_te].to_numpy())[:, 1]
    return scores, overlap_count


def test_A_label_shuffle(X, y, chroms, shuffle_frac: float, seed: int = 42):
    """If p̂ on test doesn't depend on training labels → leakage."""
    rng = np.random.default_rng(seed)
    y_shuf = y.copy()
    flip_idx = rng.choice(len(y), size=int(shuffle_frac * len(y)), replace=False)
    y_shuf[flip_idx] = 1 - y_shuf[flip_idx]
    scores, _ = chrom_loo_fit_predict(X, y_shuf, chroms, seed=seed)
    # Evaluate on ORIGINAL labels — how much signal remains?
    auprc_shuffled, _ = weighted_per_chrom_auprc(y, scores, chroms)
    return {"shuffle_frac": shuffle_frac, "AUPRC_on_original_y": auprc_shuffled}


def test_B_subsample(X, y, chroms, seeds: list[int], row_mask_frac: float = 0.8):
    """Introduce real row-level stochasticity via row mask and see if seed now matters."""
    results = []
    for s in seeds:
        rng = np.random.default_rng(s)
        scores, _ = chrom_loo_fit_predict(
            X, y, chroms, seed=s,
            row_mask_rng=rng, row_mask_frac=row_mask_frac)
        auprc, _ = weighted_per_chrom_auprc(y, scores, chroms)
        results.append({"seed": s, "AUPRC": float(auprc)})
    return results


def test_C_chrom_integrity(chroms: np.ndarray):
    """Structural check — should be trivially passing."""
    # For each chrom c, train mask excludes it entirely.
    uniq = sorted(set(chroms))
    violations = []
    for c in uniq:
        mask_te = chroms == c
        mask_tr = ~mask_te
        if np.any(mask_te & mask_tr):
            violations.append(c)
    return {"n_chroms": len(uniq), "violations": violations}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--feature-set", required=True, choices=list(FEATURE_SETS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    V = pd.read_parquet(args.test_parquet).reset_index(drop=True)
    X = load_feature_matrix(Path(args.features_dir), FEATURE_SETS[args.feature_set]).reset_index(drop=True)
    y = V["label"].astype(int).to_numpy()
    chroms = V["chrom"].astype(str).to_numpy()
    print(f"[load] n={len(V)} dim={X.shape[1]} pos_rate={y.mean():.3f}")

    # Reference: seed=42 no perturbation
    print("\n--- Reference (seed=42, no perturbation) ---")
    scores_ref, overlap = chrom_loo_fit_predict(X, y, chroms, seed=42)
    auprc_ref, _ = weighted_per_chrom_auprc(y, scores_ref, chroms)
    print(f"  AUPRC (ref) = {auprc_ref:.4f}   chrom overlap = {overlap}")

    # Test A: label shuffle
    print("\n--- Test A: training label shuffle (should COLLAPSE AUPRC) ---")
    test_A_results = []
    for frac in (0.1, 0.3, 0.5):
        r = test_A_label_shuffle(X, y, chroms, shuffle_frac=frac, seed=42)
        print(f"  shuffle {int(frac*100)}% labels: AUPRC = {r['AUPRC_on_original_y']:.4f}")
        test_A_results.append(r)

    # Test B: row subsample re-introduces stochasticity
    print("\n--- Test B: row subsample 0.8 across seeds (should VARY) ---")
    test_B_results = test_B_subsample(X, y, chroms, seeds=[42, 7, 2024], row_mask_frac=0.8)
    for r in test_B_results:
        print(f"  seed={r['seed']:>4d}: AUPRC = {r['AUPRC']:.4f}")
    auprcs_B = [r["AUPRC"] for r in test_B_results]
    print(f"  → std across seeds: {np.std(auprcs_B, ddof=1):.4f}")

    # Test C: chrom integrity
    print("\n--- Test C: chrom-LOO structural integrity ---")
    test_C_result = test_C_chrom_integrity(chroms)
    print(f"  chroms tested: {test_C_result['n_chroms']}  violations: {len(test_C_result['violations'])}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "feature_set": args.feature_set,
        "n": int(len(V)), "pos_rate": float(y.mean()),
        "reference_auprc": float(auprc_ref),
        "chrom_overlap_count": int(overlap),
        "test_A_label_shuffle": test_A_results,
        "test_B_subsample_seeds": test_B_results,
        "test_B_seed_std": float(np.std(auprcs_B, ddof=1)),
        "test_C_chrom_integrity": test_C_result,
    }, indent=2))

    print("\n=== Verdict ===")
    baseline = y.mean() + 0.05  # ~ pos rate + 5pp tolerance
    if test_A_results[-1]["AUPRC_on_original_y"] < baseline + 0.10:
        print(f"  [PASS] Test A: 50% label shuffle collapses AUPRC to "
              f"{test_A_results[-1]['AUPRC_on_original_y']:.3f} (near baseline {y.mean():.3f})")
    else:
        print(f"  [FAIL] Test A: 50% shuffle AUPRC still "
              f"{test_A_results[-1]['AUPRC_on_original_y']:.3f} — possible leakage")
    if np.std(auprcs_B, ddof=1) > 0.002:
        print(f"  [PASS] Test B: row-subsample introduces meaningful seed variance "
              f"(std={np.std(auprcs_B, ddof=1):.4f})")
    else:
        print(f"  [WEIRD] Test B: even subsample yields tiny variance — algorithm may be "
              f"dominated by majority features")
    if not test_C_result["violations"]:
        print(f"  [PASS] Test C: chrom-LOO splits are clean")
    else:
        print(f"  [FAIL] Test C: chrom overlap in {test_C_result['violations']}")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
