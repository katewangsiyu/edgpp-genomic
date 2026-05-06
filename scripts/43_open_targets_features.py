"""Flatten Open Targets variant.parquet annotation arrays into columnar
features suitable for tree-based ML on hg38 variants.

Source schema (variant.parquet from OT 25.06):
  - variantEffect: array of dict {method, assessment, score, normalisedScore, ...}
    methods seen: VEP, GERP, AlphaMissense, SIFT, LOFTEE, FoldX
  - alleleFrequencies: array of dict {populationName, alleleFrequency}
    populations: nfe_adj, afr_adj, amr_adj, sas_adj, eas_adj, ... (gnomAD)
  - mostSevereConsequenceId: SO term (sequence ontology)

Output flat columns (one row per variant):
    variantId, chromosome, position, ref, alt,
    af_nfe, af_overall, af_max,
    gerp, alphamissense, sift, foldx,
    is_lof_loftee, is_lof_vep,
    consequence_severity (0-15 ordinal scale, see SEVERITY_MAP),
    has_alphamissense, has_sift, has_foldx (presence flags)

Usage:
    python scripts/43_open_targets_features.py \\
        --variant-dir data/raw/open_targets/variant \\
        --out data/processed/open_targets/variant_features.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# Sequence Ontology severity ranking (higher = more severe).
# Curated from VEP impact tiers (HIGH > MODERATE > LOW > MODIFIER).
SEVERITY_MAP: dict[str, int] = {
    "transcript_ablation": 15,
    "splice_acceptor_variant": 14,
    "splice_donor_variant": 14,
    "stop_gained": 13,
    "frameshift_variant": 13,
    "stop_lost": 12,
    "start_lost": 12,
    "transcript_amplification": 11,
    "inframe_insertion": 10,
    "inframe_deletion": 10,
    "missense_variant": 9,
    "protein_altering_variant": 9,
    "splice_region_variant": 8,
    "splice_donor_5th_base_variant": 8,
    "splice_donor_region_variant": 8,
    "splice_polypyrimidine_tract_variant": 8,
    "incomplete_terminal_codon_variant": 7,
    "start_retained_variant": 7,
    "stop_retained_variant": 7,
    "synonymous_variant": 6,
    "coding_sequence_variant": 6,
    "mature_miRNA_variant": 5,
    "5_prime_UTR_variant": 5,
    "3_prime_UTR_variant": 5,
    "non_coding_transcript_exon_variant": 4,
    "intron_variant": 3,
    "NMD_transcript_variant": 3,
    "non_coding_transcript_variant": 3,
    "upstream_gene_variant": 2,
    "downstream_gene_variant": 2,
    "TFBS_ablation": 2,
    "TFBS_amplification": 2,
    "TF_binding_site_variant": 2,
    "regulatory_region_ablation": 2,
    "regulatory_region_amplification": 2,
    "feature_elongation": 1,
    "regulatory_region_variant": 1,
    "feature_truncation": 1,
    "intergenic_variant": 0,
}


def _safe_float(v) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _extract_variant_effect(arr) -> dict[str, float | bool]:
    """Pull per-method scores out of the variantEffect array of structs."""
    out = {
        "gerp": float("nan"),
        "alphamissense": float("nan"),
        "sift": float("nan"),
        "foldx": float("nan"),
        "is_lof_loftee": False,
    }
    if arr is None:
        return out
    for entry in arr:
        if entry is None:
            continue
        method = entry.get("method")
        score = entry.get("score") or entry.get("normalisedScore")
        if method == "GERP":
            out["gerp"] = _safe_float(score)
        elif method == "AlphaMissense":
            out["alphamissense"] = _safe_float(score)
        elif method == "SIFT":
            out["sift"] = _safe_float(score)
        elif method == "FoldX":
            out["foldx"] = _safe_float(score)
        elif method == "LOFTEE":
            assessment = entry.get("assessment") or ""
            if "HC" in assessment.upper() or "HIGH" in assessment.upper():
                out["is_lof_loftee"] = True
    return out


def _extract_allele_freq(arr) -> dict[str, float]:
    """Pull gnomAD AF from the alleleFrequencies array."""
    out = {"af_nfe": float("nan"), "af_overall": float("nan"),
           "af_max": float("nan")}
    if arr is None:
        return out
    seen = []
    for entry in arr:
        if entry is None:
            continue
        pop = entry.get("populationName") or ""
        af = _safe_float(entry.get("alleleFrequency"))
        if not np.isnan(af):
            seen.append(af)
        if pop == "nfe_adj":
            out["af_nfe"] = af
        elif pop == "remaining_adj":
            out["af_overall"] = af
    if seen:
        out["af_max"] = max(seen)
        if np.isnan(out["af_overall"]):
            out["af_overall"] = float(np.median(seen))
    return out


def _consequence_severity(so_id: str | None,
                          variant_effect_arr) -> tuple[int, bool]:
    """Score the most-severe consequence on an ordinal scale; flag VEP LoF."""
    label_severity = 0
    is_lof = False
    if variant_effect_arr is None:
        return 0, False
    for entry in variant_effect_arr:
        if entry is None or entry.get("method") != "VEP":
            continue
        a = (entry.get("assessment") or "").lower()
        score = SEVERITY_MAP.get(a, 0)
        label_severity = max(label_severity, score)
        if a in {"transcript_ablation", "stop_gained", "splice_donor_variant",
                  "splice_acceptor_variant", "frameshift_variant", "stop_lost"}:
            is_lof = True
    return label_severity, is_lof


def parse_one_part(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path,
                         columns=["variantId", "chromosome", "position",
                                   "referenceAllele", "alternateAllele",
                                   "variantEffect", "alleleFrequencies"])
    rows = []
    for _, r in df.iterrows():
        eff = _extract_variant_effect(r["variantEffect"])
        af = _extract_allele_freq(r["alleleFrequencies"])
        sev, is_lof_vep = _consequence_severity(None, r["variantEffect"])
        rows.append({
            "variantId": r["variantId"],
            "chromosome": str(r["chromosome"]),
            "position": int(r["position"]),
            "ref": r["referenceAllele"],
            "alt": r["alternateAllele"],
            **af,
            **eff,
            "is_lof_vep": is_lof_vep,
            "consequence_severity": sev,
            "has_alphamissense": not np.isnan(eff["alphamissense"]),
            "has_sift": not np.isnan(eff["sift"]),
            "has_foldx": not np.isnan(eff["foldx"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep-only-ot-credible", type=Path, default=None,
                    help="If given, drop variants whose variantId is not in this "
                         "credible-set parquet (saves disk for the headline run).")
    args = ap.parse_args()

    parts = sorted(args.variant_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No part-*.parquet under {args.variant_dir}")
    print(f"[plan] {len(parts)} parts under {args.variant_dir}")

    keep_set: set[str] | None = None
    if args.keep_only_ot_credible is not None:
        cs = pd.read_parquet(args.keep_only_ot_credible,
                              columns=["variantId"])["variantId"]
        keep_set = set(cs.dropna())
        print(f"[filter] keep only variantIds in {args.keep_only_ot_credible.name} "
              f"({len(keep_set):,} ids)")

    frames: list[pd.DataFrame] = []
    for p in tqdm(parts, desc="flatten"):
        sub = parse_one_part(p)
        if keep_set is not None:
            sub = sub[sub["variantId"].isin(keep_set)]
        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)
    print(f"[shape] {len(out):,} rows, {len(out.columns)} cols")
    print(f"[chroms] {sorted(out['chromosome'].unique())}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"saved: {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
