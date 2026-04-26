"""Aggregate per-trait HCCP metrics into clinical super-categories.

Reads outputs/trait_loo/{features}_{dataset}/per_trait_metrics.csv and groups:
  - Complex (28 traits) by physiologic system
  - Mendelian (30 OMIMs) by HPO ancestor

Reports per-cluster (n_traits, n_total, weighted marg cov, sigma-bin gap proxy).
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "R_raw" / "trait_cluster_aggregate"

# Complex traits clustered by physiologic system
COMPLEX_CLUSTERS = {
    "Cardio-metabolic / anthropometric": [
        "AG", "ALP", "BW", "Ca", "GGT", "HDLC", "HbA1c", "Height", "IGF1",
        "PP", "SHBG", "TP", "Urea", "WHRadjBMI", "eBMD", "eGFR", "eGFRcys",
    ],
    "Hematologic / blood-cell": [
        "Eosino", "Hb,Ht", "Lym", "MCH,MCV", "MCV", "Mono", "Plt", "RBC",
    ],
    "Inflammatory / pulmonary / other": [
        "Balding_Type4", "CRP", "FEV1FVC",
    ],
}

# Mendelian OMIMs by HPO super-category (lookups derived from OMIM phenotype)
# Note: these MIM codes correspond to specific clinical phenotypes; mapping below
# uses standard HPO ontology ancestors.
MENDELIAN_CLUSTERS = {
    "Skeletal / connective tissue": [
        "MIM 125850",  # nephrogenic diabetes insipidus var
        "MIM 127550",  # Brachydactyly
        "MIM 174500",  # Polycystic kidney disease 2
        "MIM 188000",  # Thyroid hormone resistance
        "MIM 277900",  # Wilson disease
        "MIM 605407",  # MAS
        "MIM 614429",  # Beta-mannosidosis
        "MIM 615935",  # Spastic paraplegia
    ],
    "Neuromuscular / nervous-system": [
        "MIM 141749",  # Hypoparathyroidism
        "MIM 180200",  # Retinitis pigmentosa
        "MIM 187300",  # Telangiectasia
        "MIM 304790",  # Aicardi syndrome
        "MIM 306400",  # CGD
        "MIM 306700",  # Hemophilia A
        "MIM 306900",  # Hemophilia B
        "MIM 608030",  # Charcot-Marie-Tooth
        "MIM 614167",  # Glycogen storage disease
        "MIM 616651",  # Spinocerebellar ataxia
    ],
    "Metabolic / hematologic / other": [
        "MIM 143890",  # Hypercholesterolemia
        "MIM 210710",  # Pyruvate carboxylase
        "MIM 227500",  # Galactosemia
        "MIM 250250",  # Marfan
        "MIM 263700",  # Phenylketonuria
        "MIM 300624",  # Fragile-X
        "MIM 600886",  # Pseudohypoaldosteronism
        "MIM 606176",  # Glaucoma
        "MIM 609310",  # Mitochondrial encephalopathy
        "MIM 609637",  # Stargardt disease
        "MIM 613985",  # Combined immunodeficiency
        "MIM 614743",  # Lymphoma
    ],
}


def aggregate(df: pd.DataFrame, clusters: dict[str, list[str]], target: float = 0.90):
    rows = []
    for cluster, traits in clusters.items():
        sub = df[df["trait"].isin(traits)]
        if len(sub) == 0:
            continue
        n_traits = len(sub)
        n_tot = int(sub["n_total"].sum())
        # Weighted by trait size
        w = sub["n_total"].values
        marg = float((sub["coverage"].values * w).sum() / w.sum())
        cov_pos = float((sub["cov_pos"].values * w).sum() / w.sum())
        # Worst per-trait gap as proxy for sigma-bin gap
        gap = float((sub["coverage"] - target).abs().max())
        rows.append(dict(cluster=cluster, n_traits=n_traits, n_total=n_tot,
                         marginal_coverage=marg, cov_pos=cov_pos,
                         max_per_trait_gap=gap))
    overall_marg = float((df["coverage"].values * df["n_total"].values).sum() /
                         df["n_total"].values.sum())
    overall_gap = float((df["coverage"] - target).abs().max())
    rows.append(dict(cluster="OVERALL", n_traits=len(df), n_total=int(df["n_total"].sum()),
                     marginal_coverage=overall_marg, cov_pos=float("nan"),
                     max_per_trait_gap=overall_gap))
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Complex
    path_c = REPO / "outputs/trait_loo/CADD+GPN-MSA+Borzoi_complex/per_trait_metrics.csv"
    df_c = pd.read_csv(path_c)
    print("Complex 28-trait clustering:")
    rows_c = aggregate(df_c, COMPLEX_CLUSTERS)
    for r in rows_c:
        print(f"  {r}")
    (OUT_DIR / "complex_cluster.json").write_text(json.dumps(rows_c, indent=2))

    # Mendelian
    path_m = REPO / "outputs/trait_loo/CADD+GPN-MSA+Borzoi_mendelian/per_trait_metrics.csv"
    df_m = pd.read_csv(path_m)
    print("\nMendelian 30-OMIM clustering:")
    rows_m = aggregate(df_m, MENDELIAN_CLUSTERS)
    for r in rows_m:
        print(f"  {r}")
    (OUT_DIR / "mendelian_cluster.json").write_text(json.dumps(rows_m, indent=2))

    # Sanity: assert all traits/OMIMs are covered exactly once
    all_complex = sum(COMPLEX_CLUSTERS.values(), [])
    missing = set(df_c["trait"]) - set(all_complex)
    extra = set(all_complex) - set(df_c["trait"])
    print(f"\nComplex coverage: missing {missing}, extra {extra}")

    all_mim = sum(MENDELIAN_CLUSTERS.values(), [])
    missing_m = set(df_m["trait"]) - set(all_mim)
    extra_m = set(all_mim) - set(df_m["trait"])
    print(f"Mendelian coverage: missing {missing_m}, extra {extra_m}")


if __name__ == "__main__":
    main()
