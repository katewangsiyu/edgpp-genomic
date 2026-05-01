# Phase 4: Code & Tools Survey

## Repos for Phase 3 papers

| # | Paper | Repo URL | ★ | Lang | Last update | Doc | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Kandinsky CP (Bairaktari/Wu/Wu, ICML 2025) | **none** | — | — | checked 2026-04-30 | — | Theory-only release; `NKI-AI/kandinsky-calibration` is unrelated 2023 segmentation work |
| 2 | Yao/He/Gastpar (ICLR 2026) | **none** | — | — | checked 2026-04-30 | — | Theory-heavy SGD bounds; no empirical code |
| 3 | RC3P (NeurIPS 2024) | https://github.com/YuanjieSh/RC3P | 5 | Python | 2024 | decent | Reproduces CIFAR10/100, mini-ImageNet, Food-101, EuroSAT |
| 4 | OT-CP (ICML 2025) | https://github.com/gauthierthurin/OTCP | 1 | Jupyter | 2025 | minimal | 2 notebooks; no fig/table mapping |
| 5 | Backward CP (Gauthier/Bach/Jordan, NeurIPS 2025) | https://github.com/GauthierE/backward-cp | 4 | Jupyter | 2025 | decent | BCW + CIFAR-10 const-size + adapt-size |
| 6 | Length Optimization CP (NeurIPS 2024) | https://github.com/shayankiyani98/CP | 8 | Jupyter | unstated | minimal | Single `CPL.ipynb`, empty README |
| 7 | CP²-HPD (ICLR 2025) | **none** | — | — | checked 2026-04-30 | — | `stat-ml/rcp` is the same author's *follow-up* (ICML 2025), not CP²-HPD |
| 8 | DeepWAS (Amin et al., ICML 2025) | https://github.com/AlanNawzadAmin/DeepWAS | 9 | Python | 2025 | decent | Training + semi-synthetic sim; no fig map |

## Verified active "known" repos (HCCP ecosystem)

| Repo | URL | ★ | Lang | Last release | Doc |
|---|---|---|---|---|---|
| **DEGU** (npj AI 2026) | https://github.com/zrcjessica/ensemble_distillation | 2 | Python+Jupyter | active (116 commits) | **excellent** — `paper_reproducibility/` folder |
| **Borzoi** (Nat Genet 2024) | https://github.com/calico/borzoi | 237 | Python | active | **excellent** — install/models/tutorials |
| **GPN / GPN-MSA** (Genome Biol 2025) | https://github.com/songlab-cal/gpn | 339 | Python+Jupyter | 2025-09-23 | **excellent** — GPN/GPN-MSA/PhyloGPN/GPN-Star |
| **TraitGym** (bioRxiv 2025) | https://github.com/songlab-cal/TraitGym | 17 | Jupyter+Python | active (238 commits) | **excellent** — Snakemake + HF + Colab |
| **crepes** (CP toolkit) | https://github.com/henrikbostrom/crepes | 568 | Python | v0.9.0 2025-10-08 | **excellent** — Mondrian + predictive systems |
| **MAPIE** (CP library) | https://github.com/scikit-learn-contrib/mapie | ~1500 | Python+Jupyter | v1.3.0 2026-02-03 | **excellent** — sklearn-compatible |

## Top 5 to study for HCCP implementation patterns

1. **`crepes`** — closest reference for Mondrian + predictive-systems API; we already use v0.9.0 in `R_raw/cp_baselines_h2h/`. Re-read internal Mondrian regressor before finalizing §6 baselines.
2. **`zrcjessica/ensemble_distillation` (DEGU)** — direct comparator. `paper_reproducibility/` is canonical TF/Keras heteroscedastic NLL student loss reference for our §6.4 framing of "DEGU as ally / pioneer".
3. **`YuanjieSh/RC3P`** — best-documented class-conditional CP repo. Rank-calibrated thresholding loop is template for class-cond T3' certificate code if we expose HCCP as library.
4. **`GauthierE/backward-cp`** — short, focused notebooks per experiment. Good packaging pattern for `T3'` + `T5.1` + `T5.2` reproducibility bundle.
5. **`scikit-learn-contrib/mapie`** — production-grade CP library. v1.3 abstention/risk-control API helps decide: upstream HCCP as `crepes` extension or `mapie` `ConformalRegressor` subclass for D&B Track artifact.

## Skip

- `shayankiyani98/CP` (Length-Opt) — minimal doc, single notebook
- `gauthierthurin/OTCP` — minimal doc, 2 notebooks

## Implications for HCCP

- **Yao 2026 has no public code** → easier rebuttal "their headline rate is theoretical only; we provide implementations + TraitGym empirics"
- **Kandinsky has no public code** → same; reduces empirical comparability concerns
- **DEGU has `paper_reproducibility/`** with TF/Keras → directly addressable in our §6.4 H2H reframing
- **TraitGym repo is healthy + Snakemake-based** → reproducibility story is solid; can ship HCCP as Snakemake module on top
- **3 leading CP libraries (crepes, MAPIE, +maybe puncc)** all have Mondrian primitives → upstreaming path exists for D&B Track artifact

✅ **Phase 4 complete.** ≥3 repos documented (8 candidates + 6 ecosystem = 14 total). Publication-track artifacts strategy: ship HCCP as `crepes` add-on (closest API match) for D&B Track P1.

→ Proceed to Phase 5 (Synthesis + gaps).
