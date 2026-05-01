# Phase 2: Survey — 81 Papers Clustered

**Target**: heteroscedastic conformal prediction × variant effect prediction
**Coverage**: 2021–2026, NeurIPS / ICLR / ICML + arXiv preprints
**Total papers in `paper_db.jsonl`**: 81 (255 peer-reviewed + 16 preprints, dedup from 312 raw hits)

## Year distribution
- 2021–2023: 15 (foundational CP + uncertainty)
- 2024: ~30 (frontier shift to conditional CP)
- 2025: ~22 (Kandinsky era + heteroscedastic explosion)
- 2026 (pre-press): 18 (cutting edge)

## Source distribution
- ai-paper-finder (peer-reviewed conferences): 65
- arxiv preprints: 16

---

## Cluster A — Conditional / class-cond / group-cond CP (the T5 frontier)

The most load-bearing cluster for HCCP's hero claims. Methods aim to give exact coverage *conditional on subgroups* defined either on labels (class-cond) or covariates (covariate-cond / group-cond).

| Year | Venue | Paper | Why it matters |
|---|---|---|---|
| 2025 | ICML | **Kandinsky CP: Beyond Class- and Covariate-Conditional Coverage** (Bairaktari/Wu/Wu) | **Subsumes class-cond + Mondrian as special cases**; minimax-optimal high-prob conditional bound. Direct T5 threat. |
| 2024 | NeurIPS | **CP for Class-wise Coverage via Augmented Label Rank Calibration** | Direct class-conditional baseline; competes with our class axis |
| 2024 | NeurIPS | **Conformal Classification with Equalized Coverage for Adaptively Selected Groups** | Adaptive group definition |
| 2023 | arXiv→JMLR 2025 | **Conformal Prediction with Conditional Guarantees** (Gibbs/Cherian/Candès) | Foundational text; quantile-based conditional bounds |
| 2024 | NeurIPS | **Length Optimization in Conformal Prediction** | Width as objective; relevant to "tight" hero |
| 2025 | ICLR | **Probabilistic CP with Approximate Conditional Validity** | Approximate conditional via density ratio |
| 2025 | ICML | **Rectifying Conformity Scores for Better Conditional Coverage** | Score reweighting for cond coverage |
| 2026 | ICLR | **Adaptive CP via Mixture-of-Experts Gating Similarity** | Adaptive bin partitioning — rival to our K-selection |

## Cluster B — Heteroscedastic / locally adaptive scores (HCCP's σ̂(x) lineage)

| Year | Venue | Paper |
|---|---|---|
| 2024 | ICML | Conformalized Adaptive Forecasting of Heterogeneous Trajectories |
| 2025 | NeurIPS | **Scalable and Adaptive Prediction Bands with Kernel Sum-of-Squares** |
| 2025 | ICLR | Kernel-based Optimally Weighted Conformal Time-Series Prediction |
| 2024 | ICML | Adaptive Conformal Inference by Betting |
| 2025 | NeurIPS | Bootstrap Your Uncertainty: Adaptive Robust Classification driven by Optimal Transport |
| 2026 | ICLR | JAPAN: Joint Adaptive Prediction Areas with Normalising Flow |
| 2024 | arXiv | HQR-WACI: Heteroscedastic Quantile Regression + Width-Adaptive Conformal Inference |
| 2025 | arXiv | Multivariate CP via Conformalized Gaussian Scoring (residual whitening) |
| 2025 | ICLR | CONTRA: CP Region via Normalizing Flow Transformation |

## Cluster C — CP rate / theory / non-asymptotic (the T5.1/T5.2 frontier)

| Year | Venue | Paper | Threat level |
|---|---|---|---|
| 2026 | ICLR | **Non-Asymptotic Analysis of Efficiency in Conformalized Regression** | **HIGH** — direct T5.1/T5.2 competitor |
| 2024 | NeurIPS | Generalized Fast Exact Conformalization (Diyang Li) | Solution-path geometry; relevant to oracle K\* construction |
| 2025 | ICML | **Optimal Transport-based Conformal Prediction** | Rate via OT; possible alt to T5.2 lower bound |
| 2025 | ICML | A Unified Comparative Study with Generalized Conformity Scores for Multi-Output CP | Comparative theory |
| 2025 | NeurIPS | Backward Conformal Prediction (Gauthier/Bach/Jordan) | Fixed-set-size variant |
| 2025 | ICML | Online CP via Online Optimization | Online complement |
| 2024 | ICLR | Non-Exchangeable Conformal Risk Control | Foundational |
| 2026 | ICLR | Distribution-informed Online CP | New online angle |
| 2026 | ICLR | Neural Optimal Transport Meets Multivariate CP | Multivariate extension |
| 2025 | ICML | False Coverage Proportion Control for CP | FCP control |

## Cluster D — CP under shift / robustness / OOD

| Year | Venue | Paper |
|---|---|---|
| 2025 | NeurIPS | **CP under Lévy-Prokhorov Distribution Shifts** |
| 2025 | ICLR | Wasserstein-Regularized CP under General Distribution Shift |
| 2026 | ICLR | CP with Corrupted Labels: Uncertain Imputation and Robust Re-weighting |
| 2024 | arXiv | Conformal Predictive Systems under Covariate Shift |

## Cluster F — VEP / genomic / protein language model (the application target)

| Year | Venue | Paper | Use |
|---|---|---|---|
| 2025 | ICML | **Training Flexible Models of Genetic Variant Effects from Functional Annotations using Accelerated Linear Algebra** | Direct VEP at top venue |
| 2025 | NeurIPS | **From Likelihood to Fitness: Improving VEP in Protein and Genome Language Models** | Protein-LM VEP |
| 2025 | NeurIPS | **PRSformer: Disease Prediction from Million-Scale Individual Genotypes** | Population-scale transformer |
| 2025 | ICML | **Predicting Mutational Effects on Protein Binding from Folding Energy** | Energy-based VEP |
| 2024 | arXiv | Leveraging Genomic Deep Learning Models for Non-Coding Variant Effects | Survey/method |
| 2025 | arXiv | Combining Multiplexed Functional Data to Improve Variant Classification | Multiplexed assay → variants |

## Cluster G — Selective prediction / abstention

| Year | Venue | Paper |
|---|---|---|
| 2022 | NeurIPS | Efficient Active Learning with Abstention |
| 2023 | NeurIPS | Counterfactually Comparing Abstaining Classifiers |

## Cluster H — Uncertainty / ensemble distillation (DEGU lineage)

| Year | Venue | Paper |
|---|---|---|
| 2025 | NeurIPS | **Knowledge Distillation of Uncertainty using Deep Latent Factor Model** |
| 2026 | ICLR | **Contextual Similarity Distillation: Ensemble Uncertainties with a Single Model** |
| 2024 | NeurIPS | Multi-model Ensemble Conformal Prediction in Dynamic Environments |
| 2023 | NeurIPS | A Rigorous Link between Deep Ensembles and (Variational) Bayesian Methods |
| 2022 | NeurIPS | Deep Ensembles Work, But Are They Necessary? |
| 2023 | NeurIPS | Progressive Ensemble Distillation: Building Ensembles for Efficient Inference |
| 2022 | NeurIPS | Functional Ensemble Distillation |
| 2022 | NeurIPS | Disentangling the Predictive Variance of Deep Ensembles through the NTK |

## Critical absences (notable papers not surfaced; manually flagged)

The shim/arxiv coverage missed several known critical works that exist in the bib but aren't in our paper_db:

- **DEGU (Zhou et al., npj AI 2026 vol 2 art 3)** — backend doesn't index npj AI; manually known
- **TraitGym (Benegas/Eraslan/Song, bioRxiv 2025)** — bioRxiv not indexed; manually known
- **AlphaMissense (Cheng et al., Science 2023)** — Science not indexed
- **Borzoi (Linder et al., Nature Genetics 2024)** — Nat Gen not indexed
- **GPN-MSA (Benegas et al., Genome Biol 2025)** — Genome Biol not indexed
- **Conformal Risk Control (Bates/Candès et al., JRSSB 2024)** — JRSSB not indexed
- **CQR (Romano/Patterson/Candès, NeurIPS 2019)** — pre-2021 cutoff

These are already cited in `papers/neurips2027_pathA/refs.bib` and don't need re-fetching for this survey, but Phase 3 must include several for cross-coverage.

## Phase 3 selection candidates (15 papers)

**Tier 1 (MUST READ — concurrent threats / direct competitors)**:
1. **Kandinsky Conformal Prediction** (Bairaktari/Wu/Wu, ICML 2025)
2. **Non-Asymptotic Analysis of Efficiency in Conformalized Regression** (ICLR 2026)
3. **CP for Class-wise Coverage via Augmented Label Rank Calibration** (NeurIPS 2024)
4. **CP with Conditional Guarantees** (Gibbs/Cherian/Candès, arXiv 2305.12616)

**Tier 2 (read for theory positioning)**:
5. **Probabilistic CP with Approximate Conditional Validity** (ICLR 2025)
6. **Optimal Transport-based CP** (ICML 2025)
7. **Backward Conformal Prediction** (Gauthier/Bach/Jordan, NeurIPS 2025)
8. **Adaptive CP via MoE Gating** (ICLR 2026)
9. **Length Optimization in CP** (NeurIPS 2024)

**Tier 3 (heteroscedastic + score lineage)**:
10. **Scalable and Adaptive Prediction Bands with Kernel Sum-of-Squares** (NeurIPS 2025)
11. **Multivariate CP via Conformalized Gaussian Scoring** (arXiv 2507.20941)

**Tier 4 (VEP grounding)**:
12. **Training Flexible Models of Genetic Variant Effects** (ICML 2025)
13. **From Likelihood to Fitness** (NeurIPS 2025)

**Tier 5 (DEGU lineage / uncertainty distillation)**:
14. **Knowledge Distillation of Uncertainty using Deep Latent Factor Model** (NeurIPS 2025)
15. **Contextual Similarity Distillation: Ensemble Uncertainties with a Single Model** (ICLR 2026)

✅ **Phase 2 complete.** 81 papers, ≥35 target met, 8 thematic clusters, 15 papers shortlisted for Phase 3 deep read.

→ Proceed to Phase 3 (download + read 8-15 PDFs + structured notes).
