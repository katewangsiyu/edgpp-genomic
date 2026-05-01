# Phase 1: Frontier — Heteroscedastic Conformal Prediction × Variant Effect Prediction

**Topic**: heteroscedastic conformal prediction for variant effect prediction (VEP)
**Coverage**: NeurIPS / ICLR / ICML 2024–2026 + recent arXiv preprints (2024–2026)
**Method**: paper_finder_shim (130 raw hits across 3 queries) + WebSearch supplements
**Date**: 2026-04-30

---

## TL;DR — Three Concurrent Threats Identified

| Threat | Source | Impact on HCCP |
|---|---|---|
| **T1. DEGU now in npj AI 2026** | Zhou et al., npj AI 2026 vol 2 art 3 (Feb 2026) | Published version explicitly mentions *"conformal prediction offering coverage guarantees"* — must verify whether this is integrated method or citation. Path A's "DEGU as ally / pioneer" framing still holds, but H2H story tightens. |
| **T2. Kandinsky CP (Bairaktari, Wu, Wu, ICML 2025)** | OpenReview `IHAnkPkoiX` | Subsumes class-conditional + Mondrian as special cases + claims **minimax-optimal high-probability conditional coverage bound**. **Does not address heteroscedastic σ̂(x) conditioning** — our defensive moat holds, but T5.2's "equi-bin Mondrian-K class" qualifier becomes load-bearing. |
| **T3. Non-Asymptotic Analysis of Efficiency in Conformalized Regression (ICLR 2026)** | Anon., ICLR 2026 | Direct match to our T5.1 oracle K\* / O(n^{-1/2}) rate framing. Need to check if their bound is dimension-free, what loss function, what the constant looks like. |

These three must be read in Phase 3.

---

## Trending Directions (Phase 1 themes)

### Theme A — **Conditional / class-cond / group-cond CP frontier (HIGH overlap with HCCP T5)**

The whole conformal community has converged on conditional coverage as the key open problem since Vovk's `2012b` impossibility. ICML 2025 / ICLR 2025-2026 have ≥6 papers attacking this frontier:

- **Kandinsky CP** (Bairaktari/Wu/Wu, ICML 2025) — minimax-optimal for overlapping/fractional groups
- **Conformal Prediction with Conditional Guarantees** (Gibbs/Cherian/Candès, JMLR 2025-extension of arXiv 2305.12616)
- **Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration** (NeurIPS 2024) — direct competitor to class-conditional axis
- **Conformal Classification with Equalized Coverage for Adaptively Selected Groups** (NeurIPS 2024) — adaptive group definition
- **Probabilistic Conformal Prediction with Approximate Conditional Validity** (ICLR 2025)
- **Adaptive Conformal Prediction via Mixture-of-Experts Gating** (ICLR 2026) — adaptive bin partitioning, **direct competitor to our K-selection**
- **Conformal Structured Prediction** (ICLR 2025)

### Theme B — **Heteroscedastic / locally adaptive nonconformity scores (CORE OF HCCP)**

The σ̂(x)-normalized nonconformity score line of work:

- **Heteroscedastic Quantile Regression + Width-Adaptive Conformal Inference (HQR-WACI)** (arXiv 2406.14904, multi-step ahead update) — width adapts to predictive uncertainty
- **Multivariate Conformal Prediction via Conformalized Gaussian Scoring** (arXiv 2507.20941, Aug 2025) — whitening residuals to decouple correlations and standardize local variance
- **Optimal Transport-based Conformal Prediction** (ICML 2025) — uses OT for adaptive intervals
- **Scalable and Adaptive Prediction Bands with Kernel Sum-of-Squares** (NeurIPS 2025) — adaptive width via kernel methods
- **CONTRA: Conformal Prediction Region via Normalizing Flow Transformation** (ICLR 2025) — flow-based score normalization
- **Conformal Prediction under Lévy-Prokhorov Distribution Shifts** (NeurIPS 2025) — robustness extension

### Theme C — **CP rate analysis / non-asymptotic theory (DIRECTLY HITS T5)**

- **Non-Asymptotic Analysis of Efficiency in Conformalized Regression** (ICLR 2026) — rate analysis of conformal regression
- **Generalized Fast Exact Conformalization** (Diyang Li, NeurIPS 2024) — second-order solution path geometry
- **Length Optimization in Conformal Prediction** (NeurIPS 2024) — interval length as objective
- **A Unified Comparative Study with Generalized Conformity Scores for Multi-Output CP** (ICML 2025) — comparative theory
- **Backward Conformal Prediction** (Gauthier/Bach/Jordan, NeurIPS 2025) — fixed-set-size variant

### Theme D — **VEP / genomic uncertainty quantification (APPLICATION GROUNDING)**

- **DEGU** (Zhou, Rizzo, Christensen, Tang, Koo, npj AI 2026 vol 2 art 3) — ensemble distillation, bioRxiv 2024 → npj AI Feb 2026 — see threat T1 above
- **Training Flexible Models of Genetic Variant Effects from Functional Annotations using Accelerated Linear Algebra** (ICML 2025) — direct VEP at top venue
- **From Likelihood to Fitness: Improving Variant Effect Prediction in Protein and Genome Language Models** (NeurIPS 2025) — protein-LM VEP
- **PRSformer: Disease Prediction from Million-Scale Individual Genotypes** (NeurIPS 2025) — population-genetic transformer
- **Predicting mutational effects on protein binding from folding energy** (ICML 2025)
- **VIKING: Deep variational inference with stochastic projections** (NeurIPS 2025) — probabilistic deep learning, applicable to genomics

VEP at top ML venues remains thin — most genomic VEP work goes to bioRxiv / Nature Methods / Bioinformatics / Genome Biology. Phase 2 must cover that literature via S2 / arXiv.

---

## Top Papers for Phase 3 Selection (preliminary)

15 candidates flagged for full read in Phase 3:

1. **Kandinsky Conformal Prediction** (Bairaktari/Wu/Wu, ICML 2025) — **MUST READ** (T5 threat)
2. **Non-Asymptotic Analysis of Efficiency in Conformalized Regression** (ICLR 2026) — **MUST READ** (T5 threat)
3. **DEGU** (Zhou et al., npj AI 2026) — **MUST READ** (DEGU updated framing in npj version)
4. **Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration** (NeurIPS 2024)
5. **Conformal Prediction with Conditional Guarantees** (Gibbs/Cherian/Candès, arXiv 2305.12616 / JMLR 2025)
6. **Probabilistic CP with Approximate Conditional Validity** (ICLR 2025)
7. **Adaptive Conformal Prediction via MoE Gating Similarity** (ICLR 2026)
8. **Optimal Transport-based Conformal Prediction** (ICML 2025)
9. **Multivariate CP via Conformalized Gaussian Scoring** (arXiv 2507.20941)
10. **HQR-WACI** (arXiv 2406.14904)
11. **Backward Conformal Prediction** (Gauthier/Bach/Jordan, NeurIPS 2025)
12. **Length Optimization in Conformal Prediction** (NeurIPS 2024)
13. **Conformal Classification with Equalized Coverage for Adaptively Selected Groups** (NeurIPS 2024)
14. **Training Flexible Models of Genetic Variant Effects** (ICML 2025) — VEP grounding
15. **From Likelihood to Fitness** (NeurIPS 2025) — protein-LM VEP

---

## Output Inventory

- `phase1_frontier/paper_finder_config.yaml` — 3 queries, NeurIPS/ICML/ICLR 2024-2026
- `phase1_frontier/search_results/00_heteroscedastic_conformal_prediction.jsonl` (50 hits)
- `phase1_frontier/search_results/01_variant_effect_prediction_deep_learning.jsonl` (50 hits)
- `phase1_frontier/search_results/02_class_conditional_conformal_prediction_mondrian.jsonl` (30 hits)
- **Total: 130 raw hits** (with duplicates between queries; dedupe in Phase 2)
- 15 distinct papers flagged for Phase 3 deep read
- 3 concurrent threats identified (DEGU update, Kandinsky CP, ICLR 2026 rate paper)

✅ **Phase 1 complete.** ≥10 papers identified, trending directions documented, threat assessment done.

→ Proceed to Phase 2 (broaden 2021-2026 + multi-source: arXiv + S2 to fill VEP gap).
