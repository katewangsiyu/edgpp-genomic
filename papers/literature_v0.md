# Literature v0 — Conformalized Heteroscedastic Variant Effect Prediction

**Target venue**: NeurIPS 2027 main conference
**Benchmark**: TraitGym (Mendelian matched_9 + complex_traits matched_9)
**Direct competitor**: DEGU (Zhou & Koo, npj AI 2026) — heteroscedastic NLL distillation
**Project posture**: Post-Borzoi aggregator + conformalized selective head; class-conditional coverage guarantees with heteroscedastic non-conformity scores

Harvest date: 2026-04-16. Verified links preferred; when in doubt, fall back to "search by title".

---

## 1. Core conformal prediction foundations

### 1.1 Romano, Patterson, Candès (2019) — Conformalized Quantile Regression
- **Authors**: Yaniv Romano, Evan Patterson, Emmanuel J. Candès
- **Venue**: NeurIPS 2019
- **Link (arXiv)**: https://arxiv.org/abs/1905.03222
- **Link (NeurIPS)**: https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf
- **Summary**: Combines split-conformal calibration with quantile regression. Builds prediction intervals whose width adapts to local noise, with finite-sample marginal coverage and no distributional assumptions. Introduces the CQR non-conformity score E_i = max{q̂_lo(X_i) − Y_i, Y_i − q̂_hi(X_i)}.
- **Why it matters for us**: CQR is the canonical template for heteroscedastic conformal prediction. Our "heteroscedastic VEP" head will plug DEGU-style σ(x) or a Borzoi-residual GBM into the non-conformity score σ(x)^{−1}·|p − y|, which is a direct binary-classification analogue of CQR.

### 1.2 Tibshirani, Barber, Candès, Ramdas (2019) — Conformal Prediction under Covariate Shift
- **Authors**: Ryan J. Tibshirani, Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas
- **Venue**: NeurIPS 2019
- **Link**: https://arxiv.org/abs/1904.06019
- **Summary**: Introduces weighted conformal prediction, restoring finite-sample coverage under known (or estimable) likelihood-ratio covariate shift between calibration and test distributions.
- **Why it matters**: TraitGym's `matched_9` splits induce deliberate covariate shift between training (chr 1–16) and test chromosomes (17–22, X). Also, Mendelian vs complex traits are different regimes. This is the go-to machinery if our calibration set is not exchangeable with the deployment target.

### 1.3 Barber, Candès, Ramdas, Tibshirani (2021) — Predictive Inference with the Jackknife+
- **Authors**: Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, Ryan J. Tibshirani
- **Venue**: Annals of Statistics 49(1), 486–507
- **DOI**: 10.1214/20-AOS1965
- **Link (arXiv)**: https://arxiv.org/abs/1905.02928
- **Summary**: Generalizes the jackknife by using leave-one-out predictions at the test point to account for variability in the fit. Proves rigorous (2α) coverage for symmetric algorithms with exchangeable data.
- **Why it matters**: Our chromosome-LOO CV in `09_aggregator_chrom_loo.py` is structurally jackknife-like. Jackknife+ gives the formal coverage guarantee we currently lack when we select λ by inner-CV. It is the cleanest replacement for our current ad-hoc internal CV.

### 1.4 Barber, Candès, Ramdas, Tibshirani (2023) — Conformal Prediction Beyond Exchangeability
- **Authors**: Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, Ryan J. Tibshirani
- **Venue**: Annals of Statistics 51(2), 816–845
- **DOI**: 10.1214/23-AOS2276
- **Link (arXiv)**: https://arxiv.org/abs/2202.13415
- **Summary**: Relaxes exchangeability via user-chosen weights and asymmetric algorithms. Provides a coverage gap bound expressed as a total-variation distance between calibration and test distributions.
- **Why it matters**: Genomic variants on different chromosomes are empirically not exchangeable (LD structure, chromatin context). This paper provides the theoretical stack for "coverage under structured genomic violations" — likely part of our theorem section.

### 1.5 Angelopoulos & Bates (2022) — A Gentle Introduction to Conformal Prediction and Distribution-Free UQ
- **Authors**: Anastasios N. Angelopoulos, Stephen Bates
- **Venue**: Foundations and Trends in Machine Learning 16(4)
- **Link (arXiv)**: https://arxiv.org/abs/2107.07511
- **Summary**: Self-contained tutorial covering split CP, CQR, APS, RAPS, risk control, and applications to structured outputs and distribution shift. Includes Python notebooks.
- **Why it matters**: The Rosetta stone for writing the background section. Also the cleanest reference for readers from the genomics community who are new to conformal prediction. Use it to justify notation choices in the paper.

### 1.6 Vovk, Lindsay, Nouretdinov, Gammerman (2003) — Mondrian Confidence Machine
- **Authors**: Vladimir Vovk, David Lindsay, Ilia Nouretdinov, Alex Gammerman
- **Venue**: Royal Holloway Tech Report / ICML workshop (2003)
- **Link**: Search by title (Semantic Scholar: https://www.semanticscholar.org/paper/Mondrian-Confidence-Machine-Vovk-Lindsay/abd6ce8a5484b2046584d742ef4002cac4ba0ca3). Original tech report not always archived with a stable URL.
- **Summary**: Earliest formulation of group/class-conditional (Mondrian) conformal predictors. Coverage guarantee holds within each user-defined category (e.g. label, region) rather than only marginally.
- **Why it matters**: **Central to our contribution.** TraitGym is massively imbalanced (positives are rare causal variants). Marginal 90% coverage can hide 50% coverage on the pathogenic class. Mondrian CP on the label axis fixes this. This is exactly the class-conditional conformal formulation that bumped pathogenic coverage 73%→89% in our Day 9 experiment.

### 1.7 Vovk, Gammerman, Shafer (2005) — Algorithmic Learning in a Random World
- **Authors**: Vladimir Vovk, Alexander Gammerman, Glenn Shafer
- **Venue**: Springer monograph (2nd ed. 2022)
- **Link (Springer)**: https://link.springer.com/book/10.1007/b106715
- **Link (book site)**: https://www.alrw.net/
- **Summary**: The canonical reference for conformal prediction theory: exchangeability, validity, efficiency, online CP, Mondrian CP, Venn predictors. The 2022 second edition adds three chapters on modern developments.
- **Why it matters**: Citable foundational reference. Needed for Mondrian CP provenance (Section 2.1 of the 1st ed.) and for any argument that references the online / streaming / conditional validity framework.

---

## 2. Heteroscedastic / adaptive conformal

### 2.1 Romano, Sesia, Candès (2020) — Classification with Valid and Adaptive Coverage (APS)
- **Authors**: Yaniv Romano, Matteo Sesia, Emmanuel J. Candès
- **Venue**: NeurIPS 2020 (Spotlight)
- **Link (arXiv)**: https://arxiv.org/abs/2006.02544
- **Code**: https://github.com/msesia/arc
- **Summary**: Introduces the Adaptive Prediction Set (APS) score E_i = Σ_{y: π̂(y|x) ≥ π̂(Y_i|x)} π̂(y|x). Gives adaptive set sizes that shrink on easy inputs and grow on hard inputs, with marginal coverage.
- **Why it matters**: VEP is fundamentally a binary classification task (causal vs non-causal). APS (and its RAPS extension) is the correct adaptive non-conformity score. Combining APS with the Mondrian (per-class) approach is our baseline recipe for class-conditional calibration.

### 2.2 Angelopoulos, Bates, Jordan, Malik (2021) — Uncertainty Sets for Image Classifiers (RAPS)
- **Authors**: Anastasios Angelopoulos, Stephen Bates, Jitendra Malik, Michael I. Jordan
- **Venue**: ICLR 2021 (Spotlight)
- **Link (arXiv)**: https://arxiv.org/abs/2009.14193
- **Summary**: RAPS regularizes APS by penalizing the inclusion of low-probability classes after Platt scaling, yielding smaller and more stable prediction sets while retaining exact marginal coverage.
- **Why it matters**: The stability argument (fewer "wild" large sets in the tail) is directly relevant to rare pathogenic variants. Not a drop-in for binary VEP, but gives us the design pattern for regularizing class-conditional conformal sets when class probabilities are miscalibrated (as they are for rare-positive VEP).

### 2.3 Lei, G'Sell, Rinaldo, Tibshirani, Wasserman (2018) — Distribution-Free Predictive Inference for Regression
- **Authors**: Jing Lei, Max G'Sell, Alessandro Rinaldo, Ryan J. Tibshirani, Larry Wasserman
- **Venue**: JASA 113(523), 1094–1111
- **Link (arXiv)**: https://arxiv.org/abs/1604.04173
- **Summary**: Compares full conformal, split conformal, and jackknife for regression. Introduces locally-weighted conformal scores |y − μ̂(x)| / σ̂(x) for heteroscedastic adaptivity (the direct conceptual ancestor of CQR).
- **Why it matters**: **The heteroscedastic score σ̂(x)^{−1} · residual from Section 5 is exactly our target non-conformity score.** Replace μ̂, σ̂ with Borzoi-LogReg and DEGU's σ-head (or our GBM-residual model) and we have the core method. Theoretical analysis of efficiency vs coverage is directly reusable.

---

## 3. Genomic variant effect prediction (VEP)

### 3.1 Benegas, Eraslan, Song (2025) — TraitGym
- **Title**: Benchmarking DNA Sequence Models for Causal Regulatory Variant Prediction in Human Genetics
- **Authors**: Gonzalo Benegas, Gökçen Eraslan, Yun S. Song
- **Venue**: bioRxiv 2025 (doi: 10.1101/2025.02.11.637758)
- **Link**: https://www.biorxiv.org/content/10.1101/2025.02.11.637758v2
- **Code**: https://github.com/songlab-cal/TraitGym
- **Summary**: 113 Mendelian + 83 complex traits; matched_9 positive/negative controls; chrom-stratified evaluation with `AUPRC_by_chrom_weighted_average` as the leaderboard metric. Finding: alignment-based (CADD, GPN-MSA) wins on Mendelian; Borzoi wins on complex non-disease.
- **Why it matters**: **This is our benchmark.** All experiments must report the exact leaderboard metric (cf. `project_eval_methodology_bug.md`). The Borzoi_L2_L2 zero-shot bar = 0.4356 and CADD+Borzoi LogReg SOTA = 0.7515 are our reference points.

### 3.2 Zhou, Rizzo, Koo (2026) — DEGU (Uncertainty-Aware Genomic Deep Learning with Knowledge Distillation)
- **Authors**: Jessica Zhou, Kaeli Rizzo, Peter K. Koo
- **Venue**: npj Artificial Intelligence 2, Article 3 (2026); bioRxiv 2024.11.13.623485
- **Link (journal)**: https://www.nature.com/articles/s44387-025-00053-3
- **Link (bioRxiv)**: https://www.biorxiv.org/content/10.1101/2024.11.13.623485v2.full
- **Code**: https://github.com/zrcjessica/ensemble_distillation
- **Summary**: Distills an ensemble into a single model that outputs both the ensemble mean (epistemic) and variance (aleatoric via an auxiliary σ head) using heteroscedastic NLL. Reports calibrated UQ and post-hoc conformal prediction on top.
- **Why it matters**: **Direct competitor.** DEGU already claims "conformal prediction on top of heteroscedastic distillation", but (a) no TraitGym numbers; (b) marginal coverage only, no class-conditional guarantee; (c) non-conformity score is the raw NLL, not explicitly σ-weighted. Our paper's pitch: **on TraitGym, add class-conditional (Mondrian) conformal on top of DEGU-style σ, and show the pathogenic-coverage gap that vanilla CP hides.**

### 3.3 Linder, Srivastava, Yuan, Agarwal, Kelley (2025) — Borzoi
- **Title**: Predicting RNA-seq Coverage from DNA Sequence as a Unifying Model of Gene Regulation
- **Authors**: Johannes Linder, Divyanshi Srivastava, Han Yuan, Vikram Agarwal, David R. Kelley
- **Venue**: Nature Genetics 57, 949–961 (2025); bioRxiv 2023.08.30.555582
- **Link (Nature Genetics)**: https://www.nature.com/articles/s41588-024-02053-6
- **Link (bioRxiv)**: https://www.biorxiv.org/content/10.1101/2023.08.30.555582v1
- **Code**: https://github.com/calico/borzoi
- **Summary**: CNN predicting RNA-seq coverage at 32bp resolution from 524kb input. Enformer successor that unifies transcription, splicing, polyadenylation for variant scoring.
- **Why it matters**: **Our upstream teacher.** Our aggregator in `09_aggregator_chrom_loo.py` starts from Borzoi embeddings/scores. We never retrain Borzoi — we sit on top of it and add a conformalized reliability head. Citation required for reproducing the TraitGym LogReg pipeline.

### 3.4 Rentzsch, Witten, Cooper, Shendure, Kircher (2019) — CADD
- **Title**: CADD: Predicting the Deleteriousness of Variants throughout the Human Genome
- **Authors**: Philipp Rentzsch, Daniela Witten, Gregory M. Cooper, Jay Shendure, Martin Kircher
- **Venue**: Nucleic Acids Research 47(D1), D886–D894 (2019)
- **DOI**: 10.1093/nar/gky1016
- **Link (Oxford)**: https://academic.oup.com/nar/article/47/D1/D886/5146191
- **Summary**: Integrative annotation-based scoring combining 60+ genomic features via logistic regression trained on simulated vs observed alleles. Long-standing baseline and strong on Mendelian.
- **Why it matters**: Part of the CADD+Borzoi hybrid that currently owns TraitGym SOTA (0.7515). We need to understand its feature construction to avoid leakage in our selective-head features and to explain why the CADD+Borzoi regime leaves little headroom for a residual GBM (cf. our closed λ=0 result).

### 3.5 Benegas, Albors, Aw, Ye, Song (2025) — GPN-MSA
- **Title**: A DNA Language Model Based on Multispecies Alignment Predicts the Effects of Genome-Wide Variants
- **Authors**: Gonzalo Benegas, Carlos Albors, Alan J. Aw, Chengzhong Ye, Yun S. Song
- **Venue**: Nature Biotechnology (2025); bioRxiv 2023.10.10.561776
- **Link (Nature Biotech)**: https://www.nature.com/articles/s41587-024-02511-w
- **Link (bioRxiv)**: https://www.biorxiv.org/content/10.1101/2023.10.10.561776v1
- **Code**: https://github.com/songlab-cal/gpn
- **Summary**: DNA language model trained on multiple-sequence alignments across species. Noncoding-competitive, trains in hours. Strong Mendelian scorer on TraitGym.
- **Why it matters**: Alternative teacher-side feature stream (complementary to Borzoi). Relevant if we want to extend the aggregator beyond CADD+Borzoi to include GPN-MSA. Also same group as TraitGym — useful for framing.

---

## 4. Selective classification / abstention

### 4.1 El-Yaniv, Wiener (2010) — On the Foundations of Noise-Free Selective Classification
- **Authors**: Ran El-Yaniv, Yair Wiener
- **Venue**: JMLR 11, 1605–1641 (2010)
- **Link**: https://jmlr.org/papers/v11/el-yaniv10a.html
- **Summary**: Formalizes selective classification (classification with a reject option): characterizes the risk-coverage tradeoff and gives optimal near-RC algorithms under a noise-free assumption.
- **Why it matters**: **Theoretical backbone for the "selective head" half of our paper.** Our `10_selective_head.py` uses reliability scores to decide abstention; this paper gives the formal risk-coverage objective we should be optimizing, and the notion of optimal selective risk. The conformal-calibrated version is novel.

### 4.2 Geifman, El-Yaniv (2017) — Selective Classification for Deep Neural Networks
- **Authors**: Yonatan Geifman, Ran El-Yaniv
- **Venue**: NeurIPS 2017
- **Link (arXiv)**: https://arxiv.org/abs/1705.08500
- **Code**: https://github.com/geifmany/selective_deep_learning
- **Summary**: Practical algorithm to wrap a trained DNN with a selective classifier that attains a user-specified risk level with high probability. Demonstrated on ImageNet and CIFAR.
- **Why it matters**: Closest prior art for "take a pretrained model + train a reliability head + abstain by threshold". Our contribution relative to Geifman–El-Yaniv is (a) genomic VEP domain, (b) conformal finite-sample guarantee replacing the empirical Bernstein bound, (c) class-conditional coverage for imbalanced rare-positive settings.

---

## Must-read (week 1) vs. background

### Must-read in week 1 (read in order)

1. **Angelopoulos & Bates 2022** (1.5) — tutorial. Start here to fix vocabulary (non-conformity score, split CP, APS, RAPS, Mondrian). One sitting.
2. **Lei et al. 2018** (2.3) — read Section 5 (locally-weighted / heteroscedastic conformal) carefully. This is the exact form of our non-conformity score.
3. **Romano, Sesia, Candès 2020** (2.1) — APS, because VEP is classification not regression. Use this as the working non-conformity score template.
4. **Vovk et al. 2003** (1.6) + relevant Vovk–Gammerman–Shafer 2005 chapter (1.7, ch. 4) — Mondrian CP. **This is the technical core of our class-conditional-coverage contribution.**
5. **Zhou, Rizzo, Koo 2026 — DEGU** (3.2) — the direct competitor. Map their σ-head to our non-conformity score. Identify exactly what they do NOT do (class-conditional, TraitGym, formal selective risk) so we can claim it.

### Background (skim, read when needed)

- **Romano, Patterson, Candès 2019 — CQR** (1.1): regression version of the heteroscedastic idea; cite but not central to classification VEP.
- **Barber et al. 2021 — Jackknife+** (1.3): cite for the LOO-CV story; read Section 3 (coverage proof) if we want a formal statement about our chrom-LOO.
- **Barber et al. 2023 — Beyond exchangeability** (1.4): read only when we write the "distribution shift between chromosomes" subsection of the theorem.
- **Tibshirani et al. 2019 — Covariate shift CP** (1.2): read only if reviewers push on train/test distribution shift.
- **Angelopoulos et al. 2021 — RAPS** (2.2): read for the regularization trick if our APS sets blow up in the rare-positive tail.
- **Benegas et al. 2025 — TraitGym** (3.1): already understood operationally; keep for the citation and for Methods-section reference on the matched_9 protocol.
- **Linder et al. 2025 — Borzoi** (3.3): teacher model; read only the variant-scoring section (Methods).
- **Rentzsch et al. 2019 — CADD** (3.4): read Table of features to understand CADD+Borzoi regime.
- **Benegas et al. 2025 — GPN-MSA** (3.5): optional, only if we extend the aggregator.
- **El-Yaniv & Wiener 2010** (4.1): read definitions (selective risk, coverage) and Theorem 21 (optimal selective classifier).
- **Geifman & El-Yaniv 2017** (4.2): read Section 3 (SR function, SGR algorithm); used as direct comparison baseline.

### Explicitly NOT in week 1
Conformal risk control (Angelopoulos et al. 2024), online CP, federated CP, calibration-set-free CP. Good extensions for v1 → v2 but not needed to make the NeurIPS story.
