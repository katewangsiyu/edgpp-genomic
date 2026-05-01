# Heteroscedastic Conformal Prediction for Variant Effect Prediction — Concurrent-Work Survey

**Topic slug**: `hccp_vep`
**Date**: 2026-04-30
**Coverage**: 2021–2026, NeurIPS / ICLR / ICML + arXiv preprints (2024–2026)
**Method**: 6-phase deep-research pipeline (paper_finder_shim + Semantic Scholar + arXiv + WebSearch); 8 papers read in full
**Audience**: NeurIPS 2027 main-track HCCP submission preparation (defensive concurrent-work survey)

---

## Executive Summary (TL;DR for paper authors)

A NeurIPS 2027 submission building on **Heteroscedastic Class-Conditional Conformal Prediction (HCCP)** must navigate **three concurrent threats** identified in this survey, all of which **have credible defenses**:

1. **Yao, He, Gastpar (ICLR 2026)** — proves dimension-free `O(n^{-1/2})` rate for split-CP linear quantile regression. **Same headline rate as HCCP T5.1** but on methodologically distinct family (split-CP vs equi-bin Mondrian); no class imbalance handling; no lower bound. **Defense**: explicit framing in §2 / §6.4 as complementary frontier, not subsumption. [Yao2026]
2. **Kandinsky CP (Bairaktari, Wu, Wu, ICML 2025)** — claims minimax-optimal high-prob conditional bound subsuming class-cond + Mondrian as special cases. **Static-group framework only**; does not cover data-dependent σ̂-bins; explicit dimension-dependence `O(√(d/n))`; no lower bound. **Defense**: HCCP's T5.2 lower bound *within equi-bin Mondrian-K family* lies outside Kandinsky's class. [Bairaktari2025]
3. **CP²-HPD (Plassier et al., ICLR 2025)** — soft density-ratio reweighting; empirically threatens discrete σ̂-bin Mondrian on continuous heteroscedastic data. **Defense**: H2H on TraitGym Mendelian + Complex (infrastructure exists). Add to §6.3 if reviewer flags. [Plassier2025]

**DEGU update — published in npj AI 2026** [Zhou2026]: bioRxiv 2024 → npj AI vol 2 art 3 (Feb 2026). Published version explicitly mentions "conformal prediction offering coverage guarantees." HCCP's "DEGU as ally / pioneer" framing remains valid; the H2H story tightens.

**Top finding**: HCCP sits in a **structurally distinct frontier** — equi-bin Mondrian-K class-conditional heteroscedastic CP. No concurrent method covers (heteroscedastic σ̂(x)) ∩ (π_min class imbalance) ∩ (data-dependent partition) ∩ (lower-bound theory) simultaneously. This is the defensive moat to lean on in §1 / §8 contributions.

---

## 1. Topic & Scope

**Research question**: For a NeurIPS 2027 submission on HCCP applied to TraitGym variant effect prediction, what is the concurrent-work landscape (2024-2026), and what are the empirical / theoretical threats and defenses?

**Method statement**: 6-phase pipeline.
- Phase 1 (Frontier): NeurIPS / ICLR / ICML 2024-2026, 130 raw hits.
- Phase 2 (Survey): Broaden to 2021-2026 + multi-source (paper_finder_shim + arXiv); 81 papers in `paper_db.jsonl`.
- Phase 3 (Deep Dive): 8 papers fully read.
- Phase 4 (Code): 14 repos surveyed (8 paper repos + 6 ecosystem).
- Phase 5 (Synthesis): 3-axis taxonomy + gap analysis.

---

## 2. Trending Directions (2024-2026)

The field has converged on **conditional coverage** as the post-Vovk-2012b open problem, attacked along three orthogonal axes:

| Axis | Mechanism | Representatives |
|---|---|---|
| **(A) Group structure** | Static groups, often overlapping | Kandinsky CP [Bairaktari2025], CP-Cond-Guarantees [Gibbs2025] |
| **(B) Density reweighting** | Soft continuous weights via Π̂(Y\|X) | CP²-HPD [Plassier2025], Probabilistic CP |
| **(C) Partition learning** | Data-dependent bins (σ̂-bins, equi-bin Mondrian) | **HCCP**, equi-bin Mondrian variants |

HCCP is firmly in axis (C). Kandinsky covers (A) but NOT (C). CP²-HPD covers (B) but NOT (C). **No concurrent method covers (A) ∩ (C) simultaneously**.

A second trend: **rate analysis**. Yao 2026 [Yao2026] gives split-CP dimension-free `O(n^{-1/2})`; Length-Opt CP [Kiyani2024] gives length-minimization `O(n^{-1/2})`. All three frontiers (Yao / Kandinsky / HCCP) converge to `O(n^{-1/2})` headline — differentiation must be at constant + class-restriction level, not rate level.

---

## 3. Concurrent Threats (priority order)

### Threat T1 — Yao et al., ICLR 2026 [Yao2026]

**Paper**: Non-Asymptotic Analysis of Efficiency in Conformalized Regression.
**Result**: For split CP with linear quantile regression trained via SGD,
$$
\mathbb{E}[|\mathcal{C}(X) - \mathcal{C}^*(X)|] = O(n^{-1/2} + (\sigma^2 n)^{-1} + m^{-1/2} + e^{-\sigma^2 m}), \text{ dim-free.}
$$
Constant `H = f_max / f_min` (flatness of conditional density), no input dim.

**Threat**: Same dim-free `O(n^{-1/2})` headline as HCCP T5.1.

**Defenses**:
- Methodologically distinct family: split-CP linear quantile vs equi-bin Mondrian classification.
- HCCP T5.1 oracle K\* = ⌊√(L_F R π_min n)⌋ explicit closed-form K-selection; Yao has no K to select.
- HCCP T5.2 matching lower bound; Yao gives upper bound only.
- HCCP π_min class-imbalance handling; Yao regression only.

**§ framing language**: "Yao et al. (2026) prove dimension-free O(n^{-1/2}) for split CP with linear quantile regression; we prove the same rate for equi-bin Mondrian classification with explicit π_min dependence and a matching lower bound — methodologically distinct and non-subsumed."

### Threat T2 — Kandinsky CP, ICML 2025 [Bairaktari2025]

**Paper**: Beyond Class- and Covariate-Conditional Coverage.
**Result**: Theorem 3.1: `E[wCD] ≤ ‖β‖_1 (C√(d/n) + d/n + max{α,1-α}√(2 ln(4d/δ)/n))` where `d` = VC-dim of weight basis Φ.

**Threat**: Subsumes class-cond + Mondrian as **static-group special cases** with claimed minimax-optimal conditional bound.

**Defenses**:
- Static, pre-specified groups only — Cor 3.4 explicitly assumes "calibration and test data do not include unobserved attributes." HCCP's σ̂-bins computed from chrom-LOO residuals violate this assumption.
- No matching lower bound for their class.
- Dimension-dependent `O(√(d/n))` vs HCCP T5.1 dim-free `O(n^{-1/2})`.
- Empirical: ACSIncome + CivilComments. Zero genomics / VEP.

**§ framing language**: "Kandinsky CP gives minimax-optimal coverage for static, pre-specified overlapping groups. We address a structurally distinct setting — data-dependent equi-bin Mondrian-K partitions with learned σ̂(x)-stratification — for which Kandinsky's framework does not apply."

### Threat T3 — CP²-HPD, ICLR 2025 [Plassier2025]

**Paper**: Probabilistic Conformal Prediction with Approximate Conditional Validity.
**Result**: Soft density-ratio reweighting: τ_{x,z} = inf{τ : Π̂(R_z(x;f_τ(φ))) ≥ 1-α}. Theorem 3.2: P(Y ∈ C | x,z) ≥ 1 - α - d_TV(P; Π̂) - p(x,z). Rate `O_s(√(n^{-1} log n + r_n))` dim-free.

**Threat**: **Empirical, not theoretical.** On continuous heteroscedastic data, soft density-ratio reweighting could outperform discrete σ̂-bin Mondrian — boundary artifacts at σ̂-bin edges hurt HCCP.

**Defenses**:
- HCCP gives **exact finite-sample** class-conditional coverage; CP²-HPD relies on Π̂ estimator quality (the d_TV term).
- Run H2H on TraitGym Mendelian + Complex — infrastructure exists in `R_raw/cp_baselines_h2h/`.

**§ framing language**: "CP²-HPD trades exact bin-conditional coverage for smooth heteroscedastic adaptation via density-ratio reweighting. Our discrete equi-bin Mondrian achieves exact finite-sample bin-cond coverage, with the trade-off that boundary artifacts may emerge under highly continuous σ̂(x). We empirically validate (§ 6.3) on TraitGym."

---

## 4. DEGU Update [Zhou2026]

**Critical finding**: DEGU (Zhou, Rizzo, Christensen, Tang, Koo) is **now published in npj AI 2026 vol 2 art 3 (Feb 2026)**, not just bioRxiv 2024.

**Published version explicitly mentions conformal prediction**: from EurekAlert!/npj AI summary: "DEGU-trained models provide calibrated uncertainty estimates, with conformal prediction offering coverage guarantees under minimal assumptions."

**Implication for HCCP positioning**:
- HCCP's "DEGU as ally / pioneer" framing (Phase 6 polish) **remains valid** — DEGU's mention of conformal prediction is at the conceptual level, not an integrated method.
- DEGU's published version still does **not** address heteroscedastic NLL / class-conditional / Mondrian — HCCP's surgical contributions stand.
- §6.4 H2H reframing: "DEGU pioneered ensemble distillation for genomic uncertainty and noted conformal prediction as compatible. We provide the integrated method: heteroscedastic class-conditional Mondrian conformal prediction with theoretical analysis (T5.1, T5.2) and TraitGym empirics."

---

## 5. Code-Ecosystem Findings

| Theme | Recommendation |
|---|---|
| **Yao 2026 + Kandinsky 2025 have no public code** | Reproducibility differential — HCCP can lean on shipping standalone `crepes` extension + Snakemake module |
| **DEGU has `paper_reproducibility/` folder** [Zhou2026 repo] | Direct addressable in §6.4; can H2H against actual TF/Keras DEGU-actual not just DEGU-lite |
| **TraitGym + GPN + Borzoi all healthy** | Solid reproducibility story; ship HCCP as Snakemake module on top of TraitGym |
| **3 leading CP libraries** (crepes 568★, MAPIE 1.5k★, +puncc) | All have Mondrian primitives — upstreaming path for D&B Track P1 artifact exists |

---

## 6. Recommended HCCP Positioning at NeurIPS 2027

**§1 contribution restatement**:
> "We identify the *equi-bin Mondrian-K class-conditional heteroscedastic CP family* as a structurally distinct frontier — outside both static-group methods (e.g., Kandinsky) and split-CP rate analyses (e.g., Yao et al.). Within this family we (1) prove an oracle K\* = ⌊√(L_F R π_min n)⌋ achieving dimension-free O(n^{-1/2}) suboptimality (T5.1), (2) prove a matching lower bound (T5.2), (3) derive a tractable operational binding certificate (T3'), and (4) validate on TraitGym Mendelian + complex_traits matched_9, beating the 0.900 GBM aggregator's σ̂-bin coverage gap by 33× (Complex) / 2.3× (Mendelian) over class-Mondrian baselines."

**§2 / §6.4 must explicitly reference**:
- [Bairaktari2025] — concurrent group-cond CP
- [Yao2026] — concurrent split-CP rate
- [Plassier2025] — concurrent density-ratio CP
- [Zhou2026] — DEGU updated to npj AI 2026

**§7 future work**:
- Backward HCCP — class-cond + σ̂-aware backward CP for clinical triage [Gauthier2025]
- Soft × Hard hybrid — density-ratio reweighting within Mondrian bins [Plassier2025 inspiration]

**§8 contributions order** (theory-first):
1. T5.1 + T5.2 dimension-free rate + matching lower bound
2. T3' operational binding certificate
3. VEP empirics + DEGU H2H + ProteinGym cross-domain ablation
4. Open-source `crepes`-extension artifact

---

## 7. Phase outputs inventory

| Phase | Output | Papers / records |
|---|---|---|
| 1 | `phase1_frontier/frontier.md` + 3 search jsonl | 130 raw hits, 15 flagged |
| 2 | `phase2_survey/survey.md` + `paper_db.jsonl` | 81 papers in 8 clusters |
| 3 | `phase3_deep_dive/{selection.md, deep_dive.md, papers/}` | 8 papers fully read; 8 PDFs in `papers/` |
| 4 | `phase4_code/code_repos.md` | 14 repos (8 paper + 6 ecosystem) |
| 5 | `phase5_synthesis/{synthesis.md, gaps.md}` | 3-axis taxonomy, 5 gaps |
| 6 | `phase6_report/{report.md, references.bib}` | This file + bibtex |

---

## 8. References

See `references.bib` for full BibTeX. Key citations used in this report:

- [Bairaktari2025] Bairaktari, Wu, Wu, "Kandinsky Conformal Prediction: Beyond Class- and Covariate-Conditional Coverage", ICML 2025.
- [Yao2026] Yao, He, Gastpar, "Non-Asymptotic Analysis of Efficiency in Conformalized Regression", ICLR 2026 (arXiv 2510.07093).
- [Plassier2025] Plassier et al., "Probabilistic Conformal Prediction with Approximate Conditional Validity", ICLR 2025.
- [Zhou2026] Zhou, Rizzo, Christensen, Tang, Koo, "Uncertainty-aware genomic deep learning with knowledge distillation (DEGU)", npj AI 2026 vol 2 art 3.
- [Gauthier2025] Gauthier, Bach, Jordan, "Backward Conformal Prediction", NeurIPS 2025 (arXiv 2505.13732).
- [Kiyani2024] Kiyani, Pappas, Hassani, "Length Optimization in Conformal Prediction", NeurIPS 2024.
- [RC3P2024] Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration, NeurIPS 2024.
- [Gibbs2025] Gibbs, Cherian, Candès, "Conformal Prediction with Conditional Guarantees", JMLR 2025 (arXiv 2305.12616).
- [Amin2025] Amin et al., "Training Flexible Models of Genetic Variant Effects from Functional Annotations using Accelerated Linear Algebra (DeepWAS)", ICML 2025.

✅ **Phase 6 report complete.**
