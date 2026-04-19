# NeurIPS 2027 Main — Paper Outline

**Working title (v0)**: *Conformalized Heteroscedastic Variant Effect Prediction with Local Coverage Guarantees*

**Target length**: 9 pages main + unbounded appendix (NeurIPS 2027 format)
**Submission**: ~2027-05
**Lead**: Siyu Wang (katewangsiyu)

---

## 0. Decision log

| 决策 | 选择 | 依据 |
|---|---|---|
| Base aggregator | HistGradientBoostingClassifier (depth=2, iter=100, balanced) | Day 10 GBM AUPRC 0.900 Mendelian / 0.350 Complex，显著超 TraitGym LogReg 0.7515 |
| σ̂(x) head | HistGradientBoostingRegressor on chrom-LOO \|residual\| | Day 11 验证 σ̂-bin gap 改善（Mondrian 下） |
| Score | $s(x,y) = |y - \hat p(x)| / \hat\sigma(x)$ | feature-dependent CQR binary 类比 |
| Stratification | Mondrian $(y \times \hat\sigma\text{-bin})$, **adaptive K** (T5) | Day 11 Complex σ̂-bin gap 0.020; Day 20 K-sweep: K_CV=3 (M) / K_CV=2 (C) beats K=5 by 6-7x |
| Benchmark | TraitGym Mendelian + Complex matched_9 | 官方 leaderboard；split: chr 17–22,X = val |
| Primary baseline | DEGU (Zhou & Koo, npj AI 2026) | 唯一 heteroscedastic distillation VEP 方法；abstract 提 conformal |
| External OOD | Trait-LOO + cross-dataset（非 ClinVar）| ClinVar 非编码 P/LP 样本少且 skewed splice（`reports/clinvar_holdout_investigation.md`） |

---

## 1. Sections 结构（9-page 主文）

| § | 标题 | 页数 | 主要 claim | 支撑素材 |
|---|---|---:|---|---|
| 1 | Introduction | 1.0 | Variant effect prediction 缺 uncertainty；现有 conformal 缺 feature-dependent local guarantee；我们首个 heteroscedastic + conformal joint framework on TraitGym | Fig 1 (concept pane); §9 success criteria |
| 2 | Related work | 0.5 | CP foundations (Vovk/Romano/Barber); 异方差 UQ (Nix-Weigend, DEGU); VEP benchmarks (TraitGym, DEGU) | `papers/literature_v0.md` |
| 3 | Problem formulation | 1.0 | A1 chrom-group exchangeability；A2 score stationarity；**A-SL score-Lipschitz**；target: marginal + class-cond + local coverage | `theory/formulation_v0.md` §1-2 |
| 4 | Method: HCCP | 1.5 | GBM aggregator + σ̂ regressor + Mondrian(y×σ̂-bin) score; **Algorithm 1 + adaptive K (Alg 2)**; design rationale | `theory/formulation_v0.md` §2；algorithm 1-2 |
| 5 | Theory | 2.0 | T1 marginal + T2 class-cond + **T3 local bin-cond** + **T5 adaptive K bias-variance (K*=O(√n), rate O(n^{-1/2}))** + T4 chrom-shift | `theory/t1_t2_formal_proofs.md`, `theory/t3_formal_proof.md`, **`theory/t5_adaptive_K.md`** |
| 6 | Experiments | 2.5 | AUPRC + coverage on Mendelian + Complex; σ̂-bin gap three-axis (chrom-LOO / trait-LOO / cross-dataset); ablations | Day 10–14 outputs; Tab 1-4, Fig 2-5 |
| 7 | Discussion | 0.5 | Honest limitations (marginal-TV proxy, DEGU reimpl caveat); broader impact | — |
| 8 | Conclusion | 0.25 | — | — |
| App A | T1-T4 full proofs | ∞ | — | theory/ |
| App B | Data + training details | ∞ | — | configs/ |
| App C | Per-trait / per-chrom tables | ∞ | — | outputs/ |

---

## 2. Claim stack（verify against evidence）

### C1 Main claim
**HCCP achieves three-axis local coverage gap ≤ 0.04 on TraitGym Mendelian + Complex**, while matching or exceeding SOTA aggregator AUPRC.

- Evidence: Day 10 GBM SOTA 0.900 Mendelian; Day 12 chrom-LOO σ̂-bin gap 0.020 Complex / 0.198 Mendelian (remaining gap); Day 14 trait-LOO σ̂-bin gap 0.002–0.004 (floor); Day 14 cross-dataset C→M 0.035
- Key table: `reports/day14_external_validation.md` §7

### C2 Theoretical novelty
**T3 local coverage + T5 adaptive partition**: bin-conditional finite-sample rate $1/(n_{k,b}+1)$; **optimal bin count K* = O(√n) with dimension-free O(n^{-1/2}) local coverage rate** (T5.1); matching lower bound (T5.2).

- Evidence: `theory/t3_formal_proof.md`; `theory/t5_adaptive_K.md`; K-sweep Day 20 validation
- Positioning: T3 = identification of σ̂-bin as Mondrian taxon (Vovk 2003 application); **T5 = new contribution** — first derivation of optimal Mondrian partition granularity. Dimension-free rate O(n^{-1/2}) vs RLCP's O(n^{-2/(d+2)}).
- Key differentiator from Dewolf (IMA 2025): they analyze conditional validity analytically for regression; we derive optimal K for classification with a learned σ̂ and prove matching bounds.

### C3 DEGU comparison
**DEGU heteroscedastic NLL head does not automatically confer local coverage**: Day 13 DEGU-lite ablation shows matched Mondrian(y×σ̂-bin) gap only when paired with class-cond + σ̂-bin stratification.

- Evidence: Day 13 DEGU-lite report (6/6 partition Complex)
- Positioning: "DEGU provides σ̂; conformal + class-cond + Mondrian turns it into coverage."

### C4 Benchmark audit (secondary)
**TraitGym supervised LogReg pipeline is not optimal**: GBM on same features gives +14.9pp Mendelian; CADD-alone LR (0.871) > published "SOTA" CADD+Borzoi LR (0.752).

- Evidence: Day 10 full feature-set ablation
- Positioning: "A byproduct of our work is a revised aggregator baseline."

---

## 3. Figure plan

| Fig | Content | Status |
|---|---|---|
| 1 | Concept figure: CP pipeline with σ̂ modulation; prediction set geometry | todo |
| 2 | AUPRC vs coverage (selective risk curves) Mendelian + Complex, GBM vs LogReg vs DEGU-lite | data ready |
| 3 | σ̂-bin coverage three-axis panel: chrom-LOO / trait-LOO / cross-dataset; K=5 bars | data ready |
| 4 | Per-trait coverage heterogeneity (30 OMIM Mendelian + 28 Complex) | data ready |
| 5 | Cross-dataset direction asymmetry + Barber Thm 2 proxy overlay | data ready |
| Ablation | σ̂ source × K × class-cond × stratification (2×3×2×3 grid); radar plot | partial |

---

## 4. Table plan

| Tab | Content | Status |
|---|---|---|
| 1 | Main: (AUPRC, cov, cov\|pos, σ̂-bin gap, set size dist) × (Mendelian, Complex) × (GBM, GBM+HCCP, DEGU-lite+CP, LogReg+CP) | data ready |
| 2 | Three-axis OOD summary: chrom-LOO / trait-LOO / cross-dataset × (feature set × dataset) | data ready (§7 of day14 report) |
| 3 | σ̂ source ablation: GBM / NN / dropout / ensemble | partial (DEGU-lite done) |
| 4 | **T5 K-sweep: K ∈ {2,3,5,8,10,15,20,30} × gap / n_min + theoretical curve overlay** | **done (Day 20)** |
| 5 | DEGU-lite σ̂ vs HCCP σ̂ head-to-head: same features, same Mondrian, different σ̂ source | **in progress (Day 20)** |

---

## 5. 写作 cadence（to 2027-05）

| Phase | 月 | 交付 |
|---|---|---|
| Skeleton + §2-3 draft | 2026-04 ~ 2026-05 | outline + related work + formulation done in LaTeX |
| §4-5 draft (method + theory) | 2026-06 ~ 2026-08 | T1/T2 in appendix; T3 polished |
| §6 draft + figures | 2026-09 ~ 2026-11 | main table + 5 figures； DEGU full reimpl if time |
| Internal review + revision | 2026-12 ~ 2027-02 | 至少 2 轮内审 |
| bioRxiv preprint | 2027-02 | 锁版本 |
| External review + polish | 2027-03 ~ 2027-04 | 邀 1-2 外审 |
| Submission | 2027-05 | — |

并行：NeurIPS 2026 D&B 保底（`reports/day10_ablation_paper_outline.md`），~2026-06 截稿，Day 10 材料独立打包。

---

## 6. 未决

1. 论文 title 最终版需 iterate（主 claim 突出 "local coverage" 还是 "heteroscedastic"？）
2. Figure 1 concept 图风格：是否引入 toy 2D example（ala Romano 2019）?
3. Appendix 里是否 include 完整 Day 5 falsification 故事（"scientific epistemics" 角度）？— 倾向不要，保持主线聚焦
4. DEGU full reimpl 是否进主 table？— 当前 DEGU-lite（共享 backbone）已足够 headline，full reimpl 作为 appendix 安全
