# Phase 3: Deep Dive — Structured Notes from 8 Papers

8 papers read in full (PDF). Per-paper structured notes below; **threat assessment integrated**.

---

## 1. Kandinsky Conformal Prediction: Beyond Class- and Covariate-Conditional Coverage
**Bairaktari, Wu, Wu — ICML 2025** (OpenReview `IHAnkPkoiX`)

**Problem**. Group-conditional CP coverage for **overlapping, fractional groups** G ⊆ X × Y. Strict generalization of class-cond + covariate-cond.

**Setup**. Groups static, pre-specified. Membership φ_G(x,y) = P[Z ∈ G | φ(X,Y)=φ(x,y)]. Calibration/test data assumed not to include unobserved attributes (Cor 3.4). **Critical**: σ̂-bins computed from training residuals violate this.

**Main theorem (3.1)**:
$$\mathbb{E}[\mathrm{wCD}(C, \alpha, w_\beta)] \leq \|\beta\|_1 \left(C\sqrt{\tfrac{d}{n}} + \tfrac{d}{n} + \max\{\alpha, 1-\alpha\}\sqrt{\tfrac{2\ln(4d/\delta)}{n}}\right)$$
**Not dimension-free** — `d` = VC-dim of weight basis Φ enters explicitly.

**Mondrian as special case (Cor 3.3)**: Static partition only. No data-dependent binning.

**Heteroscedasticity**: ❌ NONE.

**Empirical**: ACSIncome (income by US state, 31 groups) + CivilComments (toxicity, 16 demographic groups). **Zero genomics / VEP.**

**Connection to HCCP**:
- ❌ Does NOT prove a lower bound — T5.2 "tight rate within equi-bin Mondrian-K family" not threatened.
- ❌ Static-group framework does NOT cover data-dependent σ̂-bins.
- ✅ Their O(√(d/n)) is dimension-dependent; our T5.1 dimension-free O(n^{-1/2}) is sharper for fixed equi-bin Mondrian-K class.
- **2 defensive moats**: (a) data-dependent partition with chrom-LOO σ̂-bins lies outside Kandinsky framework; (b) dimension-free rate exploits π_min + equi-bin structure in ways their VC-based proof cannot.

**Verdict**: Concurrent threat for general group-cond CP, but **HCCP defenses hold**.

---

## 2. Non-Asymptotic Analysis of Efficiency in Conformalized Regression
**Yao, He, Gastpar — ICLR 2026** (arXiv 2510.07093)

**Problem**. Non-asymptotic deviation of CP set length from oracle interval length, for CQR & CMR trained via SGD on linear quantile/median models.

**Main theorem (3.2)** (CQR-SGD; identical form as CMR Thm 4.1):
$$\mathbb{E}\left[|\mathcal{C}(X) - \mathcal{C}^*(X)|\right] \leq O\left(n^{-1/2} + (\sigma^2 n)^{-1} + m^{-1/2} + \exp(-\sigma^2 m)\right)$$
Rate is **dimension-free** (constant H = f_max/f_min, no input dim).

**Setup**. Split CP only, linear quantile regression, SGD with η_k = 1/(λ_min f_min k). Bounded conditional density (Asn 3.3).

**Mondrian / class-cond**: ❌ NOT studied. Pure split CP.

**Heteroscedastic**: Implicit only (via density bounds f_min ≤ f_{Y|X} ≤ f_max). No explicit σ̂(x) modeling.

**Class imbalance π_min**: ❌ NOT mentioned. Regression only.

**Lower bounds**: ❌ NONE.

**Connection to HCCP**:
- ❌ Concurrent dimension-free O(n^{-1/2}) — same headline rate as T5.1 — but on a **completely different method** (split CP linear quantile regression vs equi-bin Mondrian classification).
- ✅ T5.1 oracle K* = ⌊√(L_F R π_min n)⌋ is non-trivial K-selection — they have no K to select.
- ✅ T5.2 lower bound has no analog in their work.
- **2 defensive moats**: (a) Mondrian partitioning + K-selection vs split CP; (b) explicit class imbalance π_min handling vs regression-only.

**Verdict**: **Headline-rate-collision** (both achieve O(n^{-1/2}) dimension-free). Need careful framing in §2 / §6.4 to position HCCP as Mondrian-specific complement to their split-CP-specific result. Honest related-work language: "Yao et al. (2026) prove dimension-free O(n^{-1/2}) for split CP with linear quantile regression; we prove the same rate for equi-bin Mondrian classification with explicit π_min dependence — methodologically distinct."

---

## 3. Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration (RC3P)
**NeurIPS 2024** (OpenReview `T7dS1Ghwwu`)

**Problem**. Multi-class classification with class-conditional coverage P(Y ∈ C(X) | Y=c) ≥ 1-α for each c.

**Method**. Dual calibration: (i) APS/RAPS/HPS conformity score V(X,Y); (ii) calibrated label rank k̃(c) = min{k : ε_c^k < α} per class. Prediction set: C̃(X_test) = {y : V(X_test,y) ≤ Q̂_{1-α}^class(y) AND r_f(X_test,y) ≤ k̃(y)}.

**Theorem 4.1**: Class-conditional coverage holds finite-sample, distribution-free.
**Theorem 4.3**: E[|C̃^RC3P|] ≤ E[|C̃^CCP|] when σ_y ≤ 1.

**Heteroscedastic**: ❌ NO. Score is class-stratified but **σ̂(x)-agnostic**.

**Datasets**: CIFAR-10/100, mini-ImageNet, Food-101. **No binary classification**, no imbalanced binary at π=0.10. Decay types: EXP/POLY/MAJ.

**Connection to HCCP**:
- ✅ This is the canonical B3 (class-Mondrian) baseline for HCCP §6.3.
- ❌ Cannot capture HCCP's σ̂-bin × class joint Mondrian — RC3P has K_class dimensions, HCCP has K_class × K_σ dimensions.
- **Positioning**: "RC3P achieves class-conditional coverage via label-rank augmentation, but remains σ̂-oblivious. HCCP strictly refines by jointly conditioning on class labels AND learned heteroscedasticity σ̂(x)."

---

## 4. Optimal Transport-based Conformal Prediction
**ICML 2025** (OpenReview `kEAyffH3tn`)

**Problem**. Multivariate CP via Monge-Kantorovich (MK) rank maps. Multi-output regression / multiclass classification with **marginal** coverage.

**Method**. R_n(s) = argmax_{U_i} {⟨U_i, s⟩ - φ_n(U_i)} where {U_i} are reference uniform unit-sphere vectors. Maps multivariate residuals to scalar ranks via OT theory (no full transport plan).

**Theorem 2.4 (coverage)**: α ≤ P(Y_test ∈ Ĉ_α) ≤ α + n_ties/(n_2+1). Excess from ties is O(n^{-1}).

**Theorem 3.2 (asymptotic conditional)**: Adaptive OT-CP+ with k-NN converges to nominal conditional coverage.

**Mondrian / class-cond**: Mentioned only in passing (Kuchibhotla 2020 reference). **No empirical H2H against Mondrian.**

**Heteroscedastic**: ✅ Adaptive variant (OT-CP+) handles via k-NN local quantiles.

**Computational**: O(n^3) baseline; O(n^2) with entropy regularization (Klein 2025). vs Mondrian O(n log n) — **OT is slower**.

**Connection to HCCP**:
- ❌ Different problem (multivariate marginal vs class-cond classification). No rate competition.
- Method offers no rate analysis comparable to T5.1.
- **Verdict: Orthogonal** — useful for related-work cross-reference, not direct competitor.

---

## 5. Backward Conformal Prediction
**Gauthier, Bach, Jordan — NeurIPS 2025** (arXiv 2505.13732)

**Problem**. Forward CP fixes α, lets size vary; backward CP fixes a size-rule T, lets α vary adaptively.

**Method**. E-variable E^test_y = S(X_test, y) / [(n+1)(Σ S(X_i,Y_i) + S(X_test,Y_test))]. Choose α̃ = inf{α : #{y : E^test_y < 1/α} ≤ T}. LOO estimator α̃^LOO = (1/n) Σ_j α̃_j gives concentration of order O_P(1/√n) (Thm 3.1).

**Coverage guarantee**: P(Y_test ∈ C^α̃_n(X_test)) ≥ 1 - α̃^LOO - R_δ(n) where R_δ(n) = O(1/√n).

**Heteroscedastic-aware**: Implicit via local entropy variant (App B.2.1) — H(X_i) = -Σ p̂(c) log p̂(c) → larger T for higher-uncertainty inputs. Not explicit σ̂(x).

**Class-cond / Mondrian**: ❌ NOT addressed. Marginal only. Authors note extension is feasible but not done.

**Connection to HCCP**:
- 🟰 **Complementary**, not competitive. Forward (HCCP) excels for fixed-α regulatory requirements; backward excels for size-bounded clinical triage.
- For VEP triage: backward CP gives "predicted set ≤ {Pathogenic, Likely Pathogenic, VUS}" — very actionable.
- **Future work hook**: extend HCCP to backward variant — class-cond size-rule T_c per class.

---

## 6. Length Optimization in Conformal Prediction
**NeurIPS 2024** (OpenReview `E4ILjwzdEA`)

**Problem**. Minimize average length E_X[len(C(X))] subject to coverage constraint.

**Method**. Minimax: min_{h ∈ H} max_{f ∈ F} g_α(f,h). Optimize conformity score S(·,·) jointly with hypothesis class H. F = constant for marginal; F broader for cond/group cov.

**Main theorem (4.4)**:
$$\mathbb{E}\left[f_\alpha(X)\{\mathbb{1}[Y \in C_{\text{CPL}}(X)] - (1-\alpha)\}\right] \leq \frac{c_1\sqrt{\ln(2dN(\mathcal{H},d_+,z)/n)} + c_2}{\sqrt{n}}$$
PAC-style **O(1/√n)** matching T5.1.

**Mondrian**: Not explicitly. Their structured prediction sets with optimized threshold can subsume "find best K" but they don't draw out this connection.

**Heteroscedastic**: Implicit — score S can encode σ̂(x)-normalization but no separate theory.

**Connection to HCCP**:
- 🟰 **Same rate, different framework**. They solve full minimax over score+hypothesis; HCCP fixes the score (|y - p̂|/σ̂(x)) and optimizes K within equi-bin Mondrian.
- Their framework is more general (covers covariate-shift class F), but offers no tighter rate.
- ✅ Differentiation: T5.1 oracle K* gives **explicit closed-form** K* = ⌊√(L_F R π_min n)⌋; their minimax has no closed-form K analog.
- **Positioning**: "Length-optimization CP gives O(1/√n) under generic minimax; HCCP gives explicit oracle K* for equi-bin Mondrian — orthogonal contributions on the same rate frontier."

---

## 7. Probabilistic Conformal Prediction with Approximate Conditional Validity (CP²-HPD)
**ICLR 2025** (OpenReview `Nfd7z9d6Bb`)

**Problem**. Approximate conditional validity formalized via TV distance to true conditional distribution.

**Method**. Estimate Π̂_{Y|X=x} (Mixture Density Network in experiments). Conformal score reweighted: τ_{x,z} = inf{τ : Π̂(R_z(x;f_τ(φ))) ≥ 1-α}. Scores combined via empirical + tail measure: μ_n = (1/(n+1))[Σδ_{f_τ_k^{-1}(V_k)} + δ_∞].

**Theorem 3.2**: P(Y_{n+1} ∈ C_α(X_{n+1}) | x_{n+1}, z_{n+1}) ≥ 1 - α - d_TV(P_{Y|X=x}; Π̂_{Y|X=x}) - p_{n+1}(x,z).

**Theorem 3.3 (asymptotic)**: O_s(√(n^{-1} log n + r_n)) — dimension-free.

**Heteroscedastic-aware**: ✅ Yes — Π̂_{Y|X=x} captures full σ(x) variation. No discretization.

**vs Hard Mondrian**: Implicit critique — fixed-radius CP fails on multimodal/heteroscedastic data; soft density-ratio reweighting avoids partition boundaries.

**Connection to HCCP**:
- ⚠️ **Strongest empirical threat**. Soft density-ratio reweighting could outperform discrete σ̂-bin Mondrian when:
  - σ̂(x) is highly continuous / multimodal
  - Class imbalance creates sharp coverage gradients near partition boundaries
- ✅ HCCP defense: discrete bins give **exact finite-sample** class-cond coverage; CP²-HPD relies on Π̂ estimator quality (the d_TV term).
- **Empirical experiment for §6**: H2H HCCP vs CP²-HPD on TraitGym — would be a cleaner test of partition vs density-ratio.

---

## 8. Training Flexible Models of Genetic Variant Effects from Functional Annotations (DeepWAS)
**Amin et al. — ICML 2025** (OpenReview `oOtdWiLb1e`)

**Problem**. Population-scale variant effect prediction (β coefficients) from functional annotations across all variant types.

**Method**. Transformer-CNN hybrid (Enformer-based, 1536 hidden dims) on 165-bp windows. Banded approximation of LD matrix R; iterative linear algebra (stochastic Lanczos quadrature, conjugate gradient) via CoLA library.

**Datasets**: UK Biobank (300K+ individuals), private phenotypes. **NOT TraitGym, NOT GWAS catalog, NOT gnomAD, NOT ClinVar**.

**Uncertainty**: ❌ ZERO. Point estimates β̂_m only.

**Baselines**: LDSR + GLM only. No Borzoi / GPN-MSA / AlphaMissense.

**Metrics**: % marginal likelihood increase per person. **NOT AUPRC_by_chrom_weighted_average**.

**Connection to HCCP**:
- ❌ **Fundamental orthogonality**. DeepWAS = annotation-driven population-scale effect-size estimator; HCCP = sequence-driven per-variant uncertainty-calibrated classifier.
- DeepWAS cannot serve as base predictor for HCCP σ̂-aggregation (no uncertainty estimates, different feature space).
- **§6.5 / Related-work language**: "DeepWAS (Amin et al., ICML 2025) is the only ICML 2025 VEP paper but defines its own benchmark (UK Biobank semi-synthetic, % likelihood increase) — orthogonal to TraitGym + AUPRC framework. We do not benchmark against it."

---

## Cross-cutting findings

### Concurrent threats (priority order)
1. **Yao et al. (ICLR 2026) — same dimension-free O(n^{-1/2}) rate** for split CP linear quantile reg. Need careful §2/§6.4 framing.
2. **Kandinsky CP (ICML 2025) — subsumes class-cond + Mondrian for static groups**, but does NOT cover data-dependent σ̂-bins or prove lower bounds.
3. **CP²-HPD (ICLR 2025) — empirical threat** via continuous density-ratio reweighting. Run H2H on TraitGym.

### Defensive moats confirmed
- ✅ **Heteroscedastic σ̂(x) data-dependent partition** — outside Kandinsky framework, outside Yao framework
- ✅ **Class imbalance π_min explicit handling** — outside Yao, outside Length-Opt
- ✅ **T5.2 matching lower bound** — no concurrent paper has this in our class
- ✅ **VEP / TraitGym benchmark + AUPRC_by_chrom_weighted** — DeepWAS doesn't compete

### Phase 4 GitHub leads
From these papers, GitHub repos to verify in Phase 4:
- Kandinsky CP — likely on Steven Wu's GitHub or NeurIPS supp
- Yao et al. — EPFL LINX lab repo
- RC3P — NeurIPS 2024 supp
- CP²-HPD — author's GitHub
- Backward CP — Etienne Gauthier's repo
- DeepWAS — Amin lab GitHub
- DEGU — `https://github.com/zrcjessica/ensemble_distillation` (already known)

✅ **Phase 3 complete.** 8 papers fully read, structured notes written, threats classified, defenses identified.

→ Proceed to Phase 4 (Code & Tools survey).
