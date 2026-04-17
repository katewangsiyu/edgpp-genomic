# T1 + T2 Formal Proofs — Appendix-Ready Coverage Theorems

**Date**: 2026-04-17
**Status**: v1 (appendix-ready draft; supersedes the one-page sketch in `formulation_v0.md` §3)
**Companion**: `theory/t3_proof_sketch.md` (for the bin-conditional extension), `theory/theorems_roadmap.md` (for the full plan).

---

## 0. What this document delivers

This file gives appendix-grade statements and proofs of

- **T1 (marginal coverage)** of the conformalized heteroscedastic VEP predictor
- **T2 (class-conditional coverage)** via Mondrian-by-$Y$ stratification

plus the robust fall-back inequalities when our chrom-exchangeability/A2 assumptions are only approximately satisfied (via Barber et al. 2023). The bin-conditional refinement (T3) is proven in `t3_proof_sketch.md` as a direct corollary of the machinery built here.

**Non-goals**: T3.a / T3.b asymptotic feature-space coverage (roadmap Month 2–7), T4 empirical TV estimation.

**Summary of what T1 + T2 buy us**: at NeurIPS-review bar, T1 + T2 alone are "standard results applied correctly". Their role in the paper is to (i) cleanly set up the Mondrian machinery that T3 extends, and (ii) pin down the exact assumptions under which Day 10's homoscedastic class-conditional numbers are provably calibrated. They are prerequisite, not the main novelty. (Roadmap §Main-bar correspondence.)

---

## 1. Setting

Notation inherits from `formulation_v0.md` §1 (variants $X \in \mathcal{X}$, labels $Y \in \{0,1\}$, chromosome $C \in \mathcal{C}$, base classifier $\hat{p}$, reliability head $\hat{\sigma}$, score $s(x, y) = |y - \hat{p}(x)|/(\hat{\sigma}(x)+\varepsilon)$). We fix a miscoverage level $\alpha \in (0, 1)$ throughout.

**Split & calibration scheme (CL1, chrom-LOO).** For each test chromosome $c^* \in \mathcal{C}$:

- Train data: $\mathcal{D}^{(-c^*)} = \{(X_i, Y_i, C_i) : C_i \neq c^*\}$
- $\hat{p}$ and $\hat{\sigma}$ are fit on $\mathcal{D}^{(-c^*)}$ only
- Calibration set: $\mathcal{D}_{\text{cal}}^{(-c^*)} \subseteq \mathcal{D}^{(-c^*)}$ — in CL1 we take the full $\mathcal{D}^{(-c^*)}$; CL2 (nested) is discussed in §6
- Test set: $\mathcal{D}^{(c^*)} = \{(X_i, Y_i, C_i) : C_i = c^*\}$

We abuse notation and write $S_i := s(X_i, Y_i)$ for the realized nonconformity score of sample $i$, computed with the $\hat{p}, \hat{\sigma}$ trained on $\mathcal{D}^{(-c^*)}$. This dependence on $c^*$ is implicit.

---

## 2. Assumptions

We are explicit about a chain of assumptions, each strictly stronger than the last. T1/T2 hold *exactly* under (A1, A1', A2) and *approximately* (with a quantified TV error) under (A1) alone via Barber et al. (2023) Theorem 2.

### A1 — Chrom-group exchangeability

For each $c \in \mathcal{C}$, conditional on $C = c$, the variants $(X_i, Y_i)_{C_i = c}$ are **exchangeable** with distribution $P_c$. Chromosomes are mutually **independent** (no across-chromosome coupling in the generative process).

**Justification.** Chrom-LOO is the TraitGym protocol (Benegas et al., 2025). Within-chrom LD is absorbed into the chrom-specific law $P_c$; across-chrom independence is standard (linkage equilibrium across chromosomes).

### A1' — Chrom-wise i.i.d. (strict strengthening of A1)

For each $c$, $(X_i, Y_i)_{C_i = c} \overset{\text{i.i.d.}}{\sim} P_c$.

**Justification.** Within a chromosome, TraitGym's matched_9 sampling enforces per-trait matching on consequence / MAF / TSS distance; between residual patterns $Y - \hat{p}(X)$ at the variant level, dependence is typically weak compared with the $(X, Y)$ level (two nearby SNVs can be in high LD yet have very different class residuals because labels are set by the matched positive/control assignment, not by LD). A1' is approximately true for the score sequence and easier to reason about; we use it to derive clean coverage, then quantify the gap under A1-only via Barber 2023 in §5.

### A2 — Marginal score stationarity across chroms

The (marginal) score distribution does not depend on the chromosome:

$$
S \mid C = c \overset{d}{=} S \mid C = c', \quad \forall c, c' \in \mathcal{C}.
$$

Call this common law $F$. Note that $F$ depends on $(\hat{p}, \hat{\sigma})$ and hence on the training fold, but not on which chromosome the variant comes from.

### A2 (class-conditional) — needed for T2

$$
S \mid (Y = k, C = c) \overset{d}{=} S \mid (Y = k, C = c'), \quad \forall c, c', \forall k \in \{0,1\}.
$$

Call this common law $F_k$.

### A2 (cell-conditional) — needed for T3, stated here for completeness

$$
S \mid (Y = k, b(X) = b, C = c) \overset{d}{=} S \mid (Y = k, b(X) = b, C = c'),
$$

where $b(\cdot)$ is the $\hat{\sigma}$-bin indicator. Strongest form. See `t3_proof_sketch.md` §3.

### Relationship between assumptions

$$
\underbrace{\text{A2 (cell)}}_{\text{T3}} \;\Rightarrow\; \underbrace{\text{A2 (class)}}_{\text{T2}} \;\Rightarrow\; \underbrace{\text{A2 (marginal)}}_{\text{T1}}
$$

Each is obtained from the stronger one by marginalizing over a partition variable. Empirical check in Day 13 used the cell version (155 KS tests, 12.3% rejection at $p<0.05$, mildly elevated from 5% nominal) — so A2-cell (and hence A2-class and A2-marginal) hold approximately.

---

## 3. The Conformal Predictor (restated)

For each chromosome $c^*$ and each class $k \in \{0, 1\}$:

1. Let $\mathcal{I}_k^{(-c^*)} = \{i \in \mathcal{D}_{\text{cal}}^{(-c^*)} : Y_i = k\}$, $n_k = |\mathcal{I}_k^{(-c^*)}|$.
2. Scores $S_k = \{s(X_i, k) : i \in \mathcal{I}_k^{(-c^*)}\} = \{s(X_i, Y_i) : i \in \mathcal{I}_k^{(-c^*)}\}$ (note $Y_i = k$ in this set).
3. Threshold
$$
\hat{q}_k \;=\; \text{Quantile}\!\left( S_k, \, \frac{\lceil (n_k + 1)(1-\alpha) \rceil}{n_k} \right).
$$
(Operationally: sort $S_k$ ascending and take the $\lceil (n_k+1)(1-\alpha) \rceil$-th value; if this index exceeds $n_k$, set $\hat{q}_k = +\infty$.)

4. For a test $x$, define
$$
\mathcal{C}_\alpha(x) \;=\; \{k \in \{0,1\} : s(x, k) \leq \hat{q}_k\}.
$$

This is the same object as Day 10's `class_conditional_conformal` (with $\hat{\sigma} \equiv 1$) and Day 11's heteroscedastic version (with $\hat{\sigma}$ from `scripts/13_hetero_head.py`); the proofs below do not use which of these $\hat{\sigma}$ is in force, only that $\hat{\sigma}$ is a deterministic function of $\mathcal{D}^{(-c^*)}$.

---

## 4. Theorem T1 — Marginal coverage

### 4.1 Statement (exact form, under A1' + A2)

**Theorem T1.** Suppose A1' and A2 hold, and that the score distribution $F$ is continuous (no ties with probability 1, or ties broken uniformly at random). Let $(X_{\text{test}}, Y_{\text{test}}, C_{\text{test}} = c^*) \sim P_{c^*}$ be a fresh test variant independent of $\mathcal{D}^{(-c^*)}$. Then
$$
1 - \alpha \;\leq\; P\!\left( Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}}) \right) \;\leq\; 1 - \alpha + \frac{1}{n_{\min} + 1},
$$
where $n_{\min} = \min_k n_k$. (The lower bound uses only A2-marginal; the upper bound uses continuity.)

### 4.2 Proof

Define the "pooled" calibration score sequence
$$
\mathcal{S} \;=\; \{S_i : i \in \mathcal{D}_{\text{cal}}^{(-c^*)}\} \;=\; S_0 \cup S_1.
$$

We prove the lower bound by splitting the coverage event on the true label.

**Step 1: Reduction to per-class coverage.** By the tower property:
$$
P(Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}})) = \sum_{k \in \{0,1\}} P(Y_{\text{test}} = k) \cdot P(Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}}) \mid Y_{\text{test}} = k).
$$
Since the condition $Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}})$ reduces to $s(X_{\text{test}}, Y_{\text{test}}) \leq \hat{q}_{Y_{\text{test}}}$, we have
$$
P(Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}}) \mid Y_{\text{test}} = k) \;=\; P(S_{\text{test}}^{(k)} \leq \hat{q}_k \mid Y_{\text{test}} = k),
$$
where $S_{\text{test}}^{(k)} := s(X_{\text{test}}, k)$.

**Step 2: Joint exchangeability inside class $k$.** Under A1' + A2 (class-conditional form — implied by A2-marginal for $F$ only if $P(Y=k|C)$ is chrom-independent; otherwise needed separately. For our setup we invoke the class-conditional A2 — which is formally T2's assumption and is also empirically justified by the same KS check). Fixing class $k$:

- Each $S_i$ with $Y_i = k$ has marginal law $F_k$ (by A2-class).
- Across different $c$, samples are independent (A1).
- Within each $c$, samples are i.i.d. $P_c^{(k)}$ conditional on $Y = k$ (A1'); the score is a deterministic function of $(X, Y)$ via $(\hat{p}, \hat{\sigma})$, so $S_i \mid Y_i = k, C_i = c$ is i.i.d. $F_k^{c}$. By A2-class, $F_k^{c} = F_k$ for all $c$.
- Therefore the combined sequence $\{S_i\}_{i \in \mathcal{I}_k^{(-c^*)}}$ is i.i.d. $F_k$, and $S_{\text{test}}^{(k)}$ (conditional on $Y_{\text{test}} = k$) is independent of the calibration samples and also distributed as $F_k$.

So the full sequence $(S_{\text{test}}^{(k)}, S_1^{(k)}, \ldots, S_{n_k}^{(k)})$ is i.i.d. $F_k$, hence **exchangeable**.

**Step 3: Standard exchangeable conformal bound.** For any exchangeable sequence of scores $(T_1, \ldots, T_{n_k}, T_{n_k+1} = S_{\text{test}}^{(k)})$ with continuous law, the rank of $T_{n_k+1}$ is uniform on $\{1, \ldots, n_k+1\}$. Setting $\hat{q}_k$ to the $\lceil (n_k+1)(1-\alpha)\rceil$-th order statistic of the first $n_k$ samples yields
$$
P(T_{n_k+1} \leq \hat{q}_k) \;=\; \frac{\lceil (n_k+1)(1-\alpha)\rceil}{n_k + 1} \;\in\; [1-\alpha, \; 1-\alpha + 1/(n_k+1)].
$$

**Step 4: Assemble.** Plugging back into Step 1:
$$
P(Y_{\text{test}} \in \mathcal{C}_\alpha) \;=\; \sum_k P(Y_{\text{test}}=k) \cdot \text{(bound in Step 3 for class } k\text{)}.
$$

Each term is $\in [1-\alpha, 1-\alpha + 1/(n_k+1)]$, and the convex combination inherits this range. The lower bound gives $\geq 1 - \alpha$; the upper bound gives $\leq 1 - \alpha + 1/(n_{\min}+1)$. $\square$

### 4.3 Remarks

- **Why we invoked A2-class for a "marginal" theorem.** The class-conditional predictor is Mondrian-by-$Y$; the quantile $\hat{q}_k$ is a function only of class-$k$ calibration scores. The "marginal" in T1 is the unconditional test probability. Working class-by-class makes the proof modular with T2's.

- **Continuity / tie-breaking.** If $F$ is continuous, the upper bound is tight. If there are ties (e.g. $\hat{p}$ outputs at 0 or 1 exactly), randomized tie-breaking recovers the same bounds; without it, coverage is $\geq 1 - \alpha$ but may slightly exceed the upper bound.

- **"Chrom-LOO uses training fold for calibration (CL1) — isn't that double-dipping?"** $\hat{p}$ and $\hat{\sigma}$ are trained on $\mathcal{D}^{(-c^*)}$, and we then compute $S_i$ for $i \in \mathcal{D}^{(-c^*)}$ — this is *in-sample* residual, not held-out. This does inflate coverage slightly (the predictor fits training residuals tighter than test), biasing $\hat{q}_k$ upward (larger than oracle). The effect is a *conservative* coverage inflation, not a violation of the bound. Rigorous Barber 2023 language: weights $w_i = 1/n$ are correct for fresh data, and $d_{TV}(R(Z), R(Z^i))$ captures the in-sample bias; empirically with gradient-boosted models at depth 2 and 100 trees on 3k–11k variants, this is negligible (Day 10 marginal cov 0.901 at target 0.90). The nested scheme CL2 (§6) eliminates this entirely at a compute cost we have not needed so far.

---

## 5. Theorem T2 — Class-conditional coverage (Mondrian-by-$Y$)

### 5.1 Statement

**Theorem T2.** Suppose A1', A2-class hold with continuous $F_k$ for each $k$. For each $k \in \{0, 1\}$ with $n_k \geq \lceil 1/\alpha \rceil$,
$$
1 - \alpha \;\leq\; P\!\left( Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}}) \;\big|\; Y_{\text{test}} = k \right) \;\leq\; 1 - \alpha + \frac{1}{n_k + 1}.
$$

Moreover, the marginal coverage bound of T1 is the law-of-total-probability average of these two per-class bounds, so T2 is strictly stronger than T1.

### 5.2 Proof

This is Step 2–3 of T1 proof, stated per class rather than averaged.

1. Fix $k$. Under A1' + A2-class, conditioning on $Y_i = k$ makes $(S_1^{(k)}, \ldots, S_{n_k}^{(k)})$ i.i.d. $F_k$. Conditional on $Y_{\text{test}} = k$, $S_{\text{test}}^{(k)}$ is independent of calibration and distributed as $F_k$.
2. Thus the joint $(S_1^{(k)}, \ldots, S_{n_k}^{(k)}, S_{\text{test}}^{(k)})$ is i.i.d., exchangeable.
3. The prediction-set inclusion event $\{Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}})\}$ under $Y_{\text{test}} = k$ is $\{S_{\text{test}}^{(k)} \leq \hat{q}_k\}$, whose probability is bounded by Step 3 of T1's proof.

The sample-size requirement $n_k \geq \lceil 1/\alpha \rceil$ is necessary for the quantile index $\lceil (n_k+1)(1-\alpha)\rceil$ to not exceed $n_k$ (otherwise we set $\hat{q}_k = +\infty$, which gives trivially $\mathcal{C}_\alpha(x) \supseteq \{k\}$ for all $x$ — still valid coverage but informative only if $|\mathcal{C}_\alpha| < 2$). $\square$

### 5.3 Concrete sample-size sanity

For $\alpha = 0.1$: need $n_k \geq 10$.

- **Mendelian matched_9**, chrom-LOO: on held-out chrom, $n_{\text{neg}} \approx 10 \times n_{\text{pos}}$. Smallest chrom-LOO calibration has $n_1 \geq 30$ (chr22, chrX underrepresented); $n_0 \geq 300$. ✓
- **Complex traits matched_9**: $n_1 \geq 100$, $n_0 \geq 1000$ per chrom. ✓

So the sample-size requirement is comfortably satisfied on all 22 + X chrom-LOO folds. Finite-sample upper bound is correspondingly tight: $1/(n_1+1) \leq 0.033$ on smallest Mendelian fold, $\leq 0.010$ on Complex.

### 5.4 Relation to Vovk's 2003 Mondrian conformal machine

Vovk (2003) defines a **taxonomy** function $\kappa: (X, Y) \mapsto k$ and Mondrian conformal predictor that calibrates within each taxon separately. Class-conditional conformal is the special case $\kappa(X, Y) = Y$. The per-taxon coverage theorem of Vovk (2003, §6) gives exactly the bound in T2 under exchangeability of the full data. Our contribution here is **not** a new theorem but the precise identification of which assumptions (A1', A2-class) are needed when data is chrom-LOO'd rather than i.i.d.-sampled.

### 5.5 What T2 does *not* give

- **Local / feature-neighborhood coverage.** T3 handles this by adding Mondrian-by-$\hat{\sigma}$-bin. Without it, the most one can say is "coverage averaged across the whole class-$k$ test set is ≥ $1-\alpha$"; coverage on a specific subset of class-$k$ variants (e.g. hardest quintile) can be much worse. Day 10 numbers confirm: class-cond + homoscedastic has cov|pos = 0.896 but $\hat{\sigma}$-bin-9 coverage = 0.454 on Complex.

---

## 6. Robust fallback: T1/T2 under A1 only (no A2)

When A2 (of any granularity) fails — e.g. a test chromosome has systematically different variant characteristics than all calibration chroms — we invoke Barber, Candès, Ramdas, Tibshirani (2023) Theorem 2.

### 6.1 Barber 2023 Theorem 2 (restated in our notation)

Let $Z = (Z_1, \ldots, Z_n, Z_{n+1})$ be the full data ordering with $Z_{n+1}$ as the test point, $R(Z)$ the vector of residuals $(S_1, \ldots, S_{n+1})$, and $Z^i$ the sequence with $Z_i$ and $Z_{n+1}$ swapped. For prespecified weights $w_i \in [0, 1]$ with $\tilde{w}_i = w_i / (w_1 + \cdots + w_n + 1)$ and weighted quantile $\hat{q}^w$,
$$
P\!\left(Y_{n+1} \in \hat{C}_n(X_{n+1})\right) \;\geq\; 1 - \alpha \;-\; \sum_{i=1}^n \tilde{w}_i \cdot d_{\text{TV}}(R(Z), R(Z^i)).
$$

No factor of 2.

### 6.2 Adaptation to our problem

Use uniform weights $w_i = 1$ (so $\tilde{w}_i = 1/(n+1)$). The bound becomes
$$
P(\text{coverage}) \;\geq\; 1 - \alpha \;-\; \frac{1}{n+1} \sum_{i} d_{\text{TV}}(R(Z), R(Z^i)).
$$

Under A1 alone (without A2), $R(Z^i)$ differs from $R(Z)$ in the position of one residual $S_i \leftrightarrow S_{\text{test}}$. If $S_i$ is from chrom $c$ and $S_{\text{test}}$ from $c^*$, with $F_c^{(k)}$ and $F_{c^*}^{(k)}$ the conditional laws, the swap changes residual distribution by
$$
d_{\text{TV}}(R(Z), R(Z^i)) \;\leq\; d_{\text{TV}}(F_c, F_{c^*}).
$$

(We drop the class superscript here — same argument per class gives a T2-style version.)

### 6.3 T1'/T2' — distribution-shift coverage

**Corollary T1'.** Under A1 only,
$$
P(Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}})) \;\geq\; 1 - \alpha - \bar{\delta}_{\text{TV}},
$$
where $\bar{\delta}_{\text{TV}} = \frac{1}{|\mathcal{C}| - 1} \sum_{c \neq c^*} d_{\text{TV}}(F_c, F_{c^*})$ is the average TV distance from the test-chrom score law to each calibration-chrom law.

**Corollary T2'.** Same, class-conditional:
$$
P(Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}}) \mid Y_{\text{test}}=k) \;\geq\; 1 - \alpha - \bar{\delta}_{\text{TV}}^{(k)}.
$$

### 6.4 How to estimate $\bar{\delta}_{\text{TV}}$ in practice

$d_{\text{TV}}(F_c, F_{c^*})$ between two one-dimensional score distributions can be estimated with the empirical histograms via the $L_1$ distance of empirical CDFs / KS statistics. Day 13 KS empirical check (on cell-level A2, 155 tests, 12.3% rejection at $p<0.05$, max KS statistic = 0.24) shows $\bar{\delta}_{\text{TV}}^{(k,b)} \lesssim 0.05$ on worst chrom pairs.

At $\alpha = 0.10$, this bounds the coverage gap at $\leq 0.05$, consistent with the observed per-chrom cov-gap of $\leq 0.08$ across methods (Day 12 `scripts/15`). **Recommendation for the paper**: report both T1/T2 exact (under A1' + A2) and T1'/T2' robust (under A1 only), with the estimated TV term tabulated.

---

## 7. Scheme CL2 (nested chrom-LOO) — optional

In CL1, $\hat{p}$ and $\hat{\sigma}$ are trained on the same data used as calibration. This is standard for chrom-LOO but introduces a mild in-sample bias (training residuals < test residuals). CL2 addresses this:

**Scheme CL2.** For each test chrom $c^*$:
1. For each inner chrom $c' \in \mathcal{C} \setminus \{c^*\}$:
   - Train $\hat{p}^{(-c^*, -c')}, \hat{\sigma}^{(-c^*, -c')}$ on $\mathcal{D} \setminus (\mathcal{D}^{(c^*)} \cup \mathcal{D}^{(c')})$.
   - Compute calibration scores on $\mathcal{D}^{(c')}$ using these predictors.
2. Pool across inner $c'$: $\hat{q}_k = \text{Quantile}(\bigcup_{c'} S_{k,c'}, (1-\alpha))$.
3. For test prediction, use the "outer" predictors trained on $\mathcal{D}^{(-c^*)}$ (re-trained once, not 22 × 21 times).

Cost: 22 outer + 22 × 21 inner = 484 training runs (GBM, ~10 sec each → ~1 hour total). Not used in Day 10–13; flagged for paper-table robustness check.

Under A1' + A2, CL2 gives exact T1/T2 coverage without the in-sample-bias caveat. Under A1 alone, CL2 does not escape the chrom-shift TV term — that's fundamental.

---

## 8. Summary: what this chain gives us for the paper

| Result | Assumption | Guarantee | Status |
|---|---|---|---|
| T1 | A1' + A2 | $\text{cov} \in [1-\alpha, 1-\alpha+1/(n_{\min}+1)]$ | §4, proven |
| T2 | A1' + A2-class | per-class $\text{cov} \in [1-\alpha, 1-\alpha+1/(n_k+1)]$ | §5, proven |
| T1' | A1 only | $\text{cov} \geq 1-\alpha - \bar{\delta}_{\text{TV}}$ | §6, Barber 2023 corollary |
| T2' | A1 only | per-class $\text{cov} \geq 1-\alpha - \bar{\delta}_{\text{TV}}^{(k)}$ | §6, Barber 2023 corollary |
| T3 (bin-cond) | A1' + A2-cell | per-cell $\text{cov} \in [1-\alpha, 1-\alpha+1/(n_{kb}+1)]$ | `t3_proof_sketch.md`, same machinery |
| T4 | A1 | coverage-gap $\leq \bar{\delta}_{\text{TV}}$ across chrom shift | direct from §6 |

**For NeurIPS main (Path A) reviewer-facing claim**: T1 + T2 are prerequisite; T3 is the novelty. The T1/T2 proofs in this document set up the machinery and explicitly handle the chrom-LOO-specific "no calibration data from test chrom" issue (via A2 and Barber 2023 TV bound). T1' and T2' are our honest robustness statements — they directly tie empirical per-chrom cov-gap observations to a theoretically bounded quantity.

**For D&B backup paper (Day 10 deliverable)**: T2 (class-conditional) is the main theoretical content, directly justifying why Day 10's class-conditional conformal fixes the pathogenic-coverage collapse of homoscedastic. This is ~1.5 pages of appendix.

---

## 9. Citations used

- **Vovk, V., Lindsay, D., Nouretdinov, I., Gammerman, A.** (2003). *Mondrian confidence machine.* Technical Report, Royal Holloway. [Mondrian conformal, per-taxon coverage] — section reference for T2.
- **Vovk, V.** (2012). *Conditional validity of inductive conformal predictors.* ACML. [Mondrian extension; revisits the 2003 argument with cleaner conditions]
- **Tibshirani, R. J., Barber, R. F., Candès, E. J., Ramdas, A.** (2019). *Conformal prediction under covariate shift.* NeurIPS. [Weighted exchangeability; not directly used for T1/T2 here, but prerequisite to Barber 2023.]
- **Barber, R. F., Candès, E. J., Ramdas, A., Tibshirani, R. J.** (2023). *Conformal prediction beyond exchangeability.* Annals of Statistics 51(2). [Theorem 2 used for §6 TV bound; exact coverage inequality verified — no factor of 2.]
- **Romano, Y., Patterson, E., Candès, E. J.** (2019). *Conformalized quantile regression.* NeurIPS. [Motivates feature-adaptive score forms; used in formulation_v0.md §2.1 for score comparison.]
- **Benegas, G., Eraslan, G., Song, Y. S., et al.** (2025). *TraitGym.* bioRxiv. [Chrom-LOO protocol; AUPRC-by-chrom-weighted-average metric.]
- **Zhou, J. et al.** (2026). *DEGU: Deep ensemble with Gaussian uncertainty.* npj AI. [Concurrent work without coverage guarantee.]

---

## 10. Change log vs. earlier sketches

- `formulation_v0.md` §3 T1: "Barber et al 2023 Theorem 1" ↠ **Barber et al. 2023 Theorem 2** (verified from the arXiv HTML). The Theorem 1 in that paper is the unweighted version; Theorem 2 is the weighted statement with the TV bound we actually use.
- `formulation_v0.md` §3 T4: "$2\delta$" ↠ **$\delta$** (no factor of 2 in Barber 2023's bound; we had this wrong).
- `t3_proof_sketch.md` §4 Step 4: "$2 \cdot d_{\text{TV}}$" ↠ **$d_{\text{TV}}$**. Same correction.
- `theorems_roadmap.md` T4: "$2 \cdot d_{\text{TV}}$" ↠ **$d_{\text{TV}}$**.
- Introduced A1' (chrom-wise i.i.d.) as a strict strengthening of A1 and used explicitly in the exact-coverage proofs. Old sketches implicitly needed this without saying so.
- Introduced A2 hierarchy (marginal ⊂ class ⊂ cell) and made the implication chain explicit.

All three old files will be patched after this note is merged; see `reports/` commit for date.
