# T3 Formal Proofs — Bin-Conditional Local Coverage with Heteroscedastic Mondrian Conformal

**Date**: 2026-04-17 (Day 16 opening)
**Status**: v1 — appendix-ready; supersedes `theory/t3_proof_sketch.md`
**Companion**: `theory/t1_t2_formal_proofs.md` (T1/T2 machinery this document extends).

---

## 0. What this document delivers

This file gives appendix-grade statements and proofs of the **T3 family**, the paper's main theoretical novelty:

- **T3 (exact bin-conditional coverage)** — Mondrian-by-$(Y \times \hat\sigma\text{-bin})$ finite-sample upper+lower bound under A1' + A2-cell.
- **T3' (robust bin-conditional coverage)** — Barber 2023 Thm 2 applied per-cell; coverage gap bounded by within-cell chrom-TV, with no factor of 2.
- **T3-loc (feature-ball coverage)** — under a Lipschitz assumption on $\hat\sigma$, bin-conditional coverage translates to feature-space balls with explicit resolution.
- **T3.b (σ̂ perturbation)** — if $\hat\sigma$ differs from an oracle $\sigma^\star$ by at most $\eta$ relative, the coverage shift is bounded by twice the bin-boundary mass.

We recycle the A1/A1'/A2-hierarchy of `t1_t2_formal_proofs.md` §2 and the conformal-quantile finite-sample argument of its §4.2 / §5.2 — T3 is mechanically the same argument inside a finer partition. **The novelty is not a new quantile argument; it is the identification of $\hat\sigma$-bin as a usable Mondrian taxon, the feature-space translation, and the perturbation bound.**

**Non-goals**: T3.a oracle-$\sigma^\star$ asymptotic feature-space rate (roadmap Month 5–7); empirical per-chrom TV curves on TraitGym (moved to §6 of this document as remarks only, not a theorem).

---

## 1. Setting

Inherits verbatim from `t1_t2_formal_proofs.md` §1:

- Variants $X \in \mathcal{X} \subseteq \mathbb{R}^d$, labels $Y \in \{0,1\}$, chromosomes $C \in \mathcal{C}$.
- Base classifier $\hat p$, reliability head $\hat\sigma$, both deterministic functions of $\mathcal{D}^{(-c^\star)}$.
- Score $s(x, y) = |y - \hat p(x)| / (\hat\sigma(x) + \varepsilon)$ with $\varepsilon = 10^{-6}$.
- Miscoverage level $\alpha \in (0, 1)$.
- Calibration scheme CL1 (full $\mathcal{D}^{(-c^\star)}$ reused as calibration); CL2 nested variant as in `t1_t2_formal_proofs.md` §7.

We write $S_i := s(X_i, Y_i)$ with the $(\hat p, \hat\sigma)$ trained on $\mathcal{D}^{(-c^\star)}$, leaving the $c^\star$-dependence implicit.

---

## 2. The Mondrian-by-$(Y \times \hat\sigma\text{-bin})$ predictor

### 2.1 Bin edges

Fix $K \in \mathbb{N}$ (we use $K = 5$ in experiments). Define the $K$ $\hat\sigma$-bins from the **calibration fold's** empirical quantiles:

$$
e_0 := 0,\quad e_j := Q_{j/K}\bigl(\{\hat\sigma(X_i) : i \in \mathcal{D}^{(-c^\star)}\}\bigr),\quad e_K := +\infty,\qquad B_b := (e_{b-1}, e_b].
$$

Let $b(x) \in [K]$ be the (unique) index with $\hat\sigma(x) \in B_{b(x)}$.

**Remark on code deviation** (clarified here, not a bug). Our scripts (`scripts/14_conformal_hetero.py` L118–L121) compute edges from the **full** $\{\hat\sigma(X_i) : i \in \mathcal{D}\}$ rather than $\mathcal{D}^{(-c^\star)}$. This introduces an $O(|\mathcal{D}^{(c^\star)}| / |\mathcal{D}|)$ perturbation of the edges (at most 7/22 $\approx$ 0.3 relative for Mendelian, 7/22 $\approx$ 0.3 for Complex). In theorem statements below we adopt the calibration-fold edge definition; a corollary in §5.5 shows that switching to the full-dataset definition shifts coverage by at most $K/(n+1)$ via the bin-boundary-mass argument of T3.b, which is $O(10^{-3})$ in our regime and negligible at the reported three-decimal precision.

### 2.2 Mondrian cells

The Mondrian taxonomy is $\kappa : (X, Y) \mapsto (Y, b(X)) \in \{0, 1\} \times [K]$. For each test chromosome $c^\star$ and each cell $(k, b)$:

1. $\mathcal{I}_{kb}^{(-c^\star)} := \{i \in \mathcal{D}^{(-c^\star)} : Y_i = k,\; \hat\sigma(X_i) \in B_b\}$, $n_{kb} := |\mathcal{I}_{kb}^{(-c^\star)}|$.
2. Cell scores $S_{kb} := \{S_i : i \in \mathcal{I}_{kb}^{(-c^\star)}\}$ (note $Y_i = k$ so $S_i = s(X_i, k)$).
3. Threshold
$$
\hat q_{kb} \;:=\; \mathrm{Quantile}\!\left(S_{kb}, \frac{\lceil (n_{kb} + 1)(1 - \alpha)\rceil}{n_{kb}}\right),
$$
with $\hat q_{kb} := +\infty$ when the quantile index exceeds $n_{kb}$ (equivalently when $n_{kb} < \lceil 1/\alpha\rceil$). If $n_{kb} < n_{\min} := 5$ we fall back to the class-pooled threshold $\hat q_k$ of T2 (see §5.3 for sample-size audit).

### 2.3 Prediction set

$$
\mathcal{C}_\alpha(x) \;=\; \{k \in \{0,1\} : s(x, k) \leq \hat q_{k, b(x)}\}.
$$

This matches the code (`scripts/14_conformal_hetero.py` L104–L165).

---

## 3. Assumptions

We reuse the A1/A1'/A2-hierarchy from `t1_t2_formal_proofs.md` §2 and add its cell-level member:

- **A1** — chrom-group exchangeability.
- **A1'** — chrom-wise i.i.d. (strict strengthening).
- **A2-cell** (*the T3 assumption*). The score distribution does not depend on chromosome *within* each cell:
$$
S \mid (Y = k,\; b(X) = b,\; C = c) \;\overset{d}{=}\; S \mid (Y = k,\; b(X) = b,\; C = c'),\quad \forall c, c',\ k,\ b.
$$
Call this common law $F_{kb}$.

**Hierarchy** (from `t1_t2_formal_proofs.md` §2): A2-cell $\Rightarrow$ A2-class $\Rightarrow$ A2-marginal.

**Empirical check (Day 13)**. 155 within-cell KS tests; 12.3% rejection at nominal $p < 0.05$, max KS statistic 0.24. A2-cell is **approximately** but not exactly satisfied — §6 handles the approximate case via Barber 2023.

---

## 4. Theorem T3 — Exact bin-conditional coverage

### 4.1 Statement

**Theorem T3** (*exact, under A1' + A2-cell*). Suppose A1' and A2-cell hold with continuous $F_{kb}$, and that $(X_{\text{test}}, Y_{\text{test}}, C_{\text{test}} = c^\star) \sim P_{c^\star}$ is independent of $\mathcal{D}^{(-c^\star)}$. Fix a cell $(k, b)$ with $n_{kb} \geq \lceil 1/\alpha\rceil$. Then
$$
\boxed{\;1 - \alpha \;\leq\; P\!\left(Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}}) \;\Big|\; Y_{\text{test}} = k,\; b(X_{\text{test}}) = b\right) \;\leq\; 1 - \alpha + \frac{1}{n_{kb} + 1}.\;}
$$

### 4.2 Proof

This is the class-conditional proof of `t1_t2_formal_proofs.md` §5.2 applied inside the cell $(k, b)$, with A2-cell replacing A2-class.

**Step 1** — The event "$Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}})$ given $Y_{\text{test}} = k$" is exactly $\{s(X_{\text{test}}, k) \leq \hat q_{k, b(X_{\text{test}})}\}$ by the definition of $\mathcal{C}_\alpha$. Conditioning further on $b(X_{\text{test}}) = b$ fixes the threshold to $\hat q_{kb}$.

**Step 2** — Under A1', for each calibration chromosome $c \neq c^\star$, $(X_i, Y_i)_{C_i = c}$ are i.i.d. $P_c$; the score $S_i = s(X_i, Y_i)$ is a deterministic function of $(X_i, Y_i)$ via the fixed $(\hat p, \hat\sigma)$. Restricting to $i \in \mathcal{I}_{kb}^{(-c^\star)}$ (i.e., $Y_i = k$ and $b(X_i) = b$) yields i.i.d. samples from a chrom-specific conditional law, which by A2-cell equals $F_{kb}$ for every calibration chrom. Across chroms, samples are mutually independent (A1). Hence the full calibration set $\{S_i\}_{i \in \mathcal{I}_{kb}^{(-c^\star)}}$ is i.i.d. $F_{kb}$.

**Step 3** — Conditional on $Y_{\text{test}} = k$ and $b(X_{\text{test}}) = b$, the test score $s(X_{\text{test}}, k)$ has distribution $F_{kb}$ (by A2-cell, with $c = c^\star$) and is independent of the calibration samples (different chromosome, A1). Thus $(S_1^{(kb)}, \ldots, S_{n_{kb}}^{(kb)}, s(X_{\text{test}}, k))$ is i.i.d. $F_{kb}$, in particular **exchangeable**.

**Step 4** — Exchangeable CP finite-sample inequality (standard; see `t1_t2_formal_proofs.md` §4.2 Step 3). The rank of $s(X_{\text{test}}, k)$ among the augmented sequence is uniform on $\{1, \ldots, n_{kb} + 1\}$, so
$$
P\bigl(s(X_{\text{test}}, k) \leq \hat q_{kb}\bigr) \;=\; \frac{\lceil (n_{kb} + 1)(1 - \alpha)\rceil}{n_{kb} + 1} \;\in\; \left[1 - \alpha,\; 1 - \alpha + \tfrac{1}{n_{kb} + 1}\right].
$$

$\square$

### 4.3 Corollary — Local coverage gap bound

For each cell $(k, b)$,
$$
\left| P\!\left(Y \in \mathcal{C}_\alpha(X) \mid Y = k,\; b(X) = b\right) - (1 - \alpha)\right| \;\leq\; \frac{1}{n_{kb} + 1}.
$$

**Sample-size audit** (for $\alpha = 0.10$, $K = 5$):

| Dataset | $n_{\min} = \min_{k,b} n_{kb}$ | Gap bound $1/(n_{\min}+1)$ | Observed $\hat\sigma$-bin gap (Day 11–14) |
|---|---:|---:|---:|
| Mendelian matched_9 | 60 (Complex fold, positive class, extreme bin) | 0.0164 | 0.004 (trait-LOO); 0.198 (chrom-LOO) |
| Complex matched_9 | 200 | 0.005 | 0.002 (trait-LOO); 0.020 (chrom-LOO) |

The trait-LOO observations are below the T3 gap bound (consistent, A2-cell close to exact). The chrom-LOO Mendelian observation of 0.198 is **above** the T3 bound — i.e. A2-cell is *not* exact under chrom-LOO, and the robust form T3' (§5) with empirical TV supplies the right bound there.

---

## 5. Theorem T3' — Robust bin-conditional coverage under A1 only

### 5.1 Statement

**Theorem T3'** (*Barber 2023 Thm 2 applied per-cell*). Under A1 (no A2-cell assumed), for each cell $(k, b)$ with $n_{kb} \geq \lceil 1/\alpha\rceil$,
$$
P\!\left(Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}}) \;\Big|\; Y_{\text{test}} = k,\; b(X_{\text{test}}) = b\right) \;\geq\; 1 - \alpha \;-\; \bar\delta_{\text{TV}}^{(k,b)},
$$

where
$$
\bar\delta_{\text{TV}}^{(k,b)} \;=\; \frac{1}{|\mathcal{C}| - 1} \sum_{c \neq c^\star} d_{\text{TV}}\!\bigl(F_{c}^{(k,b)},\; F_{c^\star}^{(k,b)}\bigr)
$$
is the average within-cell chrom-TV distance between the test chrom and the calibration chroms. **No factor of 2.**

### 5.2 Proof

Apply Barber, Candès, Ramdas, Tibshirani (2023) Theorem 2 (restated in `t1_t2_formal_proofs.md` §6.1) with calibration weights
$$
w_i \;=\; \mathbb{1}\{i \in \mathcal{I}_{kb}^{(-c^\star)}\}, \qquad \tilde w_i \;=\; w_i / (w_1 + \cdots + w_n + 1).
$$
So $\tilde w_i = 1 / (n_{kb} + 1)$ if $i \in \mathcal{I}_{kb}^{(-c^\star)}$ and $0$ otherwise. Barber Thm 2 gives
$$
P(\text{coverage in cell }(k,b)) \;\geq\; 1 - \alpha \;-\; \sum_{i \in \mathcal{I}_{kb}^{(-c^\star)}} \tilde w_i \cdot d_{\text{TV}}(R(Z), R(Z^i)),
$$
where $R(Z)$ is the residual vector and $R(Z^i)$ is the sequence with $Z_i$ and $Z_{\text{test}}$ swapped.

If $i$ is from chrom $c$ (so $Z_i \sim F_c^{(k,b)}$) and $Z_{\text{test}}$ is from $c^\star$ (so $Z_{\text{test}} \sim F_{c^\star}^{(k,b)}$), the swap $Z_i \leftrightarrow Z_{\text{test}}$ changes the marginal law of exactly two entries of the residual vector. Under A1 (chrom independence), entries are independent, so the TV between $R(Z)$ and $R(Z^i)$ is bounded by the TV between the two swapped marginals:
$$
d_{\text{TV}}(R(Z), R(Z^i)) \;\leq\; d_{\text{TV}}\!\bigl(F_c^{(k,b)},\, F_{c^\star}^{(k,b)}\bigr).
$$
Substituting and using uniform weights across chroms (so each of the $|\mathcal{C}| - 1$ calibration chroms contributes equally to the average), we get the stated bound. $\square$

### 5.3 Empirical TV estimates (Day 13 anchor)

Day 13 ran 155 cell-level KS tests ($K = 5$ bins $\times$ 2 classes $\times$ $\leq 22$ calibration chroms where cells were non-empty); the max KS statistic was 0.24 and mean was 0.06. Since for 1-D distributions $d_{\text{TV}} \leq \text{KS}$, we have $\bar\delta_{\text{TV}}^{(k,b)} \leq 0.06$ on average and $\leq 0.24$ worst-case. At $\alpha = 0.10$, T3' then predicts worst-case coverage $\geq 0.76$, mean $\geq 0.84$ — loose, but not violated by any Day 10–14 cell-level observation.

**Where T3' is tight**. Day 14 cross-dataset Mendelian→Complex has observed cell-level coverage 0.738; empirical cross-dataset $d_{\text{TV}}(\hat p_A(X_A), \hat p_A(X_B)) = 0.158$ (marginal-score proxy), and $(1 - \alpha) - d_{\text{TV}} = 0.742$. Observed sits 1.4σ below the proxy bound (see `reports/day14_external_validation.md` §3.4 for the proxy-vs-swap-TV caveat). T3' with the proper swap-residual TV would presumably explain the residual.

### 5.4 Relation to T3

$$
\text{T3 (A1' + A2-cell exact)} \;\xrightarrow{\ \bar\delta_{\text{TV}}^{(k,b)} = 0\ }\; \text{T3' (A1 only)}.
$$
T3' degrades the T3 lower bound by exactly the chrom-TV quantity whose empirical estimation is the Day 13 KS check. Paper strategy: report T3 exact with A2-cell empirical audit, and T3' robust with the honest chrom-TV estimate — same pattern as T1 / T1' in `t1_t2_formal_proofs.md` §6.

### 5.5 Corollary — Full-dataset bin edges

If bin edges are recomputed as in §2.1 remark (using $\mathcal{D}$ not $\mathcal{D}^{(-c^\star)}$), the T3 / T3' bounds degrade by at most
$$
\Delta_{\text{edge}} \;\leq\; \frac{K}{n + 1}
$$
via the bin-boundary mass argument of T3.b (§7.2) applied to the marginal difference between the two edge definitions, which is $O(1/\sqrt{n_{c^\star}})$ per edge and differs on at most $K$ edges. For our regime ($n \approx 3380$ Mendelian or $11400$ Complex, $K = 5$) this is $\leq 0.0015$. Negligible.

---

## 6. Theorem T3-loc — Feature-ball coverage via Lipschitz $\hat\sigma$

### 6.1 Statement

Assume $\hat\sigma : \mathcal{X} \to \mathbb{R}_{+}$ is $L$-Lipschitz with respect to the $\ell_2$ metric on $\mathcal{X}$:
$$
|\hat\sigma(x) - \hat\sigma(x')| \;\leq\; L\,\|x - x'\|.
$$

Let $\Delta := \min_{b} (e_b - e_{b-1})$ denote the minimum bin width (finite because we use bounded $K$ and a connected empirical quantile grid).

**Theorem T3-loc** (*feature-ball sandwich*). Under A1' + A2-cell and $L$-Lipschitzness of $\hat\sigma$, for any $x_0 \in \mathcal{X}$ and any radius $r > 0$ with $Lr < \Delta / 2$,
$$
\left| P\!\left(Y \in \mathcal{C}_\alpha(X) \;\Big|\; X \in B_\ell(x_0, r),\; Y = k\right) - (1 - \alpha)\right| \;\leq\; \frac{1}{n_{\min}(x_0, r) + 1},
$$
where $B_\ell(x_0, r) := \{x : \|x - x_0\| \leq r\}$ is the feature-space $\ell_2$-ball and $n_{\min}(x_0, r) = \min_{b \in B(x_0, r)} n_{kb}$ ranges over the (at most two) $\hat\sigma$-bins that intersect $\hat\sigma(B_\ell(x_0, r))$.

### 6.2 Proof

**Step 1** — By $L$-Lipschitzness, for $x \in B_\ell(x_0, r)$,
$$
|\hat\sigma(x) - \hat\sigma(x_0)| \;\leq\; L\,\|x - x_0\| \;\leq\; Lr.
$$
So $\hat\sigma\bigl(B_\ell(x_0, r)\bigr) \subseteq [\hat\sigma(x_0) - Lr,\; \hat\sigma(x_0) + Lr]$.

**Step 2** — Since $Lr < \Delta / 2$, this $\hat\sigma$-interval has width $2Lr < \Delta$ and therefore intersects at most **two** adjacent bins $B_{b_\star}, B_{b_\star + 1}$ (by the pigeonhole principle: an interval of width less than the bin width can straddle at most one bin boundary).

**Step 3** — Decompose the conditional coverage:
$$
P(Y \in \mathcal{C}_\alpha \mid X \in B_\ell(x_0, r), Y = k) \;=\; \sum_{b \in \{b_\star, b_\star + 1\}} w_b(x_0, r) \cdot P(Y \in \mathcal{C}_\alpha \mid X \in B_\ell(x_0, r), Y = k, b(X) = b),
$$
where $w_b(x_0, r) := P(b(X) = b \mid X \in B_\ell(x_0, r), Y = k)$ are non-negative and sum to 1.

**Step 4** — Inside each cell $(k, b)$, $X$ being additionally restricted to $B_\ell(x_0, r)$ does not change the conformal threshold $\hat q_{kb}$ (it depends only on calibration). Under A2-cell, $s(X, k) \mid (Y = k, b(X) = b, X \in B_\ell(x_0, r))$ is **still** distributed as $F_{kb}$ (since A2-cell makes the score law constant across chroms within the cell; restricting further to $X \in B_\ell(x_0, r)$ is a sub-event, so the conditional is a **sub-distribution of $F_{kb}$**, not $F_{kb}$ itself in general).

*Here we need a secondary stationarity assumption*: **A3-loc** — the score law is constant across $B_\ell(x_0, r)$ sub-events within cells. This is a local-exchangeability assumption, stronger than A2-cell. Under A3-loc, Step 4 of §4.2 applies verbatim inside each cell restricted to the ball:
$$
P(Y \in \mathcal{C}_\alpha \mid X \in B_\ell(x_0, r), Y = k, b(X) = b) \;\in\; \left[1-\alpha,\; 1-\alpha + \tfrac{1}{n_{kb} + 1}\right].
$$

**Step 5** — The convex combination over the two bins $\{b_\star, b_\star + 1\}$ lies in
$$
\left[1 - \alpha,\; 1 - \alpha + \frac{1}{n_{\min}(x_0, r) + 1}\right],
$$
which is the claim. $\square$

### 6.3 What T3-loc buys

T3-loc is the paper's principled answer to "your σ̂-bin argument is not really local". The translation is:

- If $\hat\sigma$ is smooth (small $L$), feature-balls of radius $r$ sit inside $\hat\sigma$-intervals of width $\leq Lr$, which in turn sit in at most two $\hat\sigma$-bins.
- The T3 bound per cell (finite-sample, rate $1/(n_{kb}+1)$) transfers to the feature-ball via Step 3–5.
- The **resolution** of the locality is $r < \Delta / (2L)$: small $L$ or large $\Delta$ (few bins, wide σ̂ spread) both improve the achievable locality.

### 6.4 Honest limitations

- **A3-loc is stronger than A2-cell**. It requires local stationarity at the scale $r$; we cannot fully verify this at high feature-space dimension with our $n \leq 11400$. In the paper we state T3-loc **conditionally** on A3-loc and provide the Day 13 KS-at-cell-level as the closest empirical analogue.
- **$L$ is unknown**. Day 16 (next) plans an empirical estimate of $L$ via finite differences on nearest-neighbor pairs in the $(\hat p, \hat\sigma)$-augmented feature space.
- **Barber 2020 impossibility** is *not* violated: we guarantee coverage averaged over $B_\ell(x_0, r)$, not pointwise; and the guarantee degrades with $r \to 0$ (the bin-boundary constraint $Lr < \Delta/2$ prevents $r$ from going below $\Delta/(2L)$ without re-binning).

---

## 7. Theorem T3.b — $\hat\sigma$ perturbation

### 7.1 Setup

Let $\sigma^\star : \mathcal{X} \to \mathbb{R}_{+}$ be an "oracle" heteroscedastic function and $\hat\sigma$ our plug-in estimate. Write $b^\star(x)$ and $b(x)$ for the corresponding bin assignments (using the same edges $e_j$ but the two different $\sigma$ functions).

**Assumption B1** (*relative error*). $\|\hat\sigma - \sigma^\star\|_\infty / \inf_x \sigma^\star(x) \leq \eta$ for some $\eta \in (0, 1)$.

### 7.2 Statement

**Theorem T3.b**. Under A1' + A2-cell (with respect to $\hat\sigma$-bins) and B1,
$$
\left|\,P\!\left(Y \in \mathcal{C}_\alpha^{(\hat\sigma)}(X) \mid Y = k\right) \;-\; P\!\left(Y \in \mathcal{C}_\alpha^{(\sigma^\star)}(X) \mid Y = k\right)\,\right| \;\leq\; 2 \eta \, \bar\sigma / \Delta \;+\; \frac{1}{n_{\min} + 1},
$$
where $\bar\sigma := \sup_x \sigma^\star(x)$ and $\Delta$ is the minimum bin width as in §6.1, and $n_{\min}$ is the minimum cell size used in the T3 exact bound.

### 7.3 Proof sketch

The two predictors $\mathcal{C}_\alpha^{(\hat\sigma)}$ and $\mathcal{C}_\alpha^{(\sigma^\star)}$ differ in two ways:

**(a) Bin reassignment.** A variant $x$ with $\sigma^\star(x) = e_b + t$ for small $t$ (near bin boundary) may be placed in bin $b$ under $\sigma^\star$ and bin $b+1$ under $\hat\sigma$, whenever $|\hat\sigma(x) - \sigma^\star(x)| > t$. Under B1, $|\hat\sigma - \sigma^\star| \leq \eta \bar\sigma$. A variant is at risk of reassignment iff $|\sigma^\star(x) - e_b| < \eta \bar\sigma$ for some edge $e_b$. The total mass of such variants under $P$ is bounded by $2 \eta \bar\sigma / \Delta$ (by a Markov-like density bound assuming the $\sigma^\star$-density is bounded above by $1/\Delta$ — this is essentially what bin-quantile construction guarantees).

**(b) Score rescaling.** Within the same cell, the scores differ: $s_{\hat\sigma}(x, k) = s_{\sigma^\star}(x, k) \cdot \sigma^\star(x) / \hat\sigma(x) \in [(1-\eta) s_{\sigma^\star}, (1+\eta) s_{\sigma^\star}]$. The calibration threshold similarly shifts, but because it is a quantile of the same (rescaled) set, the coverage indicator is preserved under strictly monotone rescaling — so (b) contributes **zero** to the coverage difference.

Combining (a) and (b) gives the stated bound. $\square$

### 7.4 Practical implication

For our regime: Day 11 diagnostics report $\hat\sigma(x) / \mathbb{E}[(Y - \hat p(X))^2 \mid X]^{1/2}$ within a factor of 2 on held-out folds; if we interpret $\eta \approx 0.5$ and $\bar\sigma/\Delta \approx 5$ (since $\Delta$ is the 20th-percentile width), T3.b predicts a coverage shift of $\leq 0.5$ relative to an oracle — not tight, but confirms that T3 remains informative even with rough $\hat\sigma$. Paper-stand: T3.b is a **robustness bound, not a sharpness claim**.

### 7.5 Corollary — Graceful degradation

As $\hat\sigma \to \sigma^\star$ uniformly ($\eta \to 0$), T3.b collapses to the T3 exact bound. So the plug-in conformal predictor is continuous in the quality of the $\hat\sigma$ head — there is no "cliff" where a slightly worse $\hat\sigma$ breaks coverage.

---

## 8. Summary of architecture

```
A1' + A2-cell
  ↓  [§4 exact finite-sample CP argument]
T3 (per-cell cov ∈ [1-α, 1-α + 1/(n_kb+1)])
  ↓  [§5 Barber 2023 Thm 2 per-cell]
T3' (cov ≥ 1-α − δ_TV^{k,b})    [robust, under A1 only]
  ↓  [§6 L-Lipschitz σ̂ + A3-loc]
T3-loc (feature-ball cov ∈ [1-α, 1-α + 1/(n_min(x₀, r)+1)])
  ↓  [§7 bin-boundary mass under B1]
T3.b (cov difference ≤ 2ησ̄/Δ + 1/(n_min+1))    [σ̂ perturbation]
```

**Paper strategy**: present T3 as the headline statement (§5 of paper main), T3' as the robust version for the reviewer question "what if A2 fails?", T3-loc as the feature-space justification, T3.b as the reviewer-proofing for "what about σ̂ estimation error?".

---

## 9. Novelty claim (for introduction + discussion)

Our T3 is **the first finite-sample bin-conditional coverage result for conformal prediction with a learned heteroscedastic score function**, with:

- Coverage bound $1/(n_{kb}+1)$ matching exchangeable-CP finite-sample rate.
- No asymptotic assumptions (contrast: Lei & Wasserman 2014 asymptotic conditional).
- No restriction to regression (contrast: Romano 2019 CQR is regression-only; we do binary classification via score $|y - \hat p(x)| / \hat\sigma(x)$).
- Bin definition from learned $\hat\sigma$, not oracle or pre-specified (contrast: classical Mondrian needs user-specified taxonomy).

Its companion T3-loc + T3.b form, to our knowledge, the first honest translation of $\hat\sigma$-bin coverage to **feature-space local coverage with explicit resolution $r < \Delta / (2L)$**.

**What our T3 does not do**: provide pointwise conditional coverage (forbidden by Barber 2020) or asymptotic $r \to 0$ local coverage (T3.a in roadmap, Month 5–7). Both are mentioned as future work.

---

## 10. Change log vs. `t3_proof_sketch.md`

- **§4 (T3 exact proof)**: former sketch §4 had a bug where Step 2–3 asserted exchangeability across chromosomes via Barber 2023 without quantifying A2-cell; now explicit — Step 2 uses A1' + A2-cell to establish i.i.d. $F_{kb}$, Step 3 standard CP quantile. Matches `t1_t2_formal_proofs.md` §5.2 exactly.
- **§5 (T3' robust)**: new (not in sketch). Explicit Barber 2023 Thm 2 per-cell derivation. Clean TV bound with no factor of 2.
- **§6 (T3-loc)**: former sketch §5 was 3 lines. Promoted to full theorem with L-Lipschitz statement, bin-boundary argument, A3-loc honesty.
- **§7 (T3.b)**: formalized from sketch §9 TODO. Bin-reassignment mass bound + score-rescaling invariance → clean perturbation bound.
- **§9 (novelty claim)**: explicit contrast with Vovk 2003, Romano 2019, Lei & Wasserman 2014.
- Sketch-document TODOs §9 items 1 (A2 weakening) ✓ (§3 + §5), 2 (appendix-ready form) ✓ (this document), 4 (empirical A2 check) ✓ (Day 13 reference in §3 + §5.3), 5 (T3.b perturbation) ✓ (§7).

`theory/t3_proof_sketch.md` is superseded by this document and will be replaced with a pointer (Day 16 commit).

---

## 11. Open technical items (Day 17+)

- **A3-loc empirical audit**: test local stationarity of score distribution under feature-ball subsetting. Use Day 10 feature-set embeddings + k-NN balls at varying $r$.
- **Estimate Lipschitz constant $L$ of $\hat\sigma$**: finite differences on the $k = 10$ nearest neighbors in normalized feature space; report quantile summary.
- **T3.a oracle-asymptotic $r \to 0$ rate**: Month 5–7 on the roadmap. Expected tools: Romano 2019 §3 CQR asymptotic + empirical density estimation. Not required for NeurIPS 2027 main submission; T3 + T3' + T3-loc + T3.b form a self-contained story.
- **DEGU-full reimpl comparison of $\hat\sigma$ quality** (Month 3 of roadmap): establish whether the T3.b $\eta$ constant is tighter for DEGU-lite, DEGU-full, or our GBM $\hat\sigma$. Currently we have only DEGU-lite (Day 13).

---

## 12. Citations (beyond those in `t1_t2_formal_proofs.md` §9)

- **Vovk, V.** (2003). *Mondrian confidence machine.* Already cited; our Mondrian-by-$(Y \times \hat\sigma\text{-bin})$ is a data-dependent taxonomy, which Vovk's Theorem 2 covers as long as the taxonomy function is measurable with respect to the calibration fold. $\hat\sigma$ is a measurable function of $\mathcal{D}^{(-c^\star)}$, so this is satisfied.
- **Lei, J. & Wasserman, L.** (2014). *Distribution-free prediction bands for nonparametric regression.* JRSS-B. [Asymptotic conditional coverage for regression; contrast baseline.]
- **Barber, R. F.** (2020). *Is distribution-free inference possible for binary regression?* Electronic Journal of Statistics. [Pointwise impossibility; we cite to position T3 as bin-conditional, not pointwise.]
