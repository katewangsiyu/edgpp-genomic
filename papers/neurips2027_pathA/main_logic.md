# Logic Verification Report

**Paper:** main.tex (NeurIPS 2026 main-track submission, conformalized heteroscedastic VEP)
**Date:** 2026-05-02

## Part 1 — Argument Chain View

| Section | Primary Claim | Connects To | Status |
|---------|--------------|-------------|--------|
| Abstract | HCCP is the only single-fold partition we evaluated holding class-cond + bin-local coverage on TraitGym; tight finite-sample bound within equi-bin Mondrian-$K$ class with $\pi_{\min}^{-1/2}$ explicit | Introduction | Connected |
| Introduction | VEP needs joint class-cond + bin-local CP under imbalance + heteroscedasticity; HCCP achieves it via $(y \times \shat\text{-bin})$ Mondrian; T5.1 + T5.2 prove tight rate within the family; T3$'$ certifies under exchangeability violations | Problem formulation | Connected |
| Problem formulation (§3) | Defines $\D$, A1/A1$'$/A2-cell/A-SL assumptions; KS audit gives 46.2% Mendelian / 10.4% Complex rejection; Mondrian prediction set + small-cell T2 fallback | Method (§4) / Theory (§5) | Connected |
| Method (§4) | HCCP three-stage pipeline; joint $(y \times b)$ partition is unique refinement admitting both T2 and T3; $\hat K(c_{\mathrm{outer}})$ via proper nested chrom-LOO | Theory (§5) | Connected |
| Theory (§5) | T3 exact bin-conditional under A1$'$ + A2-cell; T3$'$ robust certificate under A1$'$ alone with $-\bar\delta_{\mathrm{TV}}$ slack; T5.1 oracle $K^\star$ + T5.2 within-class tight $\Omega(n^{-1/2})$ on the equi-bin Mondrian-$K$ class | Experiments (§6) | Connected |
| Experiments (§6) | Tab 1 main: HCCP $\shat$-bin gap 0.020 Mendelian / 0.004 Complex; H2H: only single-fold partition holding $\mathrm{cov}_{\|Y=1} \geq 0.85$ + gap $\leq 0.20$ on both; M$\to$C documented as joint-failure mode; ProteinGym 27/50 within $\pm 0.02$, 44/50 within $\pm 0.05$ | Discussion (§7) | Connected |
| Discussion (§7) | 78% singletons at $\alpha=0.10$ Mendelian; A-SL load-bearing only for T5; explicit limitations (TV proxy, $K^\star$ asymptotic, DEGU port, ProteinGym outliers); broader impact + ACMG positioning | Conclusion (§8) | Connected |
| Conclusion (§8) | Restates three contributions: tight rate within Mondrian-$K$ class with $\pi_{\min}^{-1/2}$ explicit; HCCP framework + T3$'$; TraitGym + ProteinGym + synthetic validation with base-predictor decoupling | — | Terminal |

## Part 2 — Categorized Issue List

### Argument Chain Gaps

No issues identified in this category.

### Unsupported Claims

No issues identified in this category.

### Terminology Inconsistencies

No issues identified in this category.

### Number Contradictions

**Issue NC-1: Stale $\shat$-Mondrian minority coverage in Related Work (§2)**
**Section:** [Related work, §2 paragraph 2 "Class-conditional and imbalanced CP"], [Introduction, §1 paragraph 2], [Experiments, §6.2 head-to-head]
**Problem:** §2 states "$\shat$-Mondrian preserves local coverage but loses $\mathrm{cov}_{|Y=1}$ **down to 0.66 at $\pi_{\min} = 0.10$**". §1 paragraph 2 reports the same B2 $\shat$-Mondrian quantity on Complex as "$\mathrm{cov}_{|Y=1} = 0.62$"; §6.2 H2H reports "$\mathrm{cov}_{|Y=1} = 0.624$ on Complex". The headline operating point for Complex is $\pi_{\min} = 0.10$, so all three should reference the same number. 0.62 (§1) and 0.624 (§6.2) round consistently; 0.66 (§2) does not — neither does it match the imbalance-sweep minimum 0.650 at $\pi_{+} = 0.05$ reported in §6.2.
**Why this matters:** Reviewers cross-check Related Work positioning numbers against Experiment tables. A 0.04 discrepancy on a load-bearing single-axis-Mondrian failure number undermines the §2 Pareto-wall narrative that motivates the joint partition.
**Suggestion:** Replace "0.66" with "0.62" (matching §1 / §6.2 H2H Complex headline), or restate as "the imbalance sweep (App.~\ref{app:tables}, Tab.~\ref{tab:imbsweep}) drives B2 down to 0.65 at $\pi_{+} = 0.05$" if that was the intended reading.
> **[Chinese]** 问题：§2 写"$\shat$-Mondrian 在 $\pi_{\min}=0.10$ 下 minority coverage 跌到 0.66"，但 §1 段 2 同一个 B2 σ̂-Mondrian 数字是 "0.62"，§6.2 H2H 是 "0.624"，imbalance 扫到 $\pi_+=0.05$ 极端时也只到 0.650。三个数字 (0.62 / 0.624 / 0.66) 应同源，0.66 是 outlier。为什么重要：审稿人会逐项核对 Related Work 数字与 §6 表，0.04 的差距会被记下来扣 Pareto-wall narrative 的分。建议：§2 改为 "0.62"（与 §1/§6.2 一致），或改写为"imbalance 扫到 $\pi_+=0.05$ 时跌到 0.65"——明确指向哪个 operating point。

**Issue NC-2: Stale class-Mondrian gap range in Related Work (§2)**
**Section:** [Related work, §2 paragraph 2], [Experiments, §6.2 head-to-head], [Tab 1 in §6.1]
**Problem:** §2 states "class-Mondrian preserves $\mathrm{cov}_{|Y=1}$ but loses bin-local coverage (**gap $0.32$--$0.45$**)". §6.2 H2H gives B3 class-Mondrian as "$13.4\times$/$2.4\times$" the HCCP gap, with HCCP at $0.060$ Complex / $0.173$ Mendelian — back-computing yields B3 $\approx 0.80$ Complex / $0.42$ Mendelian (range $0.42$--$0.80$). Tab 1 row "Class-cond.\ (Day 10)" gives $\shat$-bin gap $0.46$ Mendelian / $0.51$ Complex. Neither set of operative numbers ($0.42$--$0.80$ from H2H; $0.46$--$0.51$ from Tab 1) overlaps with §2's "$0.32$--$0.45$".
**Why this matters:** Same as NC-1: §2 is the framing layer reviewers cross-reference against tables. The asserted range understates the actual class-Mondrian gap; if §6.2's $13.4\times$ ratio is the headline, claiming class-Mondrian gap ≤ 0.45 in §2 weakens the §1 ratio.
**Suggestion:** Update §2 to either "gap $0.46$--$0.51$" (Tab 1 single-seed Class-cond) or "gap $0.42$--$0.80$" (§6.2 H2H bootstrap mean) — pick one source and cite it.
> **[Chinese]** 问题：§2 说 class-Mondrian "gap $0.32$--$0.45$"。从 §6.2 H2H ratio 反推 B3 class-Mondrian 是 0.42 (Mendelian) / 0.80 (Complex)；Tab 1 "Class-cond. (Day 10)" 行是 0.46 / 0.51。两套数字都和 §2 的 "0.32-0.45" 对不上。为什么重要：§2 低估了 class-Mondrian 的 gap，反而会削弱 §1 引用 H2H 的 13.4× 比值（如果 class-Mondrian gap 真的只有 0.45，13.4× = 0.060，是数据；但 §2 写得保守反而让 reviewer 怀疑数字源）。建议：§2 改为 "gap $0.46$--$0.51$"（用 Tab 1 single-seed）或 "gap $0.42$--$0.80$"（用 §6.2 H2H bootstrap mean），二选一并标明出处。

---

Found 2 issues: 0 AC-, 0 UC-, 0 TI-, 2 NC-.
