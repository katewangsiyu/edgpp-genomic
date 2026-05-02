# Logic Verification Report — FINAL (Q-phase)

**Paper:** main.tex (NeurIPS 2026 main-track submission, conformalized heteroscedastic VEP)
**Date:** 2026-05-02 (post Q-phase polish)
**Status:** all 4 issue categories scanned; 3 issues found and fixed in this pass

## Part 1 — Argument Chain View

| Section | Primary Claim | Connects To | Status |
|---------|--------------|-------------|--------|
| Abstract | HCCP is the only single-fold partition we evaluated holding cov$\|Y{=}1\|\geq 0.85$ + $\shat$-bin gap $\leq 0.20$; tight finite-sample $O(n^{-1/2})$ rate within equi-bin Mondrian-$K$ family with $\pi_{\min}^{-1/2}$ explicit | Introduction | Connected |
| Introduction (§1) | VEP needs joint class-cond + bin-local CP under imbalance + heteroscedasticity; HCCP achieves it via $(y \times \shat\text{-bin})$ Mondrian; T5.1 + T5.2 prove tight rate within the family; T3$'$ certifies under exchangeability violations | Problem formulation | Connected |
| Problem formulation (§3) | Defines $\D$, A1/A1$'$/A2-cell/A-SL assumptions; KS audit gives 46.2% Mendelian / 10.4% Complex rejection; Mondrian prediction set + small-cell T2 fallback | Method (§4) / Theory (§5) | Connected |
| Method (§4) | HCCP three-stage pipeline; joint $(y \times b)$ partition is unique refinement admitting both T2 and T3; $\hat K(c_{\mathrm{outer}})$ via proper nested chrom-LOO | Theory (§5) | Connected |
| Theory (§5) | T3 exact bin-conditional under A1$'$ + A2-cell; T3$'$ robust certificate under A1$'$ alone with $-\bar\delta_{\mathrm{TV}}$ slack; T5.1 oracle $K^\star$ + T5.2 within-class tight $\Omega(n^{-1/2})$ on equi-bin Mondrian-$K$ class | Experiments (§6) | Connected |
| Experiments (§6) | Tab 1 main: HCCP marginal-bin gap 0.020 Mendelian / 0.004 Complex; Tab h2h cell-level worst-(k,b) gap 0.173 / 0.060; only single-fold partition holding both targets; M$\to$C documented as joint-failure; ProteinGym 27/50 within $\pm 0.02$ | Discussion (§7) | Connected |
| Discussion (§7) | 78% singletons at $\alpha=0.10$ Mendelian; A-SL load-bearing only for T5; explicit limitations (TV proxy, $K^\star$ asymptotic, DEGU port, ProteinGym outliers); broader impact + ACMG positioning | Conclusion (§8) | Connected |
| Conclusion (§8) | Restates three contributions: tight rate within Mondrian-$K$ class with $\pi_{\min}^{-1/2}$ explicit; HCCP framework + T3$'$; TraitGym + ProteinGym + synthetic validation with base-predictor decoupling | — | Terminal |

## Part 2 — Categorized Issue List

### Argument Chain Gaps

No issues identified in this category.

### Unsupported Claims

**Issue UC-1: §6.5 "within 1.4 std of Barber bound" — std not reported**
**Section:** [Experiments §6.5 failure mode], [App C.6 Tab crossdataset]
**Problem:** §6.5 claimed "Marginal coverage drops to 0.738, within 1.4 std of the Barber 2023 Thm 2 marginal-coverage proxy bound (0.742)". The Tab crossdataset (App C.6) reports point estimates (0.738 / 0.742) with no std. The "1.4 std" qualifier was unsupported.
**Why this matters:** Reviewers will look for the std backing in App C and find none; the "1.4 std" framing implies a statistical-significance interpretation that the table does not warrant.
**Suggestion:** Report the numeric distance directly without invoking std.
**Resolution:** ✓ Q-phase fix applied — "within 1.4 std of" → "only 0.004 below"
> **[Chinese]** 问题：§6.5 用 "within 1.4 std" 描述 M→C marginal coverage 与 T4 bound 的接近度, 但 std 没在 Tab crossdataset 报告。为什么重要：reviewer 找 std backing 找不到, "1.4 std" 隐含统计显著性解读但表无支撑。建议：改 "only 0.004 below"。Q 阶段已修。

### Terminology Inconsistencies

No issues identified in this category.

### Number Contradictions

**Issue NC-1: Fig bootstrap_density caption mixed two different metrics**
**Section:** [App C.7 Fig bootstrap_density caption, line 322]
**Problem:** Caption stated "Bootstrap distribution of HCCP $\shat$-bin gap... the all-chrom point estimate $0.100$ sits on the right tail of the distribution, while the bootstrap mean $0.020$ is closer to the centre". The "0.100" is the per-chrom marginal-coverage gap (max over chromosomes of |cov - 0.90|, chr6 in Tab perchrom), but the caption's distribution context is HCCP $\shat$-bin gap (marginal-bin metric, Tab 1 = 0.020 ± 0.012). Two different metrics conflated in a single descriptive sentence.
**Why this matters:** Reviewer reads caption, looks at Fig, then cross-references Tab 1 (0.020) vs Tab perchrom (0.100); the inconsistency reads as either a typo or a deeper protocol confusion.
**Suggestion:** Use a single consistent metric in the caption — either σ̂-bin gap (marginal-bin), with single-seed point estimate 0.041 from Tab degu, or per-chrom marginal-coverage gap, with the bootstrap mean from a separately defined chrom-level distribution.
**Resolution:** ✓ Q-phase fix applied — "all-chrom point estimate 0.100" → "single-seed all-chrom point estimate 0.041 (Tab degu)"; chr20 → chr9 (Tab perchrom chr20 |gap| = 0)
> **[Chinese]** 问题：Fig bootstrap_density caption 把 chr-level marginal-coverage gap (0.100) 与 σ̂-bin gap distribution (bootstrap mean 0.020) 两个不同 metric 混入一句。为什么重要：reviewer 看 caption + Fig + 翻 Tab 1/Tab perchrom 会发现数字不对应。建议：metric 统一为 σ̂-bin gap (Tab degu single-seed 0.041 vs Tab 1 bootstrap mean 0.020). Q 阶段已修。

**Issue NC-2: Fig bootstrap_density caption listed chr20 as high-gap**
**Section:** [App C.7 Fig bootstrap_density caption]
**Problem:** Caption listed "high-gap chromosomes (chr6 / chr19 / chr20, see Tab.~\ref{tab:perchrom})". But Tab perchrom shows chr20 cov = 0.900, |gap| = 0.000 — chr20 hits the target exactly, not a "high-gap" chromosome. Top-3 under-coverers are chr6 (0.100), chr19 (0.073), chr9 (0.058).
**Why this matters:** A reviewer cross-checking the caption against Tab perchrom finds an immediate contradiction. This is a textbook stale data leak from Phase 4 (before Tab perchrom was finalized).
**Suggestion:** Replace "chr20" with "chr9" to match Tab perchrom data.
**Resolution:** ✓ Q-phase fix applied — "chr6 / chr19 / chr20" → "chr6, chr19, chr9"
> **[Chinese]** 问题：Fig bootstrap_density caption 列 'chr20' 为 high-gap, 但 Tab perchrom chr20 cov=0.900 |gap|=0 是完美 target. 为什么重要：reviewer 一对照 Tab perchrom 即发现矛盾, 是 Phase 4 之前 stale 数据漏修。建议：'chr20' → 'chr9' (Tab perchrom 第三大 |gap|). Q 阶段已修。

---

## Cross-section number consistency final verification

All headline numbers single-source verified after Q-phase:

| Number | Single source | Verified appearances |
|---|---|---|
| chr6 max per-chrom gap = 0.100 | Tab perchrom (chr6 cov 0.800) | §1 Fig 1, §5 T3', §6.4 OOD, §A T3' empirical, App B KS interp, App C Tab perchrom + Fig 9 + Fig bootstrap_density (after Q) |
| Mendelian σ̂-bin gap (marginal, K=5) = 0.020 ± 0.012 | Tab 1 (§6.1) | §6.1 narrative, App C Fig bootstrap_density (after Q), App E Fig caption + interpretation (after P) |
| Mendelian σ̂-bin gap (cell-level worst-(k,b), K_eval=3) = 0.173 ± 0.192 | Tab h2h (App C.1) | abstract (implied via "≤ 0.20"), §6.1 mention, §6.2 H2H, §8 conclusion |
| Complex σ̂-bin gap (marginal, K=5) = 0.004 ± 0.002 | Tab 1 | §6.1, App C Fig bootstrap_density |
| Complex σ̂-bin gap (cell-level worst-(k,b), K_eval=5) = 0.060 ± 0.031 | Tab h2h | §6.1 mention, §6.2 H2H, §8 conclusion |
| KS audit Mendelian rejection rate = 46.2% / max KS = 0.434 | Tab ks_audit | §3 setup, §A T3' empirical, App B (multiple) |
| KS audit Complex rejection = 10.4% / max KS = 0.315 | Tab ks_audit | §3 setup, App B |
| M→C marginal coverage = 0.738 / T4 bound = 0.742 / distance = 0.004 | Tab crossdataset | §6.5 failure mode (after Q-phase 1.4-std → 0.004) |
| ProteinGym mean cov = 0.887 / 27/50 within ±0.02 / 44/50 within ±0.05 / 6/50 outlier below 0.85 | Tab proteingym | §1 C3 + §6.4, App E (multiple) |
| Mendelian AUPRC: 0.752 (LogReg) / 0.900 (GBM) / +14.8 pp | Tab 1 | abstract (implied via Tab 1), §6.1, App C tab:hccp_logreg |
| π_min ≈ 0.10 = TraitGym minority prevalence | TraitGym dataset definition | abstract S2, §1 + Tab imbsweep π_+=0.10 row |

**Consistency:** ✓ all numbers single-sourced and cross-cited correctly.

---

Found 3 issues (after P-phase deeper audit):
- 0 AC-
- 1 UC-1 (§6.5 "1.4 std" — fixed)
- 0 TI-
- 2 NC-1, NC-2 (Fig bootstrap_density 2 issues — fixed)

All resolved in Q-phase commit.
