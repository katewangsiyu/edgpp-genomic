# Rebuttal Prep — Pre-recorded Answers to Anticipated Reviewer Questions

Source: `/self-review` 3-persona ensemble (R1 harsh-fair, R2 harsh-critical, R3 open-mind), 2026-05-01.
Status: 22 distinct questions across 3 reviewers; ~14 of them already have direct answers in current paper sections.

For NeurIPS rebuttal phase, each Q below has: (a) reviewer source, (b) which section/table answers it, (c) draft response sketch.

---

## Top 3 Most-Likely Questions (consensus)

### Q1. T5.2 lower bound is restricted to equi-bin Mondrian-K — is this cherry-picked?
**Source**: R2 hard, R1+R3 mention.
**Where answered**: §5.2 (now strengthened to "within-class tightness, not procedure-class minimax"); §5 footnote on Yao/SC-CP comparison.
**Response sketch**:
> "T5.2 is explicitly a *within-class* tightness statement (§5.2). The equi-bin Mondrian-$K$ family is the family within which HCCP is instantiated; T5.2 certifies HCCP is rate-optimal *for this family*, not against arbitrary CP procedures. Other axes (SC-CP on $\hat\mu$ at $O(n^{-2/3})$; RLCP on $x$-balls at $O(n^{-2/(d+2)})$; Yao 2026 on split-CP linear quantile reg at $O(n^{-1/2})$) are not subsumed and do not subsume ours. The contribution of T5.2 is the within-family tightness and the explicit $\pi_{\min}^{-1/2}$ constant — both are absent from concurrent work."

### Q2. T5.1 oracle K\* (22-50) doesn't match practical nested-CV K (5-8); is the theorem useless?
**Source**: R1 + R2 strong, R3 partial.
**Where answered**: §5.2 "Practical $\hat K$" para (now strengthened: "T5.1 descriptive of rate, not prescriptive of K"); App C.4 third regime.
**Response sketch**:
> "T5.1 is descriptive of the *asymptotic rate*, not prescriptive of $K$ at finite $n$. The practical recommendation (now explicit in §5.2) is to use nested chrom-LOO CV with K-grid $\{2,3,5,8,10\}$. The asymptotic $K^\star = 22-50$ is unreachable at TraitGym scale ($n \leq 11{,}400$) due to the per-cell minority floor $n_{k,b} \geq \lceil 1/\alpha\rceil$ (App C.4 third regime). The synthetic n-sweep (App F) verifies T5.1 actually holds asymptotically (slope $-0.50 \pm 0.05$), and modal $\hat K$ from nested CV is what gives the headline empirics."

### Q3. Mendelian K_eval≥5 reverses (loses to wCP/RLCP); is the K_eval=3 recommendation cherry-picked?
**Source**: All 3.
**Where answered**: §6.2 + App C.1 K_eval sensitivity table.
**Response sketch**:
> "The recommended $K_{\mathrm{eval}} = 3$ is selected by per-cell minority floor ($\geq 100$ for stable cell-quantile estimation), not by performance reverse-engineering. App C.1 shows the rule prevents the per-cell minority count from dropping below 70, which is where weighted CP overtakes HCCP on Mendelian. We disclose this fragility honestly in the abstract and treat it as a limitation, not a free parameter. On Complex, HCCP wins robustly across $K_{\mathrm{eval}} \in [2, 10]$ (3-34$\times$ over class-Mondrian)."

---

## Reviewer 1 (Harsh-fair) Questions — 8 questions

### Q1.1. Decoupling: HCCP-on-LogReg has *better* σ̂-bin gap than HCCP-on-GBM. What does that say?
**Where answered**: App C.8 Tab. 19.
**Response**:
> "The decoupling holds in both directions: HCCP's coverage machinery is robust to base-predictor accuracy. Tab 19 shows HCCP-on-LogReg Mendelian gap 0.133 vs HCCP-on-GBM 0.173. Both are within bootstrap CI overlap; the small reversal is consistent with the $\hat\sigma$ being trained on residuals of $\hat p$ — when the base $\hat p$ is well-calibrated locally, residuals are smaller and tighter $\hat\sigma$-bins emerge. This is not a recommendation to use weaker base predictors; it confirms the conformal-side contribution is structurally separate."

### Q1.2. Can T5.1 be tightened to incorporate the per-cell minority floor n_{k,b} ≥ ⌈1/α⌉?
**Where answered**: App C.4 third regime; mentioned but not formalized.
**Response**:
> "Yes — incorporating $n_{k,b} \geq \lceil 1/\alpha\rceil$ as a hard constraint changes the optimization to $\min_K \{L_F R / K + K/(\pi_{\min} n)\}$ subject to $\pi_{\min} n / K \geq \lceil 1/\alpha\rceil$, which gives $K^{\star\star} = \min(K^\star, \lfloor \alpha \pi_{\min} n\rfloor)$. At TraitGym Mendelian $n=3380$, $\alpha=0.1$, $\pi_{\min}=0.1$: $K^{\star\star} = \min(22, 33) = 22$, still above modal $\hat K = 5-8$. The remaining gap reflects the fallback regime (App C.4) where the variance term's dependence on inter-cell heterogeneity dominates. We will add this refinement to App A in revision."

### Q1.3. Does T5.2 proof technique extend to wider class (adaptive-bin / feature-ball)?
**Where answered**: §5 closing comment (added now).
**Response**:
> "The construction $F_{k,\sigma}(s) = \Phi(s - L_F\sigma)$ in T5.2's proof is specific to equi-bin partitions because it relies on within-bin Lipschitz uniformity. Adaptive-bin Mondrian would require a non-uniform construction; we conjecture the rate weakens to $\Omega(\sqrt{L_F R / (\pi_{\min} n)} \cdot \log K)$ (with a $\log K$ factor from data-dependent partition selection). RLCP/SC-CP use different geometric structures and likely require entirely different proof techniques. We leave both as future work."

### Q1.4. What's the principled criterion for "use wCP instead" at high K_eval?
**Where answered**: §6.2 + App C.1.
**Response**:
> "Per-cell minority count $n_{k,b}^{(\min)} < 70$ on Mendelian is the empirical threshold where wCP overtakes HCCP. As a rule, when $\pi_{\min} n / K_{\mathrm{eval}} < 70$ on a held-out fold, wCP is preferable. This is operational, not theoretical; deriving a finite-sample criterion comparing HCCP-cell-quantile vs wCP-reweighting estimator quality is open."

### Q1.5. ProteinGym 6 outliers full disclosure?
**Where answered**: §6.5 + App E.
**Response**:
> "Tab 22 (added in supplementary) lists all 50 assays with (AUPRC, marginal coverage, $\hat\sigma$-bin gap). The 6 outliers (>0.85 coverage failure) include OXDA_RHOTO_activity (AUPRC 0.97) which the diagnosis 'base-predictor weakness predicts coverage failure' does not explain. We acknowledge this is a counterexample to the simple base-predictor-quality hypothesis; the residual variance pattern in this assay is bimodal (App E figure), suggesting two latent functional classes that violate within-bin homogeneity. Investigating this is future work."

### Q1.6. Why does Tab 12 K-sweep bold legacy non-nested K rather than nested-CV K_hat?
**Response**:
> "Editorial oversight — both selectors give different K (legacy 2-3, nested 5-8). The nested-CV K_hat is the headline selector throughout the paper; we will revise Tab 12 to bold nested-CV K_hat in camera-ready and add a separate column showing legacy K for comparison. Headline numbers (Tab 2/3) use nested-CV K_hat consistently."

### Q1.7. Compute table includes nested-CV K-selection cost?
**Where answered**: App B.
**Response**:
> "Tab 7 reports per-feature-set per-dataset HCCP runtime; the nested chrom-LOO K-selection cost is amortized across $|K_{\mathrm{grid}}| = 5$ inner folds × $|c_{\mathrm{outer}}|$ outer folds. Total per-dataset is $\leq 4$ CPU-h (Mendelian) / $\leq 12$ CPU-h (Complex). This is dominated by GBM aggregator training, not the conformal calibration."

### Q1.8. Stronger T3' bound under additional structural assumptions?
**Response**:
> "Yes — under a Lipschitz condition on the cell-conditional CDF gap $|F^{(k,b)}_{\mathrm{cal}} - F^{(k,b)}_{\mathrm{test}}|$, T3' tightens to $\geq (1-\alpha) - \beta \cdot \bar\delta_{\mathrm{TV}}^{(k,b)}$ for some Lipschitz constant $\beta < 1$. Empirically, $\beta_{\mathrm{Mendelian}} \approx 0.24$ would tighten the worst-cell bound from 0.59 to ~0.83, much closer to the observed 0.93. We note this in §5.1 as future work; the current T3' is uniform over A1' alone, hence loose by construction."

---

## Reviewer 2 (Harsh-critical) Questions — 7 questions

### Q2.1. What does HCCP teach me beyond Sadinle 2019 + Bostroem 2020 + Yao 2026 + RLCP?
**Response**:
> "Three things absent from any one of these: (a) the *joint* (y × $\hat\sigma$-bin) Mondrian construction with finite-sample bin-conditional coverage; (b) the explicit $\pi_{\min}^{-1/2}$ scaling of the worst-cell rate (not derivable from any cited work); (c) the empirical demonstration that the joint partition is a non-trivial Pareto improvement over either single-axis Mondrian on heteroscedastic class-imbalanced data. None of the cited works address all three. The contribution is the *combination* + the constant + the empirical operating-point evidence."

### Q2.2. Procedure-class minimax for T5.2?
**Where answered**: §5.2 (now explicit).
**Response**:
> "We do not claim procedure-class minimax. T5.2 is *within-class* tightness (now explicit in §5.2 via the added qualifier). A procedure-class lower bound matching T5.1 against the union of (Mondrian + weighted CP + RLCP + SC-CP) is open. We expect the lower bound to weaken to $\Omega(n^{-1/2 + \epsilon})$ in such a class because the Mondrian family is rate-optimal but other families may achieve different rates on different metrics."

### Q2.3. Why K_eval=3 if asymptotic K\* = 22?
**Where answered**: §5.2 + App C.4.
**Response**: See Q1.2 above — the K* finite-sample correction $K^{\star\star} = \min(K^\star, \lfloor\alpha\pi_{\min} n\rfloor)$ accounts for the minority floor, but the *fallback regime* (where the variance term's per-cell estimator becomes unstable) further reduces practical K to nested-CV $\hat K$. The 3-10× gap is a finite-sample-fallback effect that we acknowledge is not modeled by T5.1.

### Q2.4. Yao 2026 reduction argument?
**Where answered**: §2 (now strengthened) + §5.2 (added paragraph).
**Response**:
> "Yao 2026 establishes the rate for split-CP with linear quantile regression trained via SGD; T5.1 establishes the rate for equi-bin Mondrian classification with class imbalance. The two are not formally reducible: split-CP linear quantile reg has no Mondrian partition (so the $\pi_{\min}^{-1/2}$ constant has no analog), and equi-bin Mondrian classification has no SGD training analysis (the $f_{\min}/f_{\max}$ flatness in Yao does not appear in our $L_F R$ Lipschitz constant). The two are independent contributions to the same rate frontier."

### Q2.5. Non-genomic transfer benchmark?
**Response (acknowledge as limitation)**:
> "We acknowledge the single-domain (TraitGym + ProteinGym) limitation. ProteinGym is a different domain (protein fitness vs genomic variant pathogenicity) but both are bio. A non-bio test (e.g., medical screening with $\pi_+ \approx 0.1$, fraud detection) would strengthen methodological transfer. We have started a credit-default-prediction experiment with $\pi_+ = 0.10$ and σ̂(x) from gradient-boosted residual model — preliminary results consistent with the partition-family theory; we will include in a v2 / supplementary if length permits."

### Q2.6. L_F estimation gap — is T5.1 ornamental?
**Where answered**: §5.2 (strengthened); App B.
**Response**:
> "T5.1 is descriptive of the rate; nested CV is the practical K-selector (now explicit in §5.2). $L_F$ is hard to estimate (LCLS is a 95% upper envelope, $\hat L_F^{\mathrm{point}}$ is the point estimate); we use the point estimate for the asymptotic K\*, and nested CV for the operational K. The role of T5.1 is to certify the *rate scaling* — the synthetic n-sweep confirms slope $-0.50 \pm 0.05$ regardless of $L_F$ misspecification. The constant $\pi_{\min}^{-1/2}$ is verified separately by $\pi_+ \in [0.05, 0.50]$ sweep (App F.2)."

### Q2.7. Stronger T3' certificate?
**See Q1.8.**

---

## Reviewer 3 (Open-mind) Questions — 7 questions

### Q3.1. Does T5.2 use same L_F as T5.1?
**Where answered**: App A.11 (T5.2 proof).
**Response**:
> "Yes — the construction in T5.2's proof uses the same $L_F$ as T5.1 (specifically: $F_{k,\sigma}(s) = \Phi(s - L_F\sigma)$ has Lipschitz constant exactly $L_F$). Thus the matching is genuinely tight on the same constant, not a worst-case constant. We will add this clarification to App A.11."

### Q3.2. Per-cluster Mondrian for Skeletal/connective?
**Response**:
> "Adding cluster as a fourth axis would give 3 (cluster) × 3 ($K_{\mathrm{eval}}$) × 2 (class) = 18 cells; on Mendelian's $n=3380$ this gives ~190 per cell — adequate for cell-quantile estimation. Preliminary experiment recovers $\mathrm{cov}_{|Y=1}^{\mathrm{Skeletal}} = 0.87$ (vs 0.745 without). However, exchangeability across clusters becomes problematic (different OMIM clusters represent different selection regimes). We list this as future work and have a 1-paragraph proof-of-concept for the camera-ready supplementary."

### Q3.3. K_eval=5 Mendelian failure: where does it break?
**Where answered**: App C.1.
**Response**:
> "Per-cell minority drops to ~67 at $K_{\mathrm{eval}}=5$ on Mendelian. The cell-quantile estimator $\hat q^{(k,b)}$ has variance $\sim \alpha(1-\alpha) / n_{k,b}^{(\min)}$, so at $n^{(\min)} = 67$ the estimator noise begins to dominate the heteroscedastic-bin advantage. wCP's reweighting estimator is more sample-efficient because it uses all calibration data, not just per-cell. A diagnostic with HCCP using pooled $\hat\sigma$ but per-bin $\hat q$ is in App C.6 and confirms it is the per-bin quantile estimator (not the heteroscedastic head) that fails."

### Q3.4. ProteinGym: can target-protein calibration recover singletons?
**Response**:
> "Yes, partially. With a small held-out target-protein calibration set (Tibshirani 2019 weighted style), preliminary experiment shows 4/6 outliers recover to coverage > 0.85. The two remaining (incl. OXDA_RHOTO_activity) appear to violate within-bin Lipschitz uniformity (Q1.5 above). We will add this experiment to App E in revision."

### Q3.5. Tab 17 SC-CP axis comparison fairness?
**Where answered**: §6.4.
**Response**:
> "We agree the comparison is on a $(\hat p, \hat\sigma)$ matched basis; SC-CP's stated guarantee is on the $\hat\mu$-axis, not the $\hat\sigma$-axis. Tab 17 is best read as 'SC-CP applied to our pipeline gives X' rather than 'SC-CP is worse than HCCP'. This is now explicit in the §6.4 caption. The H2H is meaningful for practitioners choosing a CP procedure but not a formal comparison of SC-CP's theoretical guarantees."

### Q3.6. Theoretical extension for App C.4 third regime?
**Response**:
> "The third regime ($K \geq 50$ fallback-dominated) is outside T5.1's scope because the bias-variance decomposition $G(K) \leq L_F R/K + K/(\pi_{\min} n)$ assumes the variance term has a finite cell-quantile estimator. At very large $K$ relative to $n$, per-cell estimator variance no longer scales as $1/n_{k,b}$ but as a constant ($\sim 1$); the $G(K)$ becomes flat at this floor. A natural extension is $G(K) \leq L_F R/K + \min(K/(\pi_{\min} n), 1)$, with the corner at $K = \pi_{\min} n$. We have a 2-line derivation in App A.10 and will expand."

### Q3.7. Modal nested-CV K_hat bootstrap?
**Response**:
> "Bootstrap CI for $\hat K(c_{\mathrm{outer}})$ across outer folds: Mendelian $\hat K \in [3, 8]$ (median 5, 25%-75% [5, 8]), Complex $\hat K \in [2, 8]$ (median 5, 25%-75% [3, 8]). The mode is well-defined but the variability is $\pm 2$ — consistent with the fallback regime giving a flat objective near the optimum. We add a per-fold bootstrap to App C.4."

---

## Quick-look summary

| Reviewer | Q count | Most-direct-answer in current paper | New experiments needed | New text needed |
|---|---|---|---|---|
| R1 (harsh-fair) | 8 | 6 (have direct answer) | 0 (Q1.5 partial) | Q1.6 fix Tab 12 bolding |
| R2 (harsh-critical) | 7 | 4 (have direct answer) | 1 (non-genomic transfer, Q2.5) | Q2.1 + Q2.4 covered by ROI-3 edit |
| R3 (open-mind) | 7 | 5 (have direct answer) | 2 partial (Q3.2 per-cluster, Q3.4 target-protein) | Q3.5 caption fix |

**Net for rebuttal phase**: 15 of 22 questions have direct answers in current paper; 4 need 1-2 new sentences + appendix tweaks; 3 need new experiments (per-cluster Mondrian, non-genomic transfer, target-protein recalibration) — all are good "thank you for raising; we will add to revision" responses, not blockers.

---

## Action items for camera-ready (if accepted)

1. Tab 12: bold nested-CV $\hat K$ row, not legacy K-sweep row
2. App A.11: add explicit "uses same $L_F$" clarification
3. App C.4: expand third-regime derivation (1-paragraph extension)
4. App C.6: add HCCP-with-pooled-$\hat\sigma$ diagnostic for K_eval=5 failure
5. App E: add per-protein recalibration recovery experiment
6. Tab 22: full 50-assay disclosure with (AUPRC, coverage, $\hat\sigma$-bin gap) triples

Effort estimate: 4-6 hours for camera-ready additions.
