# Phase 5: Synthesis — Taxonomy of Conditional/Heteroscedastic CP for VEP

Cross-paper analysis built on Phase 3 deep reads + Phase 4 code landscape.

## 5.1 Taxonomy of post-2024 conditional CP methods

The frontier has **converged on three orthogonal axes** for handling conditional coverage:

| Axis | Mechanism | Representative methods | Strengths | Weaknesses |
|---|---|---|---|---|
| **(A) Group structure** | Static, pre-specified subsets | Kandinsky CP (Bairaktari/Wu/Wu, ICML 2025); CP with Conditional Guarantees (Gibbs/Cherian/Candès) | Clean theory, minimax bounds | Requires groups known a priori; data-dependent partitions break the framework |
| **(B) Density reweighting** | Soft, continuous weights via Π̂(Y\|X) | CP²-HPD (Plassier et al., ICLR 2025); Probabilistic CP | Adapts to multimodal/heteroscedastic continuously | Depends on Π̂ estimator quality (the d_TV term) |
| **(C) Partition learning** | Data-dependent bins (e.g., σ̂-bins) | **HCCP (ours)**; equi-bin Mondrian variants | Exact finite-sample coverage per bin; matches σ̂(x) heteroscedasticity | Boundary artifacts; K-selection nontrivial |

HCCP sits in axis (C). Kandinsky covers (A) but not (C). CP²-HPD covers (B) but not (C). **No concurrent method covers (A) ∩ (C) simultaneously** — this is HCCP's structural niche.

## 5.2 Comparative table — concurrent threats

| Method | Venue | Rate | Dim-free | Heterosc | Cls imbal | Lower bd | Mondrian | Has code |
|---|---|---|---|---|---|---|---|---|
| **Kandinsky CP** | ICML 25 | O(√(d/n)) | ❌ | ❌ | ❌ | ❌ | static-only | ❌ |
| **Yao et al. (Non-Asymp)** | ICLR 26 | O(n^{-1/2}) | ✅ | implicit | ❌ | ❌ | ❌ | ❌ |
| **RC3P (class-rank)** | NeurIPS 24 | finite-sample | n/a | ❌ | ❌ | ❌ | class-only | ✅ |
| **OT-CP** | ICML 25 | O(n^{-1}) ties | n/a | ✅ (k-NN) | ❌ | ❌ | ❌ | ⚠ minimal |
| **Backward CP** | NeurIPS 25 | O(n^{-1/2}) LOO | n/a | implicit (entropy) | ❌ | ❌ | ❌ | ✅ |
| **Length-Opt CP** | NeurIPS 24 | O(n^{-1/2}) | partial | implicit | ❌ | ❌ | ❌ | ⚠ minimal |
| **CP²-HPD** | ICLR 25 | O(√(log n / n + r_n)) | ✅ | ✅ (density) | ❌ | ❌ | ❌ critique | ❌ |
| **HCCP (ours)** | NeurIPS 27 (target) | **O(n^{-1/2})** | ✅ | ✅ explicit σ̂(x) | ✅ π_min | ✅ T5.2 | ✅ data-dep | ✅ (planned crepes ext) |

Bold cells: HCCP-unique entries.

## 5.3 Theoretical map — three rate frontiers

The post-2024 rate analyses cluster into three "rate frontiers":

**Frontier-1 (Forward split-CP linear-quantile, Yao 2026)**:
$$O\big(n^{-1/2} + (\sigma^2 n)^{-1} + m^{-1/2} + e^{-\sigma^2 m}\big), \text{ dim-free}$$

**Frontier-2 (Group-cond static-partition VC, Kandinsky 2025)**:
$$O\big(\|\beta\|_1 \sqrt{d/n}\big), \text{ d = VC-dim of weight basis}$$

**Frontier-3 (Equi-bin Mondrian-K class-cond heteroscedastic, HCCP T5.1)**:
$$O\big(L_F R \pi_{\min}^{-1/2} n^{-1/2}\big), \text{ dim-free, } K^* = \lfloor\sqrt{L_F R \pi_{\min} n}\rfloor$$

**Cross-frontier observation**: All three converge to O(n^{-1/2}) — this is the new headline rate for non-asymptotic CP analysis. **Differentiation must be at the constant level + class restriction level**, not the rate level. HCCP T5.2 lower bound *within equi-bin Mondrian-K family* is what makes our contribution honest — we don't claim universal optimality, only optimality within a well-defined family.

## 5.4 Empirical positioning — where HCCP wins / risks losing

**Strong wins (Phase 3 confirmed)**:
1. **VEP / TraitGym benchmark** — DeepWAS doesn't compete (different benchmark); only our paper engages TraitGym + AUPRC_by_chrom_weighted + chrom-LOO + class-cond on imbalanced binary
2. **Heteroscedastic + class-imbalanced joint regime** — Kandinsky/Yao/Length-Opt all miss this combo
3. **T5.2 lower bound** — no concurrent paper has matching lower bound

**Empirical risks (Phase 3 flagged)**:
1. **CP²-HPD (ICLR 2025)** — soft density-ratio reweighting could outperform discrete σ̂-bin Mondrian on continuous heteroscedasticity. **Mitigation**: H2H on TraitGym Mendelian + Complex (we have the infrastructure). Need to add to §6.3 if reviewer asks.
2. **Backward CP** — for clinical triage framing, fixed-size sets are more actionable. **Mitigation**: position as complementary (§6.5 ProteinGym + §7 future-work hook to "backward HCCP")

**Theory risks (Phase 3 flagged)**:
1. **Yao 2026 headline-rate collision** — both achieve dim-free O(n^{-1/2}). **Mitigation**: §2 / §6.4 framing: "Yao prove O(n^{-1/2}) for split-CP linear quantile reg; we prove same rate for equi-bin Mondrian classification with explicit π_min handling — methodologically distinct and non-subsumed."
2. **Kandinsky lower-bound expansion in v2** — if a concurrent v2 adds matching lower bounds, our T5.2 niche shrinks. **Mitigation**: ship NeurIPS 2027 sub before any v2; our "equi-bin Mondrian-K family" qualifier remains protective.

## 5.5 What the field is missing (gap analysis)

See `gaps.md` for the 5 identified research gaps and the 1-2 directly addressed by HCCP.

## 5.6 Recommendation for HCCP positioning at NeurIPS 2027

**Frame as Frontier-3 + the operational instantiation**:
- §1 contribution: "We identify the equi-bin Mondrian-K class-conditional heteroscedastic CP family as a structurally distinct frontier (vs static-group Kandinsky; vs split-CP Yao); within this family we prove an oracle K\* with dim-free O(n^{-1/2}) rate and matching lower bound."
- §2 / §6.4 must explicitly reference Yao 2026 + Kandinsky 2025 + CP²-HPD 2025 as concurrent → HCCP positioning
- §7 "Future work" must mention Backward CP as natural extension (size-bounded HCCP for clinical triage)
- §8 contributions order: T5.1+T5.2 theory hero, then operational binding T3', then VEP empirics — **theory-first** to differentiate from DEGU's empirics-first npj AI 2026 paper

✅ **Phase 5 synthesis complete.**
