# Phase 5: Gap Analysis

Five research gaps identified by Phase 3 deep reads + Phase 4 code survey. Two are directly addressed by HCCP; three are open for future work.

## Gap 1 — Heteroscedastic + class-imbalanced joint regime (ADDRESSED by HCCP)

**Observation**. No concurrent paper handles **explicit σ̂(x) heteroscedasticity AND π_min class imbalance simultaneously**. Yao 2026 is regression-only (no class imbalance). Kandinsky 2025 is classification but σ̂-agnostic and π_min-agnostic. RC3P is class-Mondrian but σ̂-agnostic. CP²-HPD is heteroscedastic-aware but lacks class-cond + lower bound.

**HCCP contribution**: Equi-bin Mondrian-K stratified by both class label y and σ̂(x)-bin. T5.1 explicitly shows π_min^{-1/2} factor in the rate constant.

## Gap 2 — Lower bounds for partition-based class-cond CP (ADDRESSED by HCCP)

**Observation**. Yao 2026, Length-Opt CP, OT-CP, RC3P, Kandinsky CP **all give upper bounds only**. Kandinsky's matching lower bound exists for static groups, but the proof framework does not extend to data-dependent partitions or σ̂-bins.

**HCCP contribution**: T5.2 matching lower bound *within the equi-bin Mondrian-K family*. The qualifier matters — we do not claim universal optimality, only optimality within a structurally distinct family.

## Gap 3 — Backward CP × class-conditional (OPEN, future work)

**Observation**. Backward CP (Gauthier/Bach/Jordan, NeurIPS 2025) is currently **marginal-only**. Authors note class-cond extension is feasible but not done. For VEP triage, class-cond fixed-size sets are highly actionable (e.g., per-class size = 2: {Pathogenic, Likely Pathogenic} vs {Benign, Likely Benign}).

**Future work**: "Backward HCCP" — class-cond + σ̂-aware backward CP. Mention in §7 future-work.

## Gap 4 — Soft (density-ratio) ↔ Hard (Mondrian) hybrid (OPEN)

**Observation**. CP²-HPD uses continuous density-ratio reweighting; HCCP uses discrete σ̂-bins. There is **no published method that combines both** — e.g., density-ratio within bins for smooth boundary handling. Such a hybrid could capture continuous heteroscedasticity AND retain finite-sample bin-level coverage.

**Future work**: Hybrid HCCP-CP² — soft reweighting within hard Mondrian bins. Could be a follow-up paper after NeurIPS 2027.

## Gap 5 — Public reproducible code for theory-only papers (OBSERVATIONAL)

**Observation**. Two of three top theoretical threats (Yao 2026, Kandinsky 2025) **have no public code**. Only OT-CP, RC3P, Length-Opt, Backward CP, DeepWAS released code. CP²-HPD also has none.

**Implication**: HCCP's NeurIPS 2027 submission can lean on **empirical reproducibility as a differentiator**. Plan to ship:
- Standalone `crepes` extension implementing HCCP
- Snakemake module integrating with TraitGym repo
- All `R_raw/cp_baselines_h2h/`, `R_raw/asl_audit/`, `R_raw/synthetic_n_sweep/` as published artifact

This addresses NeurIPS 2027 reproducibility checklist directly.

## Summary

| # | Gap | Status |
|---|---|---|
| 1 | Heterosc + class-imbal joint | ✅ ADDRESSED by HCCP T5.1 |
| 2 | Lower bound for partition class-cond CP | ✅ ADDRESSED by HCCP T5.2 |
| 3 | Backward CP × class-cond | ⏳ OPEN — §7 future work hook |
| 4 | Soft × Hard hybrid CP | ⏳ OPEN — follow-up paper |
| 5 | Reproducibility differential vs Yao/Kandinsky/CP²-HPD | 📌 EXPLOIT in submission |

✅ **Phase 5 gaps complete.**
