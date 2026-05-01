# Phase 3: Selection — 8 Papers for Deep Read

**Picked**: 8 papers (Phase 3 gate ≥8 ✓)
**Strategy**: balance 4 axes — concurrent threats, theory positioning, score lineage, application grounding
**Rationale per pick** below.

| # | Paper | Venue | Why |
|---|---|---|---|
| 1 | **Kandinsky CP: Beyond Class- and Covariate-Conditional Coverage** (Bairaktari/Wu/Wu) | ICML 2025 | **Concurrent T5 threat** — claims to subsume class-cond + Mondrian as special cases with minimax-optimal rate. Need to verify exact assumptions to defend our "equi-bin Mondrian-K class" qualifier. |
| 2 | **Non-Asymptotic Analysis of Efficiency in Conformalized Regression** | ICLR 2026 | **Concurrent T5.1/T5.2 threat** — direct rate analysis. Need to check if their bound is dimension-free, which loss, and whether it's tight on equi-bin Mondrian-K family. |
| 3 | **CP for Class-wise Coverage via Augmented Label Rank Calibration** | NeurIPS 2024 | **Direct class-conditional baseline.** Our paper's §6.3 H2H needs this as B3 (class-Mondrian) defense. |
| 4 | **Optimal Transport-based Conformal Prediction** | ICML 2025 | **Theory rival** — OT-based rate proof; might give tighter rate than ours in some settings. Need to position against §6.4 SC-CP comparison. |
| 5 | **Backward Conformal Prediction** (Gauthier/Bach/Jordan) | NeurIPS 2025 | **Adaptive coverage** — our HCCP fixes coverage and varies set size; backward fixes set size and varies coverage. Could be complementary or competitor in clinical triage framing. |
| 6 | **Length Optimization in Conformal Prediction** | NeurIPS 2024 | **Width as objective** — our T5.1/T5.2 bound G(K\*); their work makes width the explicit minimization target. Need to position relative to T5 oracle. |
| 7 | **Probabilistic CP with Approximate Conditional Validity** | ICLR 2025 | **Soft conditional approach** — uses density-ratio reweighting; alternative to our hard Mondrian partition. |
| 8 | **Training Flexible Models of Genetic Variant Effects** (ICML 2025) | ICML 2025 | **VEP grounding** — only ICML 2025 VEP paper. Need to read to position our TraitGym evaluation framing. |

## Methodology

For each: read abstract + introduction + main method/theorem + key experimental table. Prioritize **assumption boundary** (which settings does their result hold in?) and **failure modes** (where does ours win?).

Output: structured notes per paper in `deep_dive.md` covering:
1. Problem
2. Contributions (1-3 sentences)
3. Methodology (formal setup + key idea)
4. Experiments (datasets + results headline)
5. Limitations (acknowledged + ours-noted)
6. Connection to HCCP (threat / complementary / orthogonal)
