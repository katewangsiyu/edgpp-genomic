# T3 Proof Sketch — SUPERSEDED

**This document is superseded by `theory/t3_formal_proof.md` (Day 16, 2026-04-17).**

The formal document includes:
- **§4 T3 (exact)** — bin-conditional coverage under A1' + A2-cell with upper+lower bound $\pm 1/(n_{kb}+1)$
- **§5 T3' (robust)** — Barber 2023 Thm 2 applied per-cell; coverage gap ≤ within-cell chrom-TV
- **§6 T3-loc** — feature-ball sandwich via $L$-Lipschitz $\hat\sigma$, with explicit resolution $r < \Delta / (2L)$
- **§7 T3.b** — $\hat\sigma$ perturbation bound $\leq 2\eta\bar\sigma/\Delta + 1/(n_{\min}+1)$

The corrections over the sketch (Day 13) are listed in `t3_formal_proof.md` §10 change log — notably:
- §4 Step 2–3 fixed: exact proof now uses A1' + A2-cell cleanly, no Barber 2023 smuggled in
- T3' factored out as the robust cousin (Barber 2023 Thm 2, no factor of 2)
- T3-loc feature-space translation with the additional A3-loc assumption made honest
- T3.b perturbation formalized

Reason for keeping this stub: external references (e.g. `t1_t2_formal_proofs.md` §8 summary table, `formulation_v0.md`) still link here.

---

If you are looking for the **current T3 content**, open `theory/t3_formal_proof.md`.
