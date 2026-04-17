# Theorem Roadmap — Path A

**目的**：列出每个定理的**期望形式、证明工具、预估难度、时间排期**。作为理论工作的 Gantt chart。

---

## T1 — Marginal Coverage

**形式**（final form）:
$$
P\left( Y \in \mathcal{C}_\alpha(X) \right) \geq 1 - \alpha
$$

under Assumption A1 (chrom-group exchangeability) + standard split-conformal calibration.

**证明工具**：Barber et al 2023 Theorem 1 的直接套用。核心是把他们的 "i.i.d." 替换为 "chrom-conditional exchangeability"。

**难度**：★☆☆☆☆

**时间**：1 周

**风险**：低。几乎是 textbook result。

**写作长度**：1 页（正文）+ 0.5 页（证明 in appendix）

---

## T2 — Class-Conditional Coverage

**形式**:
$$
P\left( Y \in \mathcal{C}_\alpha(X) \mid Y = k \right) \geq 1 - \alpha, \quad \forall k \in \{0, 1\}
$$

**证明工具**：Vovk 2003 Mondrian CM。按 $Y$ 分层 → 每个类是独立的 Mondrian taxon。

**前置条件**：每个类别 calibration set 非空（$n_k \geq \lceil 1/\alpha \rceil$）。

**难度**：★★☆☆☆

**时间**：1 周

**风险**：低。standard Mondrian argument。

**写作长度**：1 页 + 0.5 页

---

## T3 — Local (Feature-Neighborhood) Coverage ★★★ MAIN NOVELTY

**状态更新（Day 16, 2026-04-17）**: T3 核心 + T3' robust + T3-loc 特征空间 + T3.b σ̂ 扰动 已 appendix-ready（`theory/t3_formal_proof.md`）。剩余理论工作是 T3.a oracle-asymptotic 版（下文，Month 5–7）。

### T3 (finite-sample bin-conditional) — ✅ DONE

**形式**:
$$
\Bigl|P\!\left(Y \in \mathcal{C}_\alpha(X) \mid Y = k,\; b(X) = b\right) - (1-\alpha)\Bigr| \;\leq\; \frac{1}{n_{kb}+1}
$$

**工具**：Vovk 2003 Mondrian CM 的直接套用（with taxonomy $\kappa(X,Y) = (Y, b(X))$）。对 A1 only 情况用 Barber 2023 Thm 2 per-cell 给 robust 版。

**位置**：`theory/t3_formal_proof.md` §4 (exact), §5 (robust), §6 (feature-space), §7 (perturbation).

**难度**：★★☆☆☆（比之前 roadmap 估计简单 — 关键洞察：σ̂-bin 是合法 Mondrian taxon，不需要 CQR-style 局部 quantile 推导）。

### T3.a — Oracle asymptotic version（未做，Month 5–7）

**形式**（assuming $\hat{\sigma} = \sigma_{\text{true}}$）:
$$
\left| P\left( Y \in \mathcal{C}_\alpha(X) \mid X \in B(x_0, r) \right) - (1 - \alpha) \right| \leq \epsilon_1(r, \alpha)
$$

where $\epsilon_1(r, \alpha) \to 0$ as $r \to 0$ (at known rate).

**证明工具**：
- Romano et al 2019 CQR §3 已经做了 CQR 的 asymptotic conditional coverage
- 我们需要把它推广到我们的 $s = |y - \hat{p}| / \hat{\sigma}$ score

**关键技术引理**：
- (L1) $\hat{p}$ 的一致估计率（Lipschitz on $\mathcal{X}$ 或 Hölder）
- (L2) $\hat{\sigma}$ 逼近 oracle 的 rate
- (L3) Quantile estimation error from finite calibration

**难度**：★★★★☆

**时间**：3–4 周

**风险**：中高。local conditional coverage 严格意义在有限样本不可能（Barber 2020 impossibility）。必须改为"局部平均"版本或要求 $x_0$ 带某种"density robust"条件。

**降级备选**：若严格证明太难 → asymptotic 版 with $r \to 0$ rate + 实证验证。

**写作长度**：2–3 页 + 3–5 页 appendix

（原 T3.b realistic version 的 σ̂ 估计误差 perturbation 已在 `t3_formal_proof.md` §7 以有限样本形式证完，不再归入 T3.a asymptotic 分支。）

---

## T4 — Chrom-Shift Robustness

**形式**:
$$
P_{c^*}\!\left( Y \in \mathcal{C}_\alpha(X) \right) \;\geq\; (1 - \alpha) - d_{\text{TV}}(P_{c^*}, P_{\text{cal}}) - O(n^{-1/2})
$$

(Barber et al. 2023 Thm 2; **no factor of 2** — earlier drafts of this roadmap had this wrong.)

where $P_{\text{cal}} = \text{mixture of calibration chroms' distributions}$.

**证明工具**：Barber et al 2023 Theorem 3 (distribution shift via TV) 直接套。

**实证补充**：Estimate $d_{\text{TV}}(P_c, P_{c'})$ for all chrom pairs on TraitGym → show all pairs have small TV → shift bound is tight.

**难度**：★★★☆☆

**时间**：1–2 周 (证明) + 1 周 (实证 TV estimation)

**风险**：中。证明 standard，难在 TV estimation 是 feature-space 的高维问题 — 可以用 classifier-based TV estimator (Kim & Scott 2019)。

**写作长度**：1.5 页 + 2 页 appendix

---

## 时间排期（Day 16 刷新）

| 月 | T1 / T2 | T3 / T3' / T3-loc / T3.b | T3.a asym | T4 |
|---|---|---|---|---|
| 1 | ✅ draft (`t1_t2_formal_proofs.md` Day 14) | ✅ sketch (`t3_proof_sketch.md` Day 13) | — | ✅ sketch Day 14 |
| **2 (当前)** | ✅ appendix-ready | ✅ appendix-ready (`t3_formal_proof.md` Day 16) | — | ✅ in `t1_t2_formal_proofs.md` §6 |
| 3 | paper §5 正文化 | paper §5 正文化 | 🔄 start（Romano 2019 §3 精读） | TV empirics sweep |
| 4 | — | — | 🔄 oracle lemma | ✅ done |
| 5–6 | — | — | 🔄 main proof + polish | — |
| 7 | — | — | done 或降级 future work | — |

**注**: 原 roadmap 把 T3.b 视作 crown jewel + 8 月路径。Day 16 发现 T3.b 有限样本形式可用简单 bin-reassignment 论证 (§7 of `t3_formal_proof.md`)，不必等 T3.a asymptotic — 提前完成。剩余 T3.a asymptotic 是 "锦上添花"，不是 blocker。

---

## 证明依赖图（Day 16 刷新）

```
         T1 (marginal CP)
         /       \
        T2       T4 (chrom shift, Barber 2023 Thm 2)
        |
        |
        T3 (bin-cond, exact under A2-cell)     ◀── main novelty headline
      /   \
  T3' (robust,   T3-loc (feature-ball via σ̂ Lipschitz)
   A1 only)      |
                 |
               T3.b (σ̂ perturbation)
                 |
                 |
              T3.a (local asymptotic r→0, future work)
```

T3 是 headline；T3' / T3-loc / T3.b 是 reviewer-proofing；T3.a 是延伸但非必需。

---

## 和 NeurIPS 主会 bar 的对应（Day 16 更新）

| Theorem set | 对应 reviewer 问题 | 充分性 |
|---|---|---|
| T1 only | "你保证了 coverage 吗？" | 不够 — standard result |
| T1 + T2 | "class imbalance 怎么办？" | 勉强 — Vovk 2003 已知 |
| T1 + T2 + T4 | "分布偏移怎么办？" | 仍不够 — Barber 2023 已做 |
| T1 + T2 + T4 + **T3** | "local coverage 呢？" | 达 novelty threshold — 有限样本 bin-conditional 是 new |
| T1 + T2 + T4 + **T3 + T3' + T3-loc + T3.b** (当前) | "σ̂ 学错怎么办？feature space？A2 违反？" | **NeurIPS main bar 够** — 三个 reviewer 问题全部 covered |
| + T3.a asymptotic | "pointwise limit 呢？" | strong accept 候选 |

**结论（Day 16 更新）**: **T1 + T2 + T4 + T3 + T3' + T3-loc + T3.b 已全部 appendix-ready**。NeurIPS 主会理论 bar 已达。T3.a 是延伸工作，不再是 blocker。
