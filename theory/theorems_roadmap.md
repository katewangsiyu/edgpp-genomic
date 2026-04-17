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

### T3.a — Oracle version

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

### T3.b — Realistic version (with $\hat{\sigma}$ estimation error)

**形式**:
$$
\left| P\left( Y \in \mathcal{C}_\alpha(X) \mid X \in B(x_0, r) \right) - (1 - \alpha) \right| \leq \epsilon_1(r, \alpha) + \epsilon_2(\|\hat{\sigma} - \sigma_{\text{true}}\|_\infty)
$$

**证明工具**：Perturbation analysis on T3.a + Lipschitz continuity of coverage in score function.

**难度**：★★★★★

**时间**：2–3 周 (after T3.a)

**风险**：高。这是整个 paper 的 crown jewel。如果只能证 T3.a，就把 T3.b 放进 Discussion / Future Work，以 empirical evidence 支撑。

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

## 时间排期

| 月 | T1 | T2 | T3.a | T3.b | T4 |
|---|---|---|---|---|---|
| 1 | ✅ draft | ✅ draft | — | — | — |
| 2 | polish | polish | 🔄 start | — | 🔄 start |
| 3 | done | done | 🔄 technical lemmas | — | TV empirics |
| 4 | — | — | 🔄 main proof | — | ✅ draft |
| 5 | — | — | polish | 🔄 start | polish |
| 6 | — | — | done | 🔄 main proof | done |
| 7 | — | — | — | polish | — |
| 8 | — | — | — | done | — |

总计 8 个月理论工作（可与实验并行）。

---

## 证明依赖图

```
         T1 (marginal CP)
         /       |
        /        |
  T2 (class-cond)  T4 (chrom shift)
        \
         \
       T3.a (local, oracle σ)
          \
           \
          T3.b (local, realistic σ)
```

T3.b 依赖 T3.a（直接扩展）。T3.a 实务上不依赖 T2，但行文上应放在 T2 后面。T4 与 T3 并行独立。

---

## 和 NeurIPS 主会 bar 的对应

| Theorem | 对应 reviewer 问题 | 充分性 |
|---|---|---|
| T1 only | "你保证了 coverage 吗？" | 不够 — reviewer 会说"这是 standard result" |
| T1 + T2 | "class imbalance 怎么办？" | 勉强 — 但这是 Vovk 2003 的已知结果 |
| T1 + T2 + T4 | "分布偏移怎么办？" | 仍不够 — Barber 2023 已经做过 |
| **T1 + T2 + T3.a + T4** | 充分 | T3.a 是 **novel contribution** |
| **T1 + T2 + T3.a + T3.b + T4** | strong accept 候选 | T3.b 是 **novelty + practical** |

结论：**T3 必须做到 a.（b. 能做就做）**。T3 不做 = 降级投 NeurIPS D&B。
