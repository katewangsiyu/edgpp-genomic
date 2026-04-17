# Formulation v0 — Conformalized Heteroscedastic Variant Effect Prediction

**目的**：把 Path A 的 problem / method / theoretical targets 写成严格数学形式，作为后续定理证明与实验设计的锚点。**本稿不追求完美，只追求不含糊。**

**对比参照**：
- Romano, Patterson, Candès 2019 (CQR)
- Tibshirani, Barber, Candès, Ramdas 2019 (weighted CP)
- Barber, Candès, Ramdas, Tibshirani 2023 (CP beyond exchangeability)
- Vovk 2003 (Mondrian CM)
- Zhou et al. 2026 (DEGU)

---

## 1. Setup

### 1.1 Data

设变异特征 $X \in \mathcal{X} \subseteq \mathbb{R}^d$，标签 $Y \in \{0, 1\}$（0 = control, 1 = pathogenic）。每个变异关联一个染色体 $C \in \mathcal{C} = \{1, 2, \ldots, 22, X\}$。

观察到数据 $\mathcal{D} = \{(X_i, Y_i, C_i)\}_{i=1}^n$。

TraitGym chrom-LOO 协议：测试集 chromosome $c^* \in \mathcal{C}$，训练集 $\mathcal{D}^{(-c^*)} = \{i : C_i \neq c^*\}$，测试集 $\mathcal{D}^{(c^*)} = \{i : C_i = c^*\}$。

### 1.2 Key assumption — chrom-group exchangeability

**Assumption A1 (Chrom-group exchangeability)**. 以染色体为单位，条件于染色体标识 $C$，来自该染色体的样本内部是可交换的（exchangeable），但 **不同染色体之间不可交换**。

$$
(X_i, Y_i) \mid C_i = c \overset{\text{exch}}{\sim} P_c, \qquad P_c \neq P_{c'} \text{ in general}.
$$

**为什么合理**：同一染色体内变异因连锁不平衡（LD）形成 dependency structure，但当我们以变异为单位评估时，同一染色体内的变异在 chrom-LOO 协议下被视为一个 block。

**为什么重要**：这是 chrom-LOO 评估下 conformal 保证的根基。标准 conformal（Vovk, Romano 等）假设**全局 i.i.d.**；我们的 chrom-group 假设比 i.i.d. **弱**，比完全 arbitrary shift **强**。

### 1.3 Base aggregator

给定训练数据 $\mathcal{D}^{(-c^*)}$，学一个 probabilistic classifier $\hat{p}: \mathcal{X} \to [0, 1]$，使 $\hat{p}(x) \approx P(Y = 1 \mid X = x)$。

具体实现（Day 10 已验证）：
- $\hat{p}$ = `HistGradientBoostingClassifier(depth=2, iter=100, class_weight="balanced")`
- Feature sets: CADD, Borzoi_L2_L2, CADD+Borzoi, CADD+GPN-MSA+Borzoi

### 1.4 Heteroscedastic reliability head

第二个模型 $\hat{\sigma}: \mathcal{X} \to \mathbb{R}_+$，预测 **chrom-LOO residual 的期望尺度**：

$$
\hat{\sigma}(x) \approx \sqrt{\mathbb{E}[(Y - \hat{p}(X))^2 \mid X = x]}.
$$

训练：在 chrom-LOO 残差 $r_i = |Y_i - \hat{p}^{(-C_i)}(X_i)|$ 上拟合 regressor。推荐两种 objective：

**Option A — Gaussian log-likelihood (Nix-Weigend / DEGU hetero baseline)**:
$$
\mathcal{L}_{\text{GLL}}(\hat{\sigma}) = \sum_i \left[ \frac{r_i^2}{2 \hat{\sigma}(X_i)^2} + \log \hat{\sigma}(X_i) \right] + \lambda \|\theta\|^2
$$

**Option B — Pinball loss on |residual| quantile (CQR-style)**:
$$
\mathcal{L}_{\text{pin}}(\hat{\sigma}) = \sum_i \rho_{1-\alpha}(r_i - \hat{\sigma}(X_i))
$$

v0 采用 Option A（与 DEGU hetero baseline 同构，方便对照）。

---

## 2. Conformal Prediction Set

### 2.1 Nonconformity score

给定 $(x, y)$ 与已训练好的 $(\hat{p}, \hat{\sigma})$，定义 **feature-dependent nonconformity score**：

$$
\boxed{s(x, y) := \frac{|y - \hat{p}(x)|}{\hat{\sigma}(x) + \epsilon}}
$$

其中 $\epsilon > 0$ 是小常数防除零（实务取 $\epsilon = 10^{-6}$）。

**与已有 score 的区别**：
- Split conformal (Vovk 2005): $s = |y - \hat{p}(x)|$ — 与 $x$ 无关
- CQR (Romano 2019): $s = \max(\hat{q}_{\alpha/2}(x) - y, y - \hat{q}_{1-\alpha/2}(x))$ — 用 quantile 而非 variance
- DEGU (Zhou 2026) 若用 split conformal: $s = |y - \hat{p}(x)|$ with $\hat{\sigma}$ 单独 report 但不进 score
- **Ours**: 用 $\hat{\sigma}$ 作为 score 的 **local scale**，prediction set 大小随 $x$ 自适应

### 2.2 Class-conditional calibration (Mondrian stratification)

对每个类别 $k \in \{0, 1\}$，用 calibration set $\mathcal{D}_{\text{cal}}^{(k)} = \{i \in \mathcal{D}_{\text{cal}} : Y_i = k\}$ 分别 calibrate。

设 $n_k = |\mathcal{D}_{\text{cal}}^{(k)}|$，计算 score 集合 $S_k = \{s(X_i, Y_i) : i \in \mathcal{D}_{\text{cal}}^{(k)}\}$。

阈值：
$$
\hat{q}_k = \text{Quantile}\left( S_k, \ \lceil (n_k + 1)(1 - \alpha) \rceil / n_k \right)
$$

### 2.3 Prediction set

对新样本 $x \in \mathcal{D}_{\text{test}}^{(c^*)}$：

$$
\boxed{\mathcal{C}_\alpha(x) := \{k \in \{0, 1\} : s(x, k) \leq \hat{q}_k\}}
$$

即 $k \in \mathcal{C}_\alpha(x)$ 当且仅当：
$$
\frac{|k - \hat{p}(x)|}{\hat{\sigma}(x)} \leq \hat{q}_k
\iff
|k - \hat{p}(x)| \leq \hat{q}_k \cdot \hat{\sigma}(x).
$$

**几何意义**：对于 $\hat{\sigma}(x)$ 大的点（模型不自信），容忍度 $\hat{q}_k \cdot \hat{\sigma}(x)$ 大 → 更容易包含两个标签 → prediction set 更保守（{0,1}）。

### 2.4 Calibration set 构造 (chrom-LOO)

在 chrom-LOO 下 **calibration set 不能 i.i.d. 采样**。两种 scheme：

**Scheme CL1 (Leave-one-chrom-out, 一次通过)**:
对每个 test chrom $c^*$，用所有 $C_i \neq c^*$ 的样本作为 calibration。$\hat{p}, \hat{\sigma}$ 都在这个集合上训练。**问题**：calibration 数据与训练 $\hat{p}, \hat{\sigma}$ 的数据重合，会 overfit。

**Scheme CL2 (Nested leave-one-chrom-out)**:
两层：外层 test chrom $c^*$；内层再留出 $c' \neq c^*$ 作为 calibration，$c'' \neq c^*, c' \neq c''$ 作为训练。对 $c'$ 取遍所有选择 → 得 aggregated calibration。**代价**：22 × 21 = 462 次训练。

v0 推荐 **Scheme CL1**（接受轻微 overfit，换低 compute），在 Theorem T1' 讨论其 coverage bias 上界。

---

## 3. Theoretical Targets (Theorem List)

### T1 Marginal coverage (prerequisite)

**Theorem T1**. 在 Assumption A1 下，若 calibration 和 test 来自**同一** $P_{c^*}$，且 score 函数 $s$ 对 $(\hat{p}, \hat{\sigma})$ 是可测的，则：
$$
P\left( Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}}) \right) \geq 1 - \alpha
$$

**证明草图**：标准 split conformal + A1' (chrom-wise i.i.d.) + A2 (score stationarity)；appendix-ready 版本见 `theory/t1_t2_formal_proofs.md` §4。注意 Barber 2023 里是 Theorem 2（weighted, 带 TV 项）而非 Theorem 1；之前 sketch 写错。

**状态**：appendix-ready v1 已落盘（2026-04-17）。

### T2 Class-conditional coverage

**Theorem T2**. 在 A1 下：
$$
P\left( Y_{\text{test}} \in \mathcal{C}_\alpha(X_{\text{test}}) \mid Y_{\text{test}} = k \right) \geq 1 - \alpha, \quad \forall k \in \{0, 1\}.
$$

**证明草图**：Vovk 2003 Mondrian CM 的直接推广。按 $Y$ 分层相当于 Mondrian partition。需要假设每个类别 calibration 非空（$n_k \geq 1/\alpha$）。

**状态**：appendix-ready v1 已落盘（`theory/t1_t2_formal_proofs.md` §5）。

### T3 Local (feature-neighborhood) coverage ★★ 主贡献

**Theorem T3 (informal)**. 在 A1 + $\hat{\sigma}$ 逼近真实条件方差的条件下，对任意 $x_0 \in \mathcal{X}$ 和邻域 $B(x_0, r) = \{x : \|x - x_0\| \leq r\}$：
$$
\left| P\left( Y \in \mathcal{C}_\alpha(X) \mid X \in B(x_0, r) \right) - (1 - \alpha) \right| \leq \epsilon_1(r) + \epsilon_2(\hat{\sigma})
$$

其中 $\epsilon_1(r) \to 0$ as $r \to 0$，$\epsilon_2(\hat{\sigma}) \to 0$ as $\hat{\sigma} \to \sigma_{\text{true}}$。

**证明思路**：
1. 局部 conditional coverage 的 oracle rate（如果 $\hat{\sigma} = \sigma_{\text{true}}$）
2. $\hat{\sigma}$ 的估计误差对局部 coverage 的扰动
3. 把两部分合并

**难点**：
- Local conditional coverage 严格意义上在有限样本不可能保证（Barber 2020 的 impossibility result）。我们要的是 **局部平均** coverage 的 bound，而非 pointwise conditional。
- 需要对 $\mathcal{X}$ 的度量结构做假设（可能需要 Lipschitz 条件或 bounded density）。

**状态**：主要 novelty；需要 2–3 月工作。可能要降级为 asymptotic result 或条件在 Hölder class。

### T4 Chrom-shift robustness ★ 次贡献

**Theorem T4 (informal)**. 若 $P_{c^*}$ 与 calibration 使用的 $P_{c'}$ 在 total variation 距离 $\leq \delta$：
$$
P_{c^*}\!\left( Y \in \mathcal{C}_\alpha(X) \right) \;\geq\; (1 - \alpha) - \delta - \text{finite-sample}.
$$

**证明思路**：Barber et al 2023 Theorem 2 （weighted, $\sum \tilde{w}_i d_{TV}$ bound，**no factor of 2** — 详见 `theory/t1_t2_formal_proofs.md` §6）。需要估计 chrom 之间的 TV 距离（实证 + bound）。

**状态**：标准结果套用；写作约 1.5–2 页。实证需要跨 chrom TV 距离估计。

---

## 4. 与 DEGU 的 formal 区分

| 维度 | DEGU | Our Method |
|---|---|---|
| Uncertainty source | Ensemble disagreement (epistemic) | Learned heteroscedastic σ(x) (mixed) |
| Training | Two-stage: ensemble train + distill | Two-stage: base p̂ train + σ̂ fit |
| Output | (μ, σ) via MSE regression | p̂ (classifier) + σ̂ (regressor) |
| Conformal guarantee | Vanilla split (abstract claim) | **Feature-dep + class-cond + local bound** |
| Coverage theorems | None proven | T1–T4 |
| Test task | MPRA regression (DeepSTARR, lentiMPRA) | VEP binary classification (TraitGym) |
| Test metric | MSE / Pearson r | AUPRC + coverage (marginal/class/local) |
| # parameters student | Same as teacher | Can be smaller (GBM 100 trees) |

---

## 5. 实证设计 formal 版

### 5.1 Metrics

1. **AUPRC_by_chrom_weighted_average** (TraitGym 口径) — 用 $\hat{p}$ 的原始输出
2. **Marginal coverage**: $\frac{1}{n_{\text{test}}} \sum_i \mathbb{1}[Y_i \in \mathcal{C}_\alpha(X_i)]$
3. **Class-conditional coverage**: 同上 per class
4. **Local coverage (new metric)**: 对每个 partition $\pi_j \subseteq \mathcal{X}$（如 consequence 类型、TSS 距离 bin），测 $\text{cov}(\pi_j)$。metric = $\max_j |\text{cov}(\pi_j) - (1 - \alpha)|$ + $\text{std}_j$。
5. **Set size distribution**: fraction of {∅}, {0}, {1}, {0,1}
6. **Selective AUPRC**: AUPRC on predictions where $|\mathcal{C}_\alpha(x)| = 1$ (高置信预测)

### 5.2 Baselines

| Baseline | 简写 |
|---|---|
| TraitGym LogReg + split conformal | LR+CP |
| TraitGym LogReg + CV+ | LR+CV+ |
| GBM + split conformal | GBM+CP |
| GBM + class-cond conformal (Day 10) | GBM+CCCP |
| GBM + CQR score (adapted) | GBM+CQR |
| DEGU reimpl + split conformal | DEGU+CP |
| **Ours: GBM + σ̂(x) + class-cond** | **HCCP** |
| **Ours: GBM + σ̂(x) + class-cond + T3 constraint** | **HCCP-L** |

### 5.3 Ablations

- σ̂ source: GBM regressor / small MLP / bootstrap variance / dropout / DEGU-style ensemble
- Class-cond 有/无
- ε scale 敏感性
- Calibration scheme: CL1 vs CL2
- Feature set × model 4×2 = 8 configurations

### 5.4 Datasets

- TraitGym Mendelian matched_9 (n=3380) [primary]
- TraitGym Complex traits matched_9 (n=11400) [primary]
- ClinVar pathogenic hold-out [generalization check]
- gnomAD rare variants [OOD stress, optional]

---

## 6. Open Questions

1. **$\hat{\sigma}$ 应该与 $\hat{p}$ 共享特征还是独立？** v0 独立。Joint training 留给 v1。
2. **$\epsilon$ 常数的合理值？** 需要 sensitivity 分析。
3. **T3 的 Lipschitz 假设在 TraitGym feature space 是否成立？** 需要实证检查（estimate local density gradient）。
4. **Class imbalance (10% pos) 下 Mondrian 的 $n_1 \geq 1/\alpha$ 要求是否 binding？** 对 α = 0.1 需要 $n_1 \geq 10$，实际 $n_1 \approx 300$ in Mendelian test — 充裕。
5. **Chrom 独立假设的实际 TV 距离？** 实证。
6. **是否需要把 conformal score 改成 APS-style 以获得 adaptive behavior？** 这是潜在 v1 方向。

---

## 7. v0 → v1 Next Steps

Week 1–2:
- [ ] 深读 Romano 2019 CQR §3（score 形式）+ Barber 2023 §3 (distribution shift)
- [ ] Implement $\hat{\sigma}$ head prototype：GBM regressor on Day 10 residuals
- [ ] Empirical sanity check：$\hat{\sigma}(x)$ 在 $\hat{p}(x)$ 极端值附近是否给出合理 scale？

Week 3–4:
- [ ] Write out T1 + T2 full proofs (standard results)
- [ ] Start T3 proof outline + identify required technical lemmas
- [ ] Initial experiments with HCCP on TraitGym Mendelian

Week 5–8:
- [ ] DEGU reimpl (feature-level L1 variant per `papers/degu_reproduction_plan.md` §4.1)
- [ ] T3 proof first draft
- [ ] Full ablation table

Week 9–12:
- [ ] ClinVar hold-out experiments
- [ ] T4 proof
- [ ] Write paper skeleton
- [ ] Internal review

---

## Appendix A — Notation Table

| Symbol | Meaning |
|---|---|
| $X \in \mathcal{X}$ | Variant feature vector |
| $Y \in \{0, 1\}$ | Pathogenic label |
| $C \in \mathcal{C}$ | Chromosome ID |
| $\hat{p}: \mathcal{X} \to [0, 1]$ | Base probabilistic classifier |
| $\hat{\sigma}: \mathcal{X} \to \mathbb{R}_+$ | Heteroscedastic reliability head |
| $s(x, y)$ | Nonconformity score |
| $\hat{q}_k$ | Class-$k$ quantile threshold |
| $\mathcal{C}_\alpha(x)$ | Prediction set at miscoverage $\alpha$ |
| $\mathcal{D}^{(-c)}, \mathcal{D}^{(c)}$ | Train / test split for chrom $c$ |
| $P_c$ | Chrom-$c$ conditional distribution |

---

## Appendix B — 与 Day 10 代码的 mapping

| Formulation 符号 | Day 10 code |
|---|---|
| $\hat{p}$ (base aggregator) | `scripts/11_aggregator_gbm.py` 的 HistGB |
| $s(x, y)$ with $\hat{\sigma} \equiv 1$ | `scripts/12_conformal.py` 的 `lac_score` |
| Class-conditional calibration | `scripts/12_conformal.py` 的 `class_conditional_conformal` |
| $\hat{\sigma}$ (new in v1) | TODO: `scripts/13_hetero_head.py` |
| HCCP (new in v1) | TODO: `scripts/14_conformal_hetero.py` |

即：Day 10 是 Path A 的**特例** $\hat{\sigma}(x) \equiv 1$。v1 的工作是加上非常数 $\hat{\sigma}$ 并证明它带来更好的 local coverage。
