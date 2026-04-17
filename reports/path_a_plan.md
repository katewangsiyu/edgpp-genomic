# Path A — Conformalized Heteroscedastic VEP 项目总览

**目标 venue**: NeurIPS 2027 main conference（截稿 ~2027-05）
**保底 venue**: NeurIPS 2026 D&B Track（Day 10 素材，截稿 ~2026-06）
**期刊保底**: npj AI（直接对标 DEGU）

---

## 1. Problem Statement

**研究问题**：给定基因组变异 $x \in \mathcal{X}$，在严格覆盖率保证下输出 pathogenic 预测集合 $\mathcal{C}_\alpha(x) \subseteq \{0, 1\}$，使得：

1. **Marginal coverage**：$P(y \in \mathcal{C}_\alpha(x)) \geq 1 - \alpha$
2. **Class-conditional coverage**：$P(y \in \mathcal{C}_\alpha(x) \mid y = k) \geq 1 - \alpha, \forall k \in \{0, 1\}$
3. **Local (feature-dependent) coverage**：对 $x$ 空间的任意局部区域 $B \subseteq \mathcal{X}$，$P(y \in \mathcal{C}_\alpha(x) \mid x \in B) \approx 1 - \alpha$
4. **Chrom-exchangeability shift robustness**：在 chrom-LOO 评估下（train chroms 和 test chroms 不共享）依然成立

**已有工作的缺口**：
- TraitGym supervised LogReg pipeline：无 uncertainty 量化
- DEGU (Zhou & Koo 2026 npj AI)：heteroscedastic NLL 预测 $\sigma(x)$，但 **no coverage guarantee**
- 标准 CQR (Romano et al. 2019)：只保证 marginal，class imbalance 下 Cov|pos 崩塌
- 我们 Day 10 的 class-conditional conformal：实证有效但 score 与 $x$ 无关，局部覆盖率未知

**我们的贡献**（预期）：
1. 首个 heteroscedastic + conformal joint framework for VEP，variant-level granularity
2. 新 nonconformity score：$s(x, y) = |y - \hat{p}(x)| / \sigma(x)$，feature-dependent
3. Theorem：在 chrom-exchangeability 假设下 marginal + class-cond + local coverage 的 joint bound
4. 实证：TraitGym Mendelian + Complex + ClinVar hold-out 三数据集 SOTA on (AUPRC, calibration, selective risk)
5. **Benchmark audit 副产出**：Day 10 揭露的 TraitGym LogReg 在高维退化问题

---

## 2. Method 草图

### 2.1 Base aggregator
沿用 Day 10 的 `HistGradientBoosting`（或替换为小 NN 以便 joint training）预测 $\hat{p}(x) \in [0, 1]$。

### 2.2 Heteroscedastic reliability head
训练 $\sigma(x) \in \mathbb{R}_+$ 预测 **chrom-LOO residual magnitude**：

$$
\mathcal{L}_\text{het}(\sigma) = \mathbb{E}_x \left[ \frac{(y - \hat{p}(x))^2}{2 \sigma(x)^2} + \log \sigma(x) \right]
$$

其中 $\hat{p}(x)$ 的 chrom-LOO 预测作为 target。两种实现：
- A. Second GBM regressor（CPU, 快）
- B. Joint NN with aggregator + variance heads（GPU, 灵活）

### 2.3 Heteroscedastic conformal score
$$
s(x, y) = \frac{|y - \hat{p}(x)|}{\sigma(x)}
$$

Class-conditional 分层：对每个 $k \in \{0, 1\}$ 单独 calibrate，得到阈值 $\hat{q}_k$。

### 2.4 Prediction set
$$
\mathcal{C}_\alpha(x) = \{k : s(x, k) \leq \hat{q}_k\}
$$

局部大小 ∝ $\sigma(x)$：模型不自信区域得到大集合（{0,1}），自信区域得到 singleton。

---

## 3. 理论 Targets

| 定理 | 形式 | 工具 | 难度 |
|---|---|---|---|
| T1 Marginal coverage | $P(y \in \mathcal{C}_\alpha(x)) \geq 1 - \alpha$ | Chrom-group exchangeability + Barber 2023 split conformal | Medium |
| T2 Class-conditional coverage | $P(y \in \mathcal{C}_\alpha(x) \mid y = k) \geq 1 - \alpha$ | Mondrian conformal (Vovk 2003) 推广 | Medium |
| T3 Local conditional coverage | 对 $\|x - x_0\| < r$ 的邻域平均覆盖率 $\approx 1 - \alpha$ | Kernel-weighted conformal (Tibshirani 2019) | Hard |
| T4 Chrom-shift robustness | 对测试 chrom 的 coverage 退化 bounded by TV distance | Distribution-shift conformal (Barber 2023) | Hard |

T3 和 T4 是主要 novelty，T1/T2 是 prerequisite。

---

## 4. 实验 Matrix

### 4.1 Aggregator baselines
- LogReg (TraitGym 官方): Borzoi_L2_L2, CADD, CADD+Borzoi, CADD+GPN-MSA+Borzoi
- GBM (Day 10 our): 同上 feature sets
- DEGU (重现): heteroscedastic NLL CNN, 同样的 chrom-LOO 协议

### 4.2 Conformal baselines
- Split conformal (Vovk 2005)
- CV+ (Barber 2021)
- CQR (Romano 2019) —— regression 改造成 binary
- Mondrian conformal (Vovk 2003)
- Class-conditional (Day 10 our)
- **Ours**: heteroscedastic + class-cond + feature-dependent

### 4.3 Metrics
- AUPRC_by_chrom_weighted_average (TraitGym 口径)
- Marginal coverage + |gap|
- Class-conditional coverage (both classes)
- **Local conditional coverage** (new metric): variance across x-space partitioning by (chrom, consequence, tss_dist bin)
- Selective risk (AUPRC @ various coverage)
- Set size distribution + decisiveness index

### 4.4 Datasets
- **Primary**: TraitGym Mendelian matched_9 (n=3380)
- **Primary**: TraitGym Complex traits matched_9 (n=11400)
- **Hold-out**: ClinVar pathogenic subset (需下载)
- **OOD stress**: gnomAD rare variants（如果时间允许）

### 4.5 Ablations
- `σ(x)` from different heads: GBM vs NN vs bootstrap vs dropout
- Class-cond 有 vs 无
- Feature-dep 有 vs 无（vanilla split conformal 对照）
- Chrom-LOO vs random-split（揭示 chrom-shift 差异）

---

## 5. Timeline（不在乎时间，但要有 checkpoint）

| Phase | 内容 | 预计 |
|---|---|---|
| 1. 文献 + formulation | Romano/Tibshirani/Barber 精读；DEGU 重现草稿；problem statement | 1 月 |
| 2. 方法实现 + 初步实验 | Hetero head + conformal 联合 pipeline；与 baselines 对比 | 2 月 |
| 3. 理论推导 | T1–T4 证明 + counterexample | 2 月 |
| 4. 拓展实验 | ClinVar + gnomAD + 完整 ablation | 1.5 月 |
| 5. 写作 + iteration | 初稿 → 内审 → 外审 → 投稿 | 2 月 |
| **并行 保底** | Day 10 素材 → NeurIPS 2026 D&B 投稿 | 1 月（2026 春） |

**总计**：8.5 月。NeurIPS 2027 截稿 2027-05，从 2026-04 开始留足 13 月缓冲。

---

## 6. 风险 & 备选

| 风险 | 概率 | 应对 |
|---|---|---|
| DEGU 代码不公开，复现数字不准 | 中 | 按论文 reimplement，在 ablation 里标 "our reimplementation" |
| T3 local coverage 理论太难 | 高 | 降级为 asymptotic result 或实证验证 + conjecture |
| heteroscedastic head 过拟合 | 中 | cross-validation 选 head complexity；GBM 的 depth=2 已在 aggregator 表现稳定 |
| ClinVar 数据对齐 TraitGym 变异失败 | 中 | 改用 TraitGym 内部 val_chroms 作 hold-out 分析 |
| 被 scooped（conformal + genomics）| 低 | bioRxiv preprint 早发，占坑 |

---

## 7. 下一步（Day 11 立即）

1. ✅ 建立本文档
2. 🔄 文献 harvest（5 篇核心 + 10 篇扩展）
3. 🔄 DEGU 代码搜索 + 复现方案评估
4. 📝 写 formulation_v0.md（T1–T4 形式化 + 假设清单）
5. 📝 建立 `papers/`, `theory/`, `experiments/conformal/` 目录结构

---

## 8. 与 Day 10 工作的关系

Day 10 的全部 GBM + class-conditional conformal + ablation table **不被废弃**：
- 主论文：作为 baseline（GBM + class-cond）和 motivation（TraitGym 高维退化）
- D&B 保底论文：Day 10 素材独立打包
- 复用代码：`scripts/11_aggregator_gbm.py`, `scripts/12_conformal.py`, `scripts/09_aggregator_chrom_loo.py` 保留作为 backbone

Day 11+ 新增：
- `scripts/13_hetero_head.py` — heteroscedastic reliability head
- `scripts/14_conformal_hetero.py` — feature-dep conformal
- `scripts/15_eval_local_coverage.py` — 新 metric
- `scripts/16_baselines_cqr.py` — CQR reimpl for binary
- `scripts/17_degu_reimpl.py` — DEGU for TraitGym

---

## 9. Success Criteria

NeurIPS 2027 accept 的**必要条件**：
- [x] T1 marginal + T2 class-cond appendix-ready 证明（`theory/t1_t2_formal_proofs.md`, Day 14）
- [x] T3 local coverage sketch + bin-conditional bound $1/(n_{kb}+1)$（`theory/t3_proof_sketch.md`, Day 13）
- [ ] T3 完整证明（含 A2 定量弱化 + feature-space 转译）
- [x] **Heteroscedastic conformal 在 TraitGym Mendelian 上 AUPRC = 0.902**（trait-LOO, Day 14）
- [x] **σ̂-bin gap 达 finite-sample floor 0.002–0.004**（trait-LOO, Day 14）
- [x] **Chrom-LOO σ̂-bin gap 0.020**（Complex, Day 12）
- [x] **外部 OOD 验证（B1 trait-LOO + B2 cross-dataset）**：
  - Trait-LOO 2×2 matrix 完整（Day 14 + Day 15）：σ̂-bin gap 0.002–0.004 全部达 finite-sample floor；GPN-MSA 在 Complex per-trait cov std 收窄 15%（0.020→0.017）
  - Cross-dataset C→M：σ̂-bin gap 0.035
  - Cross-dataset M→C：落在 Barber 2023 Thm 2 proxy bound 1.4σ 内（CADD+Borzoi）
- [x] DEGU-lite 对比：6/6 partition on Complex（Day 13）
- [ ] DEGU full reimpl 对比（deferred，低优先级）
- [x] 消融：Homosc / Hetero / Mondrian 三种 score × class-cond 有/无（Day 11–12）

失败路径 fallback：
- 若 T3 完整证明做不出，T3.a oracle-σ 版本 + 跨三层（chrom/trait/cross-dataset）的强 empirical 构成 "strong evidence" 论据
- 未达 T3 完整证明 + 三层 empirical = 降级投 NeurIPS D&B / RECOMB / npj AI

**当前进度评估（Day 15）**：以 NeurIPS 2027 bar 衡量
- 实证：强（**three-axis local coverage gap 全部 ≤ 0.04**，trait-LOO 2×2 matrix 完整，全部达 finite-sample floor）
- 理论：中等（T1+T2 done，T3 sketch + bound 但尚未 polished）
- 写作：skeleton 落盘（`papers/neurips2027_pathA/`）
- 剩余主要工作：T3 完整证明（Month 2–6 of roadmap）+ paper writing（§3/§5/§6 正文化）
