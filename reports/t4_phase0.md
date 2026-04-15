# EDG++ Genomic — T4 Phase 0 报告

**日期**: 2026-04-15
**周期**: Day 0–8 (2026-04-09 → 2026-04-15)
**硬件**: T4×8 (SM 7.5, 无 BF16)
**评测**: TraitGym (Benegas et al., bioRxiv 2025)

---

## 1. 摘要

Phase 0 的目标是在 T4 硬件上验证 EDG++ (selective distillation + reliability gating) 能否在 TraitGym benchmark 上产生有竞争力的结果。

**结论**: 原始路线（从头蒸馏 Borzoi → CompactStudent + reliability）**不可行**。已成功 pivot 到 P1 路线（后 Borzoi 聚合器 + selective reliability head），在 Mendelian traits 上获得了稳定的 coverage-AUPRC 改进，但效应量不足以支撑 ICLR 级别论文。

---

## 2. 时间线与关键发现

### Day 0–4: 基础设施 + 首次实验

- 搭建完整 pipeline：hg38 下载、TraitGym 适配器（teacher 预计算 .npz）、CompactStudent (0.87M params)、SelectiveDistillLoss
- 3-way ablation: baseline / DEGU / EDG++
- Day 4 产出单种子头条：`score_difficulty` AUPRC_per_chrom = 0.551 (val-chroms only)
- ⚠️ 此数字后被证伪（见 Day 5）

### Day 5: Falsification (关键转折点)

三个独立发现联合证伪了 Day 4 头条：

**发现 1 — Teacher 幅度 null baseline**:
Borzoi teacher 的原始 L2-norm 在 val chroms 上 AUPRC = 0.5544，与 EDG++ `score_difficulty` (0.5508) 无显著差异。说明 reliability MLP 学到的就是 teacher 幅度本身。

**发现 2 — Side-feature 泄漏诊断**:
`side_features` 包含 `||teacher||` 和 `teacher.std`，reliability MLP 直接从输入拿到 teacher magnitude。去除后（`side_features_mode=tss_only`，仅保留 `log1p|tss_dist|`）：

| Score | AUPRC_per_chrom (full) | AUPRC_per_chrom (tss_only) |
|---|---|---|
| score_mag | ~0.127 | 0.129 |
| score_difficulty | **0.551** | **0.170** |

`score_difficulty` 从 0.551 崩塌至 0.170 (−69%)，且与 `tss_dist` 相关系数 = −0.9998。Reliability 在无泄漏时只学到了 TSS 距离的代理。

**发现 3 — 评测方法论修正**:
`07_evaluate.py` 仅在 val chroms (chr17-22,X, 1030/3380 variants) 上计算指标，而 TraitGym leaderboard 使用全测试集 (3380 variants, chrom-LOO)。Val-only 数字系统性高估：Borzoi_L2_L2 val=0.554 vs full-test=0.4356。

### Day 5 (续): CompactStudent 蒸馏不可行

5000 步长训练（10× Day 4 计算量）确认：

| 指标 | Train chroms | Val chroms |
|---|---|---|
| Per-track Pearson (student, teacher) | 0.82–0.92 | **0.02–0.06** |
| Output std | 0.64 | 0.27 (坍缩到均值) |
| score_mag AUPRC_per_chrom | — | **0.127** (≈随机) |

**根因**: 0.87M 参数 CNN 在 2350 variants 上从头训练，数据量不足训练集的 1/1000。Borzoi (400M params) 经基因组级多任务预训练获得泛化能力，CompactStudent 无法复现。

**决策**: 放弃蒸馏路线，pivot 到 P1（后 Borzoi 聚合器）。

### Day 6–7: P1 基础设施

**LeaderBoard 复现**（`09_aggregator_chrom_loo.py`）:

| Feature Set | AUPRC_per_chrom (ours) | LeaderBoard | |Δ| |
|---|---|---|---|
| Borzoi_L2_L2 (zero-shot) | 0.4356 | 0.4356 | 0.0000 ✅ |
| Borzoi (LogReg, chrom-LOO) | **0.4930** | 0.4930 | 0.0000 ✅ |
| CADD+Borzoi (LogReg) | **0.7515** | 0.7570 | 0.0055 ✅ |
| Borzoi_complex (LogReg) | **0.2974** | — | — |

Pipeline 完全匹配 TraitGym 标准：SimpleImputer(mean) → StandardScaler → LogReg(balanced), GridSearchCV C∈logspace(-8,0,10), GroupKFold(chrom)。

**Selective reliability head**（`10_selective_head.py`）:
- 外层 chrom-LOO: 训练 GBM 预测 LogReg 残差 r_i = |p_i − y_i|
- 7 个 non-teacher-mag 特征: log1p|tss_dist|, GC, CpG, priPhyloP, mamPhyloP, verPhyloP, bStatistic
- Combined confidence = |p − 0.5| + λ·(1 − r̂)
- λ 由内层 chrom-LOO CV 选择（防止 post-hoc 过拟合）

### Day 8: 多种子稳定性验证

Seeds = {42, 7, 2024}，每个跑 Mendelian + Complex。

**Mendelian matched_9 (n=3380) — 稳定 ✅**

| Coverage | Margin (baseline) | Selective (mean±std) | Δ (mean±std) |
|---|---|---|---|
| @10% | 0.8564 | 0.8649±0.002 | **+0.009±0.002** |
| @25% | 0.7542 | 0.7967±0.008 | **+0.043±0.006** |
| @50% | 0.6349 | 0.6835±0.021 | **+0.049±0.017** |
| @75% | 0.5522 | 0.5786±0.018 | **+0.027±0.015** |
| @100% | 0.4663 | 0.4663 | 0.000 (不变) |

- λ_cv: {1.0, 0.25, 1.0}
- Spearman(r̂, residual): {0.391, 0.374, 0.377}
- 所有 3 seeds × 所有 coverage levels 均 Δ > 0

**Complex traits matched_9 (n=11400) — 弱/不显著 ⚠️**

| Coverage | Margin (baseline) | Selective (mean±std) | Δ (mean±std) |
|---|---|---|---|
| @10% | 0.5589 | 0.5692±0.002 | **+0.010±0.002** |
| @25% | 0.4583 | 0.4649±0.002 | **+0.007±0.002** |
| @50% | 0.3704 | 0.3641±0.001 | **−0.006±0.001** |
| @75% | 0.3114 | 0.3136±0.001 | +0.002±0.001 |
| @100% | 0.2819 | 0.2819 | 0.000 |

- λ_cv: {2.0, 3.0, 3.0}
- Spearman(r̂, residual): {0.221, 0.226, 0.227}
- cov@50% 所有 3 seeds 均为负（系统性）
- 仅 cov@10%/25% 有小幅正向效果

---

## 3. 现有结果评估

### 优势
1. **Pipeline 可信度高**：两个 leaderboard 数字精确匹配（|Δ|=0.0000），方法论严谨
2. **Mendelian 改进稳定**：3 seeds 全部正向，cov@25% Δ=+0.043±0.006
3. **内层 CV 防泄漏**：λ 不是 post-hoc 选择，而是 inner chrom-LOO 交叉验证
4. **Falsification-first 文化**：Day 5 主动证伪了 Day 4 头条，避免了投出虚假结果

### 劣势
1. **效应量**：Mendelian Δ≈0.05 处于 TraitGym SE bars (±0.03–0.05) 边缘
2. **Complex 不工作**：cov@50% 系统性负向，限制了方法的普适性声称
3. **新颖性低**：selective prediction (Geifman & El-Yaniv 2017) 是成熟方法，GBM on conservation = feature engineering
4. **Full AUPRC_per_chrom 不变**：selective head 仅改进高置信子集的排序，不改变整体分类性能

### 预期审稿攻击
- "conservation features 已知与致病性相关，这不是新发现"
- "complex traits 不 work → 方法不泛化"
- "LogReg margin 已经做了大部分工作，GBM reliability 是锦上添花"
- "效应量在误差线内"

---

## 4. 论文路径评估

### Path A: Coverage-AUPRC benchmark paper (推荐)
- **内容**: 对所有 TraitGym 方法（Borzoi, CADD, GPN-MSA 及组合）系统评测 coverage-AUPRC；selective head 作为案例研究
- **新颖性**: medium — coverage-AUPRC 在 NLP selective prediction 中成熟，但在基因组变异评分中未被系统研究
- **投稿**: RECOMB 2027 (截稿 ~2026-11) 或 Bioinformatics
- **工作量**: 中等（需跑所有特征组合 × coverage 曲线，写 benchmark 论文框架）

### Path B: Selective head single-method paper
- **内容**: 当前方法原样包装
- **投稿**: RECOMB short paper / workshop
- **风险**: 效应量不够，reviewer 可能直接 reject

### Path C: Falsification story (workshop)
- **内容**: Day 5 falsification 作为方法论案例 — "reliability estimators in genomic distillation: a cautionary tale"
- **投稿**: ICML GenBio Workshop 2027 / ML4H
- **工作量**: 小（材料已有）

### Path D: Conformal prediction pivot
- **内容**: 用 conformal prediction 替代 GBM，获得理论 coverage guarantees
- **投稿**: ICLR 2027 (如果理论贡献成立)
- **工作量**: 大（需全新方法开发 + 理论证明）
- **风险**: 不确定 conformal 在 chrom-LOO 设定下的有效性

---

## 5. 资源决策

### T4 现状
P1 路线仅需 CPU（sklearn GBM + LogReg），T4 GPU 已不再是瓶颈。所有实验在单核上 <5min 完成。

### 5090 申请
- 如果选择 Path A/B/C：**不需要 5090**，T4/CPU 足够
- 如果选择 Path D (conformal)：可能需要更大规模计算，但 sklearn-based 方法仍可 CPU-only
- 如果重回蒸馏路线（不推荐）：需要 5090 + 大规模预训练数据
- **建议**: 暂不申请 5090，等论文路径明确后再决定

---

## 6. 下一步行动

### 立即可做
1. **CADD+Borzoi regime 测试**: 在更强的 CADD+Borzoi LogReg (AUPRC=0.751) 上跑 selective head，验证是否有 headroom
2. **所有特征组合 × coverage 曲线**: 为 Path A 准备数据

### 需要决策
3. **论文路径选择**: A / B / C / D（或组合）
4. **时间线**: RECOMB 2027 (截稿 ~2026-11) 给了 ~7 个月；ICLR 2027 (~2026-09) 给了 ~5 个月

---

## 附录: 完整数字摘要

### A1. LeaderBoard 参考 (Mendelian matched_9)

| Method | AUPRC_per_chrom | Source |
|---|---|---|
| Borzoi_L2_L2 (zero-shot) | 0.4356 | TraitGym HF (verified) |
| Borzoi (LogReg, chrom-LOO) | 0.4930 | TraitGym HF (reproduced exact) |
| CADD+Borzoi (LogReg) | 0.7570 | TraitGym HF |
| CADD+GPN-MSA+Borzoi (LogReg) | 0.7570 | TraitGym HF |

### A2. 我们的复现

| Feature Set | Dataset | AUPRC_per_chrom | AUPRC | AUROC |
|---|---|---|---|---|
| Borzoi | Mendelian | 0.4930 | 0.4663 | 0.7752 |
| Borzoi | Complex | 0.2974 | 0.2819 | 0.6835 |
| CADD+Borzoi | Mendelian | 0.7515 | 0.7230 | 0.8740 |

### A3. Selective head 多种子 (Mendelian, Borzoi regime)

| Seed | λ_cv | Spearman | cov@10% Δ | cov@25% Δ | cov@50% Δ | cov@75% Δ |
|---|---|---|---|---|---|---|
| 42 | 1.0 | 0.391 | +0.007 | +0.050 | +0.056 | +0.036 |
| 7 | 0.25 | 0.374 | +0.007 | +0.034 | +0.025 | +0.006 |
| 2024 | 1.0 | 0.377 | +0.011 | +0.043 | +0.065 | +0.038 |
| **mean** | — | 0.381 | **+0.009** | **+0.043** | **+0.049** | **+0.027** |

### A4. Selective head 多种子 (Complex, Borzoi regime)

| Seed | λ_cv | Spearman | cov@10% Δ | cov@25% Δ | cov@50% Δ | cov@75% Δ |
|---|---|---|---|---|---|---|
| 42 | 2.0 | 0.221 | +0.010 | +0.007 | −0.006 | +0.003 |
| 7 | 3.0 | 0.226 | +0.012 | +0.009 | −0.006 | +0.001 |
| 2024 | 3.0 | 0.227 | +0.009 | +0.005 | −0.008 | +0.002 |
| **mean** | — | 0.224 | **+0.010** | **+0.007** | **−0.006** | **+0.002** |
