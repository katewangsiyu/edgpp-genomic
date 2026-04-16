# Day 10 — 完整 feature set 消融 + complex conformal + 论文框架

## 1. Feature set 消融（Mendelian matched_9，n=3380）

| Feature Set | dim | LogReg AUPRC_per_chrom | GBM AUPRC_per_chrom | Δ (GBM−LR) |
|---|---:|---:|---:|---:|
| Borzoi_L2_L2 | 6 | 0.488 | 0.475 | −0.013 |
| Borzoi (L2+L2_L2) | ~3730 | 0.493 | 0.489 | −0.004 |
| CADD | 114 | **0.871** | 0.872 | +0.001 |
| CADD+Borzoi_L2_L2 | 120 | — | 0.883 | — |
| CADD+Borzoi | 7731 | 0.752 | **0.889** | **+0.137** |
| CADD+GPN-MSA+Borzoi | 8501 | 0.749 | **0.900** | **+0.151** |

**核心观察**（按重要性排序）：

1. **TraitGym 自身不一致**：TraitGym 论文把 CADD+Borzoi (0.7515) 作为 supervised SOTA 报告，但 **CADD-only LogReg (0.8713) 实际上比声称的 SOTA 高 12pp**。我们用官方 pipeline 精确复现出这个问题——**加入 Borzoi 让 LogReg 更差**。
2. **LogReg 在高维上退化的机制**：GridSearchCV 的 best C 从 CADD (C=0.13) 跌到 CADD+Borzoi (C=0.002–0.02) 再到 CADD+GPN-MSA+Borzoi (C=0.0003)。p>>n 时 L2 过度收缩所有系数 → 欠拟合。
3. **GBM 完全相反的行为**：GBM 从 CADD (0.872) → CADD+Borzoi (0.889) → CADD+GPN-MSA+Borzoi (0.900) 单调提升。GBM 的决策树天然做特征选择，不受维度诅咒影响。
4. **关键 claim**：GBM (0.900) vs 原 leaderboard SOTA (0.7515) = **+14.9pp**。且同时 vs CADD-only LogReg (0.871) = **+2.9pp**（避免 reviewer 攻击 "只是因为 LogReg 在高维差"）。
5. **GPN-MSA 边际贡献 +1.1pp**：在强分类器（GBM）下，新特征还能提供信号；在弱分类器（LogReg）下反而有害。

## 2. Feature set 消融（Complex traits matched_9，n=11400）

| Feature Set | dim | LogReg | GBM | Δ |
|---|---:|---:|---:|---:|
| Borzoi_L2_L2 | 6 | — | 0.248 | — |
| Borzoi | ~3730 | 0.297 | 0.276 | −0.021 |
| CADD | 114 | — | 0.295 | — |
| CADD+Borzoi_L2_L2 | 120 | — | 0.329 | — |
| CADD+Borzoi | 7731 | 0.351 | 0.350 | −0.001 |

**Complex traits 无 GBM 优势**：信号弱（best ~0.35）、样本更多（11400）、LogReg 正则化足够。GBM 在此处不过拟合也不提升。

## 3. Class-conditional conformal prediction

### Mendelian（n=3380，pos=10%）

| Base | Method | Coverage | Cov\|pos | Cov\|neg | Singleton | Both |
|---|---|---:|---:|---:|---:|---:|
| GBM (CADD+Borzoi) | CV+ | 0.900 | **0.728** | 0.920 | 0.915 | 0.000 |
| GBM (CADD+Borzoi) | **Class-cond** | **0.899** | **0.888** | **0.900** | **0.970** | 0.011 |
| LogReg (CADD+Borzoi) | CV+ | 0.900 | 0.601 | 0.933 | 0.977 | 0.000 |
| LogReg (CADD+Borzoi) | **Class-cond** | **0.899** | **0.894** | **0.899** | 0.541 | 0.459 |

### Complex（n=11400，pos=10%）

| Base | Method | Coverage | Cov\|pos | Cov\|neg | Singleton | Both |
|---|---|---:|---:|---:|---:|---:|
| GBM (CADD+Borzoi) | CV+ | 0.900 | **0.721** | 0.919 | 0.689 | 0.311 |
| GBM (CADD+Borzoi) | **Class-cond** | **0.900** | **0.900** | **0.900** | 0.390 | 0.610 |
| LogReg (CADD+Borzoi) | CV+ | 0.900 | **0.693** | 0.923 | 0.686 | 0.314 |
| LogReg (CADD+Borzoi) | **Class-cond** | **0.900** | **0.900** | **0.900** | 0.361 | 0.639 |

**核心观察**：

1. **CV+ 的病态一致存在**：所有 4 个配置下 Cov\|pos 都在 60–73%，远低于名义 90%。这是类别不平衡（pos 10%）下 marginal coverage 的已知缺陷。
2. **Class-conditional 普遍修复**：所有 4 个配置下 Cov\|pos ≈ Cov\|neg ≈ 0.9。
3. **Singleton 率分层**：
   - Mendelian + GBM：97%（强信号 + 强模型 → 决策果断）
   - Mendelian + LogReg：54%（强信号但模型弱 → 大量模糊）
   - Complex + GBM/LogReg：~39%（弱信号 → 诚实的不确定性）
4. **模型强度决定决策力**：强基础模型（GBM）在 Mendelian 上给出 97% 确定集合，弱基础模型（LogReg）给出 54%，说明 conformal 放大了基础模型的信息量。

## 4. 论文框架（RECOMB 2027 / ICLR 2027）

**标题候选**：
- A: "Revisiting Variant Effect Prediction: Gradient Boosting Surpasses Logistic Regression with Conformal Uncertainty"
- B: "Beyond TraitGym: Nonlinear Aggregation and Class-Conditional Conformal Prediction for Genomic VEP"
- C: "When Logistic Regression Fails: High-Dimensional Feature Aggregation and Honest Uncertainty for Variant Scoring"

**贡献声明（4 条）**：

1. **新 SOTA + TraitGym 内部不一致证据**：0.900 AUPRC_per_chrom on TraitGym Mendelian matched_9 (+14.9pp over 0.7515 官方 leaderboard SOTA)。同时揭露 TraitGym 的 supervised baseline 表内 CADD-only LogReg = 0.871 实际高于 "SOTA" CADD+Borzoi LogReg = 0.752 by 12pp，表明该 benchmark 的 pipeline 设计有隐含问题。
2. **LogReg 在高维 VEP 特征集上的普遍退化**：p=7731 时 best C 被 GridSearchCV 选到 0.002，几乎将所有系数收缩到 0；而 GBM 的决策树内建特征选择，随维度增加单调提升。**这挑战 TraitGym benchmark 的 classifier 选择**。
3. **首次 conformal prediction for genomic VEP**：CV+ 和 class-conditional conformal 在 TraitGym 上校准完全（|cov−target|<0.002），但 CV+ 在 pathogenic 类 (pos=10%) 上 Cov\|pos 只有 60–73%。
4. **Class-conditional conformal 修复 pathogenic under-coverage**：所有 4 个 (model, dataset) 配置下 Cov\|pos 统一提升到 ~0.90。对临床相关的病原变异提供类别级别保证——这是临床部署的必要条件。

**论文结构**（~8 页 RECOMB / 9 页 ICLR）：

```
1. Introduction
   - TraitGym benchmark + leaderboard status (Borzoi 0.436 zero-shot, CADD+Borzoi 0.757 supervised)
   - DEGU 作为相关工作（异方差 NLL，无 TraitGym 数字）
   - Our contributions (上述 4 条)

2. Background
   2.1 TraitGym: chrom-LOO，AUPRC_by_chrom_weighted_average 口径
   2.2 Feature sets: CADD (114), Borzoi (3730), GPN-MSA (4770)
   2.3 Conformal prediction: CV+, LAC nonconformity score

3. Methods
   3.1 GBM aggregator: HistGradientBoostingClassifier, depth=2, iter=100, class_weight=balanced
   3.2 Class-conditional conformal: 分类别 calibration 保证 class-wise coverage
   3.3 Evaluation: matched_9 (Mendelian, Complex), 严格复现 TraitGym baseline (Borzoi_L2_L2=0.493)

4. Results
   4.1 Feature set ablation (Mendelian + Complex table)
   4.2 Conformal calibration: α sweep + CV+ vs class-conditional
   4.3 Per-chrom stability
   4.4 Coverage-AUPRC selective prediction

5. Analysis
   5.1 为什么 LogReg 退化：GridSearchCV 的 C 选到极小（~0.002），相当于欠拟合
   5.2 GBM 的 depth=2 泛化机制：浅树 + 类别平衡避免过拟合，但能捕获 CADD × Borzoi 交互
   5.3 Singleton rate 分解：signal strength × model strength
   5.4 Mendelian vs Complex 对比：强信号 vs 弱信号下的 uncertainty behavior

6. Discussion
   - Benchmark design implication：TraitGym leaderboard 受 classifier 选择限制
   - 临床意义：class-conditional 给出的 Cov\|pos 是 pathogenic variant 必需
   - Limitations：matched_9 仍是 ~10% pos 的人工数据，真实稀有

7. Related Work: DEGU, selective classification, conformal prediction for medicine
8. Conclusion

Appendix:
  A. 完整 hyperparameter grid
  B. Multi-seed 稳定性（seeds 42, 7）
  C. 失败路线：selective reliability head (P1) 的 falsification
```

**亮点实验（支撑 reviewer 问题）**：
- 确定性证据：seed 42/7 给 0.889/0.889（HistGB 本身基本确定性）
- 复现验证：Borzoi_L2_L2 LogReg = 0.493 精确匹配 leaderboard 0.493
- 多维度覆盖：Mendelian（强信号）+ Complex（弱信号）双数据集
- 消融彻底：6 → 8501 维特征 5 档

**风险 & 备选**：
- 风险：reviewer 可能认为 "GBM > LogReg" 是 trivial finding
  - 反驳：关键 contribution 是 **class-conditional conformal** + **揭露 TraitGym pipeline 高维退化**
- 如果 RECOMB 不接收，可转 NAR Genomics & Bioinformatics 或 Bioinformatics
- 如果审稿要求：加 XGBoost / LightGBM 对比（预期差不多，sklearn HGB 已够）

## 5. 待办

- [ ] 更新 `outputs/aggregator_gbm/` 里面 CADD+Borzoi 的 metric 换成 seed 42（当前是默认 42 但 seed=7 结果相同，确认）
- [ ] 写 `scripts/13_compile_ablation.py` 自动生成表格（LaTeX 输出）
- [ ] 可选：跑 XGBoost 对比作为 sanity check
- [ ] 论文写作：先 Methods + Results 两章做到可读，再回头写 Introduction
- [ ] 目标 bioRxiv preprint：2026-08，给 RECOMB 2027 deadline (~2026-11) 留 3 个月 polish

## 关键数字 cheat sheet

| 口径 | 数字 |
|---|---|
| TraitGym leaderboard SOTA (supervised) | 0.757 |
| 我们的 GBM (CADD+Borzoi) | 0.889 (+13.2pp) |
| 我们的 GBM (CADD+GPN-MSA+Borzoi) | **0.900 (+14.3pp)** |
| Complex traits GBM (CADD+Borzoi) | 0.350 ≈ LogReg 0.351 |
| Conformal α=0.10 coverage | 0.899–0.900 (误差 < 0.002) |
| Class-cond Cov\|pos (all 4 configs) | 0.888–0.900 (vs CV+ 0.60–0.73) |
| Mendelian GBM + class-cond singleton | 97% (决策力极强) |
| Complex GBM + class-cond singleton | 39% (诚实不确定性) |
