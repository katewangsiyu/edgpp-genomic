# EDG++ Genomic — Claude Code 项目指南

## 硬性目标（不可动）

**论文出口**: P0 = ICLR 2027 (~2026-09 截稿) 或 RECOMB 2027 (~2026-11 截稿)。P1 = Nature Methods rolling。里程碑：2026-08 bioRxiv preprint。

**评测基准**: TraitGym (Benegas/Eraslan/Song, bioRxiv 2025)
- Mendelian matched_9（3380 变异）+ complex_traits matched_9
- 切分：`val = chr{17,18,19,20,21,22,X}`（TraitGym 惯例，不得改）
- 主指标：**`AUPRC_by_chrom_weighted_average`**（leaderboard 唯一认可口径）
- 副指标：AUPRC、AUROC、Brier、ECE

**要击败的对手**: DEGU (Zhou & Koo, npj AI 2026) —— 蒸馏 + 异方差 NLL 的直接竞品。注意：DEGU 尚无 TraitGym 公开结果。

**关键发现（Day 5–8 已验证）**:
- Day 4 头条"score_difficulty AUPRC=0.551"**已被证伪**：reliability MLP 通过 side_features 中的 ||teacher|| 泄漏了教师幅度信号；去除后崩塌至 0.170
- CompactStudent 从头蒸馏**不可行**：5000 step 训练后 train Pearson~0.9 / val Pearson~0.03（0.87M 参数 CNN 在 2350 变异上无法泛化）
- **已 pivot 到 P1 路线**：后 Borzoi 聚合器 + 选择性可靠度头

**当前方法（P1 selective head）**:
在复现 TraitGym 标准 LogReg 聚合器（Borzoi → 0.4930，完美匹配 leaderboard）基础上，训练 GBM 预测 LogReg 残差 |p_i − y_i|，combined confidence = margin + λ·reliability。λ 由内层 chrom-LOO CV 选择。多种子（3 seeds）Mendelian 稳定 Δ>0，complex 弱/不显著。

**硬件现实**: T4×8 (SM 7.5，**无 BF16 硬件**)。P1 仅需 CPU（sklearn GBM），不受 GPU 限制。

## 软性方法（可变）

- Destination 不变，path 可换。跑通全流程之前不要做二次抽象。
- Falsification-first：新想法必须先跑 null-baseline + 泄漏检查，再上多种子验证。
- 超参不预设最优值。新想法先 smoke，通过再上全量。
- P1 路线的控制变量：LogReg 聚合器完全复现 TraitGym pipeline（SimpleImputer → StandardScaler → LogReg balanced, GridSearchCV C ∈ logspace(-8,0,10), GroupKFold(chrom)），差异只在 confidence scoring。

## 当前进度（2026-04-15）

**Phase 0 — 已完成（Day 0–4）**:
- 基础设施：hg38 下载、TraitGym 适配器、CompactStudent、SelectiveDistillLoss
- 3-way ablation + Day 4 稳定性三修复

**Falsification — 已完成（Day 5）**:
- side_features 泄漏证伪 → Day 4 头条作废
- CompactStudent 蒸馏不可行（5000 step 长训练确认）
- 评测方法论修正（val-only → full-test leaderboard 口径）

**P1 Pivot — 已完成（Day 6–8）**:
- `08_zero_shot_eval.py`：Borzoi_L2_L2 zero-shot = 0.4356（精确匹配 leaderboard）
- `09_aggregator_chrom_loo.py`：Borzoi.LogReg = 0.4930（精确匹配），CADD+Borzoi = 0.7515
- `10_selective_head.py`：7feat+GBM selective head，内层 CV 选 λ
- 多种子（seeds={42,7,2024}）：Mendelian 稳定正向（cov@25% Δ=+0.043±0.006），complex 弱

**待定**:
- 论文路径选择（A: coverage-AUPRC benchmark / B: selective head / C: falsification workshop / D: conformal pivot）
- CADD+Borzoi regime 下 selective head 测试（预期 headroom 更小）
- Phase 0 报告 `reports/t4_phase0.md`

## 仓库操作

- 远程：`https://github.com/katewangsiyu/edgpp-genomic.git` —— **不是** `wsy` 仓库。
- 分支：`main`（**不是** master）。
- 提交：中文，格式 `动作：具体内容`（例 `修改：…` / `添加：…` / `删除：…`）。
- 完成一项工作后主动 `git add` 指定文件 → `commit` → `push`（同根 CLAUDE.md 规则）。
- 大文件 gitignore 已含 `*.pt`、`*.parquet`、`data/raw/`。新增大文件类型先更 gitignore 再提交。

## 代码结构速览

```
configs/            YAML 超参
scripts/            编号前缀 = 流程阶段
  06_train.py         单方法训练（baseline/degu/edgpp）— Phase 0，已 sunset
  07_evaluate.py      扫 score_* 列计算指标（仅 val-chroms，非 leaderboard 口径）
  08_zero_shot_eval.py  leaderboard 口径 zero-shot 评测 + sanity gate
  09_aggregator_chrom_loo.py  TraitGym LogReg 监督聚合器复现
  10_selective_head.py  P1 核心：7feat+GBM selective reliability head
  run_ablation.py     3-way 完整驱动（Phase 0，已 sunset）
src/edgpp_genomic/
  data/traitgym.py    数据适配器（teacher 预计算，支持 side_features_mode）
  models/             CompactStudent + reliability MLP（Phase 0 产物）
  training/selective.py  SelectiveDistillLoss（Phase 0 产物）
outputs/
  aggregator/         09 复现结果（Borzoi, CADD+Borzoi 等）
  selective/          10 selective head 结果（含多种子）
```

## 交互

- 中文优先，简洁。
- 目标/路径不清先问；路径非最短明说并给备选（承袭根 CLAUDE.md 第一性原理）。
