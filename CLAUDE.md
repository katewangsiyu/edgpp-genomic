# EDG++ Genomic — Claude Code 项目指南

## 硬性目标（不可动）

**论文出口**: P0 = ICLR 2027 (~2026-09 截稿) 或 RECOMB 2027 (~2026-11 截稿)。P1 = Nature Methods rolling。里程碑：2026-08 bioRxiv preprint。

**评测基准**: TraitGym (Benegas/Eraslan/Song, bioRxiv 2025)
- Mendelian matched_9（3380 变异）+ complex_traits matched_9
- 切分：`val = chr{17,18,19,20,21,22,X}`（TraitGym 惯例，不得改）
- 主指标：**`AUPRC_by_chrom_weighted_average`**（leaderboard 唯一认可口径）
- 副指标：AUPRC、AUROC、Brier、ECE

**要击败的对手**: DEGU (Zhou & Koo, npj AI 2026) —— 蒸馏 + 异方差 NLL 的直接竞品。

**论文头条候选（单种子，未经多 seed 验证）**:
`score_difficulty = 1 − reliability` 在 Mendelian AUPRC_per_chrom = **0.551**，DEGU 0.154 (+258%)。解释：reliability estimator 学到的是"学生对老师的拟合难度"，Mendelian 致病变异恰好位于 Borzoi 最难拟合区。**多种子 + complex_traits 跨数据集验证未完成前，这仍是 hypothesis 而非 claim。**

**硬件现实**: T4×8 (SM 7.5，**无 BF16 硬件**)。FP16 必须配 teacher z-score 归一化。5090 申请待 Phase 0 报告定。

## 软性方法（可变）

- Destination 不变，path 可换。跑通全流程之前不要做二次抽象。
- Baseline / DEGU / EDG++ 共享 student 架构、数据、优化器、step budget，**差异只在 loss**（这是控制变量的硬约束）。
- 超参不预设最优值。新想法先 smoke（~200 变异 × ~50 step），通过再上全量。
- 训练不稳定不靠 warm-up / grad-clip 压住：先定位根因（FP16 溢出 / reliability 坍缩 / 数据泄漏 / 公式错）。
- Score 公式都在 `infer_scores` 里做**后验组合**，不嵌入训练目标；新 score 列名一律 `score_*` 前缀，`07_evaluate.py` 自动扫。

## 当前进度（2026-04-13）

**已完成（Day 0–4）**:
- 基础设施：hg38 下载、TraitGym 适配器（teacher 预计算，不 forward）、CompactStudent、SelectiveDistillLoss
- 3-way ablation 驱动：`scripts/run_ablation.py`
- Day 4 稳定性三修复：teacher z-score 归一化 / w_reg 防 reliability 坍缩 / score 公式反演

**未做**:
- complex_traits_matched_9 跨数据集验证
- 多种子 (≥3 seeds) 稳定性
- Phase 0 报告 `reports/t4_phase0.md`
- 5090 申请决策

## 仓库操作

- 远程：`https://github.com/katewangsiyu/edgpp-genomic.git` —— **不是** `wsy` 仓库。
- 分支：`main`（**不是** master）。
- 提交：中文，格式 `动作：具体内容`（例 `修改：…` / `添加：…` / `删除：…`）。
- 完成一项工作后主动 `git add` 指定文件 → `commit` → `push`（同根 CLAUDE.md 规则）。
- 大文件 gitignore 已含 `*.pt`、`*.parquet`、`data/raw/`。新增大文件类型先更 gitignore 再提交。

## 代码结构速览

```
configs/            YAML 超参（t4_ablation.yaml 为当前主配置）
scripts/            编号前缀 = 流程阶段
  06_train.py         单方法训练（baseline/degu/edgpp）
  07_evaluate.py      扫 score_* 列计算全部指标
  run_ablation.py     3-way 完整驱动
src/edgpp_genomic/
  data/traitgym.py    数据适配器（teacher 预计算）
  models/student.py   CompactStudent
  models/reliability.py
  training/selective.py  SelectiveDistillLoss（EDG++ 核心）
outputs/ablation/{method}/  scores.parquet + metrics.json + 权重
```

## 交互

- 中文优先，简洁。
- 目标/路径不清先问；路径非最短明说并给备选（承袭根 CLAUDE.md 第一性原理）。
