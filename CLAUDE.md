# EDG++ Genomic — Claude Code 项目指南

## 日录组织规则(必须遵守)
本项目下所有新建文件必须按CDTR 四层组织，不要平铺在根目录。
-C_context/   #背景资料、需求文档、参考(我提供的输入)
-D_deliverables/   #我真的会拿出去用的最终产物(不超过 3-5个)
-R_raw/   中间产物、实验、草稿(过程需要但不真用)
-T_tools/   过程里做出来的可复用脚本/小工具
不确定某个新文件属于哪一层时，先问我。
## 硬性目标（不可动）

**论文出口**:
- **P0（硬目标，无时间约束）**: NeurIPS 2027 main (~2027-05) — Path A conformalized heteroscedastic VEP
- **P1（保底）**: NeurIPS 2026 D&B Track (~2026-06) — Day 10 GBM+conformal 素材独立打包
- **P2（期刊保底）**: npj AI（直接对标 DEGU）
- **里程碑**: 2026-08 bioRxiv preprint 占坑

**评测基准**: TraitGym (Benegas/Eraslan/Song, bioRxiv 2025)
- Mendelian matched_9（3380 变异）+ complex_traits matched_9
- 切分：`val = chr{17,18,19,20,21,22,X}`（TraitGym 惯例，不得改）
- 主指标：**`AUPRC_by_chrom_weighted_average`**（leaderboard 唯一认可口径）
- 副指标：AUPRC、AUROC、Brier、ECE

**要击败的对手**: DEGU (Zhou, Rizzo, Christensen, Tang, Koo, npj AI 2026) —— **Deep Ensemble with Gaussian Uncertainty**（非异方差 NLL；student 用标准 MSE 拟合 [ensemble_mean, ensemble_logvar]）。其 heteroscedastic NLL 只是对照 baseline。DEGU **abstract 已提 conformal prediction**，是 Path A 的直接威胁。
- DEGU 官方 repo：https://github.com/zrcjessica/ensemble_distillation（TF/Keras, MPRA 回归）
- DEGU **无 TraitGym 数字**，**无 Borzoi pipeline**，无分类 loss
- 差异化点：我们做 VEP 分类 + heteroscedastic NLL + class-conditional + local coverage theorem

**关键发现（Day 5–8 已验证）**:
- Day 4 头条"score_difficulty AUPRC=0.551"**已被证伪**：reliability MLP 通过 side_features 中的 ||teacher|| 泄漏了教师幅度信号；去除后崩塌至 0.170
- CompactStudent 从头蒸馏**不可行**：5000 step 训练后 train Pearson~0.9 / val Pearson~0.03（0.87M 参数 CNN 在 2350 变异上无法泛化）
- **已 pivot 到 P1 路线**：后 Borzoi 聚合器 + 选择性可靠度头

**当前方法（Path A — HCCP, Heteroscedastic Class-Conditional Conformal Prediction）**:
- Aggregator: Day 10 的 GBM (CADD+Borzoi) 0.889 / (CADD+GPN-MSA+Borzoi) **0.900** 作为 base
- Heteroscedastic reliability head: σ̂(x) 预测 chrom-LOO |residual|，两种实现（GBM regressor / joint NN）
- Feature-dep nonconformity score: s(x,y) = |y - p̂(x)| / σ̂(x)
- Mondrian (y × σ̂-bin) 联合分层 conformal calibration —— **唯一同时满足 class-cond + bin-local 的 Pareto 点**（vs RLCP/wCP/SC-CP empirical H2H）
- 理论 hero（NeurIPS 2027 main 的 load-bearing claim，Phase 4 重构后）：
  - **T3'** operational binding guarantee on Mendelian（σ̂-bin KS rejection 46%，K-invariant），是 hero 关键
  - **T5.1** $K^\star = \lfloor\sqrt{L_F R \pi_{\min} n}\rfloor$，$G(K^\star) = O(n^{-1/2})$，dimension-free
  - **T5.2** "tight rate within equi-bin Mondrian-K class"（**已弱化**：对手 class 限制为 equi-bin Mondrian-K；SC-CP 在不同 axis 上达 $O(n^{-2/3})$）
  - 辅助：T3 bin-cond exact / T1/T2 corollaries / T1'/T2'/T3'/T4 Barber 2023 robust forms / T3-loc / T3.b（全部下沉 App A）

**历史方法状态**:
- Day 0–5 CompactStudent 蒸馏 + reliability MLP：**CLOSED**（已 falsified）
- Day 6–8 P1 selective head：**CLOSED**（CADD+Borzoi λ=0，feature overlap）
- Day 9–10 GBM + class-cond conformal：**ACTIVE**，作为 Path A baseline 与 D&B 保底素材

**硬件现实**: T4×8 (SM 7.5，**无 BF16 硬件**)。Path A 的 aggregator 与 ensemble 主要 CPU（sklearn），joint NN 版 hetero head 用 GPU；DEGU reimpl 需要 TF/PyTorch + GPU。

## 软性方法（可变）

- Destination 不变，path 可换。跑通全流程之前不要做二次抽象。
- Falsification-first：新想法必须先跑 null-baseline + 泄漏检查，再上多种子验证。
- 超参不预设最优值。新想法先 smoke，通过再上全量。
- P1 路线的控制变量：LogReg 聚合器完全复现 TraitGym pipeline（SimpleImputer → StandardScaler → LogReg balanced, GridSearchCV C ∈ logspace(-8,0,10), GroupKFold(chrom)），差异只在 confidence scoring。

## 当前进度（2026-04-29）

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

**Day 20 — T5 Adaptive K + DEGU 对比（已完成）**:
- T5 理论：Mondrian bin count bias-variance tradeoff, K* = O(√n), rate O(n^{-1/2}) dimension-free
- K-sweep 验证：K_CV=3 (Mendelian) / K_CV=2 (Complex), 优于固定 K=5 达 6-7 倍
- DEGU-lite (CADD+GPN-MSA+Borzoi) head-to-head 完成，4.3× σ̂-bin gap 改善
- 文献调研完成：Dewolf 2025, Hore & Barber RLCP 2025, AlphaGenome API 可用

**Day 22 — Joint Pareto frontier 故事（commit 53dd550 / 46e0096）**:
- §6.3 H2H 用 crepes 跑 B1-B4 (vanilla split / σ̂-Mondrian / class-Mondrian / HCCP)
- 不平衡 sweep π_+ ∈ {0.05, ..., 0.50} 验证 T5.1 的 π_min^{-1/2} 显式入 constant
- π_min 显式 rate 写入 abstract：HCCP 是 imbalanced + heteroscedastic 联合下唯一可行的 single-fold partition

**Phase 3 — paper narrative 重构 + 12 实验/可视化 TODO（commit 65de2fe, 2026-04-26）**:
- Logic check 修复 4 NC + 2 AC + 1 UC issue（abstract / §1 / §6 / §7 / §8 一致性）
- Abstract 重写为 5-句 SPJ theory-first 结构
- §1 contribution 4→3 + 加 bridge 段对齐 abstract
- §5 Theorem 11→3 主文 + 6 lemma 物理下沉 App A
- DEGU 从 "villain" 重定位为 "ally / pioneer"（§1 / §2 / §6.4）
- 4 张已有图插入主文（K-sweep / Mendelian frontier / per-genomic-region heatmap / per-chrom forest）
- **新跑** RLCP/weighted CP/SC-CP empirical H2H：Complex K=2 HCCP gap 0.010 vs 三 baseline 0.79-0.80（**注**：旧 80× 数字基于 invalid CV，已被 Phase 4 推翻）
- **新跑** synthetic n-sweep：log-log slope -0.50±0.05，验证 dimension-free O(n^{-1/2}) + π_min^{-1/2}
- **新跑** trait clustering：Complex 3 cluster all marg ∈ [0.896, 0.905]；Mendelian Skeletal cluster 暴露 cov_y1=0.745（诚实报告）
- 新画 L_F estimator forest + bootstrap density figs
- 新增 App F synthetic conditional-coverage validation
- PDF 35 → 43 页，零 undefined ref，零 TODO 残留
- §5 collapsed 207 → 96 行 (-54%)

**Phase 4 — nested CV + theory rewrite（commits 3a02917→529c33f, 2026-04-29）**:
- **K-selection 修复**：旧 K_cv = argmin chrom-LOO test-set worst-cell gap **是 invalid CP exchangeability violation**（test-set tuning leak）。新方案 nested chrom-LOO inner CV with K_eval-fair scoring（`T_tools/nested_kcv_helpers.py`）
- **诚实数字替换**：旧 80× HCCP advantage **作废** → 新 33× (Complex) / 2.3× (Mendelian) over class-Mondrian
- **theory framing 重构**：T3' 升 hero（KS rejection 46% on Mendelian, K-invariant），T5.2 弱化为 "tight rate within equi-bin Mondrian-K class"
- **§6 外科手术**：DEGU-actual MPRA→TraitGym port 删除（strawman）；M→C 从 OOD 行降级为 §6.6 失败模式；AlphaGenome 移 §6.5 ablation；ProteinGym 升主文 §6.5
- **Phase 5 部分**：KS audit K-sweep on Mendelian（commit 1da8ce2）+ A_SL audit（commit 27da463, B 阶段）

**Phase 6 — audit-driven fixes（commits 4537cf2 + 08d2bf9, 2026-04-29 evening）**:
- **Phase 4 surgery 真完结**：3 audit agent (logic + self-review + citation) 揭示 memory 说删了实际还在的 §C.4 + 4 章 K_CV ∈ {2,3} / 6-7× 残留语句，全部清理
- **T3' 框架软化**：从 "binding/load-bearing guarantee" 改为 "operative certificate; non-vacuous lower bound"，明确 "loose by construction; bounds worst case, does not predict empirical gap"
- **T5.2 框架加 qualifier**：abstract / §1 / §5 / §8 / SC-CP 比较段全部加 "within the equi-bin Mondrian-K family / class"
- **K_eval formula 修正**：旧 paper 说 K_eval = floor(n·π_min/30) clamp [2,10] 数学上给 K=10/10，与实际值 K=3/5 不符；新 justification "保持 ≥100 minority/cell"
- **K_eval sensitivity sweep**（review HIGH-3 防御）：跑 K_eval ∈ {2,3,5,8,10} 加 §C.1 sensitivity table。**重大诚实 finding**：Mendelian 仅在 K_eval=3 推荐区赢 wCP 1.15×, K_eval ≥ 5 输 wCP/RLCP（per-cell minority < 70 让 hetero head 失效）；Complex robust win 3-34× 跨 K_eval [2,10]
- **数字一致性**：KS audit (Mendelian 46.2% / Complex 10.4% / 177 tests / max 0.434) 替换 stale "12.3% / 155 tests / max 0.24"; M→C 主文 vs Tab 不一致修复; δ_TV canonical 0.31; Skeletal 子簇 cov_{|Y=1}=0.745 在 §6.3 obs 2 诚实声明
- **3 citation placeholder 修复**：pcos2025conformal (Adegoke et al), lcls2024lipschitz (Huang/Roberts/Calliess TMLR), crps_binning2026 (Toccaceli)
- **AlphaGenome 防御 justification**：§6.5 加 "Why we keep CADD+GPN-MSA+Borzoi as headline" 段，conformal claim 与 base predictor 解耦
- PDF 46 页, 0 undefined ref, 0 stale Phase 4 残留

**目标**: NeurIPS 2027 main. **Theory: T5.1 oracle (dimension-free O(n^{-1/2})) + T5.2 matching lower bound on equi-bin Mondrian-K class + T3' robustness corollary**. Instantiated as HCCP, validated empirically on TraitGym (3-axis OOD) + ProteinGym (cross-domain) + synthetic n-sweep + K_eval sensitivity (Mendelian honest fragility flagged at K_eval ≥ 5)。

## 仓库操作

- 远程：`https://github.com/katewangsiyu/edgpp-genomic.git` —— **不是** `wsy` 仓库。
- 分支：`main`（**不是** master）。
- 提交：中文，格式 `动作：具体内容`（例 `修改：…` / `添加：…` / `删除：…`）。
- 完成一项工作后主动 `git add` 指定文件 → `commit` → `push`（同根 CLAUDE.md 规则）。
- 大文件 gitignore 已含 `*.pt`、`*.parquet`、`data/raw/`。新增大文件类型先更 gitignore 再提交。

## 代码结构速览

```
configs/            YAML 超参
scripts/            历史脚本（编号前缀 = 流程阶段；Phase 0-2 产物）
  06-10_*.py          Phase 0-1：训练/评测/聚合器/selective（部分已 sunset）
  11_aggregator_gbm.py    Day 10 GBM SOTA aggregator
  13_hetero_head.py       σ̂(x) head 训练
  14_conformal_hetero.py  HCCP main pipeline (Mondrian y×σ̂-bin)
  15_eval_local_coverage.py  per-axis local coverage eval
  16_degu_lite.py     DEGU-lite GBM ensemble baseline
  17_trait_loo.py     trait-LOO cross-validation
  18_cross_dataset.py M↔C cross-dataset transfer
  20_adaptive_K_sweep.py  T5 K-sweep validation
  21_crepes_baseline.py   B1-B4 crepes H2H
  22_imbalance_sweep.py   π_+ ∈ [0.05, 0.50] sweep

T_tools/            Phase 3-4 新增：可复用脚本（CDTR 规则）
  synthetic_n_sweep.py     T5 rate validation + T3.b drift（synthetic Gaussian-shift）
  cp_baselines_h2h.py      RLCP/weighted CP/SC-CP empirical H2H（支持 --K-mode {fixed, nestedcv}）
  trait_cluster_aggregate.py  28 traits + 30 OMIM 临床聚类
  lf_and_bootstrap_figs.py    L_F forest + bootstrap density 可视化
  nested_kcv_helpers.py    Phase 4: nested chrom-LOO K-selection 单一来源（211 行）

R_raw/              Phase 3-4 中间产物（CDTR 规则）
  synthetic_n_sweep/results.json
  cp_baselines_h2h/{complex,mendelian}_nestedcv_Keval{3,5,10}.json   # Phase 4 主结果
  cp_baselines_h2h/{complex_K2, mendelian_K3}.json                    # Phase 3 旧结果（参考）
  trait_cluster_aggregate/{complex, mendelian}_cluster.json

src/edgpp_genomic/
  data/traitgym.py    数据适配器（teacher 预计算，支持 side_features_mode）
  models/             CompactStudent + reliability MLP（Phase 0 产物，已 sunset）
  training/selective.py  SelectiveDistillLoss（Phase 0 产物，已 sunset）

outputs/            实验产物（按 method × dataset × feature-set 组织）
  aggregator_gbm/     Day 10 GBM 复现
  hetero_head/        σ̂(x) head 输出（含 scores_with_sigma.parquet）
  conformal_hetero/   HCCP main results
  trait_loo/          per-trait CSV
  bootstrap_ci/       B=200 chrom-bootstrap raw replicates
  crepes_baseline/    B1-B4 crepes H2H 结果
  imbalance_sweep/    π_+ sweep
  adaptive_K/         K-sweep
  L_F_audit/          L_F estimator 比较

papers/neurips2027_pathA/  论文工作目录
  main.tex / sections/{01-08,A-F}*.tex
  figures/            13 张已生成图（fig1-9 + fig_n_sweep + fig_lf_forest + fig_bootstrap_density + fig_frontier_*）
  refs.bib            120+ refs
  README.md           paper status
```

## 交互

- 中文优先，简洁。
- 目标/路径不清先问；路径非最短明说并给备选（承袭根 CLAUDE.md 第一性原理）。
