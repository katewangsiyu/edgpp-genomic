# DEGU 复现可行性评估报告

日期：2026-04-16
状态：Phase 1 文献 harvest 的一部分（task #7）

---

## 0. 关键校正（必读）

在进入细节前，必须澄清 `CLAUDE.md` 中关于 DEGU 的两处描述与事实不符，因为这会决定复现计划的走向：

| CLAUDE.md 当前描述 | 经一手源码/论文核实后的事实 |
|---|---|
| "DEGU = Distillation with Epistemic/aleatoric Gaussian Uncertainty" | **DEGU = Deep Ensemble with Gaussian Uncertainty**（源码 `degu.py` 类 docstring 原文） |
| "蒸馏 + 异方差 NLL 的直接竞品" | **DEGU 本身不用异方差 NLL**。Student 的 loss 是 **标准 MSE**，target 是 `[ensemble_mean ‖ ensemble_logvar]` 的拼接向量。异方差 NLL（Nix–Weigend 式）是论文里的 **对照 baseline**（`ResidualBind_heteroscedastic` / `MPRAnn_heteroscedastic`），结论是它比 DEGU **差且训练不稳**。 |
| "要击败的对手（在 TraitGym 上）" | **DEGU 没有在 TraitGym 上的数字**（我们 MEMORY 里已有这条记录，此处复核为真）。DEGU 的实验全在 STARR-seq / lentiMPRA / MPRA 回归任务上，不是 VEP 分类。 |

→ 影响：不能用"跑一下 DEGU 官方代码 → 得到 TraitGym 数字 → 对比我们的 GBM"的思路；官方代码根本没有 Borzoi/TraitGym pipeline。见 §3、§5。

---

## 1. 代码可得性

**官方 repo**：`https://github.com/zrcjessica/ensemble_distillation`

| 项 | 值 | 来源 |
|---|---|---|
| License | MIT | GitHub API |
| Framework | TensorFlow / Keras (TF2) | `README.md` 及 `degu.py` imports |
| Stars / Forks | 2 / 2 | GitHub API（2026-04-16 查询） |
| 创建时间 | 2023-11-10 | API `created_at` |
| 最近 push | 2025-09-15 | API `pushed_at` |
| 默认分支 | `master` | API |
| 核心文件 | `degu.py` (460 行) | raw.githubusercontent |

**论文**：
- bioRxiv v2: `https://www.biorxiv.org/content/10.1101/2024.11.13.623485v2.full`（本会话访问返回 403，但 URL 在搜索结果中直接暴露，未经我独立 verify 正文；needs manual verification 才能读正文）
- npj AI 版：`https://www.nature.com/articles/s44387-025-00053-3`（2026-01-07 发表，本会话 WebFetch 返回 303，needs manual verification 才能读正文）
- PMC 镜像：`https://pmc.ncbi.nlm.nih.gov/articles/PMC11601481/`（fetch 被 reCAPTCHA 拦，needs manual verification）
- 会议：ICLR 2025 MLGenX workshop；OpenReview id `YZBEBxtXyU`

**引用**（来自 repo README）：
```bibtex
@article{Zhou2024.11.13.623485,
    author = {Zhou, Jessica and Rizzo, Kaeli and Christensen, Trevor and Tang, Ziqi and Koo, Peter K},
    title = {Uncertainty-aware genomic deep learning with knowledge distillation},
    year = {2024},
    doi = {10.1101/2024.11.13.623485},
    publisher = {Cold Spring Harbor Laboratory},
    journal = {bioRxiv}
}
```

**口径更正**：用户原问中猜的作者"Ziqi Zhou"不对，第一作者是 **Jessica Zhou**（GitHub handle `zrcjessica`）；Ziqi (Amber) Tang 是第四作者。Peter K. Koo 是通讯作者（Koo Lab, CSHL）。

---

## 2. 架构与 Loss（基于 repo 源码第一手阅读）

### 2.1 DEGU pipeline（摘自 `degu.py` 类 docstring 及 `README.md` Quick Start）

两段式：

**Stage 1 — Teacher ensemble**（`DEGU.train_ensemble`）：
- 用同一个 base model（如 DeepSTARR）初始化 N 次（paper README 例 N=10），每次 reinit 权重后用标准 MSE 训练一遍。
- 保存 N 份权重。

**Stage 2 — Distill to student**（`DEGU.distill_student`）：
- 对训练集 x_train 用 N 个 teacher 推理，得到预测矩阵 `preds` 形状 `(N, num_samples, num_outputs)`。
- 计算两个量：
  - `ensemble_mean = np.mean(preds, axis=0)`
  - `ensemble_uncertainty = uncertainty_fun(preds, axis=0)`，其中默认 `uncertainty_fun = uncertainty_logvar`，即 `np.log(np.var(preds, axis=0))`（也可切 `uncertainty_std = np.std(preds, axis=0)`）
- 拼接：`y_train_ensemble = concat([ensemble_mean, ensemble_uncertainty], axis=1)`，形状 `(num_samples, 2 × num_outputs)`。
- 用 **普通 MSE loss** 训一个 student（same architecture as teacher，但最后一层输出维度翻倍）。

即：
```
L_student = MSE( student(x), [ensemble_mean(x) ‖ ensemble_logvar(x)] )
```

**关键点**：这不是 heteroscedastic NLL —— "uncertainty" 只是作为**额外的回归 target**（soft label），用普通 MSE 监督，而不是作为输出分布的 scale 参数出现在 log-likelihood 里。

### 2.2 与异方差 NLL 的关系

Repo 另有 `ResidualBind_heteroscedastic` / `MPRAnn_heteroscedastic` 模型（`model_zoo.py` 中；脚本 `distill_heteroscedastic_ResidualBind.sh`、`ensemble_predict_heteroscedastic_*.py`）。它们输出 `(mean, std)` 并用 heteroscedastic NLL 训练——**这是 paper 的对照组**：
- 搜索结果摘要（来自 WebSearch snippet，非原文精读）："heteroscedastic regression yielded slightly reduced predictive performance on functional activities and proved challenging to optimize due to an unstable loss function."
- 即：paper 明确选择 DEGU（ensemble 蒸馏）而非 heteroscedastic 作为主方法。

### 2.3 评测基准（paper_reproducibility/docs 确认）

Repo 的 docs 里只列了三类实验：
- `DeepSTARR_ensemble_distillation_protocol.md`（STARR-seq）
- `lentiMPRA_ensemble_distillation_protocol.md`（lentiMPRA HepG2/K562）
- `DREAM_RNN_protocol.md`

**无 TraitGym，无 Borzoi，无 VEP 分类任务**。所有数据都是 MPRA 回归类（预测序列在 MPRA assay 上的活性），label 连续、loss 是 MSE / Pearson r。与 TraitGym 的"二分类 pathogenic vs. control、AUPRC 口径"**任务形态不同**。

### 2.4 学生模型规模

- 官方 Quick Start：student 架构 = teacher 架构（DeepSTARR），只是输出层 `output_shape = num_targets * 2`。
- DeepSTARR 官方参数量 ~0.4M（我们 Phase 0 的 CompactStudent 0.87M 是类似量级）——即 **student 并非"大模型蒸成小模型"的 compression，而是"ensemble N 个模型蒸成 1 个同规模模型"的 inference-cost reduction**。
- 这与我们之前以为的"Borzoi(150M+) → 小学生"完全不是一回事。

---

## 3. 能否直接在 TraitGym 上跑 DEGU？

### 任务形态差异

| 维度 | DEGU 原任务 | TraitGym |
|---|---|---|
| 输入 | 230 bp / 200 bp DNA one-hot | 变异中心 ±N bp DNA |
| 输出 | 连续的 MPRA 活性值 | 二元 label（pathogenic=1 / control=0） |
| Loss | MSE (回归) | Log-loss / Brier / AUPRC |
| Label 来源 | 实验测量的 log2 RNA/DNA ratio | OMIM / ClinVar 疾病变异 |
| Teacher | DeepSTARR / ResidualBind / MPRAnn（全是 CNN） | **未在 DEGU 中测过**；TraitGym leaderboard 里 Borzoi 是 zero-shot baseline |
| 数据量 | ~100k–1M 序列 | Mendelian matched_9 = 3380 变异，complex = 几万 |

### 直接跑的 blocker

1. **无 Borzoi 适配**：repo 没有 Borzoi teacher 代码，Borzoi 本身权重 >1GB、架构复杂（TF saved_model 或 pytorch），且 Borzoi 的输出是数千维 track × 位点，不是 DEGU 默认预期的 `num_targets` 维标量。要把 Borzoi 接入 DEGU pipeline 需重写 teacher wrapper。
2. **无分类损失**：DEGU 代码只假设 regression（MSE）；要做 TraitGym 分类要改 loss、改 eval（AUPRC_by_chrom_weighted_average）、改 data splitter（chrom val=17–22,X）。
3. **无 TraitGym data loader**：我们的 `src/edgpp_genomic/data/traitgym.py` 是我们自己写的，DEGU 用的 h5 格式（`lentiMPRA_data_summary.md`）与 TraitGym 的 parquet feature 文件完全不同。
4. **训练集规模过小**：我们 Day 5 已经 falsified "CompactStudent 在 2350 变异上从头蒸馏 → val Pearson 0.03"。DEGU 是把 ensemble 蒸到同规模 CNN，需要 10 万量级序列。在 3380 变异上直接做 sequence-level DEGU distillation 不会 work（同 Day 5 结论）。

### GPU 需求

- DEGU 训练 = N × DeepSTARR training，每个 < 1h CPU/GPU，easy fit T4。
- 但若 teacher 换 Borzoi ensemble（N × Borzoi），Borzoi 推理就很重，**且 Borzoi 官方权重是 float32 + TF SavedModel**；我们 T4 SM7.5 没 BF16 这点对 inference 影响不大（FP32/FP16 即可），真正瓶颈是显存（Borzoi 推理 ~20 GB）和时间。

---

## 4. DEGU-lite：在我们的 feature-level pipeline 上实现"精神"

这是本报告最有价值的部分——如果我们不复现 paper 本身而是**借 DEGU 的 idea**，最低成本的头部适配方案。

### 4.1 映射

我们现有的 `11_aggregator_gbm.py` 工作在 feature 层：
- 输入 X ∈ R^{N × d}，d = 6 (Borzoi_L2_L2) 或 114 (CADD+Borzoi)
- 输出 p(pathogenic | x) ∈ [0,1]
- 模型 HistGradientBoosting，chrom-LOO 评测，已复现 TraitGym leaderboard 数字

"DEGU-lite" 在 feature 层的两种落地：

**方案 L1 — Ensemble mean + disagreement 作为 feature**
1. 训练 M 个 GBM（或 MLP），每个用不同 random seed / bootstrap 样本。
2. 在 test 上得到 `p_i(x)` for i=1..M，算 `mean = mean_i p_i(x)`、`std = std_i p_i(x)` 或 `logvar = log(var_i p_i(x))`。
3. 用 `[mean, std]` 作为 selective head 的 confidence（mean 用于 rank，std 作为 reliability 信号）。
4. 这和我们 `10_selective_head.py` 的 7-feat GBM 几乎一致，只是把 "7feat" 换成 "ensemble-based (mean, std)"。

**方案 L2 — 蒸馏到一个"既预测 p_mean 又预测 p_std"的 GBM/MLP**
1. 按 L1 得到 ensemble 的 `(mean, std)` 作为 soft target。
2. 训一个 multi-output regressor（或 2 head MLP），输入 X，target `[mean_on_train, std_on_train]`，MSE loss（保持 DEGU paper 的 loss 形式）。
3. 部署时用单模型一次前向得 `(p_hat, sigma_hat)`，用 `p_hat - λ·sigma_hat` 或 `p_hat / (1 + sigma_hat)` 做 selective ranking。

### 4.2 比较 DEGU-lite vs. 我们 10_selective_head

| 方面 | 10_selective_head (现状) | DEGU-lite L1 | DEGU-lite L2 |
|---|---|---|---|
| Reliability 来源 | GBM 预测 \|LogReg 残差\|（7feat） | ensemble disagreement | 蒸馏 ensemble disagreement |
| 需要训几个模型 | 2（LogReg + GBM） | M（e.g. 10） | M+1 |
| 推理成本 | O(1) | O(M) | O(1) |
| 动机 | 监督学习预测 "LogReg 会在哪里错" | 模型间分歧 = 认知不确定 | L1 的 compute-efficient 版 |
| TraitGym 可比性 | 直接可比（feature 层，chrom-LOO 不变） | 直接可比 | 直接可比 |

**Falsification-first 检查**：DEGU-lite L1 的 null baseline 是"M 个 GBM 的 majority vote"——如果 disagreement 不比 single-model margin 提供更多信息，那 DEGU-lite 无效。这要**先跑一个 cheap 版本（M=5, 1 seed）**，看 Δcoverage@25% 是否 > 0.01，再决定是否上多种子稳定性矩阵。

### 4.3 DEGU-lite 的"诚实声明"问题

如果写进论文：这**不是复现 DEGU**，是把 DEGU 的 ensemble-disagreement-as-uncertainty 思想移植到 feature-level classifier。Related work 里需要明说"we adapt the ensemble-disagreement signal from Zhou et al. (2026); we do not reproduce their sequence-level distillation because (a) their released code targets MPRA regression not TraitGym classification, (b) our pipeline operates on precomputed Borzoi features not raw DNA, (c) TraitGym Mendelian has only 3380 variants, insufficient for from-scratch sequence-level distillation as we demonstrated in Day 5"。

---

## 5. 三种路线的工作量估算

| 路线 | 内容 | 代码改动 | 估时 | 产出 | 风险 |
|---|---|---|---|---|---|
| **A. Full reproduction（原任务）** | 在 DeepSTARR / lentiMPRA 上跑官方 repo，验证其 paper 数字 | 0 LOC（用官方） | 2–3 天（下载数据 + ensemble 训 + distill + eval） | DeepSTARR MSE/Pearson 数字 | **与我们 TraitGym 目标不相关**；只能证明 paper 结论可复现，不帮助我们赢竞品比较 |
| **B. Faithful reimplementation on TraitGym** | 把 DEGU 思想从 MPRA regression 移植到 TraitGym 分类：Borzoi ensemble teacher → compact sequence student → chrom-LOO 评测 | ~600 LOC（Borzoi teacher wrapper + data loader 扩展 + 分类版 DEGU loss + eval hook） | **估计 2–3 周**，且成功率低（Day 5 证据显示 from-scratch distillation 在 3380 变异上 infeasible） | TraitGym AUPRC；可能是 0.43 左右（Borzoi zero-shot 水平），也可能崩塌 | 很高 —— Day 5 falsification 已经暗示此路走不通 |
| **C. DEGU-lite（feature-level adaptation）** | L1：多 seed GBM ensemble；L2：蒸到单 model。对齐我们现有 `11_aggregator_gbm.py` | ~150 LOC（一个新脚本 `13_degu_lite.py`） | **1–2 天** | 新 selective ranking 策略的 coverage-AUPRC 曲线；可以放进 Path A benchmark | 低。失败也能作为 negative result 写进 paper |

LOC 估计方法：L1 ≈ 把 `11_aggregator_gbm.py` 包一层 seed loop（~50 LOC）+ 算 mean/std + 新 confidence score + metrics（~100 LOC）。

---

## 6. 推荐路线：C（DEGU-lite），理由

从第一性原理出发，我们的真正目标是：**在 2026-08 bioRxiv preprint 之前，在 TraitGym matched_9 上有一个 reliable 的 selective prediction / uncertainty 方法，且能正当地引用并超越或配得上 DEGU**。

- **A 不解决我们的问题**：DeepSTARR 数字跟 TraitGym 无关，复现出来只是"确认 paper 没造假"。我们的 paper 里不需要这种证据。
- **B 的失败概率已知偏高**：Day 5 的 `project_day5_distillation_infeasible.md` 已经用 5000 step 的长训练证伪了"在 2350 变异上从 DNA 从头蒸馏可行"。DEGU 原论文的 MPRA 任务有 10 万 +样本，和我们的数据规模差 30 倍。硬上 B = 重复 Day 5 的失败。
- **C 是正确抽象层次**：我们的主流水线已经在 feature 层（Borzoi features + GBM）取得 SOTA（CADD+Borzoi = 0.75 LogReg / 0.90 GBM）。DEGU 的核心 idea（ensemble disagreement as calibrated uncertainty）在 feature 层同样适用，且：
  - 训练成本低：M=5–10 个 GBM 在 CPU 上几分钟跑完。
  - 与 `10_selective_head.py` 是直接可比的对照实验：same test split, same metric, different confidence source。
  - Paper narrative 清晰：我们不是在和 DEGU 比原文任务，而是说"DEGU 的不确定性思想在 VEP 分类上的 feature-level 翻译"。
  - 兼容我们已有的 multi-seed 评测框架（cov@25%, cov@50%, AUPRC）。

**立即可执行的下一步**（如用户批准 C）：

1. 新建 `scripts/13_degu_lite.py`。复用 `11_aggregator_gbm.py` 的 feature 读取 + chrom-LOO 框架。
2. L1 版本：M=10 个 seed 的 HistGBM，在每个 held-out chrom 上得 `(p_mean, p_std)`。
3. Selective metric：`confidence = p_mean` 或 `confidence = p_mean - λ·p_std`（λ 由内层 chrom-LOO 选），再算 coverage-AUPRC 曲线。
4. Null baseline（必跑）：single-seed GBM 的 margin `|p - 0.5|` 作为 confidence —— 看 ensemble std 是否真提供增量信息。
5. 若 null baseline 与 ensemble-std 差距 < 0.01 → 直接报告为 negative result，pivot 到 conformal 方向（已有 `12_conformal.py` 雏形）。

---

## 7. 开放问题（需要用户拍板）

1. **CLAUDE.md "DEGU = 异方差 NLL" 的口径要不要校正？** 这会影响我们论文对 related work 的表述（不要把 DEGU 误写成 heteroscedastic method，否则审稿人会指出来）。
2. **是否需要我去精读 npj AI 正文？** 本次 fetch 全被 reCAPTCHA / 303 挡了。如果用户能提供 PDF 或允许我花更多时间绕 bioRxiv 的 access（例如从 PMC 换 PDF endpoint），可以把 §2 精确到 paper 给的公式。目前 §2 基于源码一手阅读，已能支撑复现决策。
3. **路线 C 的优先级与 Path A overview 的关系**：C 与 Task #6（已完成的 conformal 文献 harvest）和 `12_conformal.py` 有重叠——如果 conformal 方向已经更成熟，可以不再做 DEGU-lite，只在 paper 的 "Alternatives considered" 小节里简要交代 DEGU 与我们选择的差异。

---

## 附录 A：一手源码片段（供 §2 核查）

### A.1 Student target 构造（`degu.py` 行 ~175–185，来自 `DEGU.distill_student`）

```python
train_mean, train_unc = self.pred_ensemble(x_train, batch_size=batch_size)
y_train_ensemble = np.concatenate([train_mean, train_unc], axis=1)
...
history = train_fun(student_model, x_train, y_train_ensemble, validation_data, save_prefix)
```

### A.2 Uncertainty 函数（`degu.py` 行 ~260–300）

```python
def uncertainty_logvar(x, axis=0):
    return np.log(np.var(x, axis=axis))

def uncertainty_std(x, axis=0):
    return np.std(x, axis=axis)
```

### A.3 Student 编译用的 loss（`README.md` Quick Start）

```python
student_model = DeepSTARR(input_shape=(L,A), output_shape=num_targets*2)
student_model.compile(keras.optimizers.Adam(learning_rate=0.001), loss='mse')
```
→ 明确是 MSE，不是 NLL。

### A.4 Heteroscedastic 对照（`paper_reproducibility/code/get_heteroscedastic_ResidualBind_ensemble_std.py` 前 80 行）

```python
from model_zoo import ResidualBind_heteroscedastic
...
ensemble_preds_train = np.zeros((args.n_mods, X_train.shape[0], 2))  # (mean, std)
```
→ 确认 heteroscedastic 分支输出 2 维 (mean, std)，作为 DEGU 的对照组存在。

---

## 附录 B：未能独立核实的条目（needs manual verification）

| 条目 | 状态 |
|---|---|
| bioRxiv 全文 v2 正文（公式、数据集细节） | 403 blocked in this session |
| npj AI published 版数字与图表 | 303 redirect in this session |
| PMC 镜像 | reCAPTCHA wall |
| "DEGU 在 Borzoi 上的官方数字" | 不存在（搜索无结果 + repo 无相关脚本 + MEMORY 已记录） |
| 论文是否用 `num_ensemble=10` 作为默认 | README 示例用 10，但 paper 可能 sweep 过，需看正文 |
| Student 是否与 teacher 同架构 | README 示例是同架构；论文可能测过 compact student，needs manual verification |

如果 §2.1 的 loss 细节在正文里有额外修饰（例如 log-space 训 uncertainty head 用 log-MSE），本报告的结论不变（DEGU ≠ NLL），但 DEGU-lite 的实现细节可以对齐。
