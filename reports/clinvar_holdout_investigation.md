# ClinVar Hold-out 调研 — 结论：不建议采用，改推 trait-LOO

**日期**: 2026-04-17
**任务编号**: Day 14 任务 B
**状态**: **调研完成，建议 pivot**

---

## 1. 背景

Path A success criteria §9 列出"ClinVar hold-out 上泛化验证"。目的：为 T4 chrom-shift robustness 提供 **外部数据集** 的 OOD 验证，回答 reviewer 必问的"跨数据集泛化"。

调研目标：确认 ClinVar 作为 TraitGym 的外部 hold-out 在我们 **non-coding regulatory VEP** 任务下是否可行。

---

## 2. 关键事实

### 2.1 TraitGym Mendelian 的来源

- 正样本：**OMIM**（338 因果 non-coding 变异），非 ClinVar
- 负样本：gnomAD MAF > 5% 的匹配变异（matched_9 protocol）
- 与 ClinVar **没有直接的 source 重叠**（OMIM 与 ClinVar 部分 overlap，但 TraitGym 只取 non-coding，而 ClinVar 几乎没有 non-coding）

### 2.2 ClinVar 的 non-coding pathogenic 稀缺

来自 2023–2024 年文献综合：

| 类别 | Pathogenic/Likely Pathogenic 变异数 |
|---|---|
| 3′ UTR | **26** |
| 5′ UTR | **68** |
| Intronic（总） | 很多，但 >50% 为 splice-site（每 intron 只有 4 bases） |
| Intergenic | 接近 0 |
| Promoter | 接近 0 |

> "ClinVar 中的 non-coding pathogenic 几乎全部集中在 splice-site；UTR/promoter/intergenic 加起来 < 0.34% 的 high-confidence P/LP。" — Frontiers in Molecular Biosciences 2023

**TraitGym 作者自己的表述**（bioRxiv 2025 §Introduction）：
> "Expert-reviewed pathogenic variants in ClinVar are highly skewed towards coding and splice region variants, containing only a single promoter variant and no intergenic variants. This limitation of ClinVar is one reason why TraitGym was developed as a complementary resource."

这是 TraitGym **存在的理由**。用 ClinVar 作 TraitGym 的"外部 hold-out"，等于用"被淘汰的资源"做"更好的资源"的验证 — 任务不对等。

### 2.3 如果强行做 ClinVar hold-out，估算可用变异数

- 取 UTR P/LP 的 26+68 = **94 个正样本**
- 加 non-splice intronic ≈ 估 **100–200 个**（无精确数字，量级乐观估计）
- 需构造 matched controls：按 consequence / MAF / TSS-dist 1:9 匹配 → 总 ~1000–3000 变异

此规模对我们 conformal coverage evaluation 不够：
- α=0.1 下 per-class coverage 的标准误 $\sqrt{\alpha(1-\alpha)/n_k} \approx \sqrt{0.09/100} = 3\%$ — 跟我们在 Mendelian 上观测到的 chrom-shift 5% 相当，**统计力不够区分 SOTA 与 baseline**
- 更致命：**feature pipeline**。Borzoi precompute 覆盖 19.53M common+low-freq 变异；ClinVar rare pathogenic 可能大面积缺失，需要逐个跑 Borzoi（GPU，单变异 ~秒级，1000 变异 ~小时级）。GPN-MSA 同理。CADD 覆盖最全。

### 2.4 Coding ClinVar 可以做吗？

不可以，因为：
- 我们的 feature 是 **non-coding regulatory models**（Borzoi 预测 RNA-seq coverage，GPN-MSA 做 MSA-based）。对 coding 变异，这些 model 的表现和 AlphaMissense/ESM 不在同一水平。
- Day 10 GBM 0.900 的 AUPRC 是 **non-coding regulatory** 的结果；放到 coding 上这个数字会崩。

---

## 3. 结论

**ClinVar hold-out 不适合 Path A 任务。** 核心矛盾：

| 任务 | TraitGym | ClinVar |
|---|---|---|
| 变异类型 | Non-coding regulatory | 95%+ coding/splice |
| 特征 pipeline | Borzoi + GPN-MSA + CADD 全覆盖 | Borzoi/GPN-MSA 大面积缺失 |
| 统计规模 | Mendelian 3380 / Complex 11400 | Non-coding P/LP ~300 |
| Matched-control 协议 | 官方 matched_9 | 自行构造 |

硬跑也能跑出个数字，但是：
1. 不能有效支撑 T4 chrom-shift robustness 主张（分布偏移的是 **coding/non-coding** 而非 chrom，混淆了论证）
2. 规模不够分辨 SOTA 与 baseline
3. Feature pipeline 补全成本高（GPU 工时）

---

## 4. 推荐 pivot

放弃 ClinVar，改做 **TraitGym 内部的 3 种外部验证**，任何一种都比 ClinVar 更干净：

### Plan B1 — Trait-Leave-One-Out（推荐）

TraitGym Mendelian 有 **113 个 trait**，Complex 有 **83 个 trait**。现行 chrom-LOO 验证了跨染色体泛化；trait-LOO 验证**跨疾病泛化**（新的 novel phenotype 是 VEP 的核心应用）。

- Mendelian: 每次留 1 trait（约 3 个正样本 + 27 个 matched control），train 112 个 traits
- 113 × 重复 = 113 个 test folds
- Coverage gap across traits 可作 T4-analog 的实证（trait-shift ≈ chrom-shift 的生物学类比）

**优点**：同一 dataset、同一 feature pipeline、方法学干净
**缺点**：每 fold 样本小（约 30），conformal quantile 稳定性要 pool across traits

### Plan B2 — Cross-dataset shift (Mendelian ↔ Complex)

我们已经跑了两个 dataset（Day 11–13）。正式化：在 Mendelian 上 train + calibrate，在 Complex 上 test（和反向）。

- 是真实的 distribution shift（OMIM 离散单基因 vs. UKB 连续多基因）
- 两个 dataset 的 $d_{TV}$ 可估计（score 分布的 empirical histograms）
- 直接对应 T4 bound

**优点**：已有数据、最少额外计算、直接回答 reviewer
**缺点**：不是"外部 dataset"，是同 benchmark 的不同子任务；reviewer 可能说"太接近"

### Plan B3 — ClinVar 3′UTR subset（小规模对照）

如果 reviewer 坚持要 ClinVar 数字，可以做 26 个 P/LP 3′UTR + 234 matched benign 3′UTR 的小表格（Appendix 一张小表，不作为主论证）。需要补 feature pipeline，~半天 GPU 工时。

**优点**：打消 reviewer 疑虑
**缺点**：统计噪声大，不建议放主文

---

## 5. 时间 & 成本对比

| 方案 | 代码工作 | 计算成本 | 统计信度 | Reviewer 分 |
|---|---|---|---|---|
| ClinVar hold-out 全量 | 3–5 天（feature pipeline + matching） | ~10h GPU | 低（n≈300） | 中（但任务不对等） |
| **Plan B1 trait-LOO** | **1–2 天**（fold split + rerun） | **0 GPU**（sklearn） | **中高** | **高** |
| Plan B2 cross-dataset | **0.5 天**（已有 outputs 重组） | 0 GPU | 中 | 中高 |
| Plan B3 ClinVar 3′UTR | 3 天（feature + matching） | ~2h GPU | 低 | 中（Appendix 小表） |

**我的推荐**：**Plan B1（主） + Plan B2（补）**。Plan B3 留作 rebuttal 素材。

---

## 6. 对 path_a_plan.md §9 的修订建议

原 success criteria：

> [ ] ClinVar hold-out 上泛化验证

改为：

> [ ] **Trait-LOO** 上 coverage gap 与 chrom-LOO 相当（Plan B1）
> [ ] **Mendelian ↔ Complex cross-dataset shift** 下 coverage 保留 $(1-\alpha) - \bar{d}_{TV}$（Plan B2）
> [ ] （可选，Appendix）ClinVar 3′UTR 小规模对照表（Plan B3）

---

## 7. 下一步

**需要用户决策**：

1. 采纳推荐（pivot 到 Plan B1+B2，放弃 ClinVar），还是
2. 坚持 ClinVar hold-out（可执行但统计力不足），还是
3. 有其他外部数据源建议（例如 MPRA-based pathogenicity，或 gnomAD 罕见变异作为 OOD 负样本 stress test）？

决策后再落实 Plan B1 的 `scripts/17_trait_loo.py` 与 Plan B2 的 `scripts/18_cross_dataset.py`。
