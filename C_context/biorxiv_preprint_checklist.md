# bioRxiv preprint 占坑 checklist (target 2026-08)

## 目的

- 在 NeurIPS 2027 main (~2027-05) 投稿前 **9 个月**先用 bioRxiv 占坑
- 建立优先权（priority claim）防被同类 work scoop（DEGU 已 abstract 提 conformal future work）
- 收集社区 feedback，节省 NeurIPS rebuttal 时间

## 时间线

| 日期 | 动作 |
|---|---|
| **2026-04-29** (今天) | Phase 6 PDF 完成（46 pp, weak accept score）|
| 2026-05–06 | 实验补强（Bootstrap CI on tab:cp_baselines + 可选 DEGU end-to-end port）|
| 2026-07 | bioRxiv 元数据准备 + ORCID + 反复 self-review |
| **2026-08** | **bioRxiv 提交占坑** |
| 2026-09–2027-04 | 收集社区反馈, 整改, 准备 NeurIPS 投稿 |
| 2027-05 | NeurIPS 2027 main deadline |

## 当前 PDF 直接 bioRxiv-ready 检查

### ✅ 已就绪
- [x] PDF 46 pp, 0 undefined ref, 0 stale claim
- [x] Reviewer v2 score Weak Accept (Soundness 7/10, Presentation 8/10)
- [x] Abstract 5 句结构 (Farquhar formula)
- [x] 3 axis OOD validation + ProteinGym cross-domain
- [x] Honest limitations (§7) + Failure mode (§6.6)
- [x] K_eval sensitivity table (§C.1)
- [x] All 3 citation placeholder fixed
- [x] Phase 6 audit-driven framing softening 完结

### ⚠️ 需要补强（推荐做完再提）
- [ ] **Bootstrap CI on Tab tab:cp_baselines** (reviewer v2 MEDIUM-3): 现有单 seed point estimate；Mendelian 1.15× 边际太薄，bootstrap CI 能 consolidate 或 invalidate
  - 工作量: 1-2 GPU-hour（B=200 chrom-resamples per method）
  - 脚本: `T_tools/bootstrap_cp_baselines.py` (新写, 复用 cp_baselines_h2h.py 内核)
- [ ] **Skeletal cluster 数值化**(reviewer v2 LOW-2): "out of N tested clusters" 一句统计
  - 工作量: 30 min, 数据已在 `R_raw/trait_cluster_aggregate/mendelian_cluster.json`
- [ ] **K̂_CV 与 nested-CV K 的 reconciliation**: §1 / §6.4 残留旧 K̂_CV ∈ {2,3} 与 §5.3 modal K=5-8 的 friction
  - 工作量: 30 min, 文本一致性

### 📋 bioRxiv 投稿元数据（提交前 fill）

```yaml
# bioRxiv 提交需要的字段
title: "Conformalized Heteroscedastic Variant Effect Prediction with Local Coverage Guarantees"
short_title: "HCCP for VEP"
running_title: "Heteroscedastic Conformal VEP"

authors:
  - first_name: Siyu
    last_name: Wang
    email: wangsiyu2030@gmail.com
    orcid: TODO
    affiliation: TODO  # 论文目前 "Anonymous Authors / double-blind" — bioRxiv 必须实名
    corresponding: true

abstract: |
  # 复制 main.tex \begin{abstract}...\end{abstract} 全文

categories:
  primary: Bioinformatics
  secondary:
    - Genomics
    - Machine Learning

keywords:
  - conformal prediction
  - variant effect prediction
  - heteroscedastic uncertainty
  - class-conditional coverage
  - Mondrian calibration

license: CC-BY-4.0  # 推荐, 兼容 NeurIPS

word_count_main: TODO  # bioRxiv 要正文字数
figure_count: 13
table_count: 17
```

### 🔄 NeurIPS double-blind 适配（bioRxiv 与 NeurIPS 投稿之间的差异）

- bioRxiv: 必须实名 + 完整 affiliation
- NeurIPS 2027: 必须 double-blind, 去 affiliation
- 解决: 维护两个 main.tex 分支 (`papers/neurips2027_pathA/` 实名版本 + `papers/neurips2027_pathA_blind/` 匿名版本) 或在 main.tex 加 `\if\anon ... \else ... \fi` 条件编译

bioRxiv 不影响 NeurIPS double-blind 政策（NeurIPS 允许 arXiv/bioRxiv preprint，但作者在 reviewer-facing 版本不能 self-cite 透露身份）。

### 📤 投稿流程

1. 注册 bioRxiv 账号（用 ORCID 登录最方便）
2. 上传 PDF + supplementary files (`papers/neurips2027_pathA/main.pdf` + `D_deliverables/` figure source)
3. 选 Subject Areas: Bioinformatics + Genomics
4. License: CC-BY 4.0
5. 选 "post a copy on medRxiv" 选项 — 临床向 readership 加倍
6. 提交后 24-48h 内 screening + DOI 分配

### 🔗 NeurIPS submission 用 bioRxiv DOI

NeurIPS 投稿允许在 main paper 加脚注 `\footnote{Preprint: bioRxiv DOI}` (但 reviewer-facing 版本删除这个脚注以保 double-blind)。提交时同步引用 bioRxiv 数字以建立 priority。

## 当前推荐路径

**Path A (最简, 直接占坑)**: 当前 PDF + 实名补充 → 立即 bioRxiv. 不补 Bootstrap CI, 留到 NeurIPS rebuttal/camera-ready. 风险: reviewer v2 MEDIUM-3 仍存在.

**Path B (推荐)**: 先做 Bootstrap CI on tab:cp_baselines (1-2h) + Skeletal 数值化 (30 min) + K̂_CV reconciliation (30 min) → 月底前再 bioRxiv 提交. 总投入 3 小时, score 可能从 Weak Accept → 接近 Accept.

**Path C (over-engineering)**: 加 DEGU end-to-end port (2-3d) → bioRxiv 推迟到 2026-08 中. 不推荐, DEGU 留 NeurIPS rebuttal 阶段更有 leverage.

**默认**: Path B.

## 风险 / 反对意见

- bioRxiv 占坑是否过早？ Phase 6 后 PDF 已是 weak accept，质量足够；越早 priority 越稳。
- bioRxiv 后能改 PDF 吗？ 可以，bioRxiv 支持 multiple revisions（v1, v2, ...），DOI 不变只追 v 号。
- 会被 scoop 风险？ DEGU 在 abstract 提 CP 但官方 repo 没动；2026-08 占坑已是 8 个月窗口，足够。
