# NeurIPS 2027 Main — HCCP (Conformalized Heteroscedastic VEP)

目标：2027-05 主会提交。当前完成度 ~90% (Day 21+)。

## 文件结构

```
papers/neurips2027_pathA/
├── README.md                  # 本文件（复现指南）
├── outline.md                 # 主 outline + claim stack + timeline
├── main.tex                   # LaTeX 入口（24 pages: 9 main + 15 appendix）
├── refs.bib                   # 参考文献（完成）
├── sections/
│   ├── 01_introduction.tex    # DONE: 4-contribution list，T5 前置
│   ├── 02_related_work.tex    # DONE: CP / hetero UQ / VEP + 2025 文献
│   ├── 03_formulation.tex     # DONE: A1/A1'/A2 hierarchy + notation
│   ├── 04_method.tex          # DONE: Algorithm 1 + Mondrian rationale
│   ├── 05_theory.tex          # DONE: T1-T5 + robust + T3-loc + T3.b
│   ├── 06_experiments.tex     # DONE: main + 3-axis + 4 ablations + bootstrap CI
│   ├── 07_discussion.tex      # DONE: limitations (i-vi) + broader impact
│   ├── 08_conclusion.tex      # DONE
│   ├── A_proofs.tex           # DONE: T1-T5 完整证明
│   ├── B_data.tex             # DONE: TraitGym + KS audit per-chrom
│   ├── C_tables.tex           # DONE: per-chrom + K-sweep + L_F audit + DEGU
│   └── D_alpha_sweep.tex      # DONE: α ∈ {0.01..0.50} sensitivity
├── figures/                   # 9 figures + figD_alpha_sweep
└── tables/                    # 表格源自 outputs/
```

## 一键复现

```bash
conda activate edgpp_t4
# 审计所有表格/图的源文件
python scripts/33_reproduce_paper_tables.py
```

输出示例：
```
=== Table 3: Main results ===
  mendelian,  CADD+GPN-MSA+Borzoi,  AUPRC=0.900, cov=0.902, σ̂-gap=0.077
  complex,    CADD+GPN-MSA+Borzoi,  AUPRC=0.347, cov=0.901, σ̂-gap=0.023

=== Table 7: K-sweep ablation ===
  mendelian: K_CV=3, K*_oracle=88, L_F=49.44, gap@K5=0.285, gap@K_CV=0.047
  complex:   K_CV=2, K*_oracle=110, L_F=21.80, gap@K5=0.042, gap@K_CV=0.005
```

## Pipeline (端到端)

```bash
# 1. 数据 & 特征（一次性）
bash scripts/download_data.sh

# 2. 聚合器 → σ̂ 头 → Mondrian conformal
python scripts/11_aggregator_gbm.py --feature-set CADD+GPN-MSA+Borzoi --dataset mendelian
python scripts/13_hetero_head.py --base-scores outputs/aggregator_gbm/.../scores.parquet ...
python scripts/14_conformal_hetero.py --sigma-scores outputs/hetero_head/.../scores_with_sigma.parquet ...

# 3. 三轴 OOD
python scripts/17_trait_loo.py ...
python scripts/18_cross_dataset.py ...

# 4. T5 + bootstrap CI
python scripts/20_adaptive_K_sweep.py ...
python scripts/29_L_F_estimator.py ...
python scripts/34_bootstrap_ci.py ...

# 5. KS audit
python scripts/27_ks_audit_per_chrom.py ...

# 6. α sweep figure
python scripts/28_alpha_sweep_figure.py
```

## 作为 pip 包使用 HCCP

```python
from edgpp_genomic.hccp import HCCPClassifier

clf = HCCPClassifier(alpha=0.10, n_sigma_bins="auto")
clf.fit(X_cal, p_cal, y_cal, chroms_cal)
pred_sets = clf.predict_set(X_test, p_test)  # list of {0}, {1}, {0,1}, set()
metrics = clf.evaluate_coverage(X_test, p_test, y_test)
```

API 参考：`src/edgpp_genomic/hccp/`
- `HCCPClassifier` — 高级封装
- `SigmaHead` — 异方差头训练
- `mondrian_calibrate` / `predict_set_from_calibration` — 裸 conformal
- `oracle_K` / `select_K_cv` — T5 K 选择

## 编译

```bash
cd papers/neurips2027_pathA
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## 证据 anchors（正文数字 → 来源文件）

| 位置 | 数字 | 源文件 |
|---|---|---|
| Abstract AUPRC 0.900 | Mendelian SOTA | `outputs/aggregator_gbm/CADD+GPN-MSA+Borzoi_mendelian/metrics.json` |
| Table 3 bootstrap CI | cov ± std | `outputs/bootstrap_ci/*.json` |
| Table 4 trait-LOO gap 0.002–0.004 | finite-sample floor | `outputs/trait_loo/*/trait_loo_results.json` |
| Table 7 K-sweep | bias-variance | `outputs/adaptive_K/*/adaptive_K_results.json` |
| Table 8 DEGU vs HCCP | 4.3× | `outputs/degu_comparison_summary.json` |
| Fig D α sweep | σ̂-gap ≤ 0.04 | `outputs/conformal_hetero/*/alpha_sweep_mondrian` |
| App B.4 KS audit | 46.2% / 10.4% | `outputs/ks_audit/*_ks_per_chrom_n20.json` |
| App C.2 L_F 4 estimators | 2.4 – 49.7 | `outputs/L_F_audit/*_L_F.json` |

## 当前 review 响应状态

2026-04-23 审稿响应清单（见 commit df6051b, 本 commit）：

- [x] draft TODO 清理
- [x] 多 seed → bootstrap CI（HistGB 确定性，bootstrap 是更合适的方差源）
- [x] α sweep 曲线 Appendix D
- [x] L_F 多估计器审计
- [x] A2-cell per-chrom KS breakdown
- [x] Related work 扩展 5 处
- [x] Broader Impact 段落
- [x] T5 framing 前置（abstract + intro）
- [x] T3-loc 降级 + Barber 2020 boxed positioning
- [x] Dimension-free rate 诚实化
- [x] σ̂ target 与 score function ablations
- [x] DEGU PyTorch port（代码，等 GPU 空跑）
- [x] AlphaGenome API 脚手架（等 API key）
- [x] HCCP pip 包化 + reproducibility script

待办（等 GPU 空）：
- [ ] DEGU-actual 真实训练（`scripts/31_degu_pytorch.py`，T4 × 2h）
- [ ] AlphaGenome 特征 extraction（`scripts/32_alphagenome_features.py`，需 API key）
- [ ] ProteinGym 异源 benchmark（数据下载 + 适配）

## 下一步

1. 跑 DEGU-actual & AlphaGenome（GPU 就绪后）
2. 逐 section ppw-polish 润色
3. Bibliography 去重 & 格式化
4. NeurIPS 2027 style 替换（官方模板释出后）
