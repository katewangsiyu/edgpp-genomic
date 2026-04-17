# NeurIPS 2027 Main — paper skeleton

目标：2027-05 提交。本目录是论文骨架与写作工作区。

## 文件结构

```
papers/neurips2027_pathA/
├── README.md                  # 本文件
├── outline.md                 # 主 outline + claim stack + timeline
├── main.tex                   # LaTeX 入口（9 页 main + unbounded appendix）
├── refs.bib                   # 参考文献（stub，iter-by-iter 填充）
├── sections/
│   ├── 01_introduction.tex    # stub：3 段落框架
│   ├── 02_related_work.tex    # stub：3 buckets (CP / hetero UQ / VEP)
│   ├── 03_formulation.tex     # stub：Assumption A1/A2 + 符号
│   ├── 04_method.tex          # stub：Algorithm 1 + Mondrian rationale
│   ├── 05_theory.tex          # stub：T1/T2/T3/T4 theorem statements
│   ├── 06_experiments.tex     # stub：main table + 3-axis OOD + ablations
│   ├── 07_discussion.tex      # stub
│   ├── 08_conclusion.tex      # stub
│   ├── A_proofs.tex           # stub，port from theory/
│   ├── B_data.tex             # stub
│   └── C_tables.tex           # stub
├── figures/                   # 按 outline.md §3 规划
└── tables/                    # 按 outline.md §4 规划
```

## 当前状态（Day 15）

- [x] Outline + claim stack 落盘 (`outline.md`)
- [x] LaTeX skeleton + 所有 section 以 `\TODO{}` 标注 anchor 建立
- [x] refs.bib minimal stubs
- [ ] 实际正文文字
- [ ] Figures
- [ ] Tables

## 证据 anchors（正文每个 `\TODO{}` 会引用）

| Section | 主要 source |
|---|---|
| §1 Intro | `reports/path_a_plan.md` §1, CLAUDE.md |
| §2 Related | `papers/literature_v0.md` |
| §3 Formulation | `theory/formulation_v0.md` §1–2 |
| §4 Method | `theory/formulation_v0.md` §2, scripts/11–14 |
| §5 Theory | `theory/t1_t2_formal_proofs.md`, `theory/t3_proof_sketch.md` |
| §6 Experiments | `reports/phase2_day11_hetero_conformal.md`, `reports/day14_external_validation.md` §7 |
| App A | `theory/t1_t2_formal_proofs.md` (direct port) |
| App B | `configs/`, `CLAUDE.md` §Hardware |
| App C | `outputs/{aggregator, selective, trait_loo, cross_dataset}/*.csv` |

## 编译

尚未集成真正的 LaTeX 编译（需 NeurIPS 2027 style 文件或用 2024 fallback）。等 style 释出后：

```bash
cd papers/neurips2027_pathA
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## 下一步（Day 15+）

1. ~~起骨架~~ ✓
2. 正文草稿：§3 formulation（从 theory/formulation_v0.md 直接 port）
3. §5 theory：T1/T2 正文版（1 page），appendix port（theory/）
4. §6 experiments：填入 Day 10–14 已有数字
5. Figure 1 concept 图（Intro）
6. Figure 2–5 生成脚本（figures/make_figures.py）
