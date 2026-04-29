# DEGU end-to-end CNN+ensemble port 实验设计

## 背景

Reviewer v2（commit 4537cf2 timeline）MED-6 critique:
> "DEGU comparison is a strawman. Authors only port the σ̂ mechanism (ensemble distillation) to a matched GBM backbone... a direct port of DEGU's CNN+conformal abstract-promised pipeline to TraitGym is the threat-model the paper needs to neutralize."

Reviewer v2 (post-Phase 6) 不再把这个标为 must-fix（自我评分 Soundness 已 7/10），但 rebuttal 阶段仍是高 leverage 防御。

**结论**: 不是 NeurIPS 投稿前必须做的，**留到 NeurIPS rebuttal 阶段使用**。这份文档是设计稿，不是要现在跑。

## 目的

证明 DEGU 在 TraitGym 上即使 end-to-end 用 CNN + ensemble + conformal，AUPRC / coverage 都不如 HCCP 的 GBM-on-pre-computed-features 路线。

DEGU 论文（Zhou et al, npj AI 2026）原本在 MPRA 回归任务上做：
- Architecture: Multi-task CNN consuming 200bp DNA + ATAC-seq tracks
- Loss: heteroscedastic NLL on log fitness
- Distillation: M=10 teacher CNN ensemble → student CNN with [mean, logvar] head
- Conformal: abstract 提到 "future work" — 没实际跑

我们的 port 要做：
1. 同样 architecture（CNN, M=10 ensemble, distillation step）
2. 改成 binary classification on TraitGym (BCE + class-balanced)
3. 套上 conformal calibration（split-CP at minimum，可选 Mondrian）

## 实验设计

### Phase A — Replication baseline (1 day)

目标: 让 DEGU CNN 在 TraitGym 复现一个 baseline 数字（论文已删除的 §C.4 给的 0.140 / 0.111 是早期实验，这次重做更严谨）

- Dataset: TraitGym Mendelian + Complex matched_9 (3380 + 11400 variants)
- Input: ref + alt 200bp DNA windows (one-hot, 8 channels), ATAC-seq tracks if available
- Architecture: copy DEGU Borzoi-lite CNN（conv → residual → softmax pooling）
- Training: M=5 teachers (random seeds {42, 7, 2024, 100, 200}), 50 epochs each, T4 GPU
- Distillation: student CNN with [logit, logvar] head trained to match teacher [mean, std]
- Conformal: split-CP only (DEGU's intended pipeline)
- Output: AUPRC, AUROC, marginal coverage, σ̂-bin gap

预期结果: AUPRC ~0.15 (Mendelian), ~0.12 (Complex). Far below GBM 0.900 / 0.353.

### Phase B — Steel-man DEGU (1-1.5 day)

让 DEGU 用 HCCP 同样的 Mondrian 分割 / nested CV / K_eval, 看 calibration 是否能补足 base predictor 弱

- Same DEGU CNN + ensemble distillation as Phase A
- Conformal: HCCP-style Mondrian (y × σ̂-bin), nested CV K-selection
- Compare to: HCCP-on-GBM 主表数字
- Output: 是否 Mondrian + nested CV 能把 DEGU 的 AUPRC=0.15 cov gap 修到接近 HCCP-on-GBM 的 0.024 (Complex) / 0.163 (Mendelian)?

预期结果: 仍不行. CNN base predictor 差，conformal calibration 无法救。Mondrian gap 可能 < HCCP-on-GBM 但 AUPRC 仍 < 0.20，clinical utility 极弱。

这正是 paper §6 现在 DEGU-lite ablation 段说的："the conformal partition rides on top of a competent base predictor; conformal calibration cannot rescue a base predictor that has lost discrimination."

### Phase C — Honest writeup (0.5 day)

把 Phase A + B 数字写成 1 表 + 1 段插入 §6.5 ablation 或 App C，作为 rebuttal 时的"我们做了你们要求的 end-to-end port，结果证实 paper 现有 framing"。

## 资源估算

- GPU: T4×1, ~30 GPU-hour (5 teachers × 50 epochs × ~6 min/epoch on T4)
- CPU/RAM: 32GB（DEGU CNN 参数量 ~5M, 比 Borzoi 524kb context 小 100×）
- 时间: 2-3 day total（包含 debug, plot, writeup）

## 决策点

### 现在做？

**反对**:
- 当前 PDF 已 Weak Accept (reviewer v2)
- 两条 MEDIUM 残留（unique 措辞 + bootstrap CI）已修一半，剩 bootstrap CI 1-2h 投入更高 leverage
- DEGU end-to-end 是防御性，不是 contribution-positive

**支持**:
- bioRxiv 占坑后 8 个月时间窗口，做完更稳
- NeurIPS rebuttal 阶段做时间紧（5000 字符 + 5 天 deadline），事先做完更从容

### 推荐时序

```
当前 (2026-04-29)        → bioRxiv Path B (3h 投入)
2026-05-上               → bioRxiv 提交 v1
2026-05-中 to 2026-07    → DEGU end-to-end port (2-3 d), 加进 bioRxiv v2
2026-08                  → bioRxiv v2 上线 (含 DEGU port)
2026-09 to 2027-04       → 社区 feedback 收集 + 实验补强
2027-05                  → NeurIPS 2027 main 投稿
2027-08-09               → 假设进 rebuttal, 已经手握 DEGU port 数字
```

## 实操脚本骨架（不要现在跑）

```python
# T_tools/degu_end_to_end_port.py (设计稿)
# 1. Load TraitGym variants + extract 200bp DNA windows around variant
# 2. Build CNN ensemble (M=5)
# 3. Train each teacher with BCE + class-balanced
# 4. Distill student with [logit, logvar] head
# 5. Compute σ̂(x) = student logvar predictions
# 6. Apply HCCP Mondrian (y × σ̂-bin) calibration with nested CV K
# 7. Report AUPRC, marginal cov, σ̂-bin gap, per-chrom gap

# Inputs needed (not yet pulled):
# - hg38 reference for 200bp window extraction (already in `data/hg38_dnabert/`)
# - TraitGym matched_9 variant lists (already in `data/traitgym/`)

# Compute path:
# CUDA 11.8 + PyTorch 2.0 (env: edgpp_t4)
# T4 GPU only, no BF16 — use FP32

# Architecture:
# - Conv1D × 6 layers (256 → 512 → 1024 channels)
# - Residual connections
# - Global softmax pool
# - 2-head output: [logit_path, logvar]
# - Total params ~5M (vs Borzoi ~150M; CNN-lite by design)
```

## Reference

- Zhou et al, "Deep Ensemble with Gaussian Uncertainty for Genomic Variant Effects" npj AI 2026 (DEGU paper)
- Repo: https://github.com/zrcjessica/ensemble_distillation (TF/Keras MPRA, 我们要 port 到 PyTorch + TraitGym binary classification)
- Borzoi 524kb context: https://github.com/calico/borzoi (用作 feature pre-computation reference)
