# NeurIPS 2026 Submission Package — HCCP

提交时按 OpenReview submission form 顺序逐字段复制粘贴。

---

## 1. Title

```
Conformalized Heteroscedastic Variant Effect Prediction with Local Coverage Guarantees
```

---

## 2. TL;DR (1-2 句, optional 字段)

```
HCCP — heteroscedastic class-conditional conformal prediction with a tight finite-sample O(n^{-1/2}) rate within the equi-bin Mondrian-K family with π_min^{-1/2} explicit in the constant, validated on TraitGym + ProteinGym + synthetic.
```

---

## 3. Abstract (OpenReview-friendly, markdown + LaTeX math)

```
We introduce **HCCP** (Heteroscedastic Class-Conditional Conformal Prediction): a post-hoc framework pairing a learned variance head $\hat\sigma(x)$ with Mondrian-$(y \times \hat\sigma\text{-bin})$ calibration via a feature-dependent nonconformity score. For binary conformal classification under joint class imbalance and heteroscedasticity, existing variants trade class-conditional coverage against bin-local coverage (a Pareto wall sharpened by Barber et al. [2020]'s finite-sample pointwise impossibility); the gap binds for variant effect prediction (VEP), where TraitGym minority prevalence sits at $\pi_{\min} \approx 0.10$. Within the equi-bin Mondrian-$K$ family we prove a tight finite-sample $O(n^{-1/2})$ rate on the worst-cell coverage gap, matched *within this family* by an $\Omega(n^{-1/2})$ lower bound with $\pi_{\min}^{-1/2}$ explicit in the constant (Theorems 5.1, 5.2); the bound is dimension-free given $\hat\sigma$ is 1-D Lipschitz. On TraitGym [Benegas et al., 2025] at the recommended operating point ($K_{\mathrm{eval}} = 3$ / $5$ for Mendelian / Complex; per-cell minority $\geq 100$), HCCP is the only single-fold partition we evaluated --- across split CP, $\hat\sigma$-Mondrian [Boström & Johansson, 2020], class-Mondrian, RLCP, weighted CP, and SC-CP at matched $(\hat p, \hat\sigma)$ --- that holds $\mathrm{cov}_{|Y=1} \geq 0.85$ and $\hat\sigma$-bin gap $\leq 0.20$ on both Mendelian ($n = 3{,}380$) and Complex ($n = 11{,}400$) under $B = 200$ chromosome-bootstrap CIs. A $K_{\mathrm{eval}}$ sweep shows the Complex advantage is robust across $K_{\mathrm{eval}} \in [2, 10]$, while on Mendelian weighted CP overtakes HCCP for $K_{\mathrm{eval}} \geq 5$ (per-cell minority $< 70$); cross-domain replication on ProteinGym [Notin et al., 2023].
```

---

## 4. Keywords (8-10 个, 逗号分隔)

```
conformal prediction, uncertainty quantification, calibration, heteroscedastic, Mondrian, variant effect prediction, distribution-free inference, computational biology, imbalanced classification, machine learning for healthcare
```

---

## 5. Primary Subject Area

```
Probabilistic Methods
```

(若 dropdown 没这个, 退一档选 `Theory` 或 `Other`)

## 6. Secondary Subject Area (optional)

```
Theory
```

或 `Applications` (按 dropdown 实际选项)

---

## 7. Authors

OpenReview submission form 一般要求每位 author 有 OR profile。按以下顺序填:

| 顺序 | 名字 | Affiliation | 作用 |
|---|---|---|---|
| 1 | Siyu Wang (你) | East China Normal University | First author, corresponding |
| 2 | <你导师> | East China Normal University | Senior author |
| 3+ | <其他 co-author> | <他们 affiliation> | Contributing |

**注意**:
- 如果导师没有 OR profile, 让她/他**今晚立刻用机构邮箱 (`xxx@ecnu.edu.cn`) 注册**, 24 小时内大概率激活
- 如果你是 single author (仅你一人), 也可只填你自己 — NeurIPS 允许 single-author

---

## 8. Author emails (用于 OpenReview profile lookup)

```
- 51261300153@stu.ecnu.edu.cn (你)
- <你导师邮箱>
- <其他 co-author 邮箱>
```

---

## 9. PDF Upload

```
papers/neurips2027_pathA/main.pdf
```

**Sanity check before upload**:
- 第 1 页 author 字段是 `Anonymous Authors` 不是真名 ✓ (已 verify)
- 38 页 PDF 完整
- 字号正常, neurips_2026.sty 渲染

---

## 10. Supplementary Materials (optional)

留空 — 所有 appendix (A-F) + checklist 已经在 main.pdf 里。

---

## 11. Conflict of Interest (COI) emails

填**近 3-5 年**所有合作者的邮箱, 用于排除审稿人:

```
- <你导师邮箱>
- <你 EDG paper co-authors 邮箱>
- <你 VISNET paper co-authors 邮箱>
- <实验室同学 co-author 邮箱>
- <任何近期 paper / project 合作者邮箱>
```

**记得每个邮箱独立一行**。OpenReview 会用这个 list 在 reviewer assignment 时排除你的合作者。

如果不确定要不要列某人, **倾向多列**——保守原则, 避免审稿冲突。

---

## 12. NeurIPS Paper Checklist (16 题)

`checklist.tex` 已写好, 但 OpenReview submission form 一般要重新答 16 题。直接照抄:

| # | 类别 | 答案 | Justification (短版) |
|---|---|---|---|
| 1 | Claims | **Yes** | Abstract + §1 contributions match scope; honest reversal at K_eval≥5 disclosed in abstract |
| 2 | Limitations | **Yes** | §7 (TV proxy / K* asymptotic / DEGU port / ProteinGym 6/50) + §6.5 M→C failure mode |
| 3 | Theory: Assumptions & Proofs | **Yes** | All 4 theorems (T3, T3', T5.1, T5.2) full proofs in App A; assumption hierarchy A1/A1'/A2/A-SL stated explicitly |
| 4 | Reproducibility | **Yes** | Public datasets (TraitGym, ProteinGym), splits, HP, nested-CV, bootstrap protocol all in §method/§exp/App B |
| 5 | Open Access | **Yes** | Code as anonymized supp at submission; full release on acceptance |
| 6 | Setting | **Yes** | App B: HistGB hyperparameters + small-cell fallback + leakage sanity audit |
| 7 | Statistical Significance | **Yes** | B=200 chrom-bootstrap CIs throughout; single-seed flagged in Tab keval_sensitivity |
| 8 | Compute | **Yes** | App B compute table: HCCP entire pipeline ≤ 4 CPU-h, 16-core, no GPU required |
| 9 | Code of Ethics | **Yes** | Read & followed; uses public benchmarks |
| 10 | Broader Impacts | **Yes** | §7: clinical deployment risks + ancestry representativeness + ACMG framework |
| 11 | Safeguards | **NA** | Post-hoc framework, not a high-risk asset |
| 12 | Licenses | **Yes** | TraitGym MIT / ProteinGym CC-BY-4.0 / CADD free academic / Borzoi/GPN-MSA open / AlphaGenome Google API / crepes BSD-3 |
| 13 | New Assets | **Yes** | HCCP code + audit utilities supplied as anonymized supplement |
| 14 | Crowdsourcing/Human Subjects | **NA** | Public benchmark datasets only |
| 15 | IRB | **NA** | No human-subjects research |
| 16 | LLM Usage | **NA** | LLM only for writing assistance, not in core method |

---

## 13. 提交流程时间线

```
现在 (5-03 周日晚)        激活完成 → 准备提交
        ↓
今晚 / 明天 (5-04 周一)    在 OpenReview 提交 abstract submission
                            (title + authors + abstract + keywords + subject area)
                            ⚠️ 不一定要 PDF, NeurIPS 2026 abstract 阶段一般 PDF 可选
        ↓ 拿到 paper ID (例如 NeurIPS-2026-12345)
5-04 周一 - 5-06 周三       回到 OpenReview submission edit page
                            上传 PDF + 填 Checklist + COI
        ↓
5-05 周二 19:59 北京 ⚠️    Abstract DDL — 这一刻系统锁 abstract 字段
        ↓
5-07 周四 19:59 北京 ⚠️    Full paper DDL — 这一刻系统锁 PDF 上传
```

**安全 buffer 建议**: abstract 在 5-04 周一 提交 (DDL 前 24 小时), full paper 在 5-06 周三 提交 (DDL 前 24 小时)。**不要等到 deadline 当天** — OR 系统在 deadline 前几小时通常拥堵。

---

## 14. 提交后的 sanity check

提交完成后回到你的 paper 页面, **核对以下 8 项**:

- [ ] Title 拼写无误
- [ ] Authors 列表完整, 每位有 OR profile link
- [ ] Abstract 渲染正常 (LaTeX 数学符号显示对)
- [ ] Keywords 8-10 个
- [ ] Subject area 已选
- [ ] PDF 已上传, 第 1 页 author 字段是 "Anonymous"
- [ ] Checklist 16 项全答 (Yes/NA)
- [ ] COI emails 已填

任何一项缺 / 错, 立即 edit (deadline 前都允许 edit)。

---

## 15. 截止后

- **不要** 在 deadline 后尝试改 PDF — 系统锁了, 改不了
- 等 review timeline (NeurIPS 2026 一般 7-8 月初次 review, 9 月 rebuttal, 9-10 月 decision)
- 进 reviewer 阶段后, 你的 OR profile 也会被分配审 4 篇左右 paper (DDL 不晚于 6 月底)
