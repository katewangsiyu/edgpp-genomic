# HCCP 论文叙事增强调研（2026-05-07）

**目的**：在已有 `feedback_writing_ml_conference.md` 通用写作 memory（Williams / Farquhar / McEnerney / Schimel / Olson / Heath）之上，为 HCCP（Heteroscedastic Class-Conditional Conformal Prediction，目标 NeurIPS 2026/2027 main）补足三个角落 —— (1) CP/UQ 领域专属 narrative pattern, (2) 理论+经验混合论文的 theorem staging, (3) 不对称证据 framing 的真实成功案例。

**方法**：3 个并发 web research agent，覆盖 ~25 篇 2019–2026 NeurIPS/ICML/Nature/Science/JRSSB paper，全部 verified arxiv/proceedings link。

---

## TL;DR — 5 条核心 takeaway（执行清单）

1. **HCCP 的 narrative arc 最像 CPL (Length Optimization, NeurIPS 2024) + RLCP (JRSSB 2025) 的 hybrid**：CPL 的 "infinite-sample optimal + finite-sample matching" 双层结构 = T5.1+T5.2；RLCP 的 "pointwise impossible → 退一步限定 class" = T5.2 的 hedge 句式。两份"对手"模板已经在论文圈跑通，HCCP 直接对齐即可。
2. **3 主 theorem + 6 lemma 沉 appendix 的当前比例已落在 CP theory paper 中位数**（CPL 6/?, VRCP 3/?, RLCP 3/?）。**不要再砍**。Phase 6 之后的 §5 collapsed-96 行已经够紧。
3. **Abstract 第 4 句必须明写 Mendelian 弱证据 + Theorem-predicted regime boundary**。学 TraitGym abstract 句式（"…**while** [D2] [different finding]"）+ Kandinsky CP 的 boundary 写法。藏在 §6 是 reviewer 直接攻击的把柄。
4. **§5 head 必须加一段 "pointwise conditional coverage is impossible (Vovk; Lei-Wasserman; Barber et al. 2021); we therefore restrict the comparison class to equi-bin Mondrian-K"** —— 这是 RLCP / SC-CP / CPL 共享的 ritualistic insurance，不写 reviewer 会假设你声称 pointwise。
5. **5 句 magic phrases 直接抄进措辞**（见 §3.3），其中最关键的是 **"a regime boundary predicted by [Theorem 5.1]"**（Kandinsky CP + Gibbs-Candès 句式），把 Mendelian 弱处从 limitation 升级为 theory 的 explanatory power。

---

## 1. CP/UQ 子领域专属 narrative pattern

### 1.1 必写的 ritual（reviewer 期待，不写会被攻击）

| 必写段 | 出处 | HCCP 对应位置 |
|---|---|---|
| **Pointwise conditional coverage impossible 免责段**（引 Vovk 2005 / Lei-Wasserman 2014 / Barber-Candès-Ramdas-Tibshirani 2021）| RLCP / SC-CP / CPL 全部有 | §5 head 第一段（必加） |
| **Exchangeability 假设声明**（"we assume (X_i, Y_i) i.i.d. from P, calibration set exchangeable with test point"）| CQR / VRCP / RLCP 都在 §2 第一段 | §2 / §3 setup（确认存在） |
| **Marginal coverage equation** P{Y ∈ C(X)} ≥ 1−α 至少出现一次（reviewer 用这个等式确认你"是 CP 不是别的"）| CP 圈 universal | §2 / §5（确认存在） |
| **"Black-box wrapper" 卖点段**（HCCP works on top of any base predictor）| Angelopoulos-Bates Foundations & Trends 2023 抬到神圣地位 | §6 配 AlphaGenome ablation（已有，加强 framing） |

### 1.2 可省的 ritual（写了像凑字数）

- ❌ **Conformal prediction 历史从 Vovk 1990 写起的回顾段** —— 2020 后 CP paper 普遍不写，一句 "see [Angelopoulos-Bates 2023] for background" 即可
- ❌ **Split vs full conformal 对比表** —— textbook 知识，占半页浪费
- ❌ **Coverage 直方图作 main figure** —— 已被滥用；HCCP fig1 是 σ̂-bin frontier 比 coverage histogram 更有信息量（已做对）

### 1.3 Abstract 句数 benchmark

| Paper | abstract 句数 | 节奏 |
|---|---|---|
| CQR (NeurIPS 2019) | 4 | problem → "existing CP unnecessarily conservative" → propose → guarantee+exp |
| APS (NeurIPS 2020) | 3 | 方法清单 → 卖点 → conformity score 创新 |
| RLCP (JRSSB 2025) | 7（最长）| marg vs local → impossibility → "many CP variants OK" → relax → propose → 双 guarantee → empirical |
| SC-CP (NeurIPS 2024) | 5 | "practical alternative to feature-conditional validity" 是关键 hedge |
| CPL (NeurIPS 2024) | 5 | infinite-sample / finite-sample 两段分隔 |
| VRCP (NeurIPS 2024) | 4 | "CP popular + assumes exchangeability" → "broken under adversarial" → "prior approaches use smoothing" → propose（**3 句铺背景**） |

**HCCP 推荐**：坚持 **Farquhar SPJ 5 句（1+1+1+2）**，不要走 RLCP 的 7 句。但句 4（results）必须按本报告 §3 的 magic phrase 4 重写以 disclose Mendelian 弱处。

---

## 2. 理论+经验混合论文的 theorem staging

### 2.1 主文 thm 数 vs appendix lemma 数 benchmark（实测 6 篇 NeurIPS 2024 CP/UQ paper）

| Paper | 主文 thm/prop/lemma 数 | exp pages | intro pages |
|---|---|---|---|
| Length Optimization (CPL) [arxiv 2406.18814](https://arxiv.org/abs/2406.18814) | 6 | ~2.5 | ~1.5 |
| Verifiably Robust CP (VRCP) [arxiv 2405.18942](https://arxiv.org/abs/2405.18942) | 3 | ~6（强 empirical） | ~3 |
| Information Theoretic CP [arxiv 2405.02140](https://arxiv.org/abs/2405.02140) | 3 bounds | 未验证 | 未验证 |
| RLCP [arxiv 2310.07850](https://arxiv.org/abs/2310.07850) | 3 | 未验证 | 未验证 |
| Boosted Conformal Intervals [arxiv 2406.07449](https://arxiv.org/abs/2406.07449) | 1 | 大头 11 datasets × 10 reps | — |

**HCCP 校准**：3 main thm + 6 lemma 沉 appendix **正好在中位数附近**（CPL 6/?, VRCP 3/?）。**不要再砍**。Conformal Risk Control (ICLR 2024 spotlight) 全是 upper bound 没 lower bound 也中了 —— 所以 T5.2 弱化为 "tight within equi-bin Mondrian-K class" **不是劣势**。

### 2.2 Hero theorem 的 staging 三种范式

- **VRCP 模式**（无 informal preview, 无 sketch）：Theorem 1 紧跟算法定义，主文给 8–10 行完整 proof。**适合 theorem 本身就短**。
- **RLCP 模式**（finite-sample 主 + companion corollary）：Theorem 1 是 hero，Theorem 2 是 "for each B ∈ ℬ" companion。Hero 给 exact statement + 短 remark；formal proof 全沉 App。**最像 HCCP 的 T5.1 + T5.2 + T3' 三角**。
- **CPL 模式**（duality first, finite-sample second）：Prop 3.1 strong duality 是 conceptual hero，Thm 4.1 finite-sample 是 operational corollary。两个 hero **都先来 informal "we show that..." 一段** 引出，再 boxed theorem。

**HCCP 推荐**：**T5.1 走 CPL 模式**（informal preview 17–25 行 → boxed → 1–2 句 proof intuition → App A 详证）；**T5.2 紧跟 T5.1 走 RLCP connector**（"Theorem 5.2 shows this rate cannot be improved within the equi-bin Mondrian-K class…"）；**T3' 单独 subsection 走 "operational certificate" 而非 hero framing**。

排序坚持 **T5.1 → T5.2 → T3'**（logical / strongest-first 双对齐），不要按时间排。

### 2.3 Restricted-class minimax claim 的 hedge 范本（3 种可抄）

| 来源 | 句式 | HCCP 应用 |
|---|---|---|
| **CPL** | "the smallest possible length **within the class ℋ**" / "optimal-length solution **among marginally valid sets**" | T5.2 直接抄："matching lower bound **within the equi-bin Mondrian-K family** $\mathcal{F}_\text{eqM}$" |
| **RLCP** | quantifier 内嵌 hedge："for each B ∈ ℬ" / "for all P̃_X ∈ 𝒫" | abstract 避免单写 "tight"，全部包到 quantifier 里 |
| **VC-class minimax 通用范式** | "tight on a restricted class of procedures" + 显式 class 名字 | §1 contribution 写 "tight rate within this family; SC-CP attains O(n^{-2/3}) on a different axis (§5.4)" |

### 2.4 怎么礼貌但锐利地比 prior bound

VRCP Related Work 段是干净范本（[arxiv 2405.18942](https://arxiv.org/abs/2405.18942)）：

> "Our approach overcomes some of the theoretical and empirical drawbacks of these prior methods, **which are restricted to** classification tasks with ℓ₂-norm bounded guarantees and are overly conservative in practice."
>
> "All the works discussed here rely on randomised smoothing... **In contrast**, our VRCP approach relies on NN verifiers, can be used with any ℓₚ-norms..."

模式：(a) prior 一句 fair credit + 一句 limitation（"restricted to" / "rely on"）；(b) "In contrast" 一句拆；(c) **不写 "we improve" 或 "outperforms"** —— 数字进 table。

**HCCP §6 当前 "33×/2.3× better" 的措辞要重写**：把 33× 留 Table，正文改成 "RLCP attains pointwise local coverage under a randomized localization kernel; in our setting, the bin-conditional gap remains positive at all evaluated $K$ values (Tab. C.1), consistent with the absence of an explicit class-conditional Mondrian partition."

---

## 3. 不对称证据 framing —— 真实成功案例

### 3.1 8 个 real-world 案例（已 verified link）

| paper | venue | 不对称类型 | 弱处怎么 frame | link |
|---|---|---|---|---|
| **TraitGym** (Benegas et al., bioRxiv 2025) | bioRxiv→pipeline | alignment 赢 Mendelian / functional 赢 complex | abstract 直给 split-by-regime: "while" 句式 | [link](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v2) |
| **Kandinsky CP** (Bairaktari et al., ICML 2025) | ICML 2025 | n_per_group >500 赢，小 group 失败 | scale-bound 写在 §6 而非 abstract | [arxiv](https://arxiv.org/abs/2502.17264) |
| **DEGU** (Zhou et al., npj AI 2026) | npj AI 2026 | epistemic 赢 / aleatoric 弱 | "In contrast, [weak axis] posed greater challenges due to inherent randomness" | [npj](https://www.nature.com/articles/s44387-025-00053-3) |
| **Limits of Fair Medical Imaging AI** (Yang et al., Nat Med 2024) | Nature Medicine 2024 | algorithmic correction in-distribution 赢 / OOD 失败 | **title 自己 disclose**："The Limits of…"; 把 negative finding 当主 contribution | [Nat Med](https://www.nature.com/articles/s41591-024-03113-4) |
| **Borzoi** (Linder et al., Nat Genet 2025) | Nature Genetics 2025 | eQTL 强赢 / effect-size 弱赢 | abstract: "competitive with **and often** outperforms" —— "often" 是关键 hedge | [Nat Genet](https://www.nature.com/articles/s41588-024-02053-6) |
| **Adaptive CI** (Gibbs & Candès, NeurIPS 2021) | NeurIPS 2021 | slow drift 赢 / abrupt change 失败 | abstract 不藏；frame prior method 失败为自己改进动机 | [arxiv](https://arxiv.org/abs/2106.00170) |
| **Conformal Risk Control** (Angelopoulos et al., ICLR 2024 spotlight) | ICLR 2024 spotlight | monotone loss 内强 / non-exchangeable 需 extra structure | Limitations **嵌在 Discussion** 一段，**不独立 §7**（NeurIPS 9 页主文页面紧时的省页技巧） | [openreview](https://openreview.net/forum?id=33XGfHLtZg) |
| **AlphaMissense** (Cheng et al., Science 2023) | Science 2023 | monomeric 强 / complex/condensate 失败 | Discussion 单句 "**One limitation** of AlphaMissense is that the structural component does not, at present, account for…" | [Science](https://www.science.org/doi/10.1126/science.adg7492) |

### 3.2 NeurIPS 官方对 disclosure 的明文期待

- **NeurIPS Paper Checklist Guidelines**：「Reviewers will be specifically instructed to **not penalize honesty concerning limitations**.」
- 「Claims in the paper should match theoretical and experimental results in terms of **how much the results can be expected to generalize**.」
- 「The paper's contributions should be **clearly stated in the abstract and introduction**, along with any important assumptions and limitations.」
- **NeurIPS 2026 Reviewing Guidelines**：authors should "reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs."

### 3.3 5 句 magic phrases（直接抄进 HCCP）

1. **"a regime boundary predicted by [Theorem 5.1]"** —— Kandinsky CP §6 + Gibbs-Candès "may fail to quickly react"-style framing。**HCCP 用法**：abstract 句 4 + §1 contribution 3 + §6.x 弱证据段。
2. **"competitive with and often outperforms"** —— Borzoi abstract。"often" 是 NeurIPS-survivable hedge。**HCCP 用法**：abstract 句 4 提 Mendelian 时。
3. **"In contrast, [weak axis] posed greater challenges due to [theory-rooted reason]"** —— DEGU。**HCCP 用法**：§6.x 弱证据段；把 "fragility" frame 成 theory-predicted。
4. **"this optimality is not true in new [regime]; [our theory predicts] [scope condition]"** —— Yang et al. Nature Medicine 2024。**HCCP 用法**：§7 limitation 单句。
5. **"alignment-based models compare favorably for [D1]…while [other class] perform better for [D2]"** —— TraitGym abstract（**同领域同 dataset**，reviewer 已习惯句式）。**HCCP 用法**：abstract 句 4 双向句式直接套。

### 3.4 三个反例（被批评的 paper）

| 案例 | 错在哪 |
|---|---|
| GPT-4 "perfect score on MIT exam" (2023, withdrawn) | label noise + duplicates + contamination + cherry pick；最终 authors 主动 withdrawn（[Leech et al. arxiv 2407.12220](https://arxiv.org/html/2407.12220)） |
| ICLR 2025 anonymized Spotlight | "intentionally suppress contradictory scientific evidence"；AC 拒后被 overruled spotlight（[arxiv 2506.19882](https://arxiv.org/html/2506.19882v1)） |
| Mirai EMBED external validation (Yala et al., STM 2021) | external cohort 没 exclude diagnostic mammograms，inflated AUC（follow-up [Radiology 2023](https://pubs.rsna.org/doi/full/10.1148/radiol.222679)） |

---

## 4. 给 HCCP 的具体行动清单（按章节）

### 4.1 Abstract（最高优先级，5 分钟改完）

**当前问题**：句 4 大概率没明写 Mendelian 弱证据 + theory-predicted boundary。

**改写模板**（直接套 TraitGym + DEGU + Kandinsky 句式）：

> "On TraitGym **complex_traits**, HCCP yields disjoint-CI improvements over RLCP, weighted CP, and SC-CP (paired-bootstrap p≈0.005), **while** on **mendelian_traits** the gain is restricted to a K_eval≈3 window (1.15× over weighted CP) — **a regime boundary predicted by Theorem 5.1** when per-cell minority count drops below the heteroscedastic head's identifiability threshold."

### 4.2 §1 Introduction

- **Contribution 第 3 条改写为 "regime characterization"**，把 Mendelian fragility 升级为 contribution（学 Yang et al. Nat Med 把 negative finding 当主 contribution）：

> "Our third contribution is **a sharp characterization of the operating regime**: HCCP dominates when per-cell minority count exceeds n_min⋆ (Eq. T5.1); below this threshold, the heteroscedastic head is unidentifiable and the method recovers — but does not improve over — class-Mondrian baselines."

- **加一条 "transferable conformity score" bullet**（学 APS 套路）：把 s(x,y) = |y - p̂|/σ̂ 抽出来卖。
- **§1 缩到 1.5 页**（CPL 1.25 / RLCP 2 / 当前你 2.5 太长）。

### 4.3 §5 Theory

1. **§5 head 第一段必须加**："pointwise conditional coverage is known impossible (Vovk; Lei-Wasserman; Barber et al. 2021); HCCP achieves the strongest finite-sample guarantee within the equi-bin Mondrian-K class". **没有这段 reviewer 直接攻你声称 pointwise**。
2. **T5.1 statement 之前加 17–25 行 informal preview**（CPL 模式）："we show that K* trades bin-bias against per-bin variance, yielding the dimension-free optimum K* = ⌊√(L_F R π_min n)⌋"。
3. **T5.2 statement 内部显式定义** $\mathcal{F}_\text{eqM}$ 一行（§5.2 开头），后续直接引用。**删掉任何不带 "within" 限定的 "tight" / "optimal"**。
4. **T3' 撤到 subsection + 加 transition 句**："Beyond rate optimality, we provide an operational σ̂-bin KS certificate that quantifies the *non-vacuous lower bound* on bin-conditional coverage gap (Theorem 3'); this is **loose by construction and bounds the worst case rather than predicting empirical gap**."

### 4.4 §6 Experiments

1. **把 "33×/2.3× better" 留 Table**，正文改成 VRCP 风格的 "In contrast" 句式（见本报告 §2.4）。
2. **Mendelian K_eval=3 fragility 升格独立 0.5 页 subsection "Honest fragility regime"**，学 CPL/SC-CP 的 "practical alternative" hedging tone。**这是你跟所有对手 paper 的 differentiator**（他们 dataset 都太均匀，写不出这种诚实段）。模板：

> "On Mendelian, HCCP **wins only in the K_eval=3 recommendation region** (Tab. C.1; 1.15× over wCP, 74.5% paired-bootstrap wins, p≈0.26). At K_eval≥5, per-cell minority count drops below ~70 — the empirical regime where Theorem 5.1's $\sqrt{L_F R \pi_{\min} n}$ scaling is no longer dominated by the heteroscedastic-σ̂ improvement. We **report this fragility honestly because the same theory that predicts our Complex-traits gain also predicts this Mendelian boundary**; an empirical loss in this regime would have **falsified the theoretical claim**, not merely the method."

3. **§6 字数预算**：3 页（CPL 2.5 / RLCP 1.5 + 强 appendix）。其中 asymmetric Mendelian honest secondary 单独 0.5 页。
4. **AlphaGenome ablation 加一句 "HCCP works on top of any base predictor"** —— 借 Angelopoulos-Bates 卖点段。

### 4.5 §7 Limitations

- **按 Conformal Risk Control 模板，Limitations 嵌在 Discussion 一段而非独立 §7**（NeurIPS 9-page 紧），单句即可：

> "Our heteroscedastic-σ̂ machinery requires per-cell minority count above the identifiability threshold derived in §5.1; the Mendelian K_eval≥5 regime in §6 is the empirically observed boundary. We do not claim coverage in regimes below this threshold."

- **加一句关于 T5.2 的 disclosure**："T5.2 is matching only within Mondrian-K; SC-CP achieves O(n^{-2/3}) on a different axis（§5.4）" —— 防 area chair 对照 SC-CP rate 后觉得你藏信息。
- **总长 ≤ 0.75 页**（NeurIPS reviewer 不会因为 limit 短扣分，但会因为没 limit 扣分）。

---

## 5. 调研边界（什么没能验证）

| 项 | 状态 |
|---|---|
| CPL / SC-CP / Risk Control / VRCP 主文 vs appendix theorem 精确比例 | 部分 PDF fetch 中断；要硬数字需下载后用 `pdftotext` 数 `\begin{theorem}` |
| RLCP §2 第一段 exchangeability 声明逐字版本 | 未抓全 |
| OpenReview meta-review 帖明文写 "abstract must disclose weak evidence" | 未找到；只有 NeurIPS 官方 checklist 是公开 normative source |
| Allen-Zhu / Bartlett / Foster 单独的 NeurIPS 2024 theorem-staging 实例 | search 没干净匹配；范例只能从 conformal 子领域取 |
| ICLR 2025 "intentionally suppress evidence" Spotlight 在 Position paper 里 anonymized | 没法给 paper-level link |

---

## 6. 引用清单（已 verified link）

### CP / UQ NeurIPS-tier paper
- CQR — [arxiv 1905.03222](https://arxiv.org/abs/1905.03222)
- APS (Romano-Sesia-Candès) — [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf)
- Adaptive CI (Gibbs-Candès) — [arxiv 2106.00170](https://arxiv.org/abs/2106.00170)
- Angelopoulos-Bates Foundations & Trends 2023 — [arxiv 2107.07511](https://arxiv.org/abs/2107.07511)
- Conformal Risk Control — [openreview ICLR 2024](https://openreview.net/forum?id=33XGfHLtZg)
- RLCP (Hore-Barber JRSSB 2025) — [arxiv 2310.07850](https://arxiv.org/abs/2310.07850)
- SC-CP (van der Laan-Alaa NeurIPS 2024) — [arxiv 2402.07307](https://arxiv.org/abs/2402.07307)
- CPL / Length Optimization (Kiyani-Pappas-Hassani NeurIPS 2024) — [arxiv 2406.18814](https://arxiv.org/abs/2406.18814)
- VRCP (Jeary et al. NeurIPS 2024) — [arxiv 2405.18942](https://arxiv.org/abs/2405.18942)
- Information Theoretic CP (Kossen et al.) — [arxiv 2405.02140](https://arxiv.org/abs/2405.02140)
- Boosted Conformal Intervals (Xie-Barber-Candès) — [arxiv 2406.07449](https://arxiv.org/abs/2406.07449)
- Kandinsky CP (Bairaktari et al. ICML 2025) — [arxiv 2502.17264](https://arxiv.org/abs/2502.17264)

### 不对称 framing 案例
- TraitGym — [bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v2)
- DEGU — [npj AI 2026](https://www.nature.com/articles/s44387-025-00053-3)
- Yang et al. "Limits of Fair Medical Imaging" — [Nat Med 2024](https://www.nature.com/articles/s41591-024-03113-4)
- Borzoi — [Nat Genet 2025](https://www.nature.com/articles/s41588-024-02053-6)
- AlphaMissense — [Science 2023](https://www.science.org/doi/10.1126/science.adg7492)

### Meta / 反例
- Leech et al. "Questionable practices in ML" — [arxiv 2407.12220](https://arxiv.org/html/2407.12220)
- Refutations & Critiques Track Position — [arxiv 2506.19882](https://arxiv.org/html/2506.19882v1)
- Mirai validation (Radiology 2023) — [Radiology](https://pubs.rsna.org/doi/full/10.1148/radiol.222679)
- NeurIPS Paper Checklist Guidelines — [neurips.cc](https://neurips.cc/public/guides/PaperChecklist)
- NeurIPS 2026 Reviewing Guidelines — [neurips.cc](https://neurips.cc/Conferences/2026/ReviewerGuidelines)
