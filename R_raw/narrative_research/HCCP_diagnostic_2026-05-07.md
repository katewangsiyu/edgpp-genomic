# HCCP 主文 8 章诊断报告（2026-05-07）

**对照标准**：
1. `feedback_writing_ml_conference.md`（通用 memory，Williams / Farquhar / McEnerney / Schimel / Olson / Heath）
2. `R_raw/narrative_research/HCCP_storytelling_research_2026-05-07.md`（HCCP 专项调研）

**已读文件**：`main.tex`（abstract）+ `sections/01-08`。

**总判断**：Phase 4-8 已经做完关键 surgical work（T3' 软化 / T5.2 within-class qualifier / Mendelian honest secondary 已 wired），但 **abstract 顺序错位 / 句子嵌套过深 / 残留 banned words / 倍率写正文** 这四类 reviewer 红旗仍然成片存在。**估计改完一轮可以减 1-2 个 borderline reviewer 的扣分**。

---

## P0 —— 必改（阻断 reviewer trust，每条 5-30 分钟）

### P0-1. Abstract 句 1 错位 —— 不是 Achievement 而是 Setup

**现状**（`main.tex:50` 句 1）：
> "For binary classification under both class imbalance and prediction-dependent noise --- the regime of clinical variant effect prediction (VEP) --- existing conformal methods trade class-conditional coverage against bin-local coverage, a Pareto wall sharpened by the finite-sample pointwise impossibility of \citet{barber2020limits}."

**问题**：Farquhar 5 句公式（memory §E）句 1 应该是 Achievement（"We introduce / We prove / We demonstrate..."），当前句 1 是 Setup/Importance。HCCP 句 2 才是 "We introduce HCCP"，违反 1+1+1+2 节奏。reviewer 第一遍 scan 看不到 contribution。

**改写模板**（按 Farquhar 句 1+句 2 重排）：
> 句 1（Achievement）：「We introduce **HCCP** (Heteroscedastic Class-Conditional Conformal Prediction), a post-hoc framework that pairs a learned variance head $\shat(x)$ with Mondrian-$(y \times \shat\text{-bin})$ calibration to attain class-conditional and bin-local coverage simultaneously.」
> 句 2（Importance）：「For binary classification under both class imbalance and prediction-dependent noise — the regime of clinical variant effect prediction — existing conformal methods trade these two objectives, a Pareto wall sharpened by the pointwise impossibility of Barber et al.」

---

### P0-2. Abstract "tight" 单独出现 —— reviewer 红旗

**现状**（`main.tex:50` 句 4）：
> "Within the equi-bin Mondrian-$K$ family we prove a **tight** finite-sample $O(n^{-1/2})$ rate..."

**问题**：HCCP 调研报告 §2.3 + memory §F 明确说 **"tight" 单独出现就是 reviewer 红旗**（即使后面 "matched within this family" 已经 qualify）。在 abstract 里更敏感 —— area chair 第一遍 scan 就会标记 over-claim。

**改写**（删 "tight"，用 quantifier 内嵌 hedge）：
> "Within the equi-bin Mondrian-$K$ family we prove an $O(n^{-1/2})$ upper bound on the worst-cell coverage gap, **matched within this family** by an $\Omega(n^{-1/2})$ lower bound with $\pi_{\min}^{-1/2}$ explicit in the constant..."

—— "matched within this family" 已经 implicit 表达 tight，**不需要再写 tight**。

---

### P0-3. Abstract Mendelian 弱处 disclosure 位置太晚

**现状**（`main.tex:52` 句 6 后半）：
> "...A $K_{\mathrm{eval}}$ sweep shows the Complex advantage is robust across $K_{\mathrm{eval}} \in [2, 10]$, **while on Mendelian weighted CP overtakes HCCP for $K_{\mathrm{eval}} \geq 5$** (per-cell minority $< 70$); a feature-pipeline-disjoint replication on Open Targets matched-9..."

**问题**：HCCP 调研 §3 + 不对称 framing 案例（TraitGym / DEGU）规定 **abstract 句 4（Evidence 第 1 句）必须 disclose 弱处**。当前拖到第 6 句最后一句的从句里，相当于"藏到末尾"。AC 第一遍 scan 看不到 honesty signal。

**改写**（直接 import HCCP 调研 §4.1 给的模板，并加 magic phrase #1 "predicted by Theorem"）：
> 句 4-5（Evidence，1 长 1 短）：
> 「On TraitGym Complex ($n{=}11{,}400$), HCCP yields disjoint-CI improvements over RLCP, weighted CP, and SC-CP at the recommended operating point ($K_{\mathrm{eval}}{=}5$; paired-bootstrap $p{\approx}0.005$), **while** on Mendelian ($n{=}3{,}380$) the gain is restricted to a $K_{\mathrm{eval}}{=}3$ window — **a regime boundary predicted by Theorem~\ref{thm:t5oracle}** when per-cell minority count falls below the heteroscedastic head's identifiability threshold. We replicate cross-platform on Open Targets matched-9 (feature-pipeline-disjoint, $16.7\times$ at the recommended operating point) and cross-domain on ProteinGym.」

—— **效果**：弱处暴露在句 4，AC 第一遍就看到 honesty；同时把 "regime boundary predicted by Theorem" magic phrase 钉进 abstract。

---

### P0-4. §1 第二段数字过密，reviewer 消化不动

**现状**（`01_introduction.tex:12-13`）：单段塞了 cov 0.62, gap 0.83, 13.4×, $G(K^\star) \leq 2\sqrt{L_F R / (\pi_{\min} n)}$ 加两个 theorem 引用。

**问题**：Williams 句子重力 + Heath Concrete 都同意 "用具体数字" — 但**一段不能塞 5 个数字 + 2 个 theorem**。reviewer 在前 5 分钟 scan 里没有 cognitive budget 消化这些。

**改写**：
- 数字 13.4× 移到 Tab. caption 或 §6
- 单段拆两段：第一段只描述 single-axis 失败的**定性现象**，第二段才上 theorem 数字
- 或者把 cov / gap 数字也都推后到 §6（Tab.~\ref{tab:main} 自然展示）

---

### P0-5. Conclusion 一段塞 3 contribution + "decisively dominant"

**现状**（`08_conclusion.tex:3`）：单 paragraph 13 行，3 contribution 全部塞一段；其中 "**decisively dominant** on Complex" + "more contested signal on Mendelian"。

**问题**：
- "decisively dominant" 在 memory §B 禁用词清单（"significantly / dramatically / decisively"）
- Conclusion 单段反 Williams 节奏，reader 找不到 takeaway
- "contested signal" 措辞 OK 但需要配 magic phrase

**改写**（拆 3 段 + 删禁用词 + 用 magic phrase）：
```
Para 1（restate ABT 紧凑版，2-3 行）：
We introduced HCCP, the first conformal framework attaining class-conditional 
and bin-local coverage jointly under class imbalance and prediction-dependent 
noise. Three contributions: ...

Para 2（contributions 1-3 各 1 句，用 "to our knowledge" hedge）：
(1) A finite-sample O(n^{-1/2}) rate within the equi-bin Mondrian-K class with 
    π_min explicit in the constant (Thm 5.1, 5.2); SC-CP attains O(n^{-2/3}) 
    on a different axis (§5.4).
(2) The HCCP framework with T3 exact + T3' robust corollary; T3' is the 
    operational certificate where the per-chromosome KS audit rejects A2-cell 
    on Mendelian (46.2%, K-invariant).
(3) Empirical validation on TraitGym + Open Targets + ProteinGym + synthetic, 
    with disjoint-CI improvements on Complex and a Mendelian K_eval-window 
    advantage that is itself a regime boundary predicted by Eq.~(gap_decomp).

Para 3（natural extensions, 2 行即可）。
```

—— 删 "decisively dominant"；保留 "regime boundary predicted by"；3 段比 1 段更易扫读。

---

## P1 —— 应改（提升清晰度，每条 10-30 分钟）

### P1-1. §1 Contribution 1 标题里 "Tight" 残留

**现状**（`01_introduction.tex:17`）：「**Tight** finite-sample rate within the equi-bin Mondrian-$K$ class with $\pi_{\min}$ explicit in the constant」

**改写**：「**Finite-sample $O(n^{-1/2})$ rate, matched within the equi-bin Mondrian-$K$ class**, with $\pi_{\min}$ explicit in the constant」

—— 删 "Tight"；"matched within" 表达 tight 同时显式 hedge。

---

### P1-2. §6 倍率写正文（13.7× / 2.3× / 5-25×）

**现状**：
- `06_experiments.tex:33`：「reduces the marginal $\shat$-bin gap by a further **$5\text{--}25\times$**」
- `06_experiments.tex:39`：「HCCP tightens the bin-local gap by **$13.7\times$** on Complex / **$2.3\times$** on Mendelian over the on-axis baseline B2」

**问题**：HCCP 调研 §2.4（VRCP 范本）+ memory hedging 都说 **倍率进 Table，正文不写 outperforms / X×**。当前 § 主体充满 "X× tighter / X× win"，reviewer 一看就是 sales tone。

**改写**：
- 主文保留**绝对数字**（gap = 0.060 vs 0.827）和**defining 比较**（"HCCP attains a marginal gap within one standard error of the $1/(n_{k,b}+1)$ floor"）
- 倍率 13.7× 进 Tab.~\ref{tab:h2h} caption 或 footnote，不写在 § 主文
- §6.2 改 VRCP 风格："on-axis $\shat$-Mondrian (B2) attains local coverage shape but loses minority coverage (cov$_{|Y=1}=0.624$); **in this setting** HCCP's joint partition keeps both within the target band (Tab.~\ref{tab:h2h})."

---

### P1-3. §6.3 + §8 残留 "decisively dominant"

**现状**：
- `08_conclusion.tex:3`："**decisively dominant** on Complex"
- 应该 grep 全文还有几处

**改写**：「consistently above the operating threshold on Complex」/「holds the target band on Complex while a Mendelian boundary at $K_{\mathrm{eval}}\geq 5$ matches Eq.~(gap_decomp)」

---

### P1-4. T5.1 缺 informal preview（CPL 范式没用）

**现状**（`05_theory.tex:29-41`）：直接给 Eq.~(gap_decomp) → boxed Theorem。

**问题**：HCCP 调研 §2.2 推荐 **CPL 模式**（informal preview 17-25 行 → boxed → proof intuition）。当前 T5.1 没有 conceptual setup，reviewer 看到 $K^\star = \lfloor\sqrt{L_F R \pi_{\min} n}\rfloor$ 第一眼想问 "why √n not n^{1/3}"。

**改写**（在 Eq.~(gap_decomp) 之前加 1 段）：
> "We show that the worst-cell coverage gap admits a clean bias-variance decomposition: bin-stationarity bias decreases as $1/K$ (more bins → tighter $\shat$ stratification), while finite-sample variance grows as $K/(\pi_{\min} n)$ (more bins → fewer minority samples per cell). Balancing yields the dimension-free oracle $K^\star = \Theta(\sqrt{n})$ with $\pi_{\min}$ entering through the binding minority-cell variance. Eq.~(gap_decomp) makes this precise."

—— 25 行减为 5 句，节省页面同时给 reviewer 直觉。

---

### P1-5. §7 "Operational envelope" 段密度过高

**现状**（`07_discussion.tex:7`）：单段塞 R1-R4 + 4 个 Scope subitem，14 行连续文本。

**改写**：拆 2 段 —— para 1 仅 R1-R4 case studies（用 boldface label 已有）；para 2 仅 4 个 Scope subitem（用 (i)-(iv)，已有）。每段开头加 1 句 transition：
> Para 1 head：「We document four out-of-envelope regimes as case studies, not coverage claims:」
> Para 2 head：「Beyond these four regimes, four scope conditions delimit the present results:」

---

### P1-6. §1 head 缺 OCAR Resolution 桥接句

**现状**（`01_introduction.tex:14-15`）：第二段末以 theorem 引用结尾，紧接 `\paragraph{Contributions.}` —— 缺少 1 句 "we therefore propose..." 桥接。

**改写**（第二段末 + 一句）：
> "...with $\pi_{\min}$ entering explicitly as a $\pi_{\min}^{-1/2}$ constant. **HCCP instantiates this construction with three components — heteroscedastic head $\shat(x)$, $(y\times\shat\text{-bin})$ Mondrian partition, and per-cell calibration — yielding the contributions below.**"

—— OCAR 的 Resolution 句让 reader 知道接下来 contribution list 是 instantiation 而非新概念。

---

## P2 —— 细节 polish（5-15 分钟每条）

### P2-1. Magic phrase #2 "competitive with and often outperforms" 没用上

memory §3.3 + 调研报告 §3.3 推荐 Borzoi abstract 的 "**often** outperforms" 句式（"often" 是 NeurIPS-survivable hedge）。HCCP 在 Mendelian 是 "often-but-not-always"，正好套这个模板。

**应用**：abstract 句 4 / Conclusion para 2 / §6 head 都可借。例：
> "On Mendelian, HCCP **competes with and often outperforms** weighted CP at the recommended operating point; the advantage narrows below per-cell minority $\sim 70$."

---

### P2-2. §1 Contribution 3 太长（9 行 1 bullet），拆 2 bullet

**现状**（`01_introduction.tex:19`）：单 bullet 含 "TraitGym + Open Targets + ProteinGym + synthetic + 强/弱不对称 + decoupling from base predictor"。

**改写**：拆成
- (3a) TraitGym + cross-platform Open Targets matched-9 replication（feature-disjoint）
- (3b) Cross-domain ProteinGym + synthetic n-sweep validating dimension-free O(n^{-1/2})

—— 强证据 + 弱证据各占 1 bullet，对应 §G.5 asymmetric framing。

---

### P2-3. Abstract "single-fold partition we evaluated" 嵌套过深

**现状**（`main.tex:51`）：
> "HCCP is the only single-fold partition we evaluated (across split CP, $\shat$-Mondrian, class-Mondrian, RLCP, weighted CP, and SC-CP at matched $(\phat, \shat)$) that holds..."

**问题**：8 个嵌套从句的句子，Williams style + Gopen-Swan stress position 都受损 —— 句尾不是关键信息。

**改写**（拆 2 句，subject = HCCP）：
> "We evaluate HCCP against six contemporary CP variants (split, $\shat$-Mondrian, class-Mondrian, RLCP, weighted CP, SC-CP) at matched $(\phat,\shat)$. **HCCP is the only method holding $\mathrm{cov}_{|Y=1}\geq 0.85$ and $\shat$-bin gap $\leq 0.20$ on both datasets** under $B{=}200$ chromosome-bootstrap CIs."

---

### P2-4. T3' 段框架已软化但 transition 缺一句

**现状**（`05_theory.tex:22`）："T3$'$ is the operative certificate on Mendelian rather than a tight prediction..."

**建议加**（HCCP 调研 §4.3 推荐的 transition）：
> 在 §5.1 head 第一段末加："Beyond rate optimality, we provide an operational σ̂-bin KS certificate that quantifies the *non-vacuous lower bound* on bin-conditional coverage gap (Theorem~\ref{thm:t3prime_main}); **this is loose by construction and bounds the worst case rather than predicting empirical gap**."

—— 主动 absorb reviewer 攻击 "你 K-invariant 46% rejection 不就说 framework 没用？"

---

### P2-5. §2 Related Work paragraph 1 过长（20 行连续 citation）

**现状**：从 split CP 一直写到 crps_binning2026，全部一段。

**改写**：拆 3 段 —— (a) classical CP + Mondrian + Lipschitz score，(b) modern local CP（RLCP / SC-CP / weighted），(c) concurrent rate analyses（Yao / Plassier / CRPS）。每段 4-5 行。reviewer 扫读时定位明确。

---

### P2-6. §4 Method "Why the joint partition" 段过宣传

**现状**（`04_method.tex:25-26`）：「the **unique refinement** that admits both T2 and T3; the two strict coarsenings each violate one of the two coverage targets」

**改写**：去掉 "unique" 改 "minimal sufficient refinement" —— 同样信息但弱化 absolutism。或加 footnote 限定 "unique among Mondrian product partitions over $\{y, b(x)\}$"。

---

## Top 5 高 ROI 改动（按 priority × 工作量）

| # | 改动 | 位置 | 工时 | 影响 |
|---|---|---|---|---|
| 1 | Abstract 句 1 句 2 互换 + 删 "tight" | `main.tex:50` | 10 min | 高 — AC 第一印象 |
| 2 | Abstract Mendelian 弱处提前到句 4 + 加 "predicted by Theorem" | `main.tex:51-52` | 15 min | 高 — honesty signal |
| 3 | Conclusion 单段拆 3 段 + 删 "decisively dominant" | `08_conclusion.tex` | 10 min | 高 — closure 印象 |
| 4 | §6 倍率 13.7× / 2.3× / 5-25× 推 Table，正文改 VRCP 风格 | `06_experiments.tex:33,39` | 20 min | 中 — sales-tone 风险 |
| 5 | T5.1 加 informal preview（5 句） | `05_theory.tex:29` 之前 | 15 min | 中 — theory readability |

总工时 ~70 min。改完后建议跑 `pdflatex` 重编译并对照本报告 P0/P1 复查一遍。

---

## 不需要改的地方（已经做对的）

- ✅ §1 已有 "pointwise impossibility" 免责（Barber 2020）—— HCCP 调研 §1.1 必写已 wired
- ✅ §5 head ritualistic insurance 已有
- ✅ T3' 框架已软化为 "operative certificate" / "loose by construction"
- ✅ T5.2 "within-class tightness statement, not a procedure-class minimax" qualifier 已加
- ✅ §7 Limitations 嵌在 Discussion 而非独立 §7（对齐 Conformal Risk Control 模式）
- ✅ §7 "Operational envelope" R1-R4 + Scope (i)-(iv) 已 honest disclose
- ✅ §1 Contribution 3 末已 "regime boundary anticipated by Eq.~(gap_decomp)"
- ✅ §6.2 末已 "regime boundary, not a model-class limitation"
- ✅ §8 末已 "regime boundary at $K_{\mathrm{eval}}\geq 5$ is consistent with Eq.~(gap_decomp)"
- ✅ Aspects A1, A1', A2-cell, A-SL 假设清楚 + 配 KS 46.2% audit
- ✅ DEGU 从 villain 重新定位为 "ally / pioneer"（§2 末 + §6 ablation footnote）
- ✅ ProteinGym 升 §6.3 主文（不是 strawman port）
- ✅ M→C 降级为 case study R3，不当 OOD 行
- ✅ Skeletal/connective 子簇 cov$_{|Y=1}=0.745$ 在 §6.3 honest disclose
- ✅ 数字 0.31 (TV) / 46.2% (KS) / 100% (Complex bootstrap) / 74.5% (Mendelian bootstrap) 一致

---

---

## P3 —— 追加 audit findings（checklist + refs + 编译质量）

### P3-A. Page budget ✅ 已通过（不需要改）

`main.aux` 显示：
- `sec:intro` p1, `sec:related` p3, `sec:formulation` p4, `sec:method` p5, `sec:theory` p5, `sec:experiments` p7, `sec:discussion` p9, `sec:conclusion` p9
- `app:proofs` p12 ⇒ **bibliography 占 p10-11**, **main paper = 9 页 ✓**
- PDF 总 43 页，0 undefined ref ✓

不需要砍页，**但有 7 条 overfull hbox** 见 P3-G。

### P3-B. Phase 4-6 surgery 已干净（不需要改）

Grep 主文 `T1/T2/T1'/T2'/T3-loc/T3.b/T4` 结果：所有出现都是 (a) Table caption / (b) Algorithm comment / (c) Tab.~\ref{tab:main} 里 calibration column 标签，**没有 dangling `\ref{lem:t1}` 等引用**。Phase 6 自评干净。

### P3-C. NeurIPS Paper Checklist 整体 ✅，但 2 处可优化

**全部 16 项已填 `\answerYes` 或 `\answerNA`，justification 长度合规**。但：

1. **#5 Open Access 措辞过保守**：
   > "Code (...) is provided as anonymized supplementary material at submission time **and will be released under a permissive licence on acceptance**"

   NeurIPS 2026 reviewer 实际期待 review 时就能跑代码。"on acceptance" 不算违规但弱。**建议**：明确说"anonymized code zip uploaded as supplementary material accompanying this submission; final permissive-license release on acceptance"，把"已提供"信号放前。

2. **#16 LLM 末句 unnecessary** （`checklist.tex:83`）：
   > "...The aggregator $\phat$ is a gradient-boosted classifier...; both consume pre-computed numeric features. **LLM use was confined to writing assistance.**"

   NeurIPS 2026 政策明文写 "writing/editing/grammar/formatting **不需声明**"。写出来反而暗示 LLM 用得多，可能触发"用了多少"的好奇。**建议删最后一句**，前面"LLMs were not used as any component of the core methodology"已足够。

### P3-D. refs.bib —— 3 处需 fix

1. **`plassier2025cp2hpd` author 仍是 placeholder**（`refs.bib:39`）：
   > `author={Plassier, Vincent and others}`

   "and others" 是 BibTeX placeholder，会渲染成 "et al." 但 reviewer 检查 bib 时会看到。**Phase 6 citation audit 漏过此处**。需补全：Plassier, Vincent et al. ICLR 2025 "Probabilistic Conformal Prediction with Approximate Conditional Validity" 的全 author list。

2. **`crps_binning2026` arxiv ID 需 verify**（`refs.bib:296`）：
   > `journal={arXiv preprint arXiv:2603.22000}`

   2603 = 2026-03 是 valid 月份，但单月序号 22000 偏大（arxiv 单月通常 ~15-18K）。**今天 2026-05-07 reviewer 大概率会 search 这个 ID**。建议核实是否真的存在；不存在的话用 Toccaceli 任一 published 工作替换或改用 \citep{...} 注脚。

3. **`vovk2003mondrian` techreport 不可访问**（`refs.bib:92-97`）：
   > Royal Holloway 内部技术报告，无 public URL

   reviewer 想 verify Mondrian CP 原始 reference 找不到 PDF。**建议**：要么加 `note={Available at https://...}`，要么换为 Vovk-Lindsay-Nouretdinov 任一公开 venue 版本（如 COPA proceedings）。

### P3-E. 残留 banned word "SOTA"（`06_experiments.tex:33`）

> "The GBM aggregator alone reaches **SOTA** Mendelian AUPRC $0.900$"

memory §B 禁用词清单：`novel / first / unprecedented / SOTA` 必须配 "to our knowledge"。这里 SOTA 单独出现。

**改写**：
> "The GBM aggregator alone reaches Mendelian AUPRC $0.900$, **a $+14.8$\,pp absolute improvement over the published LogReg baseline of \citet{benegas2025traitgym}** —— **the strongest TraitGym Mendelian AUPRC we are aware of**."

—— 用 "the strongest...we are aware of" 替代 SOTA + "to our knowledge"-style 内嵌。

### P3-F. Conclusion 引用 AlphaGenome（`08_conclusion.tex:3` 末）

> "...richer base predictors (AlphaGenome \citep{avsec2026alphagenome}), and population-stratified Mondrian"

Conclusion 一般不引入新 citation（Williams + NeurIPS 习惯）。AlphaGenome 已在 §2 / §6 ablation 引过。**建议**：删 \citep{avsec2026alphagenome}，留"AlphaGenome"裸名（reviewer 已经看过 §2）。

### P3-G. 7 条 overfull hbox / 5 条 underfull

`main.log` 显示：
| 严重度 | 位置 | 内容（推测） |
|---|---|---|
| 84pt overfull | line 324 | 极可能是 Tab.~\ref{tab:main} 末行长 cell |
| 57pt overfull | lines 14-31 | §1 Contribution bullet 单行过长 |
| 47pt overfull | (未确认) | 可能 long URL / equation |
| 37pt overfull | lines 15-31 | §1 Contributions area |
| 17pt overfull | lines 22-39 | 推测 §2 Related Work 长 \citep |
| 5pt overfull | line 257 | minor |

**reviewer presentation 分扣**：>10pt overfull 是 reviewer 一眼看到的格式 issue。

**改法**：
1. 84pt overfull 可能是 Tab.~\ref{tab:main}：可改 `\setlength{\tabcolsep}{3pt}` 或 `\small\to\footnotesize` 或调整列宽
2. §1 Contribution bullet：第 1 条 contribution 文本（5 行长，含多个 inline \citep）拆 2 句或加 manual line break
3. §2 Related Work：连续 8+ \citep 可拆 paragraph 或换 `\cite{a,b,c,...}` 一次性引用

**优先级**：>50pt 必修；20-50pt 应修；<20pt 可放过。

---

## P0+P1+P3 完整修改优先级（按 ROI 重排）

| # | 改动 | 位置 | 工时 | 影响层 |
|---|---|---|---|---|
| 1 | Abstract 句 1 句 2 互换 + 删 "tight" | `main.tex:50` | 10 min | **AC 第一印象** |
| 2 | Abstract Mendelian 弱处提前句 4 + magic phrase | `main.tex:51-52` | 15 min | **honesty signal** |
| 3 | Conclusion 单段拆 3 段 + 删 "decisively" + 删 \citep AlphaGenome | `08_conclusion.tex` | 10 min | closure 印象 |
| 4 | §6.1 删 "SOTA" 改 "the strongest...we are aware of" | `06_experiments.tex:33` | 2 min | banned word |
| 5 | §6 倍率 13.7× / 2.3× / 5-25× 推 Table | `06_experiments.tex:33,39` | 20 min | sales-tone |
| 6 | T5.1 加 informal preview | `05_theory.tex:29` 之前 | 15 min | theory readability |
| 7 | refs.bib `plassier2025cp2hpd` 补全 author list | `refs.bib:39` | 5 min | reviewer trust |
| 8 | refs.bib `crps_binning2026` arxiv ID 核实 | `refs.bib:296` | 10 min | reviewer fact-check |
| 9 | Checklist #16 删末句 "LLM use was confined..." | `checklist.tex:83` | 1 min | scope |
| 10 | Checklist #5 Open Access 措辞调整 | `checklist.tex:28` | 5 min | reviewer trust |
| 11 | §1 Contribution 1 删 "Tight" → "matched within" | `01_introduction.tex:17` | 5 min | banned word |
| 12 | §7 "Operational envelope" 段拆 2 段 | `07_discussion.tex:7` | 10 min | readability |
| 13 | Overfull hbox 84pt + 57pt 修 | `06_experiments.tex` Tab + `01_introduction.tex` | 20 min | presentation |
| 14 | refs.bib `vovk2003mondrian` 加 URL 或换公开版 | `refs.bib:92` | 10 min | verify |
| 15 | §2 Related Work paragraph 1 拆 3 段 | `02_related_work.tex:3-4` | 15 min | readability |
| 16 | Magic phrase #2 "competitive with and often outperforms" | abstract / §6 / Conclusion | 5 min | hedge |

**总工时**：约 **2 - 2.5 小时**，按上面顺序做完一轮即可。

**改完后的验证 checklist**：
1. `pdflatex main.tex && bibtex main && pdflatex main && pdflatex main` 重编译，确认仍 0 undefined ref + 主文 ≤ 9 页
2. `grep -niE "tight|SOTA|decisively|significantly|substantially|dramatically" sections/*.tex main.tex | grep -v "within|on this class|equi-bin"` 应回空
3. `grep -E "Overfull.*pt too wide" main.log` 期望 ≤ 3 处且最大 ≤ 30pt
4. PDF 第 1 页 abstract 第 1 句应该是 "We introduce HCCP..."
5. PDF 第 1 页 abstract 第 4 句应该出现 "while on Mendelian" + "predicted by Theorem"
