# Handoff: 用户后续动作（Claude 不能自动完成）

## 当前状态（2026-04-29 evening）

PDF 47 页, weak accept (reviewer v2 自验证), bootstrap CI 已加 Tab tab:cp_baselines, 8 个 'and others' citation 全名补完, K_CV/nested-CV 全文 reconciliation 完成。

## 用户必须亲自完成的任务

### 1. bioRxiv preprint 提交（target 2026-08）

**Claude 不能完成的部分**：
- bioRxiv 账号注册（需要 ORCID 关联 + 邮箱验证）
- 实名 author affiliation 填写（论文目前 anonymous, NeurIPS double-blind 用; bioRxiv 必须实名）
- DOI 分配后填回主文 footnote
- 选 Subject Areas / License / 同步 medRxiv 选项

**操作步骤**:
1. 访问 https://www.biorxiv.org/submit-a-manuscript
2. ORCID 登录（已有 ORCID 跳过此步）
3. 填 author info: 中文姓名 + 拼音 + 机构
4. 上传 `papers/neurips2027_pathA/main.pdf`（47 pages）
5. Supplementary: 可选附 figure source PDFs from `papers/neurips2027_pathA/figures/`
6. Subject: Bioinformatics + Genomics
7. License: CC-BY 4.0（推荐, 兼容 NeurIPS）
8. 同步 medRxiv ✓（临床向 readership 加倍）
9. 提交后 24-48h 内 screening + DOI 分配
10. **拿到 DOI 后**: 让 Claude 把 DOI footnote 加到 main.tex 第二个版本

预计 Path B 还要 ~3h 准备工作（详见 `C_context/biorxiv_preprint_checklist.md`）：
- ✅ Bootstrap CI on tab:cp_baselines（已完成, 此次 commit b5a6sjzwe）
- ✅ Skeletal 数值化（已完成）
- ✅ K̂_CV reconciliation（已完成）
- ⏳ 实名 affiliation 填写（用户决定）

### 2. DEGU end-to-end CNN+ensemble port（2-3 day GPU）

**Claude 可以完成代码骨架**，但实际跑实验需要：
- T4×1 GPU 30 GPU-hour
- 实时 monitor + debug
- 用户在场决定是否 abort / restart

**推荐时序**:
- bioRxiv v1 占坑后（2026-05 早）
- 用户分配 GPU 时间（3 day window）
- 让 Claude 写 `T_tools/degu_end_to_end_port.py` 骨架（已设计稿在 `C_context/degu_end_to_end_port_plan.md`）
- 跑完后产出数据加入 bioRxiv v2

**触发条件**:
- 收到 NeurIPS 2027 reviewer comments 包含 "DEGU end-to-end" critique 后
- 或 bioRxiv reader 反馈缺这个对比

**优先级**: 低. 当前 PDF reviewer v2 已 weak accept, DEGU end-to-end 是防御性实验, 不是 contribution-positive.

### 3. NeurIPS 2027 投稿前 final check（2027-04）

距 NeurIPS deadline ~12 个月. 在投稿前 1 周:
1. 让 Claude 跑一轮新 reviewer simulation v3（验证 1 年累积改动是否破坏现有 weak accept score）
2. NeurIPS Paper Checklist 16 项填完（已映射, 见 `feedback_writing_ml_conference.md` §D）
3. Code of Ethics + Broader Impacts 段最后扫一遍
4. 双盲化: 删 author info + footnote DOI 到 final 版本（保留 anonymous 提交 + camera-ready 实名两份 .tex）

## Claude 已完成 / 不需要用户介入的任务

- ✅ A_SL audit + figure (commit 27da463)
- ✅ Phase 4 surgery 真正完结 (commit 4537cf2)
- ✅ T3' / T5.2 framing 软化 (commit 4537cf2 + 08d2bf9)
- ✅ K_eval sensitivity sweep + table (commit 4537cf2)
- ✅ 3 citation placeholder + 8 'and others' 全名 (commits 4537cf2 + b4cd89c)
- ✅ Reviewer v2 验证: Borderline → Weak Accept (+1 across all 4 axes) (commit b4cd89c)
- ✅ "unique" 措辞 scope qualifier (commit b4cd89c)
- ✅ Bootstrap CI on Tab tab:cp_baselines (本次 commit, 待 push)
- ✅ Skeletal cluster 数值化 (本次 commit, 待 push)
- ✅ K̂_CV/nested-CV reconciliation (本次 commit, 待 push)

## 当前推荐行动

**今天**: 让 Claude 提交本次最终 batch + push, 当日告一段落。

**本周内**: 用户决定是否 register bioRxiv 账号 (无成本) + 准备实名 affiliation 文本.

**下周或更晚**: bioRxiv v1 提交（带当前 PDF）.

**不急**: DEGU port + NeurIPS 投稿前 final check（按 deadline 倒推）.
