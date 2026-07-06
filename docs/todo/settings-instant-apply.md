# 设置页统一为 instant-apply（即时生效）

**状态**：设计定稿，待实现（2026-06-27）
**分支**：`feat/settings-instant-apply`（从 `refactor/settings-split` 切出）
**前置**：`refactor/settings-split`（Settings.tsx 5206→1044 行拆分，已 commit `11e4cb4a`）

---

## 1. 背景与动机

全局设置页约 130 个可改控件，目前混用 6 种提交/持久化模式（逐控件审计已做）：

| 模式 | 含义 | 进全局 dirty |
|---|---|:---:|
| M1 | 文本框本地缓冲，**失焦/Enter** 才写 draft | ✅ |
| M2 | onChange **即时**写 draft（含逐字文本框） | ✅ |
| M3 | onChange **即时 PUT，绕过 draft**（下载源等） | ❌ |
| M4 | 专用端点 PUT/POST/DELETE（custom 模型、字典） | ❌ |
| M5 | localStorage 即时（主题/语言/密度） | ❌ |
| M6 | runtime 动作（下载/安装/迁移/重启） | ❌ |

**痛点**：① 文本框 M1（失焦）vs M2（逐字）不统一，外观无区分；② 下载源等 M3 永不进 dirty，与同卡片其他字段语义割裂；③ training/system tab 的配置几乎全走即时 PUT，而隔壁外观相同的开关却走 draft；④ M1 失焦提交有「打字没失焦就切 tab → 静默丢失」隐患；⑤「撤销更改」对已即时落盘的 M3/M4 无效。

## 2. 调研结论（deep-research，24/25 条经 3 票对抗验证）

业界共识一句话：**保存语义按「控件类型」决定，不按「页面」决定；操作系统级设置的默认是即时生效。**

一手出处：
- Apple HIG [Settings](https://developer.apple.com/design/human-interface-guidelines/settings)：整页无 Save/Apply/Cancel、无「未保存改动」概念。
- GNOME HIG：强制 instant-apply 为 preference 窗口默认；当前版明确「文本设置在 **Return 或失焦时**提交，**绝不逐字**」。仅两种情况允许显式 Apply：操作 >1 秒，或须原子同时生效防不稳定状态。
- Microsoft Fluent 2：「理想体验里改动自动保存」，显式 Save 仅 autosave 不可能时的 fallback。Material 3：「开关效果应立即开始，无需保存」。
- NN/g：不反对即时生效，但**强制要求**配套 ① 清晰的「已保存」反馈 ② 风险改动给撤销/取消路径。
- GitHub Primer 是唯一异见（表单默认显式保存），代表 web 表单 / 网络保存世界观——不适用本地单机工具。

**决策**：本地单机 + 配置即文件（`secrets.json` 原子写、无网络往返、无多端冲突），显式保存的核心理由都不成立，且页面长得就像 OS 设置 → **统一走 instant-apply**。

## 3. 目标心智模型：3 类控件（6 模式收敛）

| 类 | 控件 | 提交时机 | 收敛自 |
|---|---|---|---|
| **A 即时** | 开关/Bool/select/radio/分段/下载源 | onChange 立即 commit | M2 + M3 + M5(secrets 部分) |
| **B 失焦** | 所有文本/数字框（密钥/阈值/路径/URL/model_id） | 本地缓冲，**onBlur/Enter** commit + 校验，**绝不逐字** | M1 泛化 + 逐字 M2 纠正 |
| **C 按钮** | runtime 动作 + 专用端点 | 显式点击（破坏性的先 confirm） | M6 + M4 |

要纠正成 B 类（当前是逐字 M2）的：HF 自定义 URL、Eval/CLTagger 的 model_id 输入、testing 的 idle_timeout/preview number 框、**LLMTaggerWorkspace 全部输入**（含 slider/number/textarea）。

## 4. 机制

### 4.1 `commitField` 统一入口
关键洞察：现有的 `setDownloadSource`（SettingsData.tsx，M3）**就是 instant-apply 的原型**——乐观更新 + PUT 单字段。把它泛化成 `commitField(section, key, value)`，放在 `SettingsData` Provider：
1. 乐观更新 `secrets` state（`setSecrets`）
2. PUT 只带改动的那个 leaf（复用 `buildPatch` 的单字段 diff 能力）
3. 成功 → 更新「已保存」状态；失败 → 回滚乐观更新 + 错误反馈
- **删除**全局 `draft` / `dirty` / `save` / `revert` / 底部保存条 / dirty badge（PR2 废弃，方向反了）。
- 控件直接读 `secrets`（不再有 draft），写走 `commitField`。

### 4.2 文本框（B 类）
- `SettingsInput` / `SensitiveInput` 保留内部本地 useState 缓冲，onBlur/Enter → `commitField`（仅提交目标从 draft 改成 commitField）。
- `TagListInput` 改失焦提交（去掉逐字 onChange）。
- 切 tab / 关抽屉前 flush 未提交的 input（消除「失焦丢编辑」隐患）。

### 4.3 状态反馈（NN/g 要求）
PageHeader 角落一个低调全局指示：`idle`（无）/ `保存中…` / `已保存 HH:MM` / `保存失败（点重试）`。不按控件逐个标，符合 autosave 惯例。

### 4.4 破坏性操作 → 事前 confirm 弹窗（NN/g 撤销路径）
| 操作 | 在哪 | confirm |
|---|---|:---:|
| 删除自定义 LLM preset | LLM workspace | ✅ |
| 重置 builtin preset 到默认 | LLM workspace | ✅ |
| 恢复 tag 词典默认（覆盖用户字典） | 打标·字典 | ✅ |
| 移除 custom 本地模型（移除注册，文件不删） | 训练·模型 | ✅ |
| 上传 tag 词典（覆盖当前） | 打标·字典 | ❌ 选文件已显式，meta 提示覆盖 |
| 切下载源 / 主模型 / 更新通道 / 候选增删 | 多处 | ❌ 改回即可 |
| runtime 重装 / 更新 / 回滚 / 迁移 / 重启 | tagging/training/system | 已有 confirm/preview |

### 4.5 不变项
- M4 专用端点（custom 模型增删、`selectUpscaler`、字典上传/重置）：保留——它们不是 `secrets` leaf，本就该走独立端点，instant-apply 下天然一致。
- M5 localStorage（主题/语言/密度）：保留即时。
- M6 runtime 动作：保留按钮形态，视觉上与设置控件区分。

## 5. 失败与并发处理
- **PUT 失败**：回滚乐观更新到改前值 + 状态指示「保存失败」+ toast 后端原因。
- **并发 PUT**：不同字段并发互不冲突（后端 deep-merge）；同字段快速改用 last-write-wins 或串行队列（实现时定，见 open questions）。
- **校验**：数字 clamp 沿用现有；URL/格式等基本校验在提交前，复杂校验依赖后端拒绝 + 回滚。

## 6. 实施分阶段（同一分支多 commit，最终一个 PR squash 到 dev）

> 统一保存语义是一个内聚 unit，不拆多 PR——否则中间态会出现「一半 instant 一半 draft」更乱。

1. **commit①**：本设计文档。
2. **commit②**：`commitField` 机制 + 全局状态指示 + `fields.tsx` 控件层（B 类失焦化、A 类即时）+ dataset/credentials 试点切换。
3. **commit③**：铺开 monitor / tagging / training / testing / preprocess（A/B 收敛 + 4 项破坏性 confirm）。
4. **commit④**：LLMTaggerWorkspace 全部文本框失焦化。
5. **commit⑤**：删除遗留 draft/dirty/save 死代码 + 更新 `Settings.test.tsx`（原测试基于「改值→点保存」，要改成「改值→即时 PUT」断言）。

## 7. Open questions
- 全局状态指示的确切位置（PageHeader 现有 topRight/actions 区？）与文案。
- 同字段快速连改的 PUT 竞态策略：串行队列 vs last-write-wins vs 短 debounce。
- 失焦提交的 flush 时机（切 tab / 关抽屉 / 组件卸载）如何稳妥触发，避免 React 卸载时 onBlur 不触发。
- 前端前置校验的范围（哪些值得前端拦，哪些交后端）。

## 参考
- 逐控件审计（9 agent，session scratchpad）：`settings-input-audit.md`
- 调研报告：deep-research task `wft7gpq4k`（一手出处见 §2）
