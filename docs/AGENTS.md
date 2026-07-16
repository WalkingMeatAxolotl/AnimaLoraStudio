# AGENTS.md — 项目代码质量与协作公约

**受众**：所有改这个仓库代码的人和工具——AI agent（Claude Code / Codex /
Cursor / Copilot / ...）、社区 contributor、maintainer。
**目标**：让 6 个月后接手代码的人不痛苦。可维护性优先于聪明。

**入口**：本文路径 `docs/AGENTS.md`，从 [`README.md`](../README.md) 的「协作公约」段、
[`CONTRIBUTING.md`](../CONTRIBUTING.md) 的「给 AI agent 的额外说明」段都可达。

如果你想让 AI 工具每次自动加载本文，可在 working tree 根目录建（不会进 git
的）`CLAUDE.md` 或 `AGENTS.md`，单行写 `@docs/AGENTS.md` 即可。

人类贡献者请同时看 [`CONTRIBUTING.md`](../CONTRIBUTING.md)（流程 / 分支 /
commit / PR / release）。本文不重复，只写代码质量、一致性、可维护性、
跨工具协作的约定。

**适用分级**：本文的对齐协议（§1）与自检清单（§5）按改动规模分级。
小改动（typo / 文档 / 单行 fix / 重命名 / 改 copy）不必走完整流程；
中大改动（新 feature / UI 新增 / 跨模块 / schema 改）才走完整流程。
规范目的是减少误会与返工，不确定时优先问 maintainer，不要被条款本身卡住。

---

## 0. 第一原则

1. **可维护性优先于巧妙**：宁可 5 行有名字的函数，不要 1 行嵌套 lambda。
   现在写的代码 6 个月后还得有人读懂。
2. **一致性 > 个人审美**：仓库已有约定（命名、目录、import 方向、错误处理
   模式、组件复用），跟着走。不"顺手优化"成别的风格。
3. **改之前先 grep**：你想新建的组件 / helper / fixture / CLI / registry，
   90% 概率已经有了。先搜。`useProjectCtx` / `SchemaForm` / `PathPicker` /
   `Toast` / `ImageGrid` / `TagEditor` / `OnnxTaggerBase` /
   `useLocalStorageState` 都已存在。
4. **改外部算法前先 verify 论文 / 上游**：InfoNoise / Prodigy / LyCORIS
   这类外部论文实现，不要凭直觉"修正"。先查论文、上游 issue、原作者
   commit，再加单测把公式 codify（参考 `tests/test_infonoise.py`）。
5. **数据流：作者写时规范化**：项目偏好结构化 yaml/schema + 工具校验 +
   派生 markdown（见 `docs/announcements/` → `CHANGELOG.md`）。**不要**
   引入 runtime 解析 free-form 文本的设计。
6. **小步走**：一个 PR = 一个 unit of work。无关改动会让 review 难做。

---

## 1. 开工前对齐协议（防需求漂移）

**适用范围**：加 feature / 改 UI/UX 流程 / 跨模块 / schema 改动。
**不适用**：typo / 文档 / 单行 bug fix / 重命名 / 改 copy / 仅加测试 —— 这些直接动手即可。

属于适用情况时，AI 先和作者对齐。对齐失败就停，不要"边写边问"，
更不要"先写一版再说"。

### 1.1 AI 要主动做的复述

写代码前，用一段话回答给作者看（让作者能立刻发现你理解错了）：

```
我理解这个任务是：
- 解决什么问题：[一句话]
- 用户从哪条路径进入：[页面 / CLI 命令 / 哪个 step]
- 用户看到的行为变化：[操作前 → 操作后]
- 影响的文件：[列表，路径精确]
- 数据 / schema / SSE 是否受影响：[是 / 否 + 哪几个]
- 测试覆盖方式：[regression / 新测试 / 手测]
- 不在范围内：[作者可能误以为也做了的事，明确排除]
```

### 1.2 AI 要主动追问的情况

- **作者给的是症状不是根因**："这个按钮点了没反应"——是 handler 没绑？
  网络错？state 没更新？先定位再问"你期望它做什么"
- **UI/UX 描述含糊**："加个按钮"——放哪？什么样式？点了什么效果？
  跟现有页面的哪个组件相似？**不要凭猜**，看
  [`architecture/studio-pipeline.md`](architecture/studio-pipeline.md)
  的 Sidebar / Stepper 结构，看现有页面（如 `studio/web/src/pages/`）的
  风格再问
- **要改的代码刚改过 / PR 还没合**：先看相关 PR 的讨论上下文，避免推翻
  没几天前的决定
- **作者似乎不知道现有约束**：例如要在 `loop.py` 加 `if optimizer_type ==
  "xxx"`——这违反 plugin 边界（见 §3.4），先告知作者，给替代方案

### 1.3 不要做的

- ❌ 不要默认作者完整描述了需求 —— 多数情况下作者给了 50%，AI 要把另
  50% 问出来或显式标记成"我假设是 X"

---

## 2. 规范文件地图（按场景查）

| 场景 | 看哪里 |
|---|---|
| 开 PR / 切分支 / 写 commit message | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| 发版 / 写 release notes / 改版本号 | [announcements/README.md](announcements/README.md)（格式）+ [announcements/CONTENT-GUIDE.md](announcements/CONTENT-GUIDE.md)（写作规范）+ [CONTRIBUTING.md §Release 流程](../CONTRIBUTING.md) |
| 加 / 改训练栈（LoRA 变体 / optimizer / scheduler / sampler / loss） | [runtime/training/README.md](../runtime/training/README.md) + [adr/0003-anima-train-refactor.md](adr/0003-anima-train-refactor.md) |
| 加 / 改 Studio Web 功能 | [architecture/studio-pipeline.md](architecture/studio-pipeline.md) + [studio/README.md](../studio/README.md) |
| 加新依赖 / 改默认行为 / 接口 / schema | 写 ADR：[adr/README.md](adr/README.md) |
| 改了用户能感知的行为 | [user-guide/](user-guide/) 同步更新 |
| 看历史决策 | [adr/0001-0005](adr/)，**老 ADR 不删不改写**，只追加状态 |
| 写 / 改测试 | [tests/conftest.py](../tests/conftest.py)（`runtime/` 已注入 sys.path） |

---

## 3. 项目特有的强约定（通用知识不写，只写这里独有的）

### 3.1 依赖方向（单向，反向禁止）

```
models  →  utils  →  runtime  →  studio  →  tools
```

- `utils/` 不 import `runtime/` 或 `studio/`
- `runtime/training/` → `utils/`，反向不发生
- `studio/services/inference_core` 复用 `utils/lycoris_adapter`，正因为
  后者不绑训练上下文

### 3.2 Sister script 契约

`runtime/anima_daemon.py` / `anima_generate.py` / `anima_reg_ai.py` 通过
`import anima_train as _T` 拿 7 个名字（`find_diffusion_pipe_root` /
`load_anima_model` / `load_vae` / `load_text_encoders` / `sample_image` /
`enable_xformers` / `resolve_path_best_effort`）。

`runtime/anima_train.py` 顶层 re-export 是契约，**可加不可减不可改签名**。
`tests/test_anima_generate_xy.py` 会捕获破坏。

### 3.3 单一权威源（改这里 → 跑同步工具，不要手改派生文件）

| 权威源 | 派生 / 引用方 | 同步方式 |
|---|---|---|
| `studio/__init__.py:__version__` | FastAPI `/api/health` → Sidebar | 同步 `studio/web/package.json` + `README.md` badge + 「## 版本」段 |
| `docs/announcements/<date>-v<ver>.md`（`tag: release`） | `CHANGELOG.md`、公告栏、GitHub Release Body | `python tools/bump_version.py bump --version X.Y.Z`（格式见 `docs/announcements/README.md`，ADR 0013） |
| `studio/schema.py:TrainingConfig` | `argparse_bridge` 生成 argparse、前端 `SchemaForm` 渲染、4 个 `validate_schema_consistency()` 校验 | 改字段时跟 4 个 plugin registry（adapters / optimizers / schedulers / losses）的 Literal 枚举**一起改**，启动期会拒 |
| `studio/secrets.py` schema | `/api/secrets` + Settings 7 tab 表单 | Pydantic 模型 + 前端表单同时改 |

### 3.4 训练栈插件边界（ADR 0003）

加变体（LoRA 类型 / optimizer / scheduler / sampler / loss）走 plugin
registry，**不动** `main()` / `phases/` / `loop.py` / `context.py`。完整
步骤见 [`runtime/training/README.md`](../runtime/training/README.md)。

防回归：`test_plugin_registry.py` 会拒以下字面量出现在错误位置：
- `phases/optimizer.py` 不该含 `if optimizer_type == "xxx"`
- `phases/models.py` 不该 `AnimaLycorisAdapter(...)`
- `sampling.py` 不该 `if sampler_name == "xxx"`

### 3.5 SSE 事件约定

加新事件类型：
1. 在 [architecture/studio-pipeline.md §6](architecture/studio-pipeline.md) 的事件目录表**先登记一行**
2. Backend 走 `studio/event_bus.py` 的 `bus.publish`
3. Worker 子进程发 typed 事件：往 stdout 写 `__EVENT__:<type>:<json>`，
   supervisor 自动注入 `job_id / project_id / version_id / kind`，worker
   **不要也不能**伪造这几个字段
4. Frontend 走 `useEventStream`（共享连接），不要自己开 `EventSource`

### 3.6 Stage 推进

后端权威，前端只读。写在 `studio/projects.py:advance_stage()`。前端
Stepper 高亮按 stage + version.stats 派生。**不要**在前端 mutate stage，
**不要**绕过 advance_stage 直接 update DB。

### 3.7 Tagger 抽象

新 ONNX tagger 继承 `OnnxTaggerBase` 自动拿线程池调度、GPU EP fallback、
模型解析。注册到 tagger registry，UI 自动列出。**不要**平行实现 ONNX
调度逻辑。

---

## 4. Commit / PR 边界

### 4.1 一个 PR = 一件事

- 加新 feature **不顺手** refactor 别处
- 修 bug **不顺手** 改命名 / 风格 / 无关代码
- 看到要改的代码周围有 typo —— 留给单独的 docs/chore PR

### 4.2 多功能 PR 要警告

如果 AI 发现作者让你做的事**实际**包含 ≥ 2 个互相不依赖的 unit（典型信号：
两组改动可以各自独立合入、影响不同模块、动机不同），**先警告作者**：

```
注意：这个任务看上去是 N 件事：
1. [描述] —— 影响 [文件]
2. [描述] —— 影响 [文件]

合并提交会让 review / revert / cherry-pick 变难。建议拆成 N 个 PR。
要继续合并提交吗？合并的理由是什么？
```

只有作者给出有效理由（必须原子改动、否则中间态会崩、强耦合）才合并。
理由写进 PR 描述。

---

## 5. PR 提交前自检

适用：中大改动。小改动（typo / 文档 / 重命名 / 仅加测试）跳过本节。

**始终检查**：

- 一个 PR 只做一件事；包含多件时理由写进描述
- 改动跟开工前对齐的目标一致，没有滑坡
- 没有"顺手"改的无关代码
- commit message / PR title 是 Conventional Commits 格式
- bug fix 加了 regression test；feat 加了测试覆盖

**按改动类型 trigger**：

- 用户可见行为变化 → 发版时在 `docs/announcements/` 写 release post（`tag: release`，见 README）
- 改了用户能感知的行为 → `docs/user-guide/` 对应章节同步
- 改了单一权威源（version / schema / secrets / SSE event）→ 派生方都同步
- 改了 plugin registry / sister script / Stepper → 见 §8 陷阱清单确认无回归
- UI 改动 → 截图放进 PR 描述
- 前端改动 → **必须用内置浏览器做视觉校验**：`npm run build` 后起真服务、
  浏览器打开受影响页面，确认改动真的渲染出来（vitest 绿 ≠ 视觉正确；
  产物可能复用文件名，注意硬刷新）。校验截图即 PR 描述用图
- 触发 ADR 条件（§7）→ 已起草新 ADR

CI 已覆盖，AI 不必重复手测：本地 tsc / lint / pytest / vitest / plugin registry 启动校验。

发现违反时列出相关项跟作者确认，不要静默通过。

---

## 6. 底线与建议

### 6.1 硬性约束

- ❌ 不直接 push `master`
- ❌ 不 `--no-verify` / 跳测试 / disable lint —— 失败修根因
- ❌ 不在 backend 之外的层直连 SQLite（只走 `studio/db.py`）

### 6.2 强烈建议避免（review 时会请你拆 / 改）

- 在 bug fix PR 里顺手 refactor
- 为"漂亮"改无关代码 / 风格 / 命名
- mock 已有 `tests/conftest.py` fixture 能覆盖的东西
- hardcode 版本号 / 路径 / schema 字段名（见 §3.3 单一权威源）

### 6.3 设计原则

- 不引入"runtime 解析 free-form 文本"的数据流（用 yaml/schema + 工具校验）
- 不"修正"外部论文实现凭直觉(先 verify 论文 / 上游)
- 不为假想的未来需求加抽象（三处相似代码再抽）

---

## 7. 决策记录（ADR）触发条件

**必须**写新 ADR：
- 加 / 换核心依赖（torch backend / 训练框架 / DB / 模型库）
- 改 SQLite schema 中已有表的列含义（不是加列，是改）
- 改 secrets.json schema 字段含义
- 改默认行为（如默认 sampler / attention backend）
- 移除已发布的端点 / CLI flag / schema 字段
- 跨模块重大重构（如 ADR 0003）

**不需要** ADR：bug fix / 加新功能（行为可选，默认关）/ 重命名内部函数 /
新组件 / 测试 / 文档。

模板见 [adr/README.md](adr/README.md)。**已 Accepted 的 ADR
正文不改**，只能在状态行追加 `Superseded by #N` / `Deprecated`。

---

## 8. 陷阱清单（AI / 新 contributor 容易踩）

| 陷阱 | 后果 | 防御 |
|---|---|---|
| 改 Stepper 步骤只改了 `components/ProjectStepper.tsx` | UI 不变（那是死代码）；该改的是 `Sidebar.tsx` 内联版 | grep step 字符串找全部渲染点 |
| 在 `Sidebar.tsx` 硬编码版本号 | 与 `/api/health` 不一致 | 永远从 `/api/health` fetch |
| 在 `runtime/training/loop.py` 加 `if optimizer_type == "xxx"` | 破坏 plugin 边界；`test_plugin_registry.py` 拒 | 加 plugin 走 `BUILDERS` dict |
| 改 `runtime/anima_train.py` re-export 的 7 个名字 | sister script 崩 | 顶层 re-export 可加不可减 |
| release post 缺 frontmatter（version/date/tag/title）| `bump_version.py validate` 拒 / 公告栏不显示 | 抄 `docs/announcements/` 现成模板，按 README 填齐 |
| Hotfix PR 里塞 feature | dev 上未 release 的功能被绑死 | hotfix 严格只修 1 个 bug |
| WD14 / CLTagger 重写 ONNX 调度 | 已有 `OnnxTaggerBase` 抽象 | 继承基类 |
| 加 SSE 事件类型不在事件目录登记 | 前后端不知道事件存在 | 按 §3.5 流程，在 architecture/studio-pipeline.md §6 表先登记 |

---

## 9. AI 的协作角色

AI agent 在帮 contributor 改这个项目时承担三件事：

1. **指出项目特有的约束**：单一权威源、依赖方向、plugin 边界、sister
   script 契约、Stage 推进等。这些是文档化的项目知识，contributor 不一
   定都看过，AI 在动手前主动 cite §3 对应章节。
2. **遇到违反约束的要求时给替代方案**：如果 contributor 想加
   `if optimizer_type == "lion"` 字面量分支，AI 解释 §3.4 plugin 边界
   并给出走 `BUILDERS` dict 的写法，不默默照做也不直接拒绝。
3. **提交 PR 前过一遍 §5 自检**：把检查结果交给 contributor。最终决定
   权在 contributor / maintainer，AI 不替他们做合 / 不合 / 重做的决定。

---

## 10. 维护这份文档

- AI / contributor 老踩同一个坑 → 加到 §8 陷阱清单
- 团队约定变了 → 改这里 + 同步 CONTRIBUTING / ADR / spec
- 新增一类规范文件 → §2 文件地图加一行
- 加新 agent 工具支持（如 Cursor `.cursor/rules/`）→ 在其约定路径放
  stub 指向本文，**不要**复制内容（会漂移）
- 不要让本文膨胀成第二本 CONTRIBUTING —— 价值在**指路 + 项目特有
  约定**，不是复制内容
- 通用 AI 工具已知的（type hints / pathlib / f-string / TS strict /
  hooks rules 这种）**不要**写进来 —— 写了等于噪音
