# 项目 / 版本 / 任务 生命周期重构（讨论稿）

> 🚦 **文档状态**：本文档是两轮讨论稿。**最终决议见 [ADR-0007](../adr/0007-project-version-lifecycle-refactor.md)**。
> 作为讨论历史保留 —— §1–§10 是一轮决议（部分已被 §11 推翻）、§11 是二轮收敛、§12 是 park 的 follow-up。
> 实施请按 ADR-0007 走，不要按本文档前 10 节走。
>
> 分支：`feat/project-queue-relation` · 起草：2026-05-22 · 二轮收敛：2026-05-23

---

## 1. 背景与动机

当前 `Project.stage` / `Version.stage` 各 9 个枚举值的状态机存在结构性问题：

- **Project 卡片在说谎**：训练失败 / 取消 / 暂停后，`projects.stage` 不被推进，徽章永远显示"训练中"+ 动画 dot，用户回来以为还在跑。
- **死状态**：`curating` / `configured` / `regularizing`（项目级）从无代码推进，3/9 颜色映射用户永远看不到。
- **跳级与自相矛盾**：上传 train 图直接跳到 `tagging`，跳过 5 个 stage —— 说明 stage 本质是"做过哪些步骤的集合"，但 UI 把它当线性进度展示。
- **无 source of truth**：`advance_stage()` 是无条件 setter，规则散在 9 个调用点，Project / Version / Task 三套状态机互不对账。
- **数据模型错位**：Project 是聚合体（1:N → Version），不应该有"过程状态"。

经过 PM / User / Designer 三方两轮辩论后形成共识：**模型层面就错了**，需要彻底重构而非补丁修复。

---

## 2. 重构核心原则

1. **Project 没有过程状态**。Project 是收纳盒，只有"存在 / 不存在"两态（不做软删除，详见 §3）。
2. **Version 是真正的"训练实验"实体**，承载完整的运行状态机。
3. **Phase（做过哪些工序）和 Status（当前运行态）必须分离** —— 这是消除"stage 跳级"矛盾的根因。
4. **Task 是事实（fact），Version.status 是派生 + cached**。supervisor 是唯一推进者，UI 永远不直接写 status。
5. **数据集（download + preprocess）属于项目级**，多个 version 共享同一份原图（与本地工具用户心智一致）。

---

## 3. 数据模型（最终决策）

### Project（极简，纯收纳盒）

```python
{
    id, title, slug, note, created_at, updated_at,
}
```

**删除字段**：`stage`、`active_version_id`

**关于 phase**：Project 表**不存任何 phase 字段**。数据集相关信息（download/ 是否非空、图片数）由 UI 实时扫文件系统计算 —— 这本质就是文件系统状态的派生，存 bool 是冗余且有漂移风险。

**关于"代表 version" / focus / pinned**：**不做**。理由：metric strip 已经回答了"哪些在跑 / 哪些挂了 / 哪些完了"这个核心问题；用户想看具体哪个在跑，点进去看 version 列表即可，**卡片上不需要"代表 version"**。pinned 更是负担（要手动操作、pin 错反而迷茫）。
- Project 详情页默认打开 version：取最近更新的 version（纯派生，不存字段）
- 项目卡片：metric strip + 数据集快照 + 最新产物入口，**无焦点行**

**关于软删除**：本次重构**不做** archive / trashed 状态。Project 只有"存在 / 物理删除"两态。未来如需软删可单独开 ADR。

### Version（核心实体）

```python
{
    id, project_id, label, note, created_at, updated_at,

    # 运行状态（状态机，详见 §4.2）
    status: "draft" | "training" | "paused"
          | "completed" | "failed" | "canceled",

    # 工序完成标记（5 个独立 bool，从筛选开始）
    # 注：数据集导入/预处理是项目级，UI checklist 动态拼为"第 0 项"
    phases: {
        curated:             bool,   # 已为本 version 选定 train 图集
        captioned:           bool,   # 训练图 caption 覆盖率达标
        regularization_ready: bool,  # 有正则集 或 显式声明不用
        training_configured: bool,   # 训练 config 存在且校验通过
        artifact_produced:   bool,   # output/*.safetensors 落盘
    },

    # 派生 / 缓存
    output_lora_path: str | None,
    last_task_id:     int | None,
    last_failure_reason: str | None,
}
```

**删除字段**：`stage`

### Task（保留 project_id 作为冗余）

```python
{
    id, project_id, version_id, kind: "train" | "reg_ai" | "generate" | ...,
    status: "pending" | "running" | "paused" | "done" | "failed" | "canceled",
    # ... 其余执行字段保持现状
}
```

**保留 `project_id` 字段**，但**作为冗余 of `versions.project_id`**：
- NOT NULL（迁移后所有 task 都必须挂在 version 下）
- 一致性约束：`task.project_id == version.project_id`，应用层强制 + DB CHECK 双重保险
- 收益：队列页 / Monitor 等高频查询零 JOIN；点击 task 跳 project 直接可用
- 代价：写入路径必须维护一致性（supervisor / API 创建 task 时强制赋值）

### project_jobs（一张表，按 kind 区分 scope）

不拆表。`project_jobs` 表的 schema 已经支持 `version_id` 可空：

| kind | scope | version_id | 完成时影响 |
|---|---|---|---|
| `download` | project | NULL | UI 实时算（无 DB phase） |
| `preprocess` | project | NULL | UI 实时算 |
| `tag` | version | NOT NULL | `version.phases.captioned = true` |
| `reg_build` | version | NOT NULL | `version.phases.regularization_ready = true` |

各 kind 共享 5 状态：`pending → running → done | failed | canceled`（无 paused）

> 命名讨论：`project_jobs` 表名其实涵盖了版本级 job，可改名 `async_jobs` 更准确。但改名涉及 migration，暂不动。

---

## 4. 状态机定义

### 4.1 Project Lifecycle

Project 没有显式状态机，只有 `created → 物理删除` 两个生命事件。卡片显示信息完全派生自 Version 聚合 + 项目级 phases。

### 4.2 Version Status（6 态）

```
                  [createVersion]
                        │
                        ▼
                  ┌─────────────┐
              ┌──▶│    draft    │ ← 合并了旧 draft + ready + preparing
              │   └──────┬──────┘   涵盖"未在训练 / 未完成"所有准备态
              │          │ submit train task
              │          ▼
              │   ┌─────────────┐  pause     ┌─────────────┐
              │   │  training   │ ─────────▶ │   paused    │
              │   │             │ ◀───────── │             │
              │   └──────┬──────┘  resume    └─────────────┘
              │          │             ↑ 合并了旧 queued + training
              │          │
              │          ├── task done ─────▶ ┌────────────┐
              │          ├── task failed ───▶ │ completed  │ ★ 终态
              │          └── task canceled ─▶ └────────────┘
              │                                ┌────────────┐
              │                                │   failed   │ ★ 终态
              │                                └────────────┘
              │                                ┌────────────┐
              │                                │  canceled  │ ★ 终态
              │                                └─────┬──────┘
              │                                      │
              └──────────────────────────────────────┘
                  用户点"再训练" → 回 draft 或直接 training
```

**合并决策**：
- `draft + ready + preparing` → **统一为 `draft`**
  - 理由：用户"还在准备阶段"是单一心智，区分子态对用户无价值
  - 命名注：`draft` 字面偏"空白"，但实际涵盖"已经准备很多还没 submit 训练"的所有非运行态。用户实际感知靠 phases checklist 进度而非 status 标签
- `queued + training` → **统一为 `training`**
  - 理由：用户视角"在跑"，等 GPU slot 和真正跑的体感无差异

**转换约束**：
- ❌ 不能从 `completed` 退回 `draft`（拒绝倒退，保护历史事实）
- ✅ 从 `completed / failed / canceled` 可启动新训练 → 回 `training`
- ✅ `resume` 只能从 `paused`
- ✅ 在 `training / paused` 时**禁止 submit 新 task**（一次一个）
- ✅ `status` 由 supervisor 推进，**UI 永远不直接写**

### 4.3 Task Status（保持现状）

```
[submit] → pending → running → done | failed | canceled
                       │
                       └─ pause ─▶ paused ─ resume ─▶ running
```

终态：`done / failed / canceled`。每个终态向所属 Version 投递 transition event。

### 4.4 project_jobs Status（一张表，按 kind 分流）

```
[create] → pending → running → done | failed | canceled
```

终态影响：

| Job kind | scope | version_id | 完成时影响 |
|---|---|---|---|
| `download` | project | NULL | UI 实时算（无 DB phase） |
| `preprocess` | version | NOT NULL | 改 `train/` 物理文件，UI 实时算（ADR 0010） |
| `tag` | version | NOT NULL | `version.phases.captioned = true` |
| `reg_build` | version | NOT NULL | `version.phases.regularization_ready = true` |

### 4.5 Phases（独立 bool，非状态机）

**存储策略**：**持久化为 DB 字段**（决策 §6.2）。仅存于 Version 表，Project 表无 phase 字段。

**Version phases**（5 个）：`curated`、`captioned`、`regularization_ready`、`training_configured`、`artifact_produced`

每个 phase 是独立 bool，无序、可重复、**可回退**。UI 用 capability checklist 展示。

**前向写入**（设为 true）：
- 对应 job 完成时由 supervisor 写
- 用户主动动作（保存 config → `training_configured = true`、curate 选图 → `curated = true`）

**回退写入**（设为 false）—— 决策 §6.2 强制做：
| 用户/系统动作 | 触发回退 |
|---|---|
| 删除训练图至 caption 覆盖率 < 阈值 | `captioned = false` |
| 删除训练 config | `training_configured = false` |
| 清空正则集 | `regularization_ready = false` |
| 删除全部训练图 | `curated = false` |
| 删除 output safetensors | `artifact_produced = false` |

回退逻辑写在每个相关 endpoint 内，作为副作用提交。

**UI checklist 第 0 项（项目数据集）动态拼装**：
- if project `download/` 为空 → 显示"导入数据集" CTA（跳项目页）
- if 非空 → 显示"✓ 数据集就绪 (N 张)"
- 这一项**不**作为 Version.phases 的字段，是 UI 派生

---

## 5. 关联关系

```
Project (id, title, slug, note)
   │  1:N
   ▼
Version (id, project_id, status, phases, output_lora_path)
   │  1:N
   ▼
Task (id, project_id, version_id, status)
                       ↑
              project_jobs (id, project_id, version_id?, kind, status)
```

- Project ─1:N→ Version：项目下多个训练实验
- Version ─1:N→ Task：一个 version 可以重训多次（前一次失败/重新调参）
- Project ─1:N→ project_jobs：表统一，按 kind 区分 scope（version_id 可空）

**关于 Task.project_id**：保留作为冗余字段，强一致约束 `task.project_id == version.project_id`。高频查询（队列页、Monitor）零 JOIN；点击 task 跳 project 链路直接可用。

---

## 6. 关键决策（已确认）

### 6.1 数据集归属 → 项目级（保留现状）

`download/` 和 `preprocess/` 留在 **项目目录**，多个 version 共享同一份原图。

- 理由：本地训练场景中"同一角色不同 version 用同批原图"是 90% 用法，独立 download 浪费磁盘且违背用户心智。
- UI 影响：
  - 项目卡片**不**直接显示数据集状态（避免噪音）
  - 项目详情页新增"训练集"区域，展示 download/preprocess 图片数、状态、操作入口
  - Version 的 capability checklist **第 0 项动态拼装**：项目无数据集 → "导入数据集" CTA；项目有数据集 → "✓ 数据集就绪"
  - Version.phases 字段从 `curated` 开始（不存数据集 phase）

### 6.2 Version Phases → 持久化存储 + 强制回退

phases 存为 DB 字段，**不实时算**。

**写入策略**：
- 前向（设 true）：job 完成 / 用户动作（保存 config、选图等）
- 回退（设 false）：删图、清 config、删正则集等动作必须同步回退对应 phase
- 一致性靠各 endpoint 显式维护，不靠"事后扫描修复"

理由：
- 写入路径明确，状态机一致性容易保证
- 读取性能：列表页可能展示几十个项目 × 多版本，每次扫文件系统不可接受
- 代价是工作量（每个回退路径要写代码），但回退逻辑都是 1 行 SQL UPDATE，可控

### 6.3 Project 不存任何 phase

数据集状态（`download/` 是否非空、图片数）由 UI 实时扫文件系统计算。

- 理由：这是文件系统状态的派生事实，存 bool 是冗余且有漂移风险
- 实现：项目详情页"训练集"区域读 `len(os.listdir(download_dir))`，列表页**不**显示这个信息
- Project 表彻底干净：`id / title / slug / note / pinned_version_id / 时间戳`

### 6.4 Version status 粒度 → 6 态

合并后从 9 态降到 6 态：
- `draft`（合并 draft + ready + preparing）
- `training`（合并 queued + training）
- `paused`
- `completed`
- `failed`
- `canceled`

### 6.5 Task.project_id 保留 + 一致性约束

不删 `task.project_id` 字段，但定位为 `versions.project_id` 的冗余。

- NOT NULL（迁移后所有 task 都必须挂在 version 下）
- 约束 `task.project_id == version.project_id`：应用层强制（创建 task 时显式赋值）+ DB CHECK 双保险
- 收益：队列页 / Monitor 等高频查询零 JOIN；点击 task 跳 project 链路直接可用

### 6.6 Project 不做软删除

不引入 `lifecycle: active/archived/trashed`，Project 只有"存在 / 物理删除"两态。未来需要时单独开 ADR。

### 6.7 project_jobs 不拆表 / 不改名

`download / preprocess / tag / reg_build` 统一放在 `project_jobs` 表，按 kind 区分 scope（`version_id` 可空）。

- 不拆表：拆表是 schema 洁癖，但多一份 supervisor / 事件 / migration 代码；一张表已经够清晰
- 不改名为 `async_jobs`：命名瑕疵不值得专门 migration；如未来搭车其它 migration 可顺手做

### 6.8 不做"代表 version" / 焦点行 / pinned

项目卡片**不**显示"哪个 version 是代表"。

- 理由：metric strip 已经回答了"哪些在跑 / 哪些挂了 / 哪些完了"这个核心问题
- 用户想看具体哪个在跑 → 点进项目详情看 version 列表，**卡片不需要"猜代表是谁"**
- pinned 让用户手动指定 → 是操作负担，pin 错反而更迷茫
- 项目详情页默认打开 version：取最近更新的（纯派生，不存字段）

### 6.9 一致性校验：完整跑

每次读 version（list / get / 任何 endpoint）都校验 `version.status` vs 最新 task.status 推导值。

- 不一致 → log warning + 以 task 推导为准修正字段 + 发 SSE 让前端 refresh
- 性能：如发现是瓶颈，加 cached 字段 `status_implied_at` 由 supervisor 写时同步更新
- 决策原则：真理优先于性能

### 6.10 数据集实时扫不加 cache

项目详情页"训练集"区域每次都实时 `os.listdir(download_dir)`，**不**加 mtime cache。

- 理由：几百个文件的 listdir 本来就是毫秒级，cache 是过度优化
- 用户改了 download/ 内容立刻看到变化，体验更诚实

### 6.11 "再训练"语义与入口

Version 在终态（`completed / failed / canceled`）时显示明确 CTA：
- `completed` → `[再训一次]`（复用 config）
- `failed` → `[查看日志] [复用配置重试] [用更小 batch 重试（OOM 时）]`
- `canceled` → `[继续训练] [复用配置重新开始]`

**重训不创建新 Version**：保留同一 version 的多次 task 历史；用户想换 config 就开新 version。

与现状的差别：现状没有"重训"概念，要重训只能重新 submit task；重构后"重训"是 Version 上的一等动作，UI 一键完成。

---

## 7. UI 派生规则（项目卡片）

Project 卡片显示完全派生，**无焦点行 / 无 pinned**：

```python
def project_card_view(project):
    versions = list_versions(project.id)
    return {
        # metric strip（核心信息）
        "total":     len(versions),
        "running":   count(v.status in ["training", "paused"]),
        "completed": count(v.status == "completed"),
        "failed":    count(v.status in ["failed", "canceled"]),

        # 数据集快照（项目级，实时扫）
        "dataset_image_count": len(os.listdir(download_dir)),

        # 最新产物入口（取最近完成的 version）
        "latest_artifact": latest_completed_version_output(versions),
    }
```

**项目详情页默认打开的 version**：取最近更新的（纯派生，不存字段）。

```python
def default_version_for_detail(project):
    versions = list_versions(project.id)
    return max(versions, key=lambda v: v.updated_at) if versions else None
```

详细 UI 设计（metric strip / capability checklist / 失败 affordance / 三级通知）见单独的 UI 设计稿（待写）。

---

## 8. 旧数据迁移映射

```
旧 projects.stage              → 全部丢弃（Project 不再有 stage）
旧 projects.active_version_id  → 删除（不需要"代表 version"概念）

旧 versions.stage → 映射:
    "created"                  → status="draft", phases 全 false
    "downloading"              → status="draft"（download 状态归项目，UI 实时算）
    "preprocessing"            → status="draft"
    "curating"                 → status="draft", phases.curated=true
    "tagging"                  → status="draft", phases.captioned=true
    "regularizing"             → status="draft", phases.regularization_ready=true
    "configured" / "ready"     → status="draft", phases.training_configured=true
    "training"                 → 查最新 task：
                                   task.status=running    → "training"
                                   task.status=paused     → "paused"
                                   task.status=done       → "completed"
                                   task.status=failed     → "failed"
                                   task.status=canceled   → "canceled"
                                   task 不存在            → "draft"
    "done"                     → status="completed", phases.artifact_produced=true

旧 tasks.project_id            → 保留字段，加 NOT NULL + CHECK 约束
                                 （从 version JOIN 回填 NULL 行）

旧 Project.phase / 数据集状态  → 不迁移（Project 表无 phase 字段，UI 实时算）
```

---

## 9. 改造范围清单（给 ADR 用）

### Backend
- [ ] `studio/projects.py`：删 stage / advance_stage / active_version_id（不再需要"代表 version"概念）
- [ ] `studio/versions.py`：重写状态机（6 态）、5 个 phases 字段、转换约束、前向+回退写入函数
- [ ] `studio/db.py` (tasks)：保留 project_id，加 NOT NULL + DB CHECK；迁移回填 NULL 行
- [ ] `studio/project_jobs.py`：不拆表；明确各 kind 的 version_id 必填/可空规则
- [ ] `studio/supervisor.py`：完成 task 时推 version.status + 相关 phases；失败/取消/暂停也要推
- [ ] `studio/server.py`：删除 9 处 `projects.advance_stage()` + 4 处 `versions.advance_stage()` 调用，改为相应 phases / version status 推进
- [ ] 各 endpoint 加 phases 回退逻辑（删图 / 删 config / 清正则集 / 删 artifact）
- [ ] 新建 DB migration：删字段 + 加字段 + 数据回填（按 §8）
- [ ] 一致性校验函数：读 version 时 assert `version.status_implied_by_tasks() == version.status`，不一致 → log + 以 task 推导为准修正 + 发 SSE

### Frontend
- [ ] 删除 `StageBadge` 组件（或保留改用于 Version status）
- [ ] 新建 `MetricStripBadge` 组件（项目卡片用）
- [ ] 新建 `PhasesChecklist` 组件（Version 详情页用，第 0 项动态拼装数据集）
- [ ] 项目详情页新增"训练集"区域（实时扫 download/ preprocess/ 显示图片数）
- [ ] `Projects.tsx` 卡片重写：metric strip + 数据集快照 + 产物入口（**无焦点行**）
- [ ] `Sidebar.tsx` 内联 Stepper 改为 Capability Checklist
- [ ] Version 详情页：终态时显示"再训练"CTA（completed/failed/canceled）
- [ ] 失败状态 affordance：静止图标 + 红 hue + 显式动词文案 + "查看日志" / "用更小 batch 重试" actions
- [ ] 三级通知（tab 红点 / 桌面通知 / 邮件）—— 可单独 ADR

### 文档 / ADR
- [ ] 本文档讨论收敛后转 `docs/adr/000X-project-version-lifecycle-refactor.md`
- [ ] 单独 UI 设计稿 `docs/design/project-card-redesign.md`
- [ ] 更新 `AGENTS.md` 第 3.6 节（Stage 推进规则全部失效）
- [ ] 更新 `architecture/studio-pipeline.md` 状态转换表

---

## 10. 决策记录（2026-05-23 已全部收敛）

- [x] **Version status 命名**：`draft`（合并旧 draft + ready + preparing）
- [x] **不做"焦点 version"**：metric strip 已经回答了核心问题，焦点行是冗余；卡片只显示 metric strip + 数据集快照 + 最新产物入口
- [x] **不做 pinned_version**：用户手动 pin 是负担，pin 错更迷茫
- [x] **不要 active_version_id 字段**：项目详情页默认打开 version 取"最近更新"，纯派生不存
- [x] **phases 回退**：做。每个删除/清空动作 endpoint 内显式 UPDATE 回退（见 §4.5 / §6.2 表）
- [x] **project_jobs 不拆表**：一张表按 kind 区分 scope，version_id 可空
- [x] **不改名 async_jobs**：命名瑕疵不值得专门 migration
- [x] **一致性校验完整跑**：每次读 version 时都校验 `version.status` vs 最新 task.status 推导值；不一致 → log + 以 task 推导为准修正 + 发 SSE
- [x] **数据集实时扫不加 cache**：`os.listdir` 几百文件本来毫秒级，mtime cache 是过度优化
- [x] **"再训练"入口**：Version 详情页终态时显示 CTA（completed → "再训一次"；failed → "查看日志/复用配置重试/用更小 batch 重试"）；不创建新 version，保留同 version 的多次 task 历史

~~**剩余无 open question，可进入 ADR。**~~
**↑ 2026-05-23 用户二轮 review 推翻，见 §11。**

---

## 11. 第二轮 open questions（2026-05-23）

> §10 的"全部收敛"在用户二轮 review 后被推翻。9 个新问题 + 2 个事实修正。
> 简单的标 ✅，复杂的标 🔥 待逐个讨论。
> **本章节所有变更只记在 §11**；前面 §1–§10 暂不动，等讨论收敛再回头统一改 §3 / §4 / §6 / §8 / §9。

### 11.0 事实修正（与 master 对账）

- **§1 错** —— "`Project.stage` / `Version.stage` 各 9 个枚举值" 不对。
  - `projects.VALID_STAGES` 真的是 9 态（`created/downloading/preprocessing/curating/tagging/regularizing/configured/training/done`）
  - `versions.VALID_STAGES` **只有 6 态**（`curating/tagging/regularizing/ready/training/done`）—— 没有 `draft` / `preparing` / `created`
- **§4.2 / §6.4 论证基准错** —— "合并旧 draft + ready + preparing → draft" 是相对**早期重构方案**说的，不是 master 现状。
  - master 实际是 6 → 新 6 的重切分，不是 9 → 6。
  - 真实映射建议（待 §11.3 收敛）：
    - `curating / tagging / regularizing / ready` → `draft`（区分靠 phases）
    - `training` → `training`
    - `done` → `completed`

### 11.1 `active_version_id` 保留 ✅

不删。语义改为 **"最后打开的 version"**（current 行为不变），不是 PM/UX 意义上的 "pinned" 或 "代表 version"。

**待回头改的章节**：
- §3 Project 字段 → 重新加回 `active_version_id`
- §6.8 → 改为 "不做 pinned / 代表 version，但保留 last-opened 轻量语义"
- §8 → 移除 "active_version_id → 删除" 一行
- §9 → backend 任务移除 "删 active_version_id"，frontend `Layout.tsx` / `Overview.tsx` / `Sidebar.tsx` 保留

### 11.2 侧栏 stepper 映射与完成判定 ✅ 已定（§11.5 + §11.8 落地）

**映射 7 步 → 新模型**：
- ① 下载 / ② 预处理 → **项目级**（在侧栏 Version 之上，跟随项目而不是 version）
- ③ 筛选 / ④ 打标 / ⑤ 编辑 / ⑥ 正则集 / ⑦ 训练 → **version phase**（5 enum cursor，§11.3-B）
- 数字 ①–⑦ **全局保留**（不重新编号，保留 master UX 心智）

**完成判定**（§11.5 派生）：
- version phase ③–⑦：`phase_index < cursor_index` 即"完成"
- 项目级 ①②：实时扫文件系统派生（§6.10）

**视觉表现**（§11.8 落定）：
- 完成 phase：**数字字符变绿**（"③" 单字符绿色，文字常规）
- 当前 cursor：所在侧栏项**背景高亮**
- 当前路由：所在侧栏项**背景高亮**（cursor 和 focus 都用背景表达；同 phase 时叠加 OK）
- 项目级 ①② 右侧括号内显示文件数（"① 下载 (80)"），不显示完成 ✓

**结论**：不需要"打勾"图标 —— 数字本身变绿 = 完成的 UI 表达。

### 11.3 Version 状态机：paused / 6→5 态合并 / 多 task 聚合

§11.0 已说明合并基准要按 master 现状重写。本节拆三个子问题，A 已定，B 待写，C 新 open。

#### 11.3-A version 级 `paused` ✅ 去掉

**决定**（2026-05-23 讨论）：去掉 version-level `paused`。`paused` 不在 version 状态机里。

**理由**：
- task-level pause（ADR 0006）已经 cover 短/中期暂停场景（case 1/2）
- "version 长期搁置"（case 3）是伪需求 —— 真要搁置就用 `completed` + note，或 cancel 重开
- 多一态多一处一致性维护（supervisor 写入 + SSE 触发 + version vs task 状态不一致 race）
- 心智一致：pause 按钮只对应一个动作（task pause），不会有"应该 pause task 还是 pause version"的犹豫

**UI 退化方案**：
- pause 按钮 → 只发 task pause（不动 version.status）
- version 卡片 / metric strip 渲染：`status === 'training' && active_task?.status === 'paused'` → 显示 "训练中 · 已暂停 Nmin" + pause icon
- 不需要 `version_state_changed` 在 task pause/resume 时触发

**最终状态机**：5 态 = `draft / training / completed / failed / canceled`

#### 11.3-B 状态模型：5 态 status + 5 enum phase ✅

**决定**（2026-05-23 讨论）：把 master `versions.stage` 一个字段**拆成两个正交字段**。

> **2026-06-04 Addendum**：ADR 0010 把预处理下沉到 version 级 `train/` 后，phase
> 序列加了 `preprocessing` 一步，现行 cursor 是 **6 enum**：
> `curating → preprocessing → tagging → editing → regularizing → ready`，
> 可跳过集合扩到 `{preprocessing, regularizing}`。详 ADR 0007 Addendum 1 +
> migration `_v11`。本节其余文字保留作历史 5-enum 推导。

**正交模型图**：
```
status (主状态机，5 态)
   preparing / training / completed / failed / canceled
       │
       └─ 仅 status=preparing 时 phase 字段才有意义
              │
              ▼
          phase (cursor，5 enum)
              curating → tagging → editing → regularizing → ready
                                                ↑
                                            仅此 phase 可跳过 (§11.5)
```

**status**：回答"运行态如何"（用于 metric strip、是否能 submit、是否撒谎）。
**phase**：回答"准备到哪一步"，对应当前侧栏 stepper ③–⑦（数据集 ①② 步已归项目级 §6.1，不入 version phase）。

**关键认知**：phase 是 **enum cursor**，**不是** 5 个独立 bool。

→ **推翻设计稿 §4.5 原方案**（5 phases bool 可乱序）。
→ "已完成集合" 如果允许跳过部分 phase 怎么存 / phase 推进规则 / 时间锁 / 强制顺序 → 全部留 **§11.5** 讨论。
→ phase 5 值中 `editing` 是 master 没有的新值（master 把"打标"和"标签编辑"都归在 `tagging`），按侧栏 step ⑤ 独立。

**迁移表（master `versions.stage` → 新 status + phase）**：

| master `versions.stage` | 新 `status` | 新 `phase` | 备注 |
|---|---|---|---|
| `curating` | `preparing` | `curating` | |
| `tagging` | `preparing` | `tagging` | 即使用户已经在编辑，新版打开仍显示 tagging，无法判别可接受 |
| `regularizing` | `preparing` | `regularizing` | |
| `ready` | `preparing` | `ready` | |
| `training` | 看 latest task | `N/A` | task `done → completed` / `failed → failed` / `canceled → canceled` / `pending/running/paused → training` |
| `done` | `completed` | `N/A` | |

**脏数据 fallback**（stage='training' 但无 task —— 几乎不会发生但 SQL 需 default）：
`status=preparing + phase=ready`。理由：让 user 看到"准备就绪可点开始"，自己决定下一步。

**status 转换约束**（task 终态独立映射，"不撒谎"原则）：
- task `done` → version `completed`
- task `failed` → version `failed`
- task `canceled` → version `canceled`
- 三个分支独立，不合并到 `completed`

**phase 命名小尾巴**（不阻塞收敛）：`ready` 字面是"已就绪"，但语义上是"用户在 /train 配 training config"，是否改名为 `configuring` 留待回头改前面章节时一并 review。

#### 11.3-C 多 task 并存时 version.status 聚合 ✅ 已定（见 §11.7 收尾）

**决定**（2026-05-23 讨论，§11.7 收敛后解开）：

- active task ≤ 1（§4.2 队列约束保证）
- `version.status` 计算逻辑：
  - 有 active task（pending / running / paused）→ `training`
  - 无 active 看最近终态 task → `completed / failed / canceled`
  - 从未有 task → `preparing`
- "看 active task" 一句话搞定，不需要复杂聚合逻辑

### 11.4 `project_jobs` 含义展开 ✅

§3 / §4.4 要补一段澄清：

> "`project_jobs` 实质上是 download / preprocess / tag / reg_build 这些 **async job 的统一表**，
>  **不是**训练 task —— 训练 task 走 `tasks` 表。两张表的关系是平行的、不同生命周期的执行单元。"

避免读者把 `project_jobs` 误认为"项目级训练任务"。

### 11.5 phase cursor 推进 / 失效逻辑

§11.3-B 已定 phase 是 enum cursor（不是 5 bool）。本节拆 sub-question：
- **11.5-A** 推进模式（已定 ✅）
- **11.5-B** 每个 phase 的完成 / 失效判定（待）
- **11.5-C** 回退动作（删图 / 删 config / 清正则集 / 删 artifact 触发 cursor 回退）（待）
- **11.5-D** "已完成集合"存储方式（如果有跳过场景，cursor 不能完全派生完成集合）（待）

#### 11.5-A 推进规则 ✅ 已定

**决定**（2026-05-23 讨论）：

**推进模式**：混合
- cursor 单向（只前进）
- 必经 phase：`curating / tagging / editing / ready`
- 可跳过 phase：`regularizing`

**推进入口**：**Header 右侧统一 "上一步 / 下一步" 按钮**（取代每页底部按钮，位置一致更醒目）。**永远可点**（不 disabled —— 对新手友好，错了给提示比沉默点不动好）。

| 按钮 | focus 状态 | 行为 |
|---|---|---|
| 下一步 | `focus < cursor` | focus++（cursor 不动，只是 navigation） |
| 下一步 | `focus == cursor` (必经 phase) | 校验完成条件；通过 → cursor++ + focus++；失败 → toast 提示（如 "训练集为空，请先选择训练集"），cursor 不动 |
| 下一步 | `focus == cursor` (可跳过 phase) | 无校验，cursor++ + focus++（实质就是 skip） |
| 上一步 | `focus > 0` | focus--（cursor 不动） |
| 上一步 | `focus == 0` | disabled |

**侧栏 phase 项**：
- cursor 之前 + cursor 当前 → 可点（focus 跳，cursor 不动）
- **cursor 之后 → disabled**，灰色不可点
- 理由：允许连续跳多步会让"中间每步的提示"无处展示，UI 行为无定义 → 不允许

**UI 双展示（彻底解耦）**：
| 元素 | 表达 |
|---|---|
| ✓ 勾选 | phase 已完成（= cursor 之前的所有 phase） |
| 背景色高亮 | cursor 当前所在 phase |
| 哪个页面在显示 | focus（URL 决定） |

三者独立维度 —— 用户在 cursor 之前 phase 上 navigate 不动 cursor，不动 ✓。

**核心概念分离**：
- **cursor** = 准备进度（最远到哪），存 DB
- **focus** = 用户当前路由（URL），不存
- **completion ✓** = cursor 之前的派生，不存
- "下一步"按钮的 advance 是 cursor 推进的**唯一入口**（除回退动作 §11.5-C）

#### 11.5-B 每个 phase 的完成判定 ✅ 已定

**决定**（2026-05-23 讨论）：

| phase | 校验条件 | 失败提示样例 |
|---|---|---|
| `curating` | `train/ ≥ 1 张图`（不能空，无 warning 阈值） | "训练集为空，请先选择训练图" |
| `tagging` | caption 文件 **100% 覆盖**（每张 train 图都有同名 .txt） | "还有 N 张未生成 caption，请重跑或删除" |
| `editing` | caption 文件 **100% 覆盖**（同 tagging，作为兜底） | 同上。大多数时间从 tagging 来时已 100% → 自动通过；仅当 user 删了 caption 时才触发 |
| `regularizing` | **无相关 reg job 处于 running / pending**（可跳过 = 不要求正则集非空） | "正则任务进行中，请等待完成" |
| `ready` | training config 文件存在 + schema 校验通过 | "请先完成训练配置" / 具体哪个字段非法 |

**ready phase + next 的特殊性**：通过校验后 = `status: preparing → training` + submit task。这是 §11.5（phase）和 §11.3（status）状态机的**接口点**，cursor 不再前进（已是最后），由 status 转换接管。

**regularizing 校验设计优点**（与 confirm dialog 对比）：
- 不强迫用户做"用 / 不用正则集"的二元决定（很多用户不想被打断）
- 只防止"job 在跑时 cursor 跳走"造成的状态错乱
- 正则集为空 vs 非空都允许通过 —— 用户的选择被尊重

**实现位置**：每个 phase 一个 `check_completion(version_id) -> (ok: bool, reason: str)`，统一放 `studio/versions_phase.py` 或类似 module。`advance_cursor()` 调用 check，失败 reason 给前端 toast。

#### 11.5-C 回退动作 ✅ 已定

**决定**（2026-05-23 讨论）：**cursor 不做主动回退**。

**理由**：
- 用户对"回退"的语义是"自由浏览已完成的页面"（即 focus 跳，已在 §11.5-A 覆盖），不是 cursor 退
- 删图 / 删 caption / 删 config 等破坏性动作，**UI 应该在动作前 confirm 提示**告知后果（user 教育）
- 用户在提示后仍然删除是 user 行为责任 → cursor 不主动校正，避免复杂的回退状态机
- 后续 next 时按 §11.5-B 校验会自然 fail 并给出明确提示 → user redo 即可

**5 个 case 行为汇总**：

| # | 触发动作 | cursor 行为 | next 时校验 |
|---|---|---|---|
| 1 | 删 train 图（→ 0 张或部分） | 不动 | curating/tagging/editing next 时 fail（按 §11.5-B 各自的覆盖率校验） |
| 2 | 删 caption (.txt) | 不动 | tagging/editing next 时 fail |
| 3 | 删 training config | 不动 | ready next 时 fail |
| 4 | 清空 / 删正则集图 | 不动 | regularizing next 仍通过（只看 no job running） |
| 5 | 删 LoRA artifact | status 不退 | status=completed 保留 + UI 派生标"产物丢失" |

**有意接受的 UI 假象**：侧栏 ✓ 仍按 cursor 派生（cursor 之前的全 ✓），即使数据已被 user 破坏。代价是 user next 时才发现需要 redo —— 这本来就是 user 自愿的破坏选择。

**配套要求**（写入 §9）：各删除 endpoint 在 UI 层必须有 confirm dialog，文案告知"删除会导致 X phase 数据不一致，需要重做才能继续推进"。

#### 11.5-D 已完成集合 / skipped 表达 ✅ 已定

**决定**（2026-05-23 讨论，作为 §11.5-C "cursor 不退" 的直接派生）：

**不需要额外存"已完成集合"或"skipped 集合"字段**。

理由：
- cursor 单向 + 永不回退 → cursor 之前的所有 phase 都视为"已通过"
- ✓ 直接派生：phase_index < cursor_index 即 ✓
- skipped vs done 在 cursor 维度**等同** —— 都是"已经通过了"

**UI 是否区分 skipped vs done**（推荐派生，不存字段）：
- 检查该 phase 的"实际数据"是否存在
  - regularizing: `reg/` 是否非空
  - tagging: caption 文件覆盖率
- 数据存在 → "✓ 已完成"；不存在 → "✓ (跳过)"
- 派生不 100% 准确不影响功能，仅 UI hint

**模型最终极简形态**：
- `version.status` (enum 5 态)
- `version.phase` (enum 5 值，仅 status=preparing 时有意义)
- 全部已完成 / skipped / 失效 信息**都派生**，不存

→ 推翻设计稿 §4.5 全表（5 bool phases + 回退表）；§4.5 的 5 bool 模型完全废弃。

### 11.6 数据集 UI 表达 ✅ 已定

**决定**（2026-05-23 讨论）：

**项目级数据集**：
- 文案：**"数据集"**（不是"训练集"）
- 显示位置：
  - 侧栏顶部（① 下载 / ② 预处理，括号显示数字）
  - 项目详情页顶部（数字 + [管理] 链接，最简）
- 不弹"数据集是项目级"教学 dialog（保持简洁，用 layout 表达从属关系）

**Version 级数据集**：
- 文案：仍叫"数据集"（同名 OK，上下文区分 —— 在 version 详情页就是 version 的）
- 显示位置：项目详情页 [详情] tab 内（grid 布局，§11.8-C）
- 内容（按 §11.8-C grid 各格）：
  - **文件夹（repeat_N_concept/）列表 + 各自文件数 + 总计**
  - **分辨率分布**（参考放大页右侧统计组件）
  - **长宽比分布**（参考裁剪页右侧分桶统计）
  - **tag 分布**（参考标签编辑页统计）
  - **正则集数量**
- **未完成项 placeholder**：每格各自的 empty state，参考关联 phase 页面已有的 empty state 风格（不统一灰底文字）

**教育机制**（隐式而非显式）：
- 项目级数据集放最上面 + 标注"项目级"
- version 详情数据集在 [详情] tab 内
- 文案保持简洁，靠 layout 层级表达从属

### 11.7 再训练数据处理 ✅ 已定

**决定**（2026-05-23 讨论）：同 version 多 task（B 模型）+ task 内**仅冻结 config**（G 简化版）。D（硬链接）方案放弃。

**冻结内容**：
- ✅ `config.yaml` —— task 创建时复制到 task snapshot 目录
- ❌ caption / 图 / 正则集 —— 都不冻结

放弃 D（硬链接全冻结）的理由：
- 跨 OS 导出失效（用户 Windows 训完 → Linux 算力平台导入 用例下崩）
- preprocess→train 既有复制链是独立问题，扯进来 scope 失控（→ §12.1）

**心智分离（核心 UX 设计）**：
- task 详情有**独立的"训练 config 记录"页面**（只读 yaml / 复用 version config 组件 readonly 模式）
- **不点 task 跳 version config 编辑页**（避免 user 误以为"task 看到的就是当时的 config"）
- 用页面分离表达：**config 是历史快照，caption / 图是 version 当前状态**（每次训练用当前数据，user 自己理解）

**重训入口**：task config 记录页面上一个按钮（命名暂"用此 config 重训"）：
- 点击 → 跳 version config 编辑页 + 自动加载这份冻结 config
- user 可编辑（也可不改）→ 正常点开始训练 → 创建新 task
- 不需要单独的"再训练"按钮（你的 Q3：正常走训练流程就是新 task）

**与设计稿 §6.11 关系**：
- "重训不创建新 Version，保留同 version 多 task 历史" → **保留** ✅
- §6.11 多个 CTA（"再训一次 / 复用配置重试 / 用更小 batch 重试"）→ 简化为一个"用此 config 重训"按钮

**实现要点**：
- task 创建（supervisor enqueue）时：`tasks/{task_id}/snapshot/config.yaml` ← cp 自 version 当前 config
- task 详情新增路由 `/projects/:pid/v/:vid/task/:tid/config`（只读）
- 配套要求（→ §9）：
  - backend：task snapshot 目录创建 + config 复制逻辑
  - frontend：task config 只读页面组件 + "用此 config 重训"按钮的路由跳转 + 自动 prefill

**实验矩阵（你的 Q4 答）**：不专门支持，user 想要 matrix 自己手动 fork version（§6.11 PP10.1 已有"从老 version 全量副本"功能）。

### 11.8 UI 重构 layout v3 ✅ 已定

**完整 layout 决定**（2026-05-23 讨论）。分 5 块：

#### A. 侧栏 layout（进入项目后）

- 顶级 tab（**项目 / 队列 / 工具 / 设置**）不收缩，始终可见
- 进入项目后，在"项目" tab 下**展开**项目内容（表示从属）
- 队列 / 测试等顶级 tab 仍可见在下方
- 视觉规则见 §11.2（完成 phase 数字变绿 / 当前页背景高亮）

```
┌─────────────────────────────┐
│  Anima · 0.10.0             │
├─────────────────────────────┤
│ ▼ 项目                       │ ← 顶级 tab，进入项目后展开
│    项目: Cosmic Kaguya       │
│                              │
│    概览           ← 当前页    │ ← 背景高亮表示 active 路由
│    ① 下载  (80)              │ ← 项目级 phase，括号为文件数
│    ② 预处理 (50)             │
│                              │
│    Version (list 全显示):    │
│     ● test                   │ ← 当前 active version
│       baseline               │
│       highlr                 │
│                              │
│    ③ 筛选                    │ ← 完成时数字变绿
│    ④ 打标                    │
│    ⑤ 编辑                    │ ← cursor，背景高亮
│    ⑥ 正则集 (可跳过)         │
│    ⑦ 训练                    │
│                              │
│   队列                        │ ← 顶级 tab，仍可见
│   测试                        │
│   工具 / 设置                │
└─────────────────────────────┘
```

#### B. Phase 页面 header

右上角推进按钮（带编号 + phase 名 + 箭头方向），替代通用"上一步 / 下一步"文案：

```
┌──────────────────────────────────────────────────┐
│ 标签编辑           [← ④ 打标]    [⑥ 正则集 →]   │
├──────────────────────────────────────────────────┤
│                                                  │
│   ... phase 页面内容 ...                          │
│                                                  │
└──────────────────────────────────────────────────┘
```

- 按钮永远可点（§11.5-A 哲学，对新手友好）
- 不可跳过 phase：next 校验失败 → toast 提示
- 可跳过 phase（regularizing）：无校验直接推进
- 最早 / 最末 phase：方向按钮 disabled

#### C. 项目详情页（侧栏点"概览"进入）

**两层结构**：上半 = 项目级信息（Project 字段全展示，除 pipeline —— Project 本来就无 pipeline §6.3），下半 = version scope（dropdown 选 version + 3 个 tab）。

```
┌─────────────────────────────────────────────────────────┐
│ 【项目级 — 不随 dropdown 变】                            │
│ keta                                                    │ ← project.title
│ slug: keta · created 2026-05-01                         │ ← slug + 时间
│ {project.note}                                          │ ← 项目 note（无 note 不显示）
├─────────────────────────────────────────────────────────┤
│ 数据集: ① 264 张 · ② 247 张        [管理]                │ ← 项目级数据集（实时扫）
│ 2 个版本                                                │ ← 版本数
├─────────────────────────────────────────────────────────┤
│ 【version 选择 — 独立于 sidebar active】                 │
│ Version: [v1.1 ▼]                  [● 已完成]            │ ← dropdown + 选中 version status
├─────────────────────────────────────────────────────────┤
│ Tabs: [详情] [Tasks] [Output]                           │ ← 全部 version scope
├─────────────────────────────────────────────────────────┤
│ [详情 tab — grid 布局，非上下排]                          │
│                                                         │
│  ┌──────────────────┬──────────────────┐                │
│  │ 文件夹 / repeat  │ tag 分布         │                │
│  │  repeat_5_… 30   │  [chart / list]  │                │
│  │  repeat_2_… 17   │                  │                │
│  │  总计 47         │                  │                │
│  ├──────────────────┼──────────────────┤                │
│  │ 分辨率分布       │ 长宽比分布       │                │
│  │  [chart]         │  [chart]         │                │
│  ├──────────────────┴──────────────────┤                │
│  │ 正则集：0 张（未生成）                │                │
│  └──────────────────────────────────────┘               │
│                                                         │
│  未完成的格子 → 各自 empty state（参考关联 phase 页面）  │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ [Tasks tab]  = 列**本 version** 的 task（不是项目下所有）│
│               表格风格同 /queue；点 task → task 详情页   │
├─────────────────────────────────────────────────────────┤
│ [Output tab] = **本 version** 的 LoRA artifact           │
│               output_lora_path + step/epoch ckpts        │
└─────────────────────────────────────────────────────────┘
```

**Dropdown 独立性**（关键决策）：
- 概览页有 local-only `selectedVersionId`，初值 = `project.active_version_id`
- dropdown 改它**只影响概览页 3 个 tab 的数据源**，**不动 sidebar active 也不发 API**
- sidebar Version list 仍然走 active version（点 sidebar version = activate）
- 用途：用户想快速对比两个 version 的数据集/task/output，不必反复 activate
- 概览页的"项目级"上半（title/slug/数据集/版本数）**不随 dropdown 变化** —— 它本来就是项目级

#### D. Task 详情页

5 个 tab：**[概览] [日志] [Monitor] [Output] [关联配置]**（最后一个新增，§11.7 决定）。

```
┌─────────────────────────────────────────────────────────┐
│ Task #45                                                │
│ Tabs: [概览] [日志] [Monitor] [Output] [关联配置]       │
├─────────────────────────────────────────────────────────┤
│ [关联配置 tab — 新增]                                    │
│                                                         │
│   只读 yaml / 复用 version config 组件 readonly 模式     │
│   （不点 task 跳 version config 编辑页 —— 心智分离）    │
│                                                         │
│   ┌────────────────────────────────┐                    │
│   │  config 只读展示...             │                    │
│   │                                │                    │
│   └────────────────────────────────┘                    │
│                                                         │
│   [套用此配置]  ← 点击跳 ⑦ 训练 phase 配置页 + 预填     │
│                  → user 编辑后正常点开始训练            │
│                  → 创建新 task（同 version 多 task）    │
└─────────────────────────────────────────────────────────┘
```

#### E. 项目列表卡片（首页 /projects）

```
┌────────────────────────────────┐
│ Cosmic Kaguya     [preparing]  │ ← 右上角 = active version 的 status
│ test                           │ ← active version 名（直接，无前缀文案）
└────────────────────────────────┘
```

- 去掉 stage badge（§1 痛点 = master 撒谎那个）
- 右上角状态 = **当前 active version** 的 status（不是项目 stage —— 项目无 stage）
- 不显示产物 / 不显示时间 / 不写"最近 active 版本"前缀
- 其它布局保持当前 UI（不重新设计卡片整体结构）

### 11.9 讨论顺序建议

低耦合先聊，UI 留最后：

1. ~~**§11.4 + §11.1**（已定 ✅，仅落笔）~~ ✅
2. ~~**§11.3** version paused 必要性 + 6 态重写~~ ✅ A/B/C 全定
3. ~~**§11.5** phases 推进 / 失效逻辑~~ ✅ A/B/C/D 全定
4. ~~**§11.7** 再训练 vs Version Up 磁盘代价~~ ✅
5. ~~**§11.6 + §11.8** UI 联动~~ ✅
6. ~~**§11.2** 侧栏 checklist~~ ✅

**§11 全部收敛 ✅** —— 下一步：回头改 §1–§10 反映 §11 决议，再转 ADR-0007。

---

## 12. Follow-up Issues（不在本次 scope 的独立改造）

§11 讨论中浮出但**不属于本次重构 scope** 的事项，独立 issue 跟踪。

### 12.1 `preprocess → train` 数据流改造

**现状**：
- `preprocess/` 项目级；`train/` version 级
- curate 选图 = `preprocess → train` 物理复制（占 2x 磁盘）
- user 改 preprocess 不反映到已选过的 train

**讨论中浮出的方向**（未定）：
- 改硬链接（省磁盘，但跨 OS export 难）
- preprocess 替代 train（version 只存 "选了哪些" 的 manifest）
- 保持现状（接受复制代价）

**触发场景**：§11.7 讨论 D 硬链接 task snapshot 方案时浮出，与 task 数据冻结无直接耦合但同涉数据流模型。

**Park 处理**：独立 issue / 单独 ADR，**不在 ADR-0007 范围**。

### 12.2 删除 endpoint 的 confirm dialog 文案审计

§11.5-C 决定 cursor 不主动回退后，前端各删除 endpoint（删图 / 删 caption / 删 config）的 confirm dialog 必须告知"删除会影响 X phase 数据一致性"。需要审计当前哪些 endpoint 有 confirm、文案是否合适。

**Park 处理**：作为 §9 frontend 改造清单的子项，留待落笔时一并处理。
