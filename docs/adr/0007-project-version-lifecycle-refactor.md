# 0007 — Project / Version / Task 生命周期重构

**状态**：Accepted（已落地：`_v8`/`_v9` migration + VersionStatus/Phase + lifecycle 路由）
**日期**：2026-05-23
**决策者**：@WalkingMeatAxolotl

> **维护约定**：本 ADR 承袭 `docs/design/project-version-task-lifecycle.md` 两轮讨论稿。
> 一轮决议在设计稿 §1–§10；二轮 review 翻盘 9 处，收敛在 §11；§12 park 的 follow-up 不在本 ADR scope。
> 已 Accept 的初版决策不修改、不删除；后续 audit / 翻盘 → 在末尾「## 增量更新」追加。

## 背景

master 当前 `Project.stage`（9 态）/ `Version.stage`（6 态）混合了"运行态"和"准备阶段"两个正交概念，结构性问题：

- **Project 卡片撒谎**：训练失败 / 取消 / 暂停后 `projects.stage` 不被推进，badge 永远显示 `training` + 动画 dot
- **死状态**：`projects.stage` 的 `curating / configured / regularizing` 从无代码推进，3/9 颜色映射用户永远看不到
- **跳级**：上传 train 图直接跳到 `tagging`，跳过 5 个 stage —— 说明 stage 本质是"做过哪些步骤的集合"，但 UI 把它当线性进度展示
- **无 source of truth**：`advance_stage()` 是无条件 setter，规则散在 13 个调用点；Project / Version / Task 三套状态机互不对账
- **数据模型错位**：Project 是聚合体（1:N → Version），不应有"过程状态"

详细痛点 + 三方（PM / User / Designer）两轮辩论过程见 `docs/design/project-version-task-lifecycle.md`，本 ADR 承袭其逻辑模型，重点固化决策和实施路径。

## 候选方案

下列每条都列了否决的方案以记录否决理由。完整辩论见设计稿 §11.x。

### Project 数据模型

- **A 加 status 字段**：项目是聚合体（1:N → Version），不该有过程状态。否决
- **B 保留 stage**：不解决根因（stage 混合状态）。否决
- **C 极简，无 stage 无 process 字段（采纳）**：所有"过程"派生自下属 versions 聚合 + 项目级数据集实时扫文件系统

### Version 状态机粒度

- **A 单 status 字段 5 态（draft / training / completed / failed / canceled）**：失去"当前在准备哪一步"信息。否决
- **B 保留 master 6 stage + training 子态按 task 推**：stage 概念仍混合两层。否决
- **C status + phase 双字段正交（采纳）**：
  - `status` (5 enum) 答"运行态如何"
  - `phase` (5 enum, 仅 preparing 时有意义) 答"准备到哪一步"

### Version 级 paused 状态

- **A 加 version-level paused**：task pause（ADR 0006）已 cover 短中期场景；"版本长期搁置"是伪需求。否决
- **B 去掉（采纳）**：task=paused 时 UI 派生 "训练中 · 已暂停 Nmin"

### Phase 模型

- **A 5 个独立 bool 可乱序**（设计稿一轮原方案）：违背用户线性心智（筛选→打标→编辑→正则→训练有强暗示）。否决
- **B enum cursor，5 值（采纳）**：`curating → tagging → editing → regularizing → ready`，cursor 单向，已完成集合直接派生

### Phase 推进模式

- **A 强制单向**：跳过正则集 / 回头改 caption 需要"phase 回退"心智重。否决
- **B 完全自由（点哪到哪）**：cursor 失去"客观进度"语义，✓ 变成"用户点过没"。否决
- **C 混合（采纳）**：cursor 单向；`regularizing` 可跳过；header"下一步"按钮**永远可点**（不 disabled，对新手友好）+ 校验失败给 toast 提示；**cursor 不主动回退**（user 自删数据后 ✓ 显示假象，等 next 校验时才发现）

### 再训练数据冻结

- **A 强制 Version Up（每次重训 = 新 version + 全套复制）**：磁盘双倍。否决
- **B 同 version 多 task 不冻结**（master 现状）：旧 task 看到 current config / 当前 train 集 → forensics 失真。否决
- **D 同 version 多 task + 全数据冻结（caption 复制 + 图硬链接）**：硬链接跨 OS export 失效（Windows 训完 → Linux 算力平台导入崩）；`preprocess→train` 既有复制链是独立问题，扯进来 scope 失控。否决
- **G 同 version 多 task + 仅冻结 config（采纳）**：
  - task 创建时 `tasks/{task_id}/snapshot/config.yaml` ← cp 自 version 当前 config
  - caption / 图 / 正则集 **不冻结**
  - **心智分离 UI**：task 详情独立 "关联配置" tab（不点 task 跳 version config 编辑页），有 "套用此配置" 按钮 → 跳 ⑦ 训练 phase + 预填 + 标准流程

### 数据集归属

- **A version 级（每 version 独立 train 集）**：不符 user 心智（同角色不同 version 用同批原图是 90% 用法），磁盘浪费。否决
- **B 项目级 + 实时扫（采纳）**：`download/` + `preprocess/` 留在项目目录；项目级 phase 不存 DB 字段，UI 实时 `os.listdir` 派生（§6.10）

### `active_version_id` 字段

- **A 删除**（设计稿一轮原方案）：连带 10+ 处代码改动，UX 退化。否决
- **B 保留，语义为 "最后打开的 version"（采纳）**：和 master 当前行为一致；不是 PM/UX 意义上的 "pinned" 或 "代表 version"

### `project_jobs` 表

- **A 拆表（download / preprocess / tag / reg_build 各一张）**：schema 洁癖，多 supervisor / 事件 / migration 代码。否决
- **B 单表 + kind 区分 scope（采纳）**：`version_id` 可空，project 级 job 为 NULL，version 级为 NOT NULL

## 决策

### 最终数据模型

**Project**（极简）：
```
id, title, slug, note, active_version_id, created_at, updated_at
```
- 删 `stage` 字段
- 保留 `active_version_id`（"最后打开"语义）
- 无任何 process / phase 字段；数据集状态由 UI 实时扫派生

**Version**（双字段正交）：
```
id, project_id, label, note, trigger_word, created_at, updated_at,
status: preparing | training | completed | failed | canceled,
phase:  curating | tagging | editing | regularizing | ready  (仅 status=preparing 时有意义),
output_lora_path, last_task_id, last_failure_reason
```
- 删 `stage` 字段
- 已完成集合 / skipped 集合 / 回退状态全部**派生**，不存

**Task**（保留现状 + 新增 snapshot）：
- task 状态机不动（ADR 0006 已定）
- 新增：task 创建时 `tasks/{task_id}/snapshot/config.yaml` ← cp version 当前 config
- 新增 endpoint：GET `/api/tasks/{tid}/snapshot/config`（只读）

**project_jobs**（不动）：单表 + kind 区分

### 状态机定义

**Version status 转换**（由 supervisor 推，UI 永不直写）：
- task `done` → `completed`
- task `failed` → `failed`
- task `canceled` → `canceled`
- 三个分支独立（不撒谎）

**Version status 派生**（无 active task 时）：
- 有 active task（pending / running / paused）→ `training`
- 无 active 看最近终态 task → `completed / failed / canceled`
- 从未有 task → `preparing`

**Phase 推进规则**（详 §11.5-A）：
- cursor 单向（只前进）
- 推进入口：phase 页面 header 右侧 "← phase 名 | phase 名 →" 按钮，永远可点
- 必经 phase (`curating / tagging / editing / ready`) 校验失败 → toast 提示
- 可跳过 phase (`regularizing`) 校验 = 无 reg job running，无 confirm dialog
- 侧栏 phase 项：cursor 之前 + 当前可点（focus 跳）；cursor 之后 disabled
- cursor 不主动回退（user 删数据后 ✓ 显示假象，下次 next 才发现）

### 各 phase 完成判定（§11.5-B）

| phase | 校验 |
|---|---|
| `curating` | `train/ ≥ 1 张图` |
| `tagging` | caption 100% 覆盖（每张 train 图都有 .txt） |
| `editing` | 同 tagging（兜底） |
| `regularizing` | 无 reg job running/pending（可跳过） |
| `ready` | config 文件存在 + schema 校验通过；next 通过 = `status: preparing → training` + submit task |

### UI Layout v3（§11.8）

详 `design §11.8` 完整 ASCII：
- **A 侧栏**：项目/队列/工具/设置顶级 tab 不收缩；进入项目后在"项目"下展开
- **B Phase 页面 header**：`[← ④ 打标]  [⑥ 正则集 →]`（带编号 + phase 名 + 箭头）
- **C 项目详情页**（侧栏点"概览"进入）：三 tab `[详情] [Tasks] [Output]`；详情 tab grid 布局；右上角 = 当前 version 的 status
- **D Task 详情页**：5 tab `[概览] [日志] [Monitor] [Output] [关联配置]`；最后一个新增，含 "套用此配置" 按钮
- **E 项目列表卡片**：去 stage badge；右上角 = active version status；不显产物 / 不显时间

## 理由

否决其他方案的核心理由汇总：

1. **Project 加 status** → 项目是聚合体，无过程状态语义
2. **Version 6 态合并** → 失去 "当前在哪一步" 信息
3. **5 bool phases** → 违背用户线性心智，✓ 表达不清
4. **Version paused** → task pause 已 cover (ADR 0006)
5. **Phase 强制单向** → 跳过场景 UX 差
6. **Phase 完全自由** → cursor 失去客观进度语义
7. **Task 全数据冻结 / 硬链接** → 跨 OS export 失效
8. **强制 Version Up 重训** → 磁盘双倍
9. **数据集 version 级** → 不符 user 心智，浪费磁盘
10. **删 `active_version_id`** → UX 退化无收益

## 后果

**好处**：
- `status` 永远不撒谎（task 终态独立映射，supervisor 是唯一推进者）
- 模型最简（status + phase + active_task 三个变量描述完整状态机，已完成 / skipped / 失效全派生）
- forensics 95% 价值（config 100% 准确；caption / 图允许失真）
- 跨 OS export 完全 OK（不依赖硬链接 / COW）
- phase header 按钮 UX 对新手友好（永远可点 + 校验提示比 disabled 沉默好）
- migration 可渐进（v8 add-only → 双写 → frontend 切 → v9 destructive）

**约束 / 新债**：
- frontend 改动量大（layout v3 五块全改 → 3 个 frontend PR）
- `_v9` migration 打破 `studio/migrations/__init__.py` 既有约定（"不允许向后改写已有列"）—— 本次显式例外
- §11.5-C 接受 "UI 假象"：cursor 不回退 → user 删数据后 ✓ 显示假象，等 next 校验才发现（设计有意接受）
- task snapshot 配套要求：各删除 endpoint 加 confirm dialog 文案（→ §12.2）

**Park 的 follow-up（不在本 ADR scope）**：
- §12.1 `preprocess → train` 数据流复制链改造（硬链接 / project 级 manifest 取代 train/）— 独立 ADR
- §12.2 删除 endpoint confirm dialog 文案审计 — §9 frontend 改造子项

## 实施计划

7 个 PR，每个独立合 dev 不破坏现状。详细 commit 拆分见设计稿 § 实施 plan 段，PR 合入顺序：

| PR | 范围 | 阻塞关系 |
|---|---|---|
| 1 | ADR + 设计稿 banner + README 索引 | 无 |
| 2 | `_v8` schema (add-only) + models 新字段 readonly | 1 |
| 3 | backend 业务逻辑：phase 校验 / supervisor 推 status / 一致性 / task snapshot | 2 |
| 4 | frontend layout v3 第一波：侧栏 + phase header + 项目详情页三 tab | 3 |
| 5 | frontend task 详情 [关联配置] tab + "套用此配置" 流程 | 3, 4 |
| 6 | frontend 项目列表卡片 | 3 |
| 7 | 清理 + `_v9` destructive migration (DROP `stage` 列) | 2, 3, 4, 5, 6 |

## Addendum 1 — `preprocessing` phase（2026-06-04）

ADR 0010 把预处理从项目 scope 下沉到 version scope（`versions/{label}/train/`
上原地做 upscale / crop）。phase 模型相应扩展一个 step：

### 修改

- **Phase 增 `preprocessing`**，插在 `curating` 之后、`tagging` 之前。完整顺序：
  `curating → preprocessing → tagging → editing → regularizing → ready`
- **可跳过集合扩到 `{preprocessing, regularizing}`**：用户不需要 upscale 也能继续
- **完成判定**：`preprocessing` 没有强制校验（同 `regularizing`）—— 跳过即下一步
- **DB migration `_v11`**：现存 `phase ∈ {tagging, editing, regularizing, ready}`
  且 train 集非空的 version 一次性回填 `preprocessing` 状态，避免老 version 在
  UI 上突然"退步"到 preprocessing 步骤

### 理由

- 预处理 = "改 train 集像素"，跟 "改 train 集图集合"（curating）和 "标 caption"
  （tagging）是三件独立动作，混进 curating 会让校验混乱（curating 校验"≥1 张图"
  vs preprocessing 校验"全部 upscale"是两套半成品状态）
- 跳过性匹配实际：很多 LoRA workflow 不 upscale 直接打标

## 参考

- `docs/design/project-version-task-lifecycle.md` —— 两轮 review 完整讨论稿
- 相关 ADR：[0006](0006-queue-pause-resume.md)（task pause/resume）/ [0004](0004-preprocess-manifest.md)（preprocess manifest）
- 实施 PR：合入时回填编号
