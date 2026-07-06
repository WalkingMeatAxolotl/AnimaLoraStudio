# 队列调研与优化 — 0.17.0 规划

> 规划文档，只整理**要解决哪些问题、它们的依赖关系、推进顺序**。
> 不涉及实现细节（UI 布局、字段名、函数签名留给各 PR 描述）。
> 现状事实带 `文件:行号` 便于核对；决策记录见 §5，未定项见 §6。

## 0. 目的与非目的

**目的**：围绕队列做一轮"调研 + 呈现优化"。核心洞察是——**后端其实已经有一套统一的
任务队列，只是前端把它藏起来了**。本版主要工作是"把已有的如实呈现 + 补分区/过滤 +
补一个定时触发机制"，而**不是重写调度**。

**非目的**：
- 不重写 supervisor 调度模型（双槽 + daemon 的三车道保留）。
- 不把 `project_jobs`（数据作业）真正合并进 `tasks` 队列（重写级，留 0.18，见 §6）。
- 不把 generate 搬进 TRAIN 槽（daemon 的模型复用是出图体验命根子，见 §1.3）。

---

## 1. 现状：一个 supervisor、两张表、三条车道

系统**不是单一队列**，而是同一个 `Supervisor` 单线程调度下的三条执行车道
（`studio/supervisor/core.py:360-380`）：

| 车道 | 表 | 类型 | 车道内 | ID 序列 |
|------|----|------|--------|---------|
| TRAIN 槽 | `tasks` | `train` / `reg_ai` | 串行（互斥，一次一个） | `tasks.id` |
| DATA 槽 | `project_jobs` | `download` / `preprocess` / `tag` / `reg_build` / `eval_*` | 串行（一次一个） | `project_jobs.id`（**独立序列**） |
| daemon | `tasks` | `generate` | 串行（一次一个） | `tasks.id`（与 train 共用） |

### 1.1 串行 / 并行

- **车道内串行**：每条车道一次只跑一个。
- **车道间并行**：三条车道各能同时跑一个 → 理论最多 3 个同时（1 训练 + 1 数据作业 + 1 出图）。
- **GPU 仲裁给并行打折**（`core.py:443-494`）：
  - `download` 纯 IO，**永远与训练并行**。
  - GPU-bound 数据作业（tag / reg_build / eval / preprocess）训练时**默认推迟**，除非
    `secrets.queue.allow_gpu_during_train`。
  - daemon 出图在训练派活前被要求**卸载让位**。
- 实际最常见的真并行是**训练 + download**；打标 / eval 一般等训练完。

### 1.2 两张表的关键差异（决定"不合并"）

| | `tasks` | `project_jobs` |
|---|---|---|
| 状态机 | pending/running/done/failed/canceled/**paused** | pending/running/done/failed/canceled（**无 paused**） |
| 能力 | reorder / hold / pause / resume / retry | 仅硬编码 `dispatch_order`，**无用户级 priority/reorder/pause** |
| ID | `tasks.id` | `project_jobs.id`（**独立自增**，与 tasks 撞号） |
| 事件 | `task_state_changed` / `monitor_progress` | `job_state_changed` / `job_log_appended` + typed 事件 |
| DAO | `studio/infrastructure/db.py` | `studio/services/projects/jobs.py` |

两张表混进一个列表会**撞 ID**（`tasks#5` ≠ `job#5`，React key 冲突 + 路由走错），这是
"不合并"的硬约束。

### 1.3 generate 为什么走 daemon 而非 TRAIN 槽

- TRAIN 槽每 task spawn 新子进程、从零加载模型；训练跑几小时，加载成本可忽略。
- generate 是短任务，若每批图都新起进程 = 每次重载 base+VAE+text encoder（几十秒），迭代
  出图体验崩坏。
- 所以 generate 用**常驻 daemon**：模型只加载一次、**跨 task 复用**（`_dispatch_generate`
  `core.py:412-432` → `_submit_to_daemon`）。daemon 空闲时领下一个 pending generate。
- "daemon 串行" = daemon 一次只接一个 generate（要求 `STATE_IDLE`），所以**多个 pending
  generate 在后端天然排队**。
- 代价：generate **无 pause**（模型不在 slot 子进程里）、输出在**内存 cache** 非磁盘、
  训练要 GPU 时 daemon 先卸载。

### 1.4 generate："多次生图" ≠ "一个 task"

- 每次 `POST /api/generate` = **一个新 task**（新 `tasks.id`）。点 3 次生成 = 3 个 generate
  task 排队。
- 但**一个 task 内可出多图**：single 的 `count=N`、XY 的 N×M 格，都是同一 task 出多图。
- daemon 是常驻进程，task 是投给它的工作单元；一个 daemon 服务很多 task，共享模型但不是
  同一 task。→ 直接定义了 P-I 的形态（§2）。

### 1.5 前端现状（缺口）

- 队列页 `Queue.tsx` 全部任务**扁平混排**（id 倒序，`Queue.tsx:186`），无"当前/等待/历史"
  分区。
- **无任何过滤/搜索/分页/虚拟滚动**；后端 `listQueue(status)` 已支持状态过滤但前端从不传。
- Task API **不暴露 `task_type`**，前端靠 `config_name` 字符串猜 `inferKind`（`Queue.tsx:40`），
  且不认 eval / generate。
- 队列默认**过滤掉 generate / reg_ai**（`lifecycle.py:90`），它们在 UI 里不可见。
- `QueueDetail` 固定 6 tab、**不按类型分支**；非训练类型进来会显示训练专用的空 tab。
- generate **没有详情页**（走独立端点 + 被过滤），且前端单 `currentTask` 模型 + 被其它 GPU
  任务阻塞，**一次只能提交一个**。

---

## 2. 问题清单（4 个方向 → 9 个问题单元）

| ID | 问题 | 来自 | 性质 | 后端改动 |
|----|------|------|------|----------|
| **P-D** | Task API 暴露 `task_type` 字段（消灭 `inferKind` 字符串猜测） | 2/3 | 基础设施 | 极小（加字段） |
| **P-A** | 队列页**分区**：当前(running) / 等待(pending) / 历史(terminal) | 1 | 前端为主 | 无 |
| **P-C** | **过滤/搜索**：类型 + 状态 + 关键词（后端 status 过滤已就绪） | 2 | 前端为主 | 无 |
| **P-E** | **性能**：历史无限增长，需分页/虚拟滚动 | 2 | 前端为主 | 可能加 limit/offset |
| **P-F** | 解除 generate/reg_ai 的 UI 过滤 → **全部显示 + 类型过滤** | 3 | 前端+API | 小 |
| **P-H** | 按类型差异化 detail（summary+log 通用、隐无关 tab、结果靠跳转） | 3 | 前端为主 | 无 |
| **P-I** | generate **多任务排队 UI**（去掉单 currentTask 阻塞） | 4 | 前端为主 | 无 |
| **P-B** | **计划任务（定时触发）**：绝对时间 + 延迟两种 | 1 | 前端+后端 | 中（加列+判时+UI） |
| **P-G** | project_jobs **可见性**：不合并，加独立只读区块 | 3 延伸 | 前端为主 | 无（复用 jobs API） |

---

## 3. 依赖关系

```
P-D (暴露 task_type)  ──┬──> P-C (类型过滤)
   [一切的地基]         ├──> P-F (带类型显示 generate/reg_ai)
                        └──> P-H (类型差异化 detail)

P-A (分区) ────────────────> P-C (过滤)   [同改 Queue.tsx，一起做最顺]

P-F (解除过滤) + P-H ──────> P-I (generate 多任务排队)
   [generate 要在队列可见且有 detail，多任务排队才有落点]

P-B (计划任务) ── 独立后端机制，UI 落队列页 → 与 P-A 共页面，逻辑解耦，可并行开发

P-G (数据作业只读区) ── 独立只读面板，复用现有 jobs API，与主列表解耦，可任意时机插入
```

**关键**：P-D 是地基，P-C/P-F/P-H 全压在它上面。project_jobs 真正合并（非本版）见 §6。

---

## 4. 推进顺序

| 批次 | 内容 | 为什么这个顺序 |
|------|------|----------------|
| **PR-1 地基** | P-D 暴露 `task_type` | 解锁后面所有类型相关功能，独立可测 |
| **PR-2 队列页核心** | P-A 分区 + P-C 过滤/搜索 + P-E 性能 | 三者同改 Queue.tsx，一起做避免反复重构；纯前端+复用后端 |
| **PR-3 统一显示** | P-F 解除过滤 + P-H(summary+log+隐 tab+跳转) | 必须一起：解除过滤前先让 detail 不露馅 |
| **PR-4 出图排队** | P-I generate 连续提交 | 站在 PR-3 之上，generate 已可见可点；后端零改 |
| **PR-5 计划任务** | P-B 定时触发（绝对时间+延迟） | 新机制，可与 PR-2~4 并行开发，最后合 |
| **可插入** | P-G 数据作业只读区块 | 独立面板，复用 jobs API，任意时机插入 |
| **0.18 候选** | project_jobs 真正合并（重写级） | 范围大、契约多，单独立项，不进 0.17 |

---

## 5. 决策记录（已定）

- **D1 不重写调度**：保留 TRAIN 槽 / DATA 槽 / daemon 三车道；0.17 是"呈现优化 + 定时"，非"重写队列"。
- **D2 generate 保留 daemon 车道**：模型复用是出图体验命根子，不搬 TRAIN 槽（§1.3）。
- **D3 reg_ai / generate 在主队列全部显示 + 类型过滤**（P-F）：它们已在 `tasks` 表，只是被
  `lifecycle.py:90` 滤掉，解除即可。
- **D4 project_jobs 不合并**（P-G）：两表 ID 撞号 + 两套事件 + job 无 reorder/pause，混列等于
  重写。本版只给一个**独立只读区块/tab**，复用现有 jobs API 纯展示（状态/进度），不参与
  分区 / reorder / 过滤。真正合并留 0.18。
- **D5 task detail 走轻方案**（P-H）：统一 QueueDetail 的 **Overview(summary) + Log tab 对所有
  类型通用**（时间/状态/来源，同 train）；非 train 类型**隐藏训练专用的 monitor/eval/snapshot
  空 tab**；详细结果**靠跳转**——加「查看结果 →」深链跳到该类型原生页（generate→出图历史、
  reg_ai→reg 页、tag→打标页、eval→EvalMetricsPanel）。**附带红利**：generate 经 P-F 解除后
  自动进 QueueDetail，白得 summary+log，结果靠跳转，**无需单独造 detail 页**。
- **D6 generate 多任务排队 = 前端去阻塞**（P-I）：后端已能排队（§1.4），前端去掉单
  `currentTask` 阻塞、允许连续提交即可，**后端零改**。
- **D7 计划任务 = 定时触发，绝对时间 + 延迟两种**（P-B）：加 `scheduled_at`，dispatcher 到点
  才把 task 转 pending；UI 支持「凌晨 3:00 开始」的绝对时间点和「N 小时后」的相对延迟。
  不做依赖触发（A 完成跑 B）——现有两条自动链（训练→eval、sample→指标）是硬编码事件驱动，
  本版不引入通用 DAG。
- **D8 计划任务实现细节**（PR-5 拍板，原 Q1）：独立 `scheduled` 状态（非 pending 子集，
  _v15 加 `scheduled_at` 列）；提升挂在 supervisor 既有 1s tick 里（`_tick` 开头
  `promote_due_scheduled`），不起新线程；设定入口**仅提交时**（训练页入队弹层），已有
  pending 不能转定时——想定时就取消重排；scheduled 行给「立即开始」（`start_now` 端点）
  和「取消」；UI 入口本版只做训练页（后端机制类型无关）；提升后 `scheduled_at` 保留作记录。

---

## 6. Open questions（待细化 / 未定）

- ~~**Q1（P-B 入口）**：定时的设定入口——是 enqueue 时一次性设定，还是也允许对已有 pending
  任务追加/修改排程？dispatcher 判时精度（1s poll 够用）？~~ **已决 → D8**（仅 enqueue 时
  设定；1s tick 复用 supervisor 主循环）。
- **Q2（P-A 历史边界）**：历史区默认展示多少 / 保留多久？是否需要"清理历史"动作？（关联 P-E 分页）
- **Q3（P-G 面板位置）**：数据作业只读区放队列页的 tab、底部折叠面板、还是侧栏？是否按 project
  过滤？
- **Q4（0.18 project_jobs 合并）**：真正合并需弥合两表 schema + 统一事件契约 + 给 job 加
  priority/reorder + 显式资源模型（独占 GPU / 共享 GPU / IO / daemon 常驻）。单独立项，可能配 ADR。
- **Q5（P-D 迁移面）**：`task_type` 暴露后，前端 `inferKind` 的所有调用点（Queue / Overview /
  VersionTasksPanel）需一并切换；旧 task 无 task_type 的兜底（DEFAULT 'train' 已在 DB 层）。
