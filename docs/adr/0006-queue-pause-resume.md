# 0006 — Queue 任务暂停 / 恢复 + 队列挂起 / 恢复调度

**状态**：Accepted（PR-1 #97 / PR-2 #98 / PR-3 #99 / PR-4 #100 全部合入 dev；PR-5 删 feature flag 默认开启）
**日期**：2026-05-18
**决策者**：@WalkingMeatAxolotl

## 背景

Queue 系统目前唯一能停训练的方式是「取消」——supervisor 发硬终止信号
（Windows `CTRL_BREAK_EVENT` / POSIX `SIGTERM`），子进程退出，state 不保存，
重新跑必须从 step 0 开始。

CLI 侧其实已经有完整的 save/resume 链路：

- `runtime/training/context.py:109` `handle_interrupt` 保存 state + LoRA +
  finish wandb；
- `runtime/training/phases/resume.py:82` `signal.signal(SIGINT,
  ctx.handle_interrupt)` 把它绑在 SIGINT 上；
- `runtime/training/phases/resume.py:60-79` 实现 `--resume-state` 加载 state +
  恢复 monitor 历史。

但这条链路**只能从控制台手按 Ctrl+C 触发**——supervisor cancel 发的
`CTRL_BREAK_EVENT` / `SIGTERM` 都不命中 SIGINT handler，绕过了 handle_interrupt。

`ResumeFieldPicker` 让用户能在**新建 task** 时手动选一个 `.pt` 续训，但这是另起
一个 task：新 task_id、新 log、loss / 监控历史断开。

`save_state_every` / `save_state_every_epochs` 周期写 `.pt`，但路径不带 task_id
（`<output_dir>/training_state_step{N}.pt`），同 version 下多 task 跑会互相覆盖。
这是个 latent bug，pause/resume 落地会把它放大成数据丢失。

**用户痛点（按频次）**：

- 训练中途想腾 GPU 跑别的（generate / 别的 LoRA） → 现在只能丢进度取消；
- 关机 / 临时离线 → 同上；
- 跑到一半 loss 不对，想停下来分析再决定 → 同上。

详细的状态机、user case、文件存放、UI 流程已在
`docs/design/queue-pause-resume-design.md` 里讨论过三轮（PM / 终端用户 / Designer
三方 review），本 ADR 承袭该文档的逻辑模型，重点固化决策和代码层面方向。

## 候选方案

### A：SIGINT 信号通道，复用 handle_interrupt 链路（采纳）

supervisor 给子进程发信号触发已有的 handle_interrupt。POSIX 走 SIGINT，Windows
走 `CTRL_BREAK_EVENT` + 子进程额外注册 SIGBREAK handler。

- 优点：复用现成保存链路；信号是标准跨进程通知机制；改动面积小。
- 缺点：Windows 上 `CREATE_NEW_PROCESS_GROUP` 收不到 `CTRL_C_EVENT`，只能收
  `CTRL_BREAK_EVENT`；Python 把它映射成 SIGBREAK 而不是 SIGINT，需要子进程额外
  注册一次。需要 spike 验证整条链路。

### B：Sentinel 文件 / 命名管道 IPC

supervisor 写一个 sentinel 文件，子进程开 watcher 线程定期 poll，看到就主动
调 handle_interrupt。

- 优点：跨平台行为完全一致，不依赖信号语义。
- 缺点：新增 IPC 通道；watcher 线程多一份 CPU 占用 + 触发延迟（poll 间隔）；
  sentinel 清理 / 残留是新问题；如果方案 A 通了就没必要。

作为方案 A spike 失败的兜底。

### C：完全重新搭 RPC（gRPC / WebSocket）

最重，过度工程。否决。

### D：Fake pause — cancel 后自动从最近 save_state_every checkpoint 续训

- 优点：零代码成本。
- 缺点：强依赖 `save_state_every`（默认 0，多数用户没开）；恢复点不精确（最近
  周期 save 可能差几百 step）；UI 撒谎说"暂停成功"实际是 cancel，长期欠债。

否决。

## 决策

采纳**方案 A**：信号通道 + 复用 handle_interrupt。具体决策汇总如下，详细论证
见设计文档相应章节（标 §N 处引用 design doc）。

1. **新增 task 状态 `paused`**（non-terminal, non-live）。不引入
   `pausing` / `resuming` 中间态。（design §1-2）
2. **新增队列挂起开关**：db kv 单 bool，跨 server 重启保留。不进 task 状态机。
   （design §3.2）
3. **术语**：任务**暂停 / 恢复**（pause / resume），队列**挂起 / 恢复调度**
   （hold / release）。中英文都故意用不同动词避免歧义。（design §3）
4. **State 文件路径**：`<output_dir>/state/task_<TID>/`，pause 文件加 `pause_`
   前缀，跟周期 save 区分。同步顺手修今天的 latent bug。（design §5.1, §5.3）
5. **Config snapshot**：pause 时落盘 `pause_step_<N>.config.json`，把当前训练
   实际在用的全部 args / dataset / sample 参数序列化。resume 严格用 snapshot，
   不读 task 表 / version 配置 / 外部 yaml。（design §5.7, §8.5）
6. **Pause 文件对生命周期**：跟随 paused 状态自动管理；resume 成功 / 彻底取消 /
   删除 task 三种情况一并删除。任何时刻一个 task 最多 1 对 pause 文件。
   （design §5.5）
7. **UI 暂停过程 modal**：点暂停立即锁屏 modal 全程引导（保存中→成功/超时/失败），
   30s 超时不默默降级 cancel，给用户三选一。（design §4.3）
8. **挂起 confirmation modal**：检测 running task 多问一句"是否同时暂停"，
   radio + 主按钮文案联动；不做"暂停全部"复合按钮。（design §4.4）
9. **挂起状态显示用 banner，不用 task chip**——banner 是 UI 元素，不是 task
   状态机一部分。（design §4.1）
10. **过早暂停防护**：UI 端 `is_pausable` 信号控制按钮可见性；API 端
    defense-in-depth 拒绝。（design §8.1）
11. **不做** server crash 自动保 state，引导用户开 `save_state_every`。（design §9）

### 后端代码方向

#### `runtime/training/context.py`

`TrainingContext.handle_interrupt` 改动：

```python
def handle_interrupt(self, sig, frame) -> None:
    if self.interrupted:
        sys.exit(1)
    self.interrupted = True

    state_path = _build_pause_state_path(self.output_dir, self.task_id, self.global_step)
    config_path = state_path.with_suffix(".config.json")

    _write_config_snapshot(config_path, self.args, self.sample_prompts)
    save_training_state(state_path, ...)  # 现有调用
    self.injector.save(...)
    self.wandb_monitor.finish()

    self._emit_event("pause_state", {
        "state_path": str(state_path),
        "config_path": str(config_path),
        "step": self.global_step,
    })
    sys.exit(0)
```

新增字段 `TrainingContext.task_id: Optional[int]`，启动时从 env `LORA_TASK_ID`
读入（supervisor spawn 时注入）。

`_build_pause_state_path` / `_write_config_snapshot` / `_emit_event` 作为模块级
helper 函数，落到 `runtime/training/state.py` 或新文件 `runtime/training/snapshot.py`。

config snapshot 内容（候选清单，最终以实现时序列化结果为准）：

- 全部 `args.*`：lr, optimizer, optimizer_args, scheduler, batch_size,
  grad_accum, max_train_steps, num_epochs, noise schedule, loss weighting,
  network_dim, network_alpha, dropout, rank, ...
- dataset_config / resolution / caption_extension / shuffle / repeat
- output_dir / output_name / sample_prompts / sample_every / save_every_n_steps
- 关键模型路径（base, vae, text encoder） — 存路径不存 hash
- random seed
- **不存**：wandb run id（已 finish），monitor live state（已 dump 在 .pt 内）

#### `runtime/training/loop.py`

周期 save 的写盘路径同步改成 per-task 子目录，命名保持 `step_<N>.pt`（无 pause
前缀，靠命名跟 pause 文件区分）。这是顺手修 latent bug，独立 PR 先 ship 更干净
（见"PR 拆分建议"）。

#### `runtime/training/phases/resume.py`

```python
def run(ctx: TrainingContext) -> None:
    # ... 现有逻辑 ...
    signal.signal(signal.SIGINT, ctx.handle_interrupt)
    if os.name == "nt":
        signal.signal(signal.SIGBREAK, ctx.handle_interrupt)  # 新增

    # 在进入 train_loop 前 emit:
    ctx._emit_event("train_loop_started", {})

    # load_training_state 成功后 emit（已存在的 load_training_state 调用之后）:
    if args.resume_state:
        # ... 现有 load ...
        ctx._emit_event("resume_state_loaded", {"path": args.resume_state})
```

#### `studio/supervisor.py`

`_Slot` dataclass 加字段：

```python
@dataclass
class _Slot:
    # ... 现有字段 ...
    pause_pending: bool = False
    pause_state_path: Optional[Path] = None
    pause_config_path: Optional[Path] = None
    pause_step: Optional[int] = None
    train_loop_started: bool = False
```

新增方法：

- `pause(task_id) -> bool`：跟 `cancel(task_id)` 平级
- `_signal_pause_async(slot)`：跟 `_signal_terminate_async` 同形，发 pause 信号，
  **超时不强杀**（让 modal 决定下一步）
- `_send_pause_signal(proc)`：Windows `CTRL_BREAK_EVENT`，POSIX `os.kill(pid, SIGINT)`

`_finish_slot` 三元分流（替换 `supervisor.py:972-977`）：

```python
if slot.pause_pending and slot.pause_state_path:
    status = "paused"
elif slot.cancel_pending:
    status = "canceled"
elif rc == 0:
    status = "done"
else:
    status = "failed"
```

paused 分支写 db 时多 set `paused_state_path` / `paused_config_path` /
`paused_step` / `paused_at`。

`_on_line` 识别新事件 `pause_state` / `train_loop_started` / `resume_state_loaded`，
更新 slot 字段：

- `pause_state` → 设 `pause_state_path` + `pause_config_path` + `pause_step`
- `train_loop_started` → 设 `train_loop_started=True`
- `resume_state_loaded` → 标记可以删旧 pause 文件对（在 _on_finish 或独立线程清）

启动 reload 时 `status='running'` 标 failed 的现有逻辑要显式跳过 `paused`。

#### Cancel 在 Windows 的信号撞车

今天 cancel 在 Windows 发 `CTRL_BREAK_EVENT`，pause 也要发它 → 子进程无法区分意图。

**决策**：cancel 在 Windows **不再发软信号**，直接走 `taskkill /T /F` 强杀进程树。
理由：cancel 语义本来就是硬中断，"先优雅再强杀"在 Windows 上没意义（30s grace
几乎都触发强杀）。`CTRL_BREAK_EVENT` 专门留给 pause。

POSIX cancel 继续发 SIGTERM（grace 后强杀），pause 发 SIGINT，互不撞。

#### `studio/db.py`

migration 加列：

- `paused_state_path TEXT NULL`
- `paused_config_path TEXT NULL`
- `paused_step INTEGER NULL`
- `paused_at REAL NULL`

```python
VALID_STATUSES = {"pending", "running", "done", "failed", "canceled", "paused"}
TERMINAL_STATUSES = {"done", "failed", "canceled"}  # 不加 paused
```

`next_pending` 不动（自然跳过 paused）。

挂起开关用 kv 存储：

```python
def get_queue_held(conn) -> bool: ...
def set_queue_held(conn, held: bool) -> None: ...
```

放新表 `app_settings(key TEXT PRIMARY KEY, value TEXT)` 或现成的 kv 表，二选一。

#### `studio/server.py` 新 endpoint

```
POST /api/queue/{task_id}/pause   → supervisor.pause(task_id)
POST /api/queue/{task_id}/resume  → 见下
POST /api/queue/hold              → db.set_queue_held(True)
POST /api/queue/release           → db.set_queue_held(False)
GET  /api/queue/hold              → {"held": bool, "pending_waiting": N}
```

`/api/queue/{id}/pause` 检查 `is_pausable`（看 supervisor slot 的
`train_loop_started`），未就绪返 409。

`/api/queue/{id}/resume` 流程：

1. 读 task 的 `paused_state_path` + `paused_config_path`。
2. 校验文件存在（不存在返 409，引导用户走 ResumeFieldPicker 起新 task）。
3. 把 task 的 status 从 paused 改回 pending。
4. cmd_builder 在下一轮调度时识别到 `paused_state_path` / `paused_config_path`：
   - 用 `paused_config_path` 的 snapshot 拼 args；
   - 唯一覆盖：`--resume-state <paused_state_path>`；
   - env 注入 `LORA_TASK_ID=<task_id>`（保持 state 子目录一致）。

`cancel_task` 增强：允许 paused → canceled 直接改 db + 清 pause 文件对（进程
已退出，不需要发信号）。

supervisor 主循环 dispatch 时检查：

```python
if db.get_queue_held(conn):
    continue  # 跳过本轮调度，已 running 的不动
```

### 前端代码方向

#### `studio/web/src/types.ts` + API client

```ts
type TaskStatus = 'pending' | 'running' | 'done' | 'failed' | 'canceled' | 'paused'
const TERMINAL: TaskStatus[] = ['done', 'failed', 'canceled']  // 不加 paused
```

`studio/web/src/api/client.ts` 加：

```ts
pauseTask: (id: number) => req(`/api/queue/${id}/pause`, { method: 'POST' }),
resumeTask: (id: number) => req(`/api/queue/${id}/resume`, { method: 'POST' }),
holdQueue: () => req(`/api/queue/hold`, { method: 'POST' }),
releaseQueue: () => req(`/api/queue/release`, { method: 'POST' }),
getQueueHold: () => req<QueueHoldState>(`/api/queue/hold`),
```

monitor SSE 协议增加 `is_pausable: boolean` 字段，由 supervisor 从
`slot.train_loop_started` 派生。

#### `studio/web/src/pages/Queue.tsx` / `QueueDetail.tsx`

- 顶部 banner（仅 `held=true` 时显示，sticky）；
- 顶部 actions：暂停 / 取消 / 挂起队列 / 恢复调度；
- 暂停按钮：`!isPausable` 时隐藏（不是 disabled）；
- paused 行内：恢复 / 彻底取消按钮；
- paused 行附信息：在 step N 暂停于 …；
- pending 行在 held=true 时附"等待恢复调度"提示。

#### 新增组件

- `PauseProgressModal.tsx`：暂停过程 modal（保存中 / 超时 / 成功 / 失败四态），
  订阅 task 的 SSE 事件流。
- `HoldQueueModal.tsx`：挂起 confirmation modal（情形 A 无 running + 情形 B
  有 running 的 radio 联动）。

#### i18n

新增 key（中英双语，英文术语用 hold / release）：

- `queue.pause` / `queue.resume` / `queue.holdQueue` / `queue.releaseQueue`
- `queue.pauseProgress.*`（modal 状态文案）
- `queue.holdModal.*`（情形 A / B 文案）
- `status.paused`
- 等

### Spike 必做（合并 ADR 后第一件事）

Windows 端验证：

1. supervisor `proc.send_signal(signal.CTRL_BREAK_EVENT)` 能否送达
   `CREATE_NEW_PROCESS_GROUP` 子进程组；
2. 子进程 Python `signal.signal(signal.SIGBREAK, handler)` 能否捕获；
3. handler 能否完整跑完 save_training_state + write snapshot 后 `sys.exit(0)`；
4. supervisor 能否正确读到子进程 stdout 上的 `__EVENT__:pause_state` 行后才走
   `_finish_slot` 标 paused。

spike 失败 → 回退方案 B（sentinel 文件 IPC），本 ADR 第二阶段决策修订。

### PR 拆分建议

1. **PR-0 spike**：仅 spike 脚本 + 报告，不动主线。
2. **PR-1 latent bug 前置修**：`runtime/training/loop.py` 周期 save 路径加
   per-task 子目录。独立 ship，干净地基。
3. **PR-2 后端骨架**：context / supervisor / db migration，新 API endpoint，
   全部带单测。feature flag `enable_pause_resume` 默认 off。
4. **PR-3 resume 路径 + cmd_builder**：端到端集成测（pause N step → resume →
   验证 global_step 从 N+1 接上 + loss 连续）。
5. **PR-4 前端 UI**：banner / 按钮 / 两个 modal / i18n。
6. **PR-5 文档 + changelog + 灰度开启 feature flag**。

每个 PR 独立可 revert，回滚粒度细。

## 理由

**为什么否决方案 B**：信号机制更标准，spike 通了就没必要再加 IPC 通道。B 保留
作 spike 失败兜底。

**为什么否决方案 D**：强依赖 `save_state_every`（默认 0），多数用户没开；恢复
点不精确；UI 撒谎欠债。

**为什么 state 文件路径要加 per-task 子目录**：今天 `save_state_every` 在同
version 多 task 场景已经互覆盖。这不是 pause/resume 引入的问题，是 latent
bug，本 feature 顺手修。把它拆 PR-1 独立先 ship 让回归风险隔离。

**为什么 config snapshot 而不复用 task config 字段**：task config 字段创建时
frozen，但有些参数来自 version / preset / 外部 yaml，存的是引用路径不是
inline value。用户改 version 配置后，按路径再次解析会拿到新值。snapshot 落盘
= 把所有引用 inline 展开成具体值，从根上跟"用户当前 config"解耦。

**为什么挂起状态不进 task 状态机**：队列挂起是 dispatcher 级别属性，跟单 task
状态无关。task 跨挂起边界状态不变（running 继续跑、paused 继续 paused）。进
状态机意味着 5 个状态变 6 个 + 全套迁移规则，没必要。

**为什么暂停过程用全程 modal 而不是按钮 + toast**：pause 期间用户没机会"反悔
不想 pause"（信号已发），强制锁屏避免误操作把进度丢了。30s 超时给用户选 [再等
30s] / [强制取消保存进度] / [终止任务] 而不是默默降级 cancel——用户点暂停的
意图就是要保进度，默默 cancel = 用户惊吓。

**为什么 cancel 在 Windows 改成 taskkill /T /F 直接走**：cancel 语义本来就是
硬中断，30s grace 几乎都触发强杀。`CTRL_BREAK_EVENT` 专门留给 pause 让信号意图
明确。POSIX 没这个问题（SIGINT vs SIGTERM 天然分流）。

## 后果

### 正面

- 现有 cancel 语义不变，用户旧习惯不受影响。
- 用户能恢复中断进度，不再"取消 = 全部白跑"。
- 跨 server 重启的 paused task 自动保留，"关机再开"工作流可用。
- 顺手修了 `save_state_every` per-task 子目录的 latent bug。
- config snapshot 设计让 paused task 跟用户后续改 config 完全解耦。
- 队列挂起独立开关让"夜间不跑"" 维护窗口"工作流可用。

### 负面 / 待评估

- Windows 信号链路通不通取决于 spike，有方案 B 兜底但需要重新走一轮设计。
- pause 文件对（.pt + .config.json）多一份磁盘占用，但跟随 task 生命周期自动清。
- supervisor `_finish_slot` 分支从二元变三元，回归风险靠单测覆盖。
- cancel 在 Windows 改成 taskkill /T /F 直接走，跳过软信号 grace 阶段；现有
  cancel 行为对用户基本无差异，但日志 / telemetry 如果有依赖 grace 阶段需迁移。
- snapshot 序列化清单的完备性需要在 PR-3 集成测里覆盖——少存一个字段就可能
  resume 行为漂移。

### 未来债（明确不在本 ADR scope）

- Wandb run id 续接（resume 起新 run，不复用 run_id）
- 批量 pause / resume
- 挂起定时自动恢复
- paused 超 X 天提醒
- 首次跑训练 UI 推荐开 `save_state_every`
- 成功指标 / 灰度遥测
- 暂停后编辑 config 再 resume（永远不支持，强制 fork）

## 不在范围

- 服务器主动 stop / crash / 断电时自动保 state（覆盖面不可控，引导用户开
  `save_state_every` 周期 checkpoint）
- "暂停全部"复合按钮（挂起 modal 多问一句已覆盖）
- 强 kill 后保留 paused 状态（强 kill 时 state 不可信，必标 canceled）
- 自动清理周期 save 文件（用户主动开的灾后恢复点，由用户管）
- pause generate / download / tag task（跑得快无意义）

## 参考

- 设计文档（三轮 review）：`docs/design/queue-pause-resume-design.md`
- 现有代码触点：
  - `runtime/training/context.py:109` `handle_interrupt`
  - `runtime/training/phases/resume.py:82` SIGINT 注册
  - `runtime/training/loop.py:271` `save_state_every` 写盘（latent bug 现场）
  - `studio/supervisor.py:952` `_finish_slot` 状态分流
  - `studio/supervisor.py:1081` `_send_terminate_signal`
  - `studio/db.py:36` `VALID_STATUSES`
  - `studio/server.py:2893` `cancel_task` endpoint（现有）
  - `studio/web/src/pages/Queue.tsx:159` 现有 cancel-only 注释
- memory：`memory/queue_pause_resume_via_sigint.md`（早期决策痕迹）
