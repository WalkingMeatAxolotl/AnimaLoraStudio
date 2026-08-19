# 日志体系目标态：统一行契约 + 显示级过滤

状态：四项关键决策已拍板（2026-08-19）；**刀 1（#507）/ 刀 2（#508）已合，刀 3 已实施**（分支 feat/logging-logview）。分刀见 §6。
触发：issue #505 的切桶释放日志无处可去 → 全仓盘点发现 8 处 `logger.debug` 在任何配置下都不可见、run.log 是无级别的裸字节流、前端 6 套日志视图零级别解析。盘点原稿在 `tmp/logging-inventory.md`（本地）。
上游：ADR 0009（logging/error system）定的 studio 侧骨架（`setup_logging` / JSON line / trace_id / 错误 envelope）不推翻，本文只补它没覆盖的子进程与显示面，并把跨进程的行契约定下来。

## 1. 用户场景

**普通用户（webui）**
- 打开任一任务的日志：每行有时间、级别、来源；报错行红、警告行黄，一眼看到「哪一步开始不对」；默认不显示调试行，报错后**在同一个视图里打开「调试」开关**就能看到调试行，不用重跑。
- 任务失败：红框里是最后一个 ERROR/Traceback 块，不是文件尾巴碰运气。
- 日志抽屉和日志 tab 是同一个东西：同样的着色、过滤、复制、下载、自动滚动开关；长任务不卡页面；刷新/断线不丢行。
- 报 issue：一个按钮导出诊断包（该任务 run.log + 时间窗内 studio.log + 版本/GPU/env 摘要）。

**终端用户 / 纯 CLI**
- 控制台里看到的与 run.log 同一套格式；不再一半 `[studio]` print、一半 logger、一半 JSON。
- 直接跑 `runtime/anima_train.py` 仍是交互式进度条；pipe 模式自动切成逐行日志。

**开发者 / 排障**
- 拿到诊断包就够：run.log 每行可定位模块；trace_id 从 toast → API → studio.log → 子进程 run.log 贯通。
- 加一条 `logger.debug` 时知道它一定会被记录、用户一开开关就能看到。

## 2. 决策记录

| # | 问题 | 决策 | 推论 |
|---|---|---|---|
| D1 | 调试日志开关放哪 | **进 UI 设置**：全局开关只负责「默认值」；每个日志视图有**独立开关**（不持久化，初值取全局）；调试行靠**显示过滤**隐藏而不是不记录 | ⇒ 供 UI 展示的记录面（run.log / daemon ring）**始终记到 DEBUG**；级别开关是显示端的事，不是生成端的事 |
| D2 | 子进程 / daemon 日志要不要落 studio.log | **不落**；各进程写各自的面，排障靠诊断包拼 | studio.log 仍只属 webui 进程；不引入跨进程写同一文件 |
| D3 | 训练进度行（step/loss/speed）算不算日志 | **算**：pipe 模式下走 logger，与其它行同契约 | 「一个格式」不变量没有例外；tty 交互模式仍用 rich/plain 进度条 |
| D4 | run.log 保留策略 | **不做 GC**；随任务删除 | ADR 0009「7 天 GC」条目作废 |

由此推出的五条不变量（也是各刀的验收标准）：

1. **一个格式**：所有进程落到任何日志面的行都是同一契约（§3.2）；结构化面（studio.log）是它的 JSON 版。裸 `print` 只剩三类合法用途：stdout 协议行（`__EVENT__:` / daemon line-JSON）、tty 交互进度、CLI 启动 banner。
2. **记录不过滤，显示才过滤**：面向 UI 的记录面永远 DEBUG；`ANIMA_LOG_LEVEL` 只管终端可读性（开发者旋钮，不进 UI）。
3. **级别是级别、tag 是主题**：`[Debug]` / `[WARN]` / `[OK]` 这类冒充级别的前缀清掉；方括号只保留主题 tag，同主题同级别。
4. **一个日志组件**：前端所有「看一段任务/进程日志」的地方用同一组件 + 同一数据 hook；数据契约「seq 单调 + 可从 seq/offset 续拉」。
5. **debug 有去处**：没有任何 `logger.debug` 是「写了永远看不到」。

## 3. 目标态

### 3.1 记录面：级别与落点

| 进程 | 落点 | 我方 logger 级别 | 第三方 logger | 终端 stderr |
|---|---|---|---|---|
| webui | `studio.log`（JSON line，50MB×5 不变） | DEBUG | `_NOISY_LOGGERS` → WARNING（列表扩到 transformers / diffusers / accelerate / peft / wandb / spandrel / matplotlib.font_manager） | console handler 级别 = `ANIMA_LOG_LEVEL`（默认 INFO）；tty → Human，pipe → 同 Human（**不再输出 JSON**，与文件重复无意义） |
| cli | 无文件 | — | 同上 | 同上 |
| worker 子进程 | 自身 stderr = `tasks/<id>/run.log`（OS 合流不变） | DEBUG（supervisor spawn 注入 `ANIMA_LOG_LEVEL=DEBUG`，`setdefault` 外部可覆盖） | 同上 + 现有 `TRANSFORMERS_VERBOSITY=error` 等 env 保留 | 即 run.log |
| anima_train / anima_reg_ai | 同上 | 同上 | 同上 | 同上 |
| anima_daemon | stderr → server ring buffer（2000 行不变） | DEBUG（daemon spawn 同样注入） | 同上；LyCORIS 掰 stderr 逻辑保留 | 即 ring |
| anima_generate / 手跑 anima_train | 终端 | `ANIMA_LOG_LEVEL` 默认 INFO | 同上 | 终端 |

规则一句话：**被 studio 拉起 → 记 DEBUG；人手拉起 → 默认 INFO；`ANIMA_LOG_LEVEL` 两边都能覆盖。** 只有这一个 env，`setup_logging` 内部读它（现在只有 webui 入口在外面读），四个 runtime 入口不再各自 `basicConfig`，改调 `setup_logging(process, file=False, console=True)`（runtime 本就 import studio.*，无额外成本；open question 1 已定）。

trace / process 注入补齐：`_spawn_job`、daemon spawn 与 `_spawn_task` 一样注入 `ANIMA_TRACE_ID` / `ANIMA_PROCESS_NAME`。

### 3.2 行契约

```
2026-08-19 14:03:22.417 INFO  training.loop: epoch=0 step=50 loss=0.031 lr=1.0e-4 speed=0.78 it/s
2026-08-19 14:03:23.001 ERROR training.phases.finalize: block swap 释放失败
Traceback (most recent call last):
  ...
```

- 格式即现有 studio Human formatter：`%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s: %(message)s`，datefmt `%Y-%m-%d %H:%M:%S`。runtime 四入口改用同一 formatter；`anima_train` 的 logger 名从 `__name__`（跑成 `__main__`）改固定 `anima_train`。
- 多行记录（traceback、rich 曲线面板）：续行无前缀；解析器把不匹配行头的行归入上一条记录、继承其级别。
- 训练 pipe 模式：`log_every` 进度行、`ctx.emit` 消息全部 `logger.info`；tty 模式不变（rich Progress / plain `\r`）。`ctx.emit` 的三路分流保留，只把最后的 `print` 分支换成 logger。
- 裸 print 合法白名单：`__EVENT__:` 协议行（`snapshot.emit_event`、`preprocess_worker`）、daemon `_emit*` line-JSON、rich/plain tty 进度、`api/main.py` banner。其余 print → logger（runtime 35 处 / studio 41 处，清单见盘点稿 §1）。
- 严重度只由级别表达：`[Debug]`（sampling.py 11 处）→ `logger.debug` 去前缀；`[WARN]` → `logger.warning`；`[OK]` print → `logger.info`。主题 tag（`[显存]` `[navit]` `[text-cache]` `[masked-loss]` …）保留；级别按事件本身定，tag 不携带级别语义。
- `error_msg` 回写：从 run.log 尾部（只读末 256KB）取**最后一个 ERROR/CRITICAL 记录块**（行头去前缀 + 续行）；块外更靠后的裸 `Traceback`（子进程未捕获异常）盖过它；都没有退回末 12 行。

### 3.3 显示面

**全局开关**：设置页一项「默认显示调试日志」（布尔，默认关），**后端字段** `Secrets.system.log_debug_default`（与设置页其它项一样走 secrets 落盘 + 现有 settings API / instant-apply），换浏览器不丢。它只影响显示默认值，不影响记录（记录恒 DEBUG，见 §3.1）。

**视图开关**：每个日志视图头部一个「调试」toggle，不持久化，挂载时取全局开关值；开 = 显示 DEBUG 及以上，关 = 显示 INFO 及以上。WARNING/ERROR 永远显示且着色（黄/红），DEBUG 行显示时用弱化色。

**统一组件** `LogView`（一个 presentational 组件 + 一个 `useLogSource` 数据 hook）：

| 能力 | 规格 |
|---|---|
| 解析 | 按 §3.2 行头正则切记录；续行并入 |
| 过滤 | 视图开关（级别阈值）；不做关键字搜索（非目标） |
| 着色 | ERROR 红 / WARNING 黄 / INFO 默认 / DEBUG 弱化；单一 token 集，不再四套颜色 |
| 尾部加载 | 初次只拉尾 N 条（默认 500），顶部「加载更早」按 offset 往前翻；客户端上限 5000 记录 |
| 增量 | SSE 按 seq 追加；`onOpen` 重连时用最后 seq/offset 补拉 |
| 操作 | 自动滚动开关、复制、下载（原始 run.log） |
| 状态 | 等待日志 / 已结束 / 断线重连中 |

替换关系：`TaskLogDrawer` 内容区、`QueueDetail` LogTab、`DaemonLogDrawer` 内容区、`useEvalLogSource` / `EvalJobsPanel` 全部改用 `LogView`（数据由 `useTaskLog` 提供）。设置页下载日志 / 更新日志 / onboarding 安装日志三个小面**保留原 `<pre>`**：内容不是契约行（downloader / updater / 安装脚本输出），LogView 的解析与工具栏在那里没有增益（刀 3 实施时定）。抽屉的开合状态机、StepShell 的 `logSources` 挂载方式不变；`LogSource` 加可选 `downloadUrl / hasMoreBefore / onLoadEarlier`。

其它显示修正：DaemonLogDrawer 渲染 `ts`；PreprocessDuplicates 的 per-line `status` 映射到级别；`PauseProgressModal` 深链改 `navigate('/queue/<id>#log')`；error toast 末尾接上已写好的 `formatErrorTraceSuffix`。

### 3.4 读取面：API 与 SSE

| 端点 / 事件 | 现状 | 目标 |
|---|---|---|
| `GET /api/logs/{task_id}` | 全文 `{content,size}` | `?tail=N`（默认 500 行）/ `?before=<offset>&limit=N`（往前翻）/ `?after=<offset>&limit=N`（断线补拉）；返回 `{lines: [{offset,text}], start_offset, end_offset, size, has_more_before}`，`offset` = 行起始字节、`end_offset` = 最后一行结束后的偏移（after 游标）；按字节切行再逐行 `clean_log_line`，与 LogTailer 同文本；末尾半行不返回；服务端仍剥 `__EVENT__:` 行；只按 64KB 块从尾部读 |
| `GET /api/logs/{task_id}/raw` | 无 | 原始文件下载（诊断包 / 下载按钮用） |
| SSE `task_log_appended` / `job_log_appended` | 两种形状（LogTailer 带 seq、daemon 回写不带） | 统一 `{type, task_id|job_id, seq, end_offset, text}`（`end_offset` 与 API 的 after 游标同坐标系，LogTailer 改按字节切行计算）；daemon 回写路径补 seq/end_offset（`fp.tell()`） |
| SSE `daemon_log_line` | `{ts, seq, line}` | 不变 |
| `GET /api/generate/daemon/logs` | `since_seq` / `limit` | 不变（已是目标形状） |
| `event_malformed` | task 路径 emit、job 路径不 emit、前端不消费 | job 路径补 emit（刀 2 已做）；前端 LogView 收到后插一条 WARNING 伪记录「事件解析失败」（刀 3） |
| `_on_task_log` / `_on_line` | try 块包住 `_on_event` | 只包解析（`_parse_event_marker`），广播移出 try（刀 2 已做） |

### 3.5 CLI / 终端

- `_say(msg, level)` 改成 `logging.getLogger("studio.cli")` 的薄包装，`level` 真起作用（info/warning/error 走对应级别，Human formatter 出前缀）；14 处裸 print 与 `pending_install` 的 10 处 `[studio]` print 收编。`--verbose` 不做，走 `ANIMA_LOG_LEVEL`。
- webui console handler 不再在 pipe 下输出 JSON（与 studio.log 重复）。
- uvicorn access log 降到 WARNING（业务日志不与 access 抢 50MB 配额；需要时 `ANIMA_LOG_LEVEL=DEBUG` 打开）。

### 3.6 诊断包

设置页 / 失败任务详情各一个入口：`GET /api/diagnostics/bundle?task_id=…` → zip：该任务 `run.log`、以任务起止时间为窗的 `studio.log` 片段、`GET /api/system/env` 已有的版本/GPU/依赖摘要、任务 config 快照。不含 secrets（wandb key 等走现有脱敏）。

### 3.7 非目标

studio.log 查看 UI；子进程写 studio.log（D2）；run.log GC（D4）；按模块配级别；日志关键字搜索；日志上报/远端；wandb 通道。

## 4. 现状 → 目标 变化清单（按代码位置）

**后端**
- `studio/infrastructure/logging.py`：`setup_logging` 内读 `ANIMA_LOG_LEVEL`（参数仍可显式覆盖）；file handler 与 console handler 分级（file DEBUG、console = env）；`_NOISY_LOGGERS` 扩表；console `auto` 在 pipe 下改 Human；uvicorn 三 logger 设 WARNING；删死参数 `extra_handlers`。
- `runtime/anima_train.py:50` / `anima_daemon.py:68` / `anima_generate.py:53` / `anima_reg_ai.py:61`：`basicConfig` → 共用 bootstrap；`anima_train` logger 固定名。
- `studio/supervisor/core.py:1267-1318`（`_spawn_task` env）与 `:866-906`（`_spawn_job`）、`services/inference/daemon.py:274-317`：注入 `ANIMA_LOG_LEVEL=DEBUG`（setdefault）+ trace/process。
- `runtime/training/loop.py:733,745`（进度行）、`context.py:147-159`（emit print 分支）→ logger；`bootstrap.py:38-108`、`training/cli.py:90,99`、`utils/optimizer_utils.py` 15 处、`utils/caption_utils.py` 6 处 → logger。
- `runtime/training/families/anima/sampling.py` 11 处 `[Debug]` → debug；`utils/lycoris_adapter.py:42`、`optimizer_utils.py:1337` `[WARN]` → warning。
- `anima_generate.py:282,472`、`anima_reg_ai.py:599` `logger.error(str(e))` → `logger.exception`。
- workers 5 处 `progress()`/`log()` 闭包与 8 处 `[error]` print → logger（进度闭包保留函数形状，内部换 logger.info）。
- `studio/cli.py:67-80` `_say` 接 logger；14 处裸 print 收编；`services/runtime/pending_install.py:74-95` 收编。
- `studio/api/routers/logs.py`：分页 + raw；`supervisor/core.py:74-101` `_tail_log_for_error_msg` 改按 ERROR 块；`core.py:1108-1133` daemon 回写补 seq/offset；`core.py:922-950` job 路径补 `event_malformed`；两处 try 范围收窄。
- `studio/infrastructure/secrets.py` `SystemConfig` 加 `log_debug_default: bool = False`；settings 端点/前端 client 类型同步。
- 新增 `studio/api/routers/diagnostics.py`（诊断包）。

**前端**
- 新增 `components/LogView.tsx` + `lib/useLogSource.ts`；`TaskLogDrawer` / `QueueDetail` LogTab / `DaemonLogDrawer` / `useEvalLogSource` 改用；三个小面复用只读模式。
- 设置页：全局开关「默认显示调试日志」读写 `system.log_debug_default`。
- `PauseProgressModal.tsx:124` 深链；`api/client.ts` error toast 接 trace suffix；删 `Toast.tsx:66,75` 死导出；孤儿 i18n `duplicates.logTitle` / `reg.aiLogTitle` / `reg.tabConfig` 清理。
- i18n：`logView.*`（调试开关、加载更早、下载、断线重连、事件解析失败）；`settings.log.*`。

**文档**
- ADR 0009 加 Addendum：记录 D1-D4、行契约、`ANIMA_LOG_LEVEL` 语义变化（生成端 → 终端可读性）、GC 条目作废、`extra_handlers` 删除。
- user-guide：「查看日志 / 调试开关 / 导出诊断包」一节。

## 5. 与 PR #506 的关系

#506 的切桶释放日志已按本文规格写成 `logger.debug`（同主题 `[显存]` tag，级别待 sysmem.py 一侧对齐）；刀 1 合入后它在 run.log 里可见，刀 3 合入后用户在视图里打开「调试」即可看到。

## 6. 分刀与验收

依赖链：刀 1 → 刀 2 → 刀 3；刀 4 独立。每刀独立可合，无临时脚手架。

| 刀 | 内容 | 验收 |
|---|---|---|
| 1 后端一致性 | §3.1 全部 + §3.2 生成端全部 + §3.5 | 单测：`setup_logging` 读 env、file/console 分级；四入口输出行匹配契约正则；spawn env 含 `ANIMA_LOG_LEVEL=DEBUG` 与 trace；runtime/ 与 studio/ 下 print 只剩白名单（AST 扫描测试锁死）。人工：跑一次训练，run.log 每行有 ts/level/name，debug 行在场 |
| 2 读取面 | §3.4 + error_msg | 单测：分页边界（空文件 / 单行 / 跨 offset）、`__EVENT__` 剥离、SSE 形状统一、error_msg 取 ERROR 块；前端 client 类型同步 |
| 3 前端 | §3.3 全部 + 深链 / toast trace / 死代码 / i18n | vitest：解析器（行头/续行/多行 traceback）、过滤阈值、着色 token、断线补拉、开关初值取全局；三处旧组件删除；真机验一次训练日志 + daemon 日志 |
| 4 诊断包 | §3.6 | 单测：zip 内容清单、时间窗切片、脱敏；真机导一次 |

## 7. Open questions（实施时定，不阻塞开工）

- ~~runtime 入口复用 `setup_logging` 还是轻量版~~：已定复用（刀 1）。
- ~~进度行 logger 名~~：已定 `training.progress`，`ctx.emit` 非 tty 出口 `training.emit`（刀 1）。
- ~~视图开关的默认值在「全局开关变化时」是否同步已打开的视图~~：已定不同步（刀 3：挂载取值；用户没动过开关之前全局值晚到会采纳一次，动过之后不回推）。
