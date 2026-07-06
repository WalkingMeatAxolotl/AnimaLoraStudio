# Queue 暂停信号链路 Spike

ADR 0006 PR-0 的验证脚本。**临时性质，验证完会随 cleanup PR 删除。**

## 这是什么

ADR 0006 决定走「信号通道 + 复用 handle_interrupt」（方案 A）。落地前要
先在 Windows 上验证整条信号链路：

```
supervisor.proc.send_signal(CTRL_BREAK_EVENT)
    ↓
[Windows: 子进程组收到]
    ↓
Python signal.signal(SIGBREAK, handler) 捕获
    ↓
handler 跑完 save_training_state + write snapshot + emit __EVENT__:pause_state
    ↓
sys.exit(0)
    ↓
supervisor stdout reader 读到事件行 → _finish_slot 标 paused
```

链路里任何一步通不了 → 整个方案 A 报废，要回退到方案 B（sentinel 文件 IPC）。

## 跑法

```bash
python tools/spike/pause_signal_parent.py
```

不要直接跑 `pause_signal_child.py`——它依赖 parent 发的信号才有意义。

跑完控制台会打印一份验证报告（6 项检查），退出码 0 = 全过、非 0 = 至少一项
失败。

`tmp/spike_state/` 是 spike 输出目录（fake .pt + config snapshot），每次跑
parent 会先清空。

## 验证项（对应 ADR §Spike 必做）

| # | 检查 | 通过判据 |
|---|------|---------|
| 1 | CTRL_BREAK_EVENT 送达 CREATE_NEW_PROCESS_GROUP 子进程组 | parent 收到 `__EVENT__:pause_state` |
| 2 | `signal.signal(SIGBREAK, handler)` 捕获 | 子进程 stdout 出现 "handler 触发" 行 |
| 3 | handler 完整 save + sys.exit(0) | 子进程 stdout 出现 "handler 完成" + `rc=0` |
| 4 | parent 解析 `__EVENT__:pause_state` payload | payload 含 `state_path` 字段 |
| 5（附加）| pause `.pt` + `.config.json` 各一份落盘 | `tmp/spike_state/state/task_42/` 下文件存在 |
| 6（附加）| `__EVENT__:train_loop_started` 事件 | parent 收到 → 验证 ADR §8.1 `is_pausable` 信号 |

## 设计意图

- **不依赖项目代码**：spike 是黑盒验证 OS + Python 信号语义，不 import
  `runtime.training.*` / `studio.supervisor`，免得验证结果被项目代码引入的状态
  污染。
- **mirror 真实参数**：`CREATE_NEW_PROCESS_GROUP` / `PYTHONUNBUFFERED=1` /
  stdout=PIPE+stderr=STDOUT 跟 `studio/supervisor.py:_popen` 对齐。
- **不模拟体积**：fake `.pt` 是几十 KB 字节，真训练 state 几十~几百 MB。
  spike 只测信号路径通不通，IO 耗时只 sleep 0.5s 模拟。
- **不依赖 GPU**：跑得起 python 解释器就行。

## 删除时机

PR-0 合入 dev 后留作"曾经验证过方案 A 可行"的快照证据。下面两个时机
任一发生时整个 `tools/spike/` 目录可以 cleanup PR 删除：

- ADR 0006 整体落地完成（PR-1 ~ PR-5 全部合入），方案 A 已生产验证。
- 或 spike 报告显示方案 A 失败，ADR 决策修订走方案 B。

验证已完成：方案 A 经生产验证、ADR 0006 已整体落地。原 spike 验证报告
（`docs/design/queue-pause-spike-report.md`）已随文档治理删除，结论归入
[ADR 0006](../../docs/adr/0006-queue-pause-resume.md)。
