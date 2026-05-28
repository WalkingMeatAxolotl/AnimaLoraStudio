"""任务调度守护线程 — PR-4 拆分。

`studio.supervisor` 1431 行原单文件按职责切到本子包：

    slot.py         _Slot dataclass + SLOT_TRAIN/DATA 常量
    cmd_builder.py  默认 cmd builder + monitor_state_path + worker EVENT 协议常量
    finalizer.py    task 终态 → version.status 映射（ADR-0007 §11.3-B）
    process.py      _kill_process_tree（跨平台杀进程树）
    core.py         Supervisor 主类（保单类不拆，状态耦合高 — 详 PR-4 决策日志）

本 `__init__.py` 兼容 shim：把全部 public name re-export 到包顶层，旧
`from studio.supervisor import Supervisor / _Slot / _default_cmd_builder
/ _maybe_finalize_version` 等 import 路径透明工作。

monkeypatch path 兼容：tests 用 `monkeypatch.setattr("studio.supervisor._secrets.load", X)`
和 `monkeypatch.setattr("studio.supervisor.subprocess.Popen", X)`，依赖
`studio.supervisor` 模块对象上的 `_secrets` / `subprocess` attribute。下面从
core.py re-export 让 lookup 命中真实 module 单例（Python 模块对象单例 →
patch 同时影响 core.py 内的调用）。
"""
from __future__ import annotations

from .cmd_builder import (
    _EVENT_MARKER,
    GPU_BOUND_JOB_KINDS,
    CmdBuilder,
    EventCallback,
    JobCmdBuilder,
    _default_cmd_builder,
    _default_job_cmd_builder,
    _resolve_monitor_state_path,
)
from .core import Supervisor, _secrets, subprocess
from .finalizer import _maybe_finalize_version
from .process import _kill_process_tree
from .slot import SLOT_DATA, SLOT_TRAIN, _Slot

__all__ = [
    "Supervisor",
    "_Slot",
    "SLOT_TRAIN",
    "SLOT_DATA",
    "GPU_BOUND_JOB_KINDS",
    "EventCallback",
    "CmdBuilder",
    "JobCmdBuilder",
    "_EVENT_MARKER",
    "_default_cmd_builder",
    "_default_job_cmd_builder",
    "_resolve_monitor_state_path",
    "_maybe_finalize_version",
    "_kill_process_tree",
    # 下方 2 个仅为 monkeypatch path 兼容暴露；不属于业务 API
    "_secrets",
    "subprocess",
]
