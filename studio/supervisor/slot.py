"""Supervisor 执行槽位（PR-4 从 supervisor.py 抽出）。

PP10.2.a 起从「单 _current_* 字段」改成「list[_Slot]」；10.2.b 拆成两槽：
  - TRAIN 槽：只跑 training tasks（db.tasks 表）
  - DATA  槽：只跑 project_jobs（download / tag / reg_build）
download 永远跟训练并行；tag / reg_build 看 settings 开关。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Optional

from ..log_tail import LogTailer, MonitorStatePoller

# 槽位名常量
SLOT_TRAIN = "train"
SLOT_DATA = "data"


@dataclass
class _Slot:
    """Supervisor 内的一个执行槽位。每个槽位最多跑 1 个子进程。"""
    name: str = "main"
    proc: Optional[subprocess.Popen] = None
    kind: Optional[str] = None  # "task" | "job"
    id: Optional[int] = None
    log_fp: Optional[Any] = None
    tailer: Optional[LogTailer] = None
    state_poller: Optional[MonitorStatePoller] = None
    cancel_pending: bool = False
    # ADR 0006 PR-2 pause/resume backend ----------------------------------
    pause_pending: bool = False
    # `__EVENT__:pause_state` payload — handle_interrupt 写好 .pt + snapshot 后
    # 子进程通过 stdout emit；_on_line 抓到后填这三个字段，_finish_slot 据此
    # 把 task 标 paused。pause_pending=True 但缺这几个字段 → 子进程退出前
    # 没来得及 emit → 视为 cancel 兜底（ADR §4.3 modal "强制取消保存进度"）。
    pause_state_path: Optional[str] = None
    pause_config_path: Optional[str] = None
    pause_step: Optional[int] = None
    # ADR §8.1 is_pausable 信号 — resume phase emit `train_loop_started` 后才
    # 允许暂停。UI 端 SSE 用这个解锁暂停按钮，API 端 defense-in-depth 拒绝
    # 过早 pause 请求。
    train_loop_started: bool = False
    # ADR 0006 Addendum 1：epoch 末尾 auto backup 完成后填这两个字段。
    # is_pausable 升级条件要求 `last_auto_epoch_state_path is not None` —— 首 epoch
    # 未结束时按钮完全隐藏，避免用户暂停后无可恢复 state。
    last_auto_epoch_state_path: Optional[str] = None
    last_auto_epoch_config_path: Optional[str] = None

    @property
    def busy(self) -> bool:
        return self.proc is not None

    def reset(self) -> None:
        self.proc = None
        self.kind = None
        self.id = None
        self.log_fp = None
        self.tailer = None
        self.state_poller = None
        self.cancel_pending = False
        self.pause_pending = False
        self.pause_state_path = None
        self.pause_config_path = None
        self.pause_step = None
        self.train_loop_started = False
        self.last_auto_epoch_state_path = None
        self.last_auto_epoch_config_path = None
