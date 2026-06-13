"""Supervisor pause/_on_task_log/_finish_slot 分流测试（ADR 0006 PR-2）。

写了三层：

1. **`_Slot` reset 行为**：新字段都被 reset 清零。
2. **`_finish_slot` 三元分流**：直接构造 slot 状态 + 调 _finish_slot，
   verify status 写 db 的逻辑（pause_pending+state_path → paused；其余照旧）。
3. **`pause()` 状态机校验**：mock 一个 slot 模拟各种 state combo，
   verify pause() 返回 (ok, reason) 是否符合预期。

完整的"发信号 → 子进程保 state → 标 paused" e2e 留 spike 脚本验证过；这里
用 mock proc + 直接构造 slot 字段绕过真子进程，让测试在没 GPU/不能起真训练
的 CI 环境也能跑。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from studio import db
from studio.supervisor import _Slot, Supervisor


@pytest.fixture
def env(tmp_path: Path):
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    logs = tmp_path / "logs"
    configs = tmp_path / "configs"
    logs.mkdir()
    configs.mkdir()
    return {"db": db_path, "logs": logs, "configs": configs}


def _new_sup(env) -> Supervisor:
    """构造一个不 start 的 Supervisor — 直接调内部方法测分支。"""
    return Supervisor(
        on_event=lambda _: None,
        cmd_builder=lambda *_: ["echo"],
        db_path=env["db"],
        logs_dir=env["logs"],
        configs_dir=env["configs"],
        poll_interval=10,
    )


# ---- _Slot 新字段 ------------------------------------------------------------


def test_slot_has_pause_fields_with_safe_defaults() -> None:
    s = _Slot()
    assert s.pause_pending is False
    assert s.pause_state_path is None
    assert s.pause_config_path is None
    assert s.pause_step is None
    assert s.train_loop_started is False


def test_slot_reset_clears_pause_fields() -> None:
    s = _Slot()
    s.pause_pending = True
    s.pause_state_path = "/x/y.pt"
    s.pause_config_path = "/x/y.config.json"
    s.pause_step = 100
    s.train_loop_started = True
    s.reset()
    assert s.pause_pending is False
    assert s.pause_state_path is None
    assert s.pause_config_path is None
    assert s.pause_step is None
    assert s.train_loop_started is False


# ---- _finish_slot 三元分流 --------------------------------------------------


def _populate_running(env, **fields) -> int:
    """db 里新建一个 running 状态的 task，返回 id。"""
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        update = {"status": "running", "started_at": time.time()}
        update.update(fields)
        db.update_task(conn, tid, **update)
    return tid


def _make_slot_with_proc(tid: int) -> _Slot:
    slot = _Slot(name="train")
    slot.kind = "task"
    slot.id = tid
    slot.proc = MagicMock()
    slot.log_fp = None
    slot.tailer = None
    slot.state_poller = None
    return slot


def test_finish_slot_paused_when_pause_pending_and_state_path(env) -> None:
    """pause_pending=True + pause_state_path 已 set → status='paused'。"""
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.pause_pending = True
    slot.pause_state_path = str(env["db"].parent / "pause_step_100.pt")
    slot.pause_config_path = str(env["db"].parent / "pause_step_100.config.json")
    slot.pause_step = 100
    # 调 _finish_slot 后 slot.reset() 会清字段，先 snapshot 期望值
    expected_state = slot.pause_state_path
    expected_config = slot.pause_config_path

    sup._finish_slot(slot, rc=0)

    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task is not None
    assert task["status"] == "paused"
    assert task["paused_state_path"] == expected_state
    assert task["paused_config_path"] == expected_config
    assert task["paused_step"] == 100
    assert task["paused_at"] is not None


def test_finish_slot_canceled_when_pause_pending_but_no_state_path(env) -> None:
    """pause_pending=True 但子进程没 emit pause_state（state_path None）→ 降级 canceled。

    ADR §4.3 modal "强制取消保存进度" 情形。
    """
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.pause_pending = True
    slot.cancel_pending = True  # modal 操作把 cancel_pending 也 set 了
    # pause_state_path 留 None

    sup._finish_slot(slot, rc=0)

    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task is not None
    assert task["status"] == "canceled"


def test_finish_slot_canceled_takes_precedence_over_rc(env) -> None:
    """cancel_pending=True 优先于 rc — rc=0 也标 canceled。"""
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.cancel_pending = True

    sup._finish_slot(slot, rc=0)
    assert _read_status(env["db"], tid) == "canceled"


def test_finish_slot_done_when_rc_zero(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)

    sup._finish_slot(slot, rc=0)
    assert _read_status(env["db"], tid) == "done"


def test_finish_slot_failed_when_rc_nonzero(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)

    sup._finish_slot(slot, rc=7)
    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task["status"] == "failed"
    assert task["exit_code"] == 7


def _read_status(db_path: Path, tid: int) -> str:
    with db.connection_for(db_path) as conn:
        t = db.get_task(conn, tid)
    return str(t["status"]) if t else ""


# ---- pause() 状态机校验 -----------------------------------------------------


def test_pause_returns_false_for_unknown_task(env) -> None:
    sup = _new_sup(env)
    ok, reason = sup.pause(99999)
    assert ok is False
    assert "not found" in reason


def test_pause_returns_false_for_pending_task(env) -> None:
    sup = _new_sup(env)
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
    ok, reason = sup.pause(tid)
    assert ok is False
    assert "not running" in reason


def test_pause_returns_false_when_train_loop_not_started(env) -> None:
    """ADR §8.1 defense-in-depth: 未进入 train_loop 不允许 pause。"""
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.train_loop_started = False
    sup._slots = [slot]

    ok, reason = sup.pause(tid)
    assert ok is False
    assert "train loop not started" in reason


def test_pause_succeeds_when_train_loop_started(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.train_loop_started = True
    sup._slots = [slot]

    ok, _reason = sup.pause(tid)
    assert ok is True
    assert slot.pause_pending is True
    # 确认确实向子进程发过信号
    slot.proc.send_signal.assert_called_once()


def test_pause_rejects_when_already_pausing(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.train_loop_started = True
    slot.pause_pending = True
    sup._slots = [slot]

    ok, reason = sup.pause(tid)
    assert ok is False
    assert "already pending" in reason


def test_pause_rejects_when_cancel_pending(env) -> None:
    sup = _new_sup(env)
    tid = _populate_running(env)
    slot = _make_slot_with_proc(tid)
    slot.train_loop_started = True
    slot.cancel_pending = True
    sup._slots = [slot]

    ok, reason = sup.pause(tid)
    assert ok is False
    assert "canceled" in reason


# ---- cancel paused task → canceled ------------------------------------------


def test_cancel_paused_task_changes_to_canceled_keeps_files(env, tmp_path) -> None:
    """ADR §5.5 + Addendum 2 修订: paused → canceled 清 paused_* 字段，
    但恢复点文件**保留**（canceled task 之后仍可 resume）。"""
    sup = _new_sup(env)
    state_pt = tmp_path / "auto_epoch_state.pt"
    state_cfg = tmp_path / "auto_epoch_state.config.json"
    state_pt.write_bytes(b"fake")
    state_cfg.write_text("{}", encoding="utf-8")

    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(
            conn, tid,
            status="paused",
            paused_state_path=str(state_pt),
            paused_config_path=str(state_cfg),
            last_state_path=str(state_pt),
            last_config_path=str(state_cfg),
            paused_step=100,
            paused_at=time.time(),
        )

    assert sup.cancel(tid) is True

    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task["status"] == "canceled"
    assert task["paused_state_path"] is None    # paused_* 字段清掉
    assert task["last_state_path"] == str(state_pt)  # 恢复点字段保留
    assert state_pt.exists()   # 文件保留（Addendum 2）
    assert state_cfg.exists()


def test_cancel_paused_task_robust_to_missing_files(env) -> None:
    """文件已被外部删 / 路径无效时 cancel 仍标 canceled，不抛错。"""
    sup = _new_sup(env)
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(
            conn, tid,
            status="paused",
            paused_state_path="/nonexistent/path.pt",
            paused_config_path="/nonexistent/path.config.json",
            paused_step=100,
        )
    assert sup.cancel(tid) is True
    assert _read_status(env["db"], tid) == "canceled"


# ---- queue_held dispatch 影响 -----------------------------------------------


def test_queue_held_returns_db_value(env) -> None:
    sup = _new_sup(env)
    assert sup._queue_held() is False
    with db.connection_for(env["db"]) as conn:
        db.set_queue_held(conn, True)
    assert sup._queue_held() is True
