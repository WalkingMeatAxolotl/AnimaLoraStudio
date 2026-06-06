"""InferenceDaemon idle timeout 自动 unload 测试。

策略：不真启 daemon 子进程，直接对 InferenceDaemon 实例做白盒测试 —
- 用 SimpleNamespace 伪造 `_proc`（非 None 满足 reschedule 检查）
- 手动设 state / model_loaded 模拟各状态
- 用极短 timeout（0.05-0.1s）等真实 Timer 触发；不用 sleep 长时间

覆盖：
- timeout=0 时不启动 timer
- model 未 load 时不启动
- state=BUSY 时不启动
- IDLE + loaded + timeout>0 → 启动；到期调 request_unload
- 状态切换（BUSY→IDLE / unloaded / stopped）正确 cancel / 重启 timer
- sync_idle_timeout_from_secrets 从 settings 读
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import pytest

from studio.services.inference.daemon import (
    InferenceDaemon,
    STATE_BUSY,
    STATE_IDLE,
    STATE_STOPPED,
    STATE_UNLOADING,
)


# 让 `_proc is not None` 满足，但不真发协议（reschedule 只看 is None；request_unload
# 会拿 _proc.stdin —— 测试里用 spy 替换 request_unload 不走到那一步）。
def _fake_proc() -> SimpleNamespace:
    return SimpleNamespace(stdin=None, poll=lambda: None)


def _make_daemon_in_state(
    *,
    state: str = STATE_IDLE,
    model_loaded: bool = True,
    with_proc: bool = True,
) -> InferenceDaemon:
    d = InferenceDaemon()
    if with_proc:
        d._proc = _fake_proc()  # type: ignore[assignment]
    d._state = state
    d._model_loaded = model_loaded
    return d


def _wait_until(predicate, timeout=2.0, interval=0.01) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------- arming 条件 -------------------------------------------------------


def test_timer_does_not_arm_when_timeout_zero() -> None:
    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(0)
    assert d._idle_timer is None


def test_timer_does_not_arm_when_model_not_loaded() -> None:
    d = _make_daemon_in_state(model_loaded=False)
    d.set_idle_timeout_seconds(5.0)
    assert d._idle_timer is None


def test_timer_does_not_arm_when_busy() -> None:
    d = _make_daemon_in_state(state=STATE_BUSY)
    d.set_idle_timeout_seconds(5.0)
    assert d._idle_timer is None


def test_timer_does_not_arm_when_no_proc() -> None:
    d = _make_daemon_in_state(with_proc=False)
    d.set_idle_timeout_seconds(5.0)
    assert d._idle_timer is None


def test_timer_arms_when_idle_and_loaded() -> None:
    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(60.0)
    try:
        assert d._idle_timer is not None
        assert d._idle_timer.is_alive()
    finally:
        if d._idle_timer is not None:
            d._idle_timer.cancel()


# ---------- 触发 unload -------------------------------------------------------


def test_timer_fires_request_unload(monkeypatch: pytest.MonkeyPatch) -> None:
    """timer 到期且仍 idle+loaded → 调 request_unload。"""
    d = _make_daemon_in_state()
    calls: list[None] = []

    def fake_unload() -> None:
        calls.append(None)

    monkeypatch.setattr(d, "request_unload", fake_unload)
    d.set_idle_timeout_seconds(0.05)

    assert _wait_until(lambda: len(calls) >= 1, timeout=1.0)
    assert len(calls) == 1


def test_timer_does_not_fire_if_state_changed_before_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """timer 到期瞬间状态已变（race window）→ 不发 unload。

    模拟：timer arm 后用户 submit_task 进 BUSY。我们手动改 state 不调
    reschedule，让 timer 残留然后到期；_on_idle_timeout 自检发现 BUSY 应放弃。
    """
    d = _make_daemon_in_state()
    calls: list[None] = []
    monkeypatch.setattr(d, "request_unload", lambda: calls.append(None))
    d.set_idle_timeout_seconds(0.05)
    # 不走 reschedule，直接改 state 模拟 race（race window 内 callback 已排队）
    with d._lock:
        d._state = STATE_BUSY
    time.sleep(0.2)  # 让 timer 跑完
    assert calls == []


def test_set_idle_timeout_zero_cancels_existing_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已经 arm 的 timer：set_idle_timeout_seconds(0) 立刻 cancel + 不再触发。"""
    d = _make_daemon_in_state()
    calls: list[None] = []
    monkeypatch.setattr(d, "request_unload", lambda: calls.append(None))
    d.set_idle_timeout_seconds(0.1)
    assert d._idle_timer is not None
    d.set_idle_timeout_seconds(0)
    assert d._idle_timer is None
    time.sleep(0.2)
    assert calls == []


# ---------- 状态切换钩子 ------------------------------------------------------


def test_busy_then_idle_rearms_timer(monkeypatch: pytest.MonkeyPatch) -> None:
    """BUSY → IDLE（task done）→ timer 重新启动。"""
    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(60.0)
    first_timer = d._idle_timer
    assert first_timer is not None

    # 模拟 submit_task：进 BUSY
    with d._lock:
        d._state = STATE_BUSY
        d._reschedule_idle_timer_locked()
    assert d._idle_timer is None  # busy 时 cancel

    # 模拟 task done：回 IDLE
    with d._lock:
        d._state = STATE_IDLE
        d._reschedule_idle_timer_locked()
    try:
        assert d._idle_timer is not None
        assert d._idle_timer is not first_timer  # 是新的 timer
    finally:
        if d._idle_timer is not None:
            d._idle_timer.cancel()


def test_unloaded_event_cancels_timer() -> None:
    """收到 unloaded → 模型走了 → timer cancel。"""
    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(60.0)
    assert d._idle_timer is not None
    # 模拟 _handle_event 的 unloaded 分支
    with d._lock:
        d._state = STATE_IDLE
        d._model_loaded = False
        d._reschedule_idle_timer_locked()
    assert d._idle_timer is None


def test_unloading_state_cancels_timer() -> None:
    """request_unload 进 UNLOADING → timer cancel。"""
    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(60.0)
    assert d._idle_timer is not None
    with d._lock:
        d._state = STATE_UNLOADING
        d._reschedule_idle_timer_locked()
    assert d._idle_timer is None


def test_stopped_cancels_timer() -> None:
    """proc 退出 → STOPPED → timer cancel。"""
    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(60.0)
    assert d._idle_timer is not None
    with d._lock:
        d._proc = None
        d._state = STATE_STOPPED
        d._model_loaded = False
        d._reschedule_idle_timer_locked()
    assert d._idle_timer is None


# ---------- secrets 同步 ------------------------------------------------------


def test_sync_idle_timeout_from_secrets_reads_minutes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sync 从 secrets.generate.idle_timeout_minutes 读，换算成秒。"""
    from studio.infrastructure import secrets as _secrets

    fake = _secrets.Secrets()
    fake.generate.idle_timeout_minutes = 7
    monkeypatch.setattr(_secrets, "load", lambda: fake)

    d = _make_daemon_in_state()
    d.sync_idle_timeout_from_secrets()
    assert d._idle_timeout_seconds == 7 * 60.0
    if d._idle_timer is not None:
        d._idle_timer.cancel()


def test_sync_idle_timeout_zero_minutes_disables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """secrets 里 idle_timeout_minutes=0 → timer 关闭。"""
    from studio.infrastructure import secrets as _secrets

    fake = _secrets.Secrets()
    fake.generate.idle_timeout_minutes = 0
    monkeypatch.setattr(_secrets, "load", lambda: fake)

    d = _make_daemon_in_state()
    d.sync_idle_timeout_from_secrets()
    assert d._idle_timeout_seconds == 0
    assert d._idle_timer is None


def test_sync_idle_timeout_handles_secrets_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """secrets.load() 抛错时 sync 不改当前值（不抛出）。"""
    from studio.infrastructure import secrets as _secrets

    d = _make_daemon_in_state()
    d.set_idle_timeout_seconds(120.0)
    assert d._idle_timeout_seconds == 120.0

    def boom() -> Any:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(_secrets, "load", boom)
    d.sync_idle_timeout_from_secrets()  # 不抛
    assert d._idle_timeout_seconds == 120.0  # 未变
    if d._idle_timer is not None:
        d._idle_timer.cancel()
