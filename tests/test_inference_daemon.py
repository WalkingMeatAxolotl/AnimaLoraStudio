"""inference_daemon 协议 + supervisor 接入测试（commit 9）。

通过 mock daemon 子进程脚本验证：
  1. spawn / ready / submit_task / done / stop 协议路径
  2. supervisor 把 generate task 推给 daemon（不占 SLOT_TRAIN）
  3. cancel pending generate / running generate（kill daemon）
  4. daemon 进程意外退出 → active task 标 failed

不跑真实模型 —— 替换 _DAEMON_SCRIPT 指向一个回声脚本。
"""
from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from studio import db
from studio.services import inference_daemon as _daemon_mod
from studio.services.inference_daemon import (
    InferenceDaemon,
    STATE_BUSY,
    STATE_IDLE,
    STATE_STOPPED,
    reset_daemon_for_test,
)


# ---------- mock daemon 脚本（无需模型，纯协议） ---------------------------------

_MOCK_DAEMON = textwrap.dedent(
    """
    import base64, json, sys, os, time
    sys.stdout.write(json.dumps({"id":"_evt","kind":"ready"}) + "\\n")
    sys.stdout.flush()
    fake_b64 = base64.b64encode(b"FAKE-PNG").decode("ascii")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        action = msg.get("action")
        rid = msg.get("id", "")
        if action == "ping":
            sys.stdout.write(json.dumps({"id":rid,"kind":"pong"}) + "\\n")
            sys.stdout.flush()
        elif action == "generate":
            tid = msg.get("task_id", 0)
            sys.stdout.write(json.dumps({"id":rid,"kind":"started","task_id":tid}) + "\\n")
            sys.stdout.flush()
            # 出 1 张图：bytes 走协议 b64 字段（commit 10），不写磁盘
            sys.stdout.write(json.dumps({"id":rid,"kind":"image_done","task_id":tid,"filename":"fake.png","path":"/anima_gen_%d/fake.png" % tid,"step":1,"total":1,"image_b64":fake_b64,"byte_size":8}) + "\\n")
            sys.stdout.flush()
            sys.stdout.write(json.dumps({"id":rid,"kind":"done","task_id":tid}) + "\\n")
            sys.stdout.flush()
        elif action == "unload":
            sys.stdout.write(json.dumps({"id":"_evt","kind":"unloaded"}) + "\\n")
            sys.stdout.flush()
        elif action == "crash":
            os._exit(1)
    """
).strip()


@pytest.fixture
def mock_daemon_script(tmp_path: Path) -> Path:
    p = tmp_path / "mock_daemon.py"
    p.write_text(_MOCK_DAEMON, encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _reset_daemon():
    """每个 test 拿到干净的 daemon singleton。"""
    reset_daemon_for_test()
    yield
    reset_daemon_for_test()


def _wait_for(predicate, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------- 协议层测试 ---------------------------------------------------------


def test_daemon_starts_and_reaches_idle(mock_daemon_script: Path) -> None:
    d = InferenceDaemon(script_path=mock_daemon_script)
    d.start()
    try:
        assert d.state == STATE_IDLE
        assert d.is_alive
    finally:
        d.stop()
    assert d.state == STATE_STOPPED


def test_submit_task_runs_to_done(mock_daemon_script: Path) -> None:
    from studio.services import generate_cache

    generate_cache.clear_all()
    d = InferenceDaemon(script_path=mock_daemon_script)
    d.start()
    events: list[dict[str, Any]] = []
    try:
        d.submit_task(
            task_id=42, config={"prompts": ["a"]}, output_dir="/tmp/x",
            on_event=events.append,
        )
        assert d.state == STATE_BUSY
        assert _wait_for(
            lambda: any(e.get("kind") == "done" for e in events), timeout=3
        ), f"events={events}"
        # 回 idle
        assert _wait_for(lambda: d.state == STATE_IDLE, timeout=2)
    finally:
        d.stop()
    kinds = [e.get("kind") for e in events]
    assert "started" in kinds
    assert "image_done" in kinds
    assert "done" in kinds
    # task_id 透传
    for e in events:
        assert e.get("task_id") == 42

    # commit 10：bytes 已入 server-side cache（mock daemon 推的是 b"FAKE-PNG" b64）
    assert generate_cache.get_image(42, "fake.png") == b"FAKE-PNG"
    # 转发给 callback 的事件不应该再带 image_b64（已被 reader 剥掉）
    image_done_events = [e for e in events if e.get("kind") == "image_done"]
    assert image_done_events
    for e in image_done_events:
        assert "image_b64" not in e
    generate_cache.clear_all()


def test_daemon_crash_emits_error(mock_daemon_script: Path) -> None:
    d = InferenceDaemon(script_path=mock_daemon_script)
    d.start()
    events: list[dict[str, Any]] = []
    try:
        # 用一个不会自然 done 的 action 让 daemon 处于 BUSY，再 crash
        # 这里直接发 crash action（mock daemon 用 _exit）
        with d._lock:  # type: ignore[attr-defined]
            d._req_seq += 1
            req_id = "task-99-x"
            from studio.services.inference_daemon import _ActiveTask
            d._active = _ActiveTask(
                task_id=99, request_id=req_id, on_event=events.append,
            )
            d._state = STATE_BUSY
            stdin = d._proc.stdin  # type: ignore[union-attr]
        stdin.write(json.dumps({"id": req_id, "action": "crash"}) + "\n")
        stdin.flush()
        assert _wait_for(
            lambda: any(e.get("kind") == "error" for e in events), timeout=3
        ), f"events={events}"
    finally:
        d.stop()
    assert d.state == STATE_STOPPED


def test_global_listener_receives_events(mock_daemon_script: Path) -> None:
    d = InferenceDaemon(script_path=mock_daemon_script)
    seen: list[dict[str, Any]] = []
    d.add_global_listener(seen.append)
    d.start()
    try:
        # ready 是 daemon 起来的第一个 _evt
        assert _wait_for(
            lambda: any(e.get("kind") == "ready" for e in seen), timeout=2
        )
    finally:
        d.stop()
    # 进程退出 → stopped 事件
    assert _wait_for(
        lambda: any(e.get("kind") == "stopped" for e in seen), timeout=2
    ), f"seen={seen}"


# ---------- supervisor 接入测试 -----------------------------------------------


def _make_generate_task(env: dict, *, cfg_overrides: dict[str, Any] | None = None) -> int:
    """造一个 task_type=generate 的 pending task + 写 config.json。"""
    cfg_dir = env["configs"]
    cfg_path = cfg_dir / "gen.json"
    cfg = {
        "transformer_path": "/x", "vae_path": "/y", "text_encoder_path": "/z",
        "prompts": ["a"], "output_dir": str(env["configs"] / "out"),
    }
    if cfg_overrides:
        cfg.update(cfg_overrides)
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="g", config_name="gen", priority=0)
        db.update_task(conn, tid, task_type="generate", config_path=str(cfg_path))
    return tid


def _patch_singleton(d: InferenceDaemon, monkeypatch) -> None:
    """让 supervisor 拿到 mock daemon 实例（不要走 spawn 真 daemon 路径）。"""
    _daemon_mod._INSTANCE = d  # type: ignore[attr-defined]


@pytest.fixture
def env(tmp_path: Path):
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    logs = tmp_path / "logs"
    configs = tmp_path / "configs"
    logs.mkdir()
    configs.mkdir()
    return {"db": db_path, "logs": logs, "configs": configs}


def _task_status(db_path: Path, task_id: int) -> str:
    with db.connection_for(db_path) as conn:
        t = db.get_task(conn, task_id)
    return (t or {}).get("status", "?")


def test_supervisor_dispatches_generate_to_daemon(env, mock_daemon_script, monkeypatch):
    from studio.supervisor import Supervisor

    d = InferenceDaemon(script_path=mock_daemon_script)
    _patch_singleton(d, monkeypatch)

    events: list[dict[str, Any]] = []
    sup = Supervisor(
        on_event=events.append,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )

    tid = _make_generate_task(env)
    sup.start()
    try:
        assert _wait_for(
            lambda: _task_status(env["db"], tid) == "done", timeout=10
        ), f"final status={_task_status(env['db'], tid)}; events={events}"
    finally:
        sup.stop()

    statuses = [e["status"] for e in events if e.get("task_id") == tid]
    assert "running" in statuses
    assert "done" in statuses

    # commit 13：supervisor 应该至少 emit 一次 daemon_state_changed
    daemon_evts = [e for e in events if e.get("type") == "daemon_state_changed"]
    assert daemon_evts, "expected daemon_state_changed events; got none"
    # 至少一个 busy=True（提交后立刻 emit）和一个 busy=False（done 后）
    busy_states = [e["busy"] for e in daemon_evts]
    assert True in busy_states
    assert False in busy_states


def test_supervisor_cancel_pending_generate(env, mock_daemon_script, monkeypatch):
    from studio.supervisor import Supervisor

    d = InferenceDaemon(script_path=mock_daemon_script)
    _patch_singleton(d, monkeypatch)
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
        poll_interval=0.05,
    )
    tid = _make_generate_task(env)
    # 不 start sup —— pending 直接 cancel
    assert sup.cancel(tid) is True
    assert _task_status(env["db"], tid) == "canceled"


def test_supervisor_train_dispatch_skips_generate(env, monkeypatch):
    """train slot 的 dispatch 不能误拉 generate task（必须留给 daemon）。"""
    from studio.supervisor import Supervisor

    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    # 一个 generate pending
    tid_gen = _make_generate_task(env)
    # _next_pending_task_in 只拉 train/reg_ai，应该是 None
    assert sup._next_pending_task_in(("train", "reg_ai")) is None
    # 拉 generate 才能找到
    found = sup._next_pending_task_in(("generate",))
    assert found is not None and found["id"] == tid_gen
