"""PR-1 C6 — trace_id 跨进程贯穿验证。

覆盖：
  - migration v10 加 tasks.request_trace_id 列
  - db.create_task 自动写当前 ContextVar trace_id
  - 直接 INSERT INTO tasks (training.py:583) 同样写
  - supervisor._spawn_task 注入 ANIMA_TRACE_ID + ANIMA_PROCESS_NAME env
  - worker bootstrap 读 env bind_trace_id
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from studio import db
from studio.infrastructure.logging import (
    PROCESS_ENV,
    TRACE_ENV,
    _reset_for_tests,
    bind_trace_id,
    get_trace_id,
    reset_trace_id,
)


# ── migration v10 ─────────────────────────────────────────────────────────


def test_migration_v10_adds_request_trace_id_column(tmp_path: Path) -> None:
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    with db.connection_for(dbfile) as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "request_trace_id" in cols, "v10 必须加 request_trace_id 列"


def test_migration_v10_is_idempotent(tmp_path: Path) -> None:
    """两次 init_db 不应崩 — ALTER TABLE ADD COLUMN 已存在时容错。"""
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    db.init_db(dbfile)  # 第二次跑
    with db.connection_for(dbfile) as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "request_trace_id" in cols


# ── db.create_task 自动写 trace_id ────────────────────────────────────────


def test_create_task_writes_bound_trace_id(tmp_path: Path) -> None:
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    _reset_for_tests()
    token = bind_trace_id("test-trace-abc123def456789012")
    try:
        with db.connection_for(dbfile) as conn:
            tid = db.create_task(conn, name="t1", config_name="c1")
            row = conn.execute(
                "SELECT request_trace_id FROM tasks WHERE id = ?", (tid,)
            ).fetchone()
            assert row["request_trace_id"] == "test-trace-abc123def456789012"
    finally:
        reset_trace_id(token)


def test_create_task_uses_bg_prefix_when_no_trace(tmp_path: Path) -> None:
    """无 contextvar bind（直接 CLI / 测试 / 后台触发）→ bg-{uuid}。"""
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    _reset_for_tests()
    assert get_trace_id() is None
    with db.connection_for(dbfile) as conn:
        tid = db.create_task(conn, name="t1", config_name="c1")
        row = conn.execute(
            "SELECT request_trace_id FROM tasks WHERE id = ?", (tid,)
        ).fetchone()
        assert row["request_trace_id"].startswith("bg-"), (
            f"无 contextvar 时应 bg- 前缀，实际 {row['request_trace_id']!r}"
        )


def test_create_task_isolates_trace_ids_between_calls(tmp_path: Path) -> None:
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    _reset_for_tests()

    token1 = bind_trace_id("trace-id-aaaa1111bbbb2222cccc")
    with db.connection_for(dbfile) as conn:
        t1 = db.create_task(conn, name="t1", config_name="c1")
    reset_trace_id(token1)

    token2 = bind_trace_id("trace-id-dddd3333eeee4444ffff")
    with db.connection_for(dbfile) as conn:
        t2 = db.create_task(conn, name="t2", config_name="c2")
    reset_trace_id(token2)

    with db.connection_for(dbfile) as conn:
        rows = list(conn.execute(
            "SELECT id, request_trace_id FROM tasks ORDER BY id"
        ))
    assert rows[0]["request_trace_id"] == "trace-id-aaaa1111bbbb2222cccc"
    assert rows[1]["request_trace_id"] == "trace-id-dddd3333eeee4444ffff"


# ── supervisor._spawn_task 注入 env ─────────────────────────────────────


def test_spawn_task_injects_trace_env_to_subprocess(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """_spawn_task 调 _popen 时 extra_env 必须含 TRACE_ENV + PROCESS_ENV。"""
    from studio.supervisor.core import Supervisor

    captured_env = {}

    def fake_popen(self_, cmd, log_fp, extra_env=None):
        captured_env.update(extra_env or {})
        proc = MagicMock()
        proc.pid = 12345
        proc.poll = lambda: None
        return proc

    monkeypatch.setattr(Supervisor, "_popen", fake_popen)
    monkeypatch.setattr(Supervisor, "_freeze_task_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(Supervisor, "_resolve_task_config_path",
                        lambda self_, t: tmp_path / "cfg.yaml")
    monkeypatch.setattr(Supervisor, "_write_task_running_to_db",
                        lambda self_, t, pid, msp: None)
    monkeypatch.setattr(Supervisor, "_make_task_log_callback",
                        lambda self_, slot, tid: lambda line: None)
    monkeypatch.setattr(Supervisor, "_make_monitor_callback",
                        lambda self_, tid: lambda d: None)
    monkeypatch.setattr("studio.supervisor.core.LogTailer",
                        lambda *a, **kw: MagicMock(start=lambda: None))
    monkeypatch.setattr("studio.supervisor.core.MonitorStatePoller",
                        lambda *a, **kw: MagicMock(start=lambda: None))
    monkeypatch.setattr("studio.supervisor.core._resolve_monitor_state_path",
                        lambda t: tmp_path / "monitor.json")
    # task_log_path 走 paths.TASKS_DIR；改到 tmp 不污染 studio_data/
    from studio.infrastructure import paths as _paths
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")

    # 假 config 文件存在
    (tmp_path / "cfg.yaml").write_text("dummy", encoding="utf-8")

    sup = Supervisor(on_event=lambda evt: None)
    sup._logs_dir = tmp_path
    slot = MagicMock()
    slot.name = "TRAIN"
    task = {
        "id": 42,
        "task_type": "tag",
        "request_trace_id": "test-trace-from-task-row-xyz",
        "config_name": "cfg",
    }
    sup._spawn_task(slot, task)

    assert TRACE_ENV in captured_env
    assert captured_env[TRACE_ENV] == "test-trace-from-task-row-xyz"
    assert PROCESS_ENV in captured_env
    assert captured_env[PROCESS_ENV] == "worker:tag/42"


def test_spawn_task_generates_bg_trace_when_task_has_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """task.request_trace_id 是 None（老 task / 测试）→ bg-{uuid} 兜底。"""
    from studio.supervisor.core import Supervisor

    captured_env = {}

    def fake_popen(self_, cmd, log_fp, extra_env=None):
        captured_env.update(extra_env or {})
        proc = MagicMock(); proc.pid = 1; proc.poll = lambda: None
        return proc

    monkeypatch.setattr(Supervisor, "_popen", fake_popen)
    monkeypatch.setattr(Supervisor, "_freeze_task_snapshot", lambda *a, **kw: None)
    monkeypatch.setattr(Supervisor, "_resolve_task_config_path",
                        lambda self_, t: tmp_path / "cfg.yaml")
    monkeypatch.setattr(Supervisor, "_write_task_running_to_db",
                        lambda self_, t, pid, msp: None)
    monkeypatch.setattr(Supervisor, "_make_task_log_callback",
                        lambda self_, slot, tid: lambda line: None)
    monkeypatch.setattr(Supervisor, "_make_monitor_callback",
                        lambda self_, tid: lambda d: None)
    monkeypatch.setattr("studio.supervisor.core.LogTailer",
                        lambda *a, **kw: MagicMock(start=lambda: None))
    monkeypatch.setattr("studio.supervisor.core.MonitorStatePoller",
                        lambda *a, **kw: MagicMock(start=lambda: None))
    monkeypatch.setattr("studio.supervisor.core._resolve_monitor_state_path",
                        lambda t: tmp_path / "monitor.json")
    # task_log_path 走 paths.TASKS_DIR；改到 tmp 不污染 studio_data/
    from studio.infrastructure import paths as _paths
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")
    (tmp_path / "cfg.yaml").write_text("dummy", encoding="utf-8")

    sup = Supervisor(on_event=lambda evt: None)
    sup._logs_dir = tmp_path
    task = {"id": 7, "config_name": "cfg"}  # 无 request_trace_id
    sup._spawn_task(MagicMock(name="TRAIN"), task)

    assert captured_env[TRACE_ENV].startswith("bg-"), (
        f"无 task.request_trace_id 应兜底 bg-，实际 {captured_env[TRACE_ENV]!r}"
    )
    # task_type 缺省 → train
    assert captured_env[PROCESS_ENV] == "worker:train/7"


# ── worker_main bind trace_id from env ─────────────────────────────────


def test_worker_main_binds_trace_id_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.workers import _base
    from studio.infrastructure import logging as _slog

    captured = {}

    def fake_setup(process, **kw):
        captured["process"] = process
        _slog._CONFIGURED_PROCESSES.add(process)

    def fake_run(job_id):
        # 在 run 内部检查 contextvar 是否已经 bind
        captured["trace_in_run"] = get_trace_id()
        return 0

    monkeypatch.setattr(_slog, "setup_logging", fake_setup)
    monkeypatch.setattr(sys, "argv", ["tag_worker.py", "--job-id", "99"])
    monkeypatch.setenv(TRACE_ENV, "supervisor-injected-trace-xyz")
    monkeypatch.setenv(PROCESS_ENV, "worker:tag/99")
    _reset_for_tests()

    with pytest.raises(SystemExit) as excinfo:
        _base.worker_main(fake_run)
    assert excinfo.value.code == 0
    assert captured["trace_in_run"] == "supervisor-injected-trace-xyz", (
        "worker_main 必须 bind_trace_id from env，让 run() 内 logger.x 自动带"
    )


def test_worker_main_generates_trace_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无 ANIMA_TRACE_ID env（独立测 worker）→ new_trace_id 兜底，不 crash。"""
    from studio.workers import _base
    from studio.infrastructure import logging as _slog

    captured = {}

    def fake_setup(process, **kw):
        _slog._CONFIGURED_PROCESSES.add(process)

    def fake_run(job_id):
        captured["trace"] = get_trace_id()
        return 0

    monkeypatch.setattr(_slog, "setup_logging", fake_setup)
    monkeypatch.setattr(sys, "argv", ["preprocess_worker.py", "--job-id", "5"])
    monkeypatch.delenv(TRACE_ENV, raising=False)
    monkeypatch.delenv(PROCESS_ENV, raising=False)
    _reset_for_tests()

    with pytest.raises(SystemExit):
        _base.worker_main(fake_run)

    assert captured["trace"] is not None
    assert len(captured["trace"]) == 24, "兜底应是 new_trace_id (24 字符 hex)"


# ── 日志目标态刀 1：_spawn_job / daemon / _popen 的 env 注入 ─────────────────


def test_spawn_job_injects_trace_and_process_env(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    """job 路径之前完全不注入 trace/process；现与 _spawn_task 对齐（bg- 兜底）。"""
    import contextlib

    from studio.supervisor.core import Supervisor

    captured_env: dict = {}

    def fake_popen(self_, cmd, log_fp, extra_env=None):
        captured_env.update(extra_env or {})
        proc = MagicMock(); proc.pid = 3; proc.poll = lambda: None
        return proc

    monkeypatch.setattr(Supervisor, "_popen", fake_popen)
    monkeypatch.setattr(Supervisor, "_make_job_log_callback",
                        lambda self_, *a: (lambda line: None))
    monkeypatch.setattr("studio.supervisor.core.LogTailer",
                        lambda *a, **kw: MagicMock(start=lambda: None))
    monkeypatch.setattr("studio.supervisor.core.db.connection_for",
                        lambda *_a, **_k: contextlib.nullcontext(MagicMock()))
    monkeypatch.setattr("studio.supervisor.core.project_jobs.mark_running", lambda *a, **k: None)

    sup = Supervisor(on_event=lambda evt: None, job_cmd_builder=lambda job: ["python", "-c", "0"])
    job = {"id": 9, "kind": "tag", "project_id": 1, "version_id": None,
           "log_path": str(tmp_path / "run.log")}
    sup._spawn_job(MagicMock(name="DATA"), job)

    assert captured_env[TRACE_ENV].startswith("bg-")
    assert captured_env[PROCESS_ENV] == "worker:tag/9"


def test_popen_injects_debug_console_level_by_default(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    """子进程 stderr 即 run.log：记录不过滤 → 注入 ANIMA_LOG_LEVEL=DEBUG；外部显式设的值不覆盖。"""
    from studio.infrastructure.logging import LOG_LEVEL_ENV
    from studio.supervisor import core as core_mod

    seen: dict = {}

    class _FakeProc:
        pid = 1

    def fake_popen(cmd, **kw):
        seen.update(kw["env"])
        return _FakeProc()

    monkeypatch.setattr(core_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(core_mod._secrets, "load", lambda: MagicMock(
        training=MagicMock(ram_guard=True), wandb=MagicMock(enabled=False)))
    sup = core_mod.Supervisor(on_event=lambda evt: None)

    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    sup._popen(["python", "-c", "0"], open(tmp_path / "a.log", "wb"))
    assert seen[LOG_LEVEL_ENV] == "DEBUG"

    seen.clear()
    monkeypatch.setenv(LOG_LEVEL_ENV, "WARNING")
    sup._popen(["python", "-c", "0"], open(tmp_path / "b.log", "wb"))
    assert seen[LOG_LEVEL_ENV] == "WARNING"


def test_daemon_start_injects_log_level_trace_process(monkeypatch: pytest.MonkeyPatch) -> None:
    from studio.infrastructure.logging import LOG_LEVEL_ENV
    from studio.services.inference import daemon as daemon_mod

    seen: dict = {}

    def fake_popen(cmd, **kw):
        seen.update(kw["env"])
        raise RuntimeError("stop here")  # 不真起进程

    monkeypatch.setattr(daemon_mod.subprocess, "Popen", fake_popen)
    monkeypatch.delenv(LOG_LEVEL_ENV, raising=False)
    monkeypatch.delenv(TRACE_ENV, raising=False)
    monkeypatch.delenv(PROCESS_ENV, raising=False)
    d = daemon_mod.InferenceDaemon()
    with pytest.raises(RuntimeError):
        d.start()
    assert seen[LOG_LEVEL_ENV] == "DEBUG"
    assert seen[TRACE_ENV].startswith("bg-")
    assert seen[PROCESS_ENV] == "anima_daemon"
