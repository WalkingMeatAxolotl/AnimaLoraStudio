"""ADR 0006 Addendum 2 — terminal-resume（failed/canceled task 可恢复）。

覆盖五块：

1. migration v13：tasks.last_state_* 列存在、默认 NULL。
2. supervisor `_persist_last_state`：auto_epoch_backup_written 事件经
   `_make_task_log_callback` 落 DB（恢复点跨进程 / 重启可查的关键）。
3. `_clear_pause_fields`：只清 db 字段、**不删文件**（Addendum 2 修订）；
   cancel paused 同样保留文件。
4. resume endpoint 状态放宽：failed/canceled + 恢复点在盘 → pending 且
   last_* 复制进 paused_*（复用 cmd_builder 管道）；done / 无恢复点拒绝。
5. `_is_resumable` 信号 + DELETE 清恢复点目录。

原子写盘（save_training_state / write_config_snapshot tmp+replace）的测试
在 test_state_atomic_save.py（依赖 torch，单独文件让无 torch 环境可跳过）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from studio import db
from studio.supervisor import Supervisor, _Slot


@pytest.fixture
def env(tmp_path: Path):
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    logs = tmp_path / "logs"
    configs = tmp_path / "configs"
    logs.mkdir()
    configs.mkdir()
    return {"db": db_path, "logs": logs, "configs": configs, "root": tmp_path}


def _new_sup(env) -> Supervisor:
    return Supervisor(
        on_event=lambda _: None,
        cmd_builder=lambda *_: ["echo"],
        db_path=env["db"],
        logs_dir=env["logs"],
        configs_dir=env["configs"],
        poll_interval=10,
    )


def _create_task(env, status: str = "pending", **fields: Any) -> int:
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        if status != "pending" or fields:
            db.update_task(conn, tid, status=status, **fields)
    return tid


def _get_task(env, tid: int) -> dict[str, Any]:
    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task is not None
    return task


def _write_recovery_pair(root: Path, tid: int) -> tuple[Path, Path]:
    """在 <root>/state/task_<tid>/ 下放一对 auto backup 文件。"""
    state_dir = root / "state" / f"task_{tid}"
    state_dir.mkdir(parents=True, exist_ok=True)
    pt = state_dir / "auto_epoch_state.pt"
    cfg = state_dir / "auto_epoch_state.config.json"
    pt.write_bytes(b"fake state")
    cfg.write_text("{}", encoding="utf-8")
    return pt, cfg


# ---------------------------------------------------------------------------
# migration v13
# ---------------------------------------------------------------------------


def test_migration_adds_last_state_columns(env) -> None:
    with db.connection_for(env["db"]) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
    for col in ("last_state_path", "last_config_path",
                "last_state_epoch", "last_state_step"):
        assert col in cols, f"missing column {col}"


def test_last_state_columns_nullable_by_default(env) -> None:
    tid = _create_task(env)
    task = _get_task(env, tid)
    for col in ("last_state_path", "last_config_path",
                "last_state_epoch", "last_state_step"):
        assert task[col] is None


# ---------------------------------------------------------------------------
# supervisor: auto_epoch_backup_written → DB 持久化
# ---------------------------------------------------------------------------


def _make_slot(tid: int) -> _Slot:
    slot = _Slot(name="train")
    slot.kind = "task"
    slot.id = tid
    slot.proc = MagicMock()
    return slot


def test_auto_epoch_backup_event_persists_to_db(env) -> None:
    """事件不只更新内存 slot，还要写 tasks.last_state_*（关机后可 resume 的根基）。"""
    sup = _new_sup(env)
    tid = _create_task(env, status="running")
    slot = _make_slot(tid)
    cb = sup._make_task_log_callback(slot, tid)

    payload = {
        "state_path": "/out/state/task_1/auto_epoch_state.pt",
        "config_path": "/out/state/task_1/auto_epoch_state.config.json",
        "epoch": 3,
        "step": 300,
    }
    cb(f"__EVENT__:auto_epoch_backup_written:{json.dumps(payload)}")

    # slot 内存态照旧（is_pausable 信号依赖）
    assert slot.last_auto_epoch_state_path == payload["state_path"]
    # DB 持久化（Addendum 2）
    task = _get_task(env, tid)
    assert task["last_state_path"] == payload["state_path"]
    assert task["last_config_path"] == payload["config_path"]
    assert task["last_state_epoch"] == 3
    assert task["last_state_step"] == 300


def test_auto_epoch_backup_event_overwrites_previous(env) -> None:
    """每 epoch 一次，新事件覆盖旧值（覆盖式单文件语义一致）。"""
    sup = _new_sup(env)
    tid = _create_task(env, status="running")
    slot = _make_slot(tid)
    cb = sup._make_task_log_callback(slot, tid)

    for epoch in (1, 2):
        cb("__EVENT__:auto_epoch_backup_written:" + json.dumps({
            "state_path": "/s/auto_epoch_state.pt",
            "config_path": "/s/auto_epoch_state.config.json",
            "epoch": epoch,
            "step": epoch * 100,
        }))
    task = _get_task(env, tid)
    assert task["last_state_epoch"] == 2
    assert task["last_state_step"] == 200


def test_persist_last_state_ignores_empty_state_path(env) -> None:
    """payload 异常（state_path 空）→ 不写 DB，不抛错。"""
    sup = _new_sup(env)
    tid = _create_task(env, status="running")
    sup._persist_last_state(tid, {"state_path": "", "epoch": 1})
    task = _get_task(env, tid)
    assert task["last_state_path"] is None


# ---------------------------------------------------------------------------
# _clear_pause_fields：清字段不删文件（Addendum 2 修订）
# ---------------------------------------------------------------------------


def test_clear_pause_fields_keeps_files(env) -> None:
    """resume 成功后恢复点文件必须保留 —— 删了会造成 resume 后到下一 epoch
    末之间无恢复点的窗口。"""
    sup = _new_sup(env)
    tid = _create_task(env)
    pt, cfg = _write_recovery_pair(env["root"], tid)
    with db.connection_for(env["db"]) as conn:
        db.update_task(
            conn, tid,
            status="running",
            paused_state_path=str(pt),
            paused_config_path=str(cfg),
            paused_step=100,
            paused_at=time.time(),
        )

    sup._clear_pause_fields(tid)

    assert pt.exists()
    assert cfg.exists()
    task = _get_task(env, tid)
    assert task["paused_state_path"] is None
    assert task["paused_config_path"] is None
    assert task["paused_step"] is None
    assert task["paused_at"] is None
    # status 不改 — caller 决定
    assert task["status"] == "running"


# cancel paused → canceled 保留文件的测试在 test_supervisor_pause.py
# （test_cancel_paused_task_changes_to_canceled_keeps_files），不重复。


# ---------------------------------------------------------------------------
# resume endpoint：状态放宽
# ---------------------------------------------------------------------------


@pytest.fixture
def server_env(env, monkeypatch):
    monkeypatch.setattr(db, "STUDIO_DB", env["db"])
    return env


def _lifecycle():
    try:
        from fastapi import HTTPException
        from studio.api.routers.queue import lifecycle
    except ImportError:
        pytest.skip("fastapi not installed")
    return lifecycle, HTTPException


def test_resume_failed_task_with_recovery_point(server_env) -> None:
    """failed + last_state_* 在盘 → pending，且 last_* 复制进 paused_*
    （cmd_builder 读 paused_state_path 注入 --resume-state）。"""
    lifecycle, _ = _lifecycle()
    tid = _create_task(server_env, status="pending")
    pt, cfg = _write_recovery_pair(server_env["root"], tid)
    with db.connection_for(server_env["db"]) as conn:
        db.update_task(
            conn, tid,
            status="failed",
            exit_code=1,
            error_msg="exit code 1",
            finished_at=time.time(),
            last_state_path=str(pt),
            last_config_path=str(cfg),
            last_state_epoch=3,
            last_state_step=300,
        )

    result = lifecycle.resume_task(tid)
    assert result["status"] == "pending"

    task = _get_task(server_env, tid)
    assert task["status"] == "pending"
    assert task["paused_state_path"] == str(pt)
    assert task["paused_config_path"] == str(cfg)
    assert task["paused_step"] == 300
    # 上一轮的尸检字段清掉
    assert task["error_msg"] is None
    assert task["exit_code"] is None
    assert task["finished_at"] is None


def test_resume_canceled_task_with_recovery_point(server_env) -> None:
    lifecycle, _ = _lifecycle()
    tid = _create_task(server_env, status="pending")
    pt, cfg = _write_recovery_pair(server_env["root"], tid)
    with db.connection_for(server_env["db"]) as conn:
        db.update_task(
            conn, tid,
            status="canceled",
            finished_at=time.time(),
            last_state_path=str(pt),
            last_config_path=str(cfg),
            last_state_step=200,
        )
    result = lifecycle.resume_task(tid)
    assert result["status"] == "pending"
    task = _get_task(server_env, tid)
    assert task["paused_state_path"] == str(pt)


def test_resume_failed_without_recovery_point_409(server_env) -> None:
    """failed 但首 epoch 没跑完（last_state_path NULL）→ 409 引导 ResumeFieldPicker。"""
    lifecycle, HTTPException = _lifecycle()
    tid = _create_task(server_env, status="failed")
    with pytest.raises(HTTPException) as exc:
        lifecycle.resume_task(tid)
    assert exc.value.status_code == 409
    assert "missing" in exc.value.detail


def test_resume_failed_with_deleted_state_file_409(server_env) -> None:
    """DB 有路径但文件被外部删 → 409。"""
    lifecycle, HTTPException = _lifecycle()
    tid = _create_task(
        server_env, status="failed",
        last_state_path="/nonexistent/auto_epoch_state.pt",
        last_config_path="/nonexistent/auto_epoch_state.config.json",
    )
    with pytest.raises(HTTPException) as exc:
        lifecycle.resume_task(tid)
    assert exc.value.status_code == 409


def test_resume_done_task_rejected(server_env) -> None:
    """done 不可 resume（语义是重训，走 retry / ResumeFieldPicker）。"""
    lifecycle, HTTPException = _lifecycle()
    tid = _create_task(server_env, status="done")
    pt, cfg = _write_recovery_pair(server_env["root"], tid)
    with db.connection_for(server_env["db"]) as conn:
        db.update_task(conn, tid, last_state_path=str(pt), last_config_path=str(cfg))
    with pytest.raises(HTTPException) as exc:
        lifecycle.resume_task(tid)
    assert exc.value.status_code == 409
    assert "not resumable" in exc.value.detail


def test_resume_failed_with_bogus_version_id_no_crash(server_env) -> None:
    """version 已被删（reconcile 返回 None）→ resume 照常成功，不抛错。"""
    lifecycle, _ = _lifecycle()
    tid = _create_task(server_env, status="pending")
    pt, cfg = _write_recovery_pair(server_env["root"], tid)
    with db.connection_for(server_env["db"]) as conn:
        db.update_task(
            conn, tid,
            status="failed",
            version_id=99999,
            project_id=99999,
            last_state_path=str(pt),
            last_config_path=str(cfg),
        )
    result = lifecycle.resume_task(tid)
    assert result["status"] == "pending"


def test_resume_failed_reconciles_version_status(server_env, monkeypatch) -> None:
    """failed 曾把 version finalize 成 failed；resume 后派生回 training。"""
    lifecycle, _ = _lifecycle()
    from studio.services.projects import projects, versions
    monkeypatch.setattr(projects, "PROJECTS_DIR", server_env["root"] / "projects")

    with db.connection_for(server_env["db"]) as conn:
        p = projects.create_project(conn, title="P1")
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
        versions.update_version(conn, v["id"], status=versions.VersionStatus.FAILED)

    tid = _create_task(server_env, status="pending")
    pt, cfg = _write_recovery_pair(server_env["root"], tid)
    with db.connection_for(server_env["db"]) as conn:
        db.update_task(
            conn, tid,
            status="failed",
            project_id=p["id"],
            version_id=v["id"],
            last_state_path=str(pt),
            last_config_path=str(cfg),
        )

    lifecycle.resume_task(tid)

    with db.connection_for(server_env["db"]) as conn:
        v2 = versions.get_version(conn, v["id"])
    assert versions.get_status(v2) == versions.VersionStatus.TRAINING


# ---------------------------------------------------------------------------
# _is_resumable 信号
# ---------------------------------------------------------------------------


def test_is_resumable_matrix(server_env, tmp_path: Path) -> None:
    lifecycle, _ = _lifecycle()
    pt = tmp_path / "auto_epoch_state.pt"
    cfg = tmp_path / "auto_epoch_state.config.json"
    pt.write_bytes(b"x")
    cfg.write_text("{}", encoding="utf-8")

    # paused 看 paused_*
    assert lifecycle._is_resumable({
        "status": "paused",
        "paused_state_path": str(pt), "paused_config_path": str(cfg),
    }) is True
    # failed/canceled 看 last_*
    for status in ("failed", "canceled"):
        assert lifecycle._is_resumable({
            "status": status,
            "last_state_path": str(pt), "last_config_path": str(cfg),
        }) is True
    # done / running / pending 永远 False（哪怕字段有值）
    for status in ("done", "running", "pending"):
        assert lifecycle._is_resumable({
            "status": status,
            "last_state_path": str(pt), "last_config_path": str(cfg),
            "paused_state_path": str(pt), "paused_config_path": str(cfg),
        }) is False
    # 文件不在 → False
    assert lifecycle._is_resumable({
        "status": "failed", "last_state_path": "/nonexistent.pt",
    }) is False
    # state 在但 config snapshot 被删 → False（严格 freeze，跟 endpoint 一致）
    cfg.unlink()
    assert lifecycle._is_resumable({
        "status": "failed",
        "last_state_path": str(pt), "last_config_path": str(cfg),
    }) is False


# ---------------------------------------------------------------------------
# DELETE 清恢复点目录
# ---------------------------------------------------------------------------


def test_delete_task_removes_state_dir(server_env, monkeypatch) -> None:
    lifecycle, _ = _lifecycle()
    # task_dir 默认指向真实 studio_data/tasks/<id>/ —— 测试里必须隔离，
    # 否则 tmp DB 的 task id 撞上真实档案会被 rmtree。
    monkeypatch.setattr(
        lifecycle, "task_dir",
        lambda tid: server_env["root"] / "tasks" / str(tid),
    )
    tid = _create_task(server_env, status="pending")
    pt, cfg = _write_recovery_pair(server_env["root"], tid)
    with db.connection_for(server_env["db"]) as conn:
        db.update_task(
            conn, tid,
            status="canceled",
            finished_at=time.time(),
            last_state_path=str(pt),
        )

    lifecycle.delete_queue_item(tid)

    assert not pt.parent.exists()
    with db.connection_for(server_env["db"]) as conn:
        assert db.get_task(conn, tid) is None


def test_delete_task_guards_against_foreign_dir(server_env, monkeypatch) -> None:
    """last_state_path 父目录名不是 task_<id> → 不删（防 DB 路径异常误删）。"""
    lifecycle, _ = _lifecycle()
    monkeypatch.setattr(
        lifecycle, "task_dir",
        lambda tid: server_env["root"] / "tasks" / str(tid),
    )
    tid = _create_task(server_env, status="pending")
    foreign = server_env["root"] / "some_user_dir"
    foreign.mkdir()
    stray = foreign / "auto_epoch_state.pt"
    stray.write_bytes(b"x")
    with db.connection_for(server_env["db"]) as conn:
        db.update_task(
            conn, tid,
            status="canceled",
            finished_at=time.time(),
            last_state_path=str(stray),
        )

    lifecycle.delete_queue_item(tid)

    assert foreign.exists()
    assert stray.exists()
