"""ADR 0006 PR-3 — resume 路径 + cmd_builder + snapshot 覆盖测试。

不起真子进程；
  - cmd_builder 单测：构造 task dict 调 `_default_cmd_builder`
  - bootstrap 覆盖单测：argparse.Namespace + sibling snapshot 落盘 + 调
    `_maybe_apply_pause_snapshot`
  - supervisor `_clear_pause_artifacts` 单测：构造 db 状态 + 调 method
  - server resume endpoint 单测：直接调函数（绕过 fastapi router，因为本
    dev env 没装 fastapi/test_client）

完整 end-to-end（pause → resume → loss 连续）依赖真训练栈，留给手测 +
PR-4 集成。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from studio import db
from studio.supervisor import Supervisor, _default_cmd_builder


# ---------------------------------------------------------------------------
# cmd_builder 加 --resume-state
# ---------------------------------------------------------------------------


def test_cmd_builder_no_resume_when_no_paused_state(tmp_path: Path) -> None:
    """正常 train task：cmd 里没 --resume-state。"""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("", encoding="utf-8")
    cmd = _default_cmd_builder({"task_type": "train", "id": 1}, cfg)
    assert "--resume-state" not in cmd


def test_cmd_builder_adds_resume_state_when_paused(tmp_path: Path) -> None:
    """task 含 paused_state_path → cmd 加 --resume-state <path>。"""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("", encoding="utf-8")
    pt = tmp_path / "pause_step_100.pt"
    cmd = _default_cmd_builder(
        {"task_type": "train", "id": 1, "paused_state_path": str(pt)},
        cfg,
    )
    assert "--resume-state" in cmd
    idx = cmd.index("--resume-state")
    assert cmd[idx + 1] == str(pt)


def test_cmd_builder_paused_path_works_for_reg_ai(tmp_path: Path) -> None:
    """reg_ai task 也支持 paused_state_path（虽然 ADR scope 只 train，
    但 cmd_builder 不区分；reg_ai 信号链路同 train）。"""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("", encoding="utf-8")
    pt = tmp_path / "pause_step_200.pt"
    cmd = _default_cmd_builder(
        {"task_type": "reg_ai", "paused_state_path": str(pt)},
        cfg,
    )
    assert "--resume-state" in cmd


def test_cmd_builder_resume_after_monitor_state_file(tmp_path: Path) -> None:
    """args 顺序：--config / --monitor-state-file / --resume-state。"""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("", encoding="utf-8")
    cmd = _default_cmd_builder(
        {
            "task_type": "train",
            "monitor_state_path": str(tmp_path / "ms.json"),
            "paused_state_path": str(tmp_path / "p.pt"),
        },
        cfg,
    )
    assert cmd.index("--config") < cmd.index("--monitor-state-file") < cmd.index("--resume-state")


# ---------------------------------------------------------------------------
# bootstrap_phase: _maybe_apply_pause_snapshot
# ---------------------------------------------------------------------------


def test_snapshot_overrides_args(tmp_path: Path) -> None:
    """sibling .config.json 存在 → snapshot 内 args 覆盖 namespace。"""
    from runtime.training.phases.bootstrap import _maybe_apply_pause_snapshot  # type: ignore[import-not-found]

    pt = tmp_path / "pause_step_100.pt"
    pt.write_bytes(b"")
    snap = pt.with_suffix(".config.json")
    snap.write_text(json.dumps({
        "version": 1,
        "args": {"lr": 5e-5, "optimizer": "Lion", "batch_size": 8},
        "sample_prompts": ["snap_prompt"],
    }), encoding="utf-8")

    args = argparse.Namespace(lr=1e-4, optimizer="AdamW", batch_size=2, resume_state=str(pt))
    _maybe_apply_pause_snapshot(args, pt)

    assert args.lr == 5e-5
    assert args.optimizer == "Lion"
    assert args.batch_size == 8
    assert args.sample_prompts == ["snap_prompt"]


def test_snapshot_missing_leaves_args_intact(tmp_path: Path) -> None:
    """sibling 不存在 → args 完全不动（ResumeFieldPicker 起新 task 兼容）。"""
    from runtime.training.phases.bootstrap import _maybe_apply_pause_snapshot

    pt = tmp_path / "training_state_step42.pt"  # 周期 save 命名，无 snapshot
    pt.write_bytes(b"")

    args = argparse.Namespace(lr=1e-4, optimizer="AdamW", resume_state=str(pt))
    _maybe_apply_pause_snapshot(args, pt)
    assert args.lr == 1e-4
    assert args.optimizer == "AdamW"


def test_snapshot_keeps_resume_state_and_config(tmp_path: Path) -> None:
    """snapshot 里的 resume_state / config 不该覆盖当前的 — 当前的是
    supervisor 刚拼好的真路径，snapshot 里的是 pause 那一刻的（已 stale）。"""
    from runtime.training.phases.bootstrap import _maybe_apply_pause_snapshot

    pt = tmp_path / "pause_step_50.pt"
    pt.write_bytes(b"")
    snap = pt.with_suffix(".config.json")
    snap.write_text(json.dumps({
        "version": 1,
        "args": {
            "lr": 1e-3,
            "resume_state": "/old/stale/path.pt",
            "config": "/old/stale.yaml",
        },
    }), encoding="utf-8")

    args = argparse.Namespace(
        lr=1e-4,
        resume_state=str(pt),
        config="/current/cfg.yaml",
    )
    _maybe_apply_pause_snapshot(args, pt)
    assert args.lr == 1e-3                # 覆盖
    assert args.resume_state == str(pt)   # 保留
    assert args.config == "/current/cfg.yaml"  # 保留


def test_snapshot_malformed_falls_back_silently(tmp_path: Path) -> None:
    """snapshot json 损坏 / schema 错 → warn log，不抛错，args 不动。"""
    from runtime.training.phases.bootstrap import _maybe_apply_pause_snapshot

    pt = tmp_path / "pause_step_10.pt"
    pt.write_bytes(b"")
    snap = pt.with_suffix(".config.json")
    snap.write_text("not valid json {{{", encoding="utf-8")

    args = argparse.Namespace(lr=1e-4, resume_state=str(pt))
    _maybe_apply_pause_snapshot(args, pt)
    assert args.lr == 1e-4


def test_snapshot_schema_wrong_falls_back(tmp_path: Path) -> None:
    from runtime.training.phases.bootstrap import _maybe_apply_pause_snapshot

    pt = tmp_path / "pause_step_10.pt"
    pt.write_bytes(b"")
    snap = pt.with_suffix(".config.json")
    snap.write_text(json.dumps({"args": "not a dict"}), encoding="utf-8")

    args = argparse.Namespace(lr=1e-4, resume_state=str(pt))
    _maybe_apply_pause_snapshot(args, pt)
    assert args.lr == 1e-4


# ---------------------------------------------------------------------------
# supervisor._clear_pause_artifacts
# ---------------------------------------------------------------------------


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


def test_clear_pause_artifacts_deletes_files_and_clears_db(env) -> None:
    sup = _new_sup(env)
    state_pt = env["root"] / "pause_step_100.pt"
    cfg_json = env["root"] / "pause_step_100.config.json"
    state_pt.write_bytes(b"fake state")
    cfg_json.write_text("{}", encoding="utf-8")

    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(
            conn, tid,
            status="running",
            paused_state_path=str(state_pt),
            paused_config_path=str(cfg_json),
            paused_step=100,
            paused_at=time.time(),
        )

    sup._clear_pause_artifacts(tid)

    assert not state_pt.exists()
    assert not cfg_json.exists()
    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task is not None
    assert task["paused_state_path"] is None
    assert task["paused_config_path"] is None
    assert task["paused_step"] is None
    assert task["paused_at"] is None
    # status 不改 — caller 决定
    assert task["status"] == "running"


def test_clear_pause_artifacts_unknown_task_noop(env) -> None:
    """task 不存在 → 静默返回，不抛错。"""
    sup = _new_sup(env)
    sup._clear_pause_artifacts(99999)


def test_clear_pause_artifacts_missing_files_robust(env) -> None:
    """文件已被外部删 → db 字段照样清，不抛错。"""
    sup = _new_sup(env)
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(
            conn, tid,
            paused_state_path="/nonexistent/path.pt",
            paused_config_path="/nonexistent/path.config.json",
            paused_step=100,
        )
    sup._clear_pause_artifacts(tid)
    with db.connection_for(env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task["paused_state_path"] is None


# ---------------------------------------------------------------------------
# resume_task endpoint —— 直接调 function（绕开 fastapi router）
# ---------------------------------------------------------------------------


@pytest.fixture
def server_env(env, monkeypatch):
    """初始化 db.STUDIO_DB monkeypatch 让 server module 用 isolated db。"""
    # ADR 0006 PR-5 删 feature flag 后这里无需注入 env，直接 monkeypatch db 路径
    if "studio.server" in sys.modules:
        del sys.modules["studio.server"]
    # 触发 server 不要起 supervisor / FastAPI client：只 import module
    monkeypatch.setattr(db, "STUDIO_DB", env["db"])
    return env


def _create_paused_task(env, state_pt: Path, cfg_json: Path) -> int:
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        db.update_task(
            conn, tid,
            status="paused",
            paused_state_path=str(state_pt),
            paused_config_path=str(cfg_json),
            paused_step=100,
            paused_at=time.time(),
        )
    return tid


def _import_server_module():
    """每次干净 import server module — 测试 isolation 用。"""
    import importlib
    if "studio.server" in sys.modules:
        del sys.modules["studio.server"]
    try:
        import studio.server as _s  # type: ignore[import-not-found]
        return _s
    except ImportError:
        pytest.skip("fastapi not installed; cannot import studio.server")


def test_resume_endpoint_rejects_unknown_task(server_env) -> None:
    server = _import_server_module()
    with pytest.raises(server.HTTPException) as exc:
        server.resume_task(99999)
    assert exc.value.status_code == 404


def test_resume_endpoint_rejects_non_paused(server_env) -> None:
    server = _import_server_module()
    with db.connection_for(server_env["db"]) as conn:
        tid = db.create_task(conn, name="t", config_name="c")
        # pending status
    with pytest.raises(server.HTTPException) as exc:
        server.resume_task(tid)
    assert exc.value.status_code == 409
    assert "not paused" in exc.value.detail


def test_resume_endpoint_rejects_when_state_file_missing(server_env) -> None:
    server = _import_server_module()
    state_pt = server_env["root"] / "pause_step_100.pt"
    cfg_json = server_env["root"] / "pause_step_100.config.json"
    cfg_json.write_text("{}", encoding="utf-8")  # state missing, config exists
    tid = _create_paused_task(server_env, state_pt, cfg_json)
    with pytest.raises(server.HTTPException) as exc:
        server.resume_task(tid)
    assert exc.value.status_code == 409
    assert "missing" in exc.value.detail


def test_resume_endpoint_rejects_when_config_snapshot_missing(server_env) -> None:
    server = _import_server_module()
    state_pt = server_env["root"] / "pause_step_100.pt"
    cfg_json = server_env["root"] / "pause_step_100.config.json"
    state_pt.write_bytes(b"")  # state exists, config missing
    tid = _create_paused_task(server_env, state_pt, cfg_json)
    with pytest.raises(server.HTTPException) as exc:
        server.resume_task(tid)
    assert exc.value.status_code == 409
    assert "snapshot missing" in exc.value.detail


def test_resume_endpoint_success_flips_status_keeps_paused_fields(server_env) -> None:
    server = _import_server_module()
    state_pt = server_env["root"] / "pause_step_100.pt"
    cfg_json = server_env["root"] / "pause_step_100.config.json"
    state_pt.write_bytes(b"")
    cfg_json.write_text("{}", encoding="utf-8")
    tid = _create_paused_task(server_env, state_pt, cfg_json)

    result = server.resume_task(tid)
    assert result["status"] == "pending"
    assert result["task_id"] == tid
    # 关键：paused_state_path 等字段保留（cmd_builder 下一轮 dispatch 才能读到）
    with db.connection_for(server_env["db"]) as conn:
        task = db.get_task(conn, tid)
    assert task["status"] == "pending"
    assert task["paused_state_path"] == str(state_pt)
    assert task["paused_config_path"] == str(cfg_json)
    assert task["paused_step"] == 100
    # finished_at / exit_code / error_msg 都 reset 了（新一轮跑）
    assert task["finished_at"] is None
    assert task["exit_code"] is None
    assert task["error_msg"] is None
