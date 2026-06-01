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
# _resolve_monitor_state_path —— task-scoped 档案（跟 version 解耦）
# ---------------------------------------------------------------------------


def test_monitor_state_path_isolated_per_task_same_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同 version 下两个 task 的 monitor_state_path 互不相同、各含自己的
    task id；都落 tasks/<id>/monitor/state.json，跟 version 标签无关。
    回归 1：旧 pre-PP6.1 实现按 version 落 monitor_state.json，第二个 task
    覆盖第一个 → 监控页 task1 看到 task2。
    回归 2：PP6.1 (v0.5.0+) 仍把 task 子目录挂在 versions/<v>/monitor/ 下，
    删 version 一并丢历史，task 之间无法对比 → 现在搬出 version 子树。"""
    from studio.supervisor.cmd_builder import _resolve_monitor_state_path

    # version_id / project_id 现在不再参与路径计算（task-scoped）
    base = {"version_id": 999, "project_id": 1}
    p1 = _resolve_monitor_state_path({**base, "id": 1})
    p2 = _resolve_monitor_state_path({**base, "id": 2})

    assert p1 != p2
    assert p1.parts[-4:] == ("tasks", "1", "monitor", "state.json")
    assert p2.parts[-4:] == ("tasks", "2", "monitor", "state.json")


def test_monitor_state_path_no_version_still_task_scoped() -> None:
    """没 version_id 也走 tasks/<id>/monitor/state.json（之前是兜底
    monitors/task_<id>/state.json，老 task DB 列保留旧值由读端兼容）。"""
    from studio.supervisor.cmd_builder import _resolve_monitor_state_path

    pth = _resolve_monitor_state_path({"id": 7})
    assert pth.parts[-4:] == ("tasks", "7", "monitor", "state.json")


def test_task_paths_helpers_form_a_consistent_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """task_monitor_state_path / task_samples_dir / task_log_path 必须落同一个
    tasks/<id>/ 根 —— 调用方依赖这点（监控页 SSE / 删除清理）。"""
    from studio.infrastructure import paths as _paths
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")

    msp = _paths.task_monitor_state_path(42)
    samples = _paths.task_samples_dir(42)
    log = _paths.task_log_path(42)
    snapshot = _paths.task_dir(42) / "snapshot" / "config.yaml"

    root = tmp_path / "tasks" / "42"
    assert msp == root / "monitor" / "state.json"
    assert samples == root / "samples"
    assert log == root / "run.log"
    # 跟 task_snapshot.snapshot_dir 拼出来的路径完全一致，确保 task 档案
    # 不被 snapshot/ 单独的 helper 拉到别处。
    from studio.services.task_snapshot import snapshot_dir
    assert snapshot_dir(42) == root / "snapshot"


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
# bootstrap_phase: _prepend_trigger_to_sample_prompts (PR #102 触发词功能)
# ---------------------------------------------------------------------------


def test_trigger_prepended_to_sample_prompt() -> None:
    from runtime.training.phases.bootstrap import _prepend_trigger_to_sample_prompts

    args = argparse.Namespace(
        trigger_word="ohwx",
        sample_prompt="1girl, masterpiece",
        sample_prompts=[],
    )
    _prepend_trigger_to_sample_prompts(args)
    assert args.sample_prompt == "ohwx, 1girl, masterpiece"


def test_trigger_prepended_to_sample_prompts_list() -> None:
    from runtime.training.phases.bootstrap import _prepend_trigger_to_sample_prompts

    args = argparse.Namespace(
        trigger_word="ohwx",
        sample_prompt="",
        sample_prompts=["a cat", "a dog"],
    )
    _prepend_trigger_to_sample_prompts(args)
    assert args.sample_prompts == ["ohwx, a cat", "ohwx, a dog"]


def test_trigger_skips_when_already_present_token_match() -> None:
    """trigger 在 prompt 里作为独立 token（逗号分隔后等值）→ 不重复 prepend。"""
    from runtime.training.phases.bootstrap import _prepend_trigger_to_sample_prompts

    args = argparse.Namespace(
        trigger_word="ohwx",
        sample_prompt="ohwx, 1girl",
        sample_prompts=["1girl, ohwx, blue eyes", "OHWX, masterpiece"],
    )
    _prepend_trigger_to_sample_prompts(args)
    assert args.sample_prompt == "ohwx, 1girl"
    assert args.sample_prompts == [
        "1girl, ohwx, blue eyes",      # 已含
        "OHWX, masterpiece",            # case-insensitive 已含
    ]


def test_trigger_empty_or_missing_is_noop() -> None:
    from runtime.training.phases.bootstrap import _prepend_trigger_to_sample_prompts

    args = argparse.Namespace(
        trigger_word="",
        sample_prompt="1girl",
        sample_prompts=["a cat"],
    )
    _prepend_trigger_to_sample_prompts(args)
    assert args.sample_prompt == "1girl"
    assert args.sample_prompts == ["a cat"]

    # 字段缺失（老 yaml） — 不抛错
    args2 = argparse.Namespace(sample_prompt="1girl", sample_prompts=["a"])
    _prepend_trigger_to_sample_prompts(args2)
    assert args2.sample_prompt == "1girl"
    assert args2.sample_prompts == ["a"]


def test_trigger_skips_empty_prompt_strings() -> None:
    """空 prompt 字符串不被填成 "trigger, "（残缺值）。"""
    from runtime.training.phases.bootstrap import _prepend_trigger_to_sample_prompts

    args = argparse.Namespace(
        trigger_word="ohwx",
        sample_prompt="",
        sample_prompts=["", "a cat", ""],
    )
    _prepend_trigger_to_sample_prompts(args)
    assert args.sample_prompt == ""
    assert args.sample_prompts == ["", "ohwx, a cat", ""]


# ---------------------------------------------------------------------------
# bootstrap_phase: _resolve_sample_seed
# ---------------------------------------------------------------------------


def test_resolve_sample_seed_zero_is_replaced_with_random_positive() -> None:
    """sample_seed=0 → 抽一个 [1, 2**31-1] 范围的具体 seed 写回 args。"""
    from runtime.training.phases.bootstrap import _resolve_sample_seed

    args = argparse.Namespace(sample_seed=0)
    _resolve_sample_seed(args)
    assert isinstance(args.sample_seed, int)
    assert 1 <= args.sample_seed <= 2**31 - 1


def test_resolve_sample_seed_explicit_value_kept() -> None:
    """用户显式给的非 0 seed → 不动，保证 reproducibility。"""
    from runtime.training.phases.bootstrap import _resolve_sample_seed

    args = argparse.Namespace(sample_seed=42)
    _resolve_sample_seed(args)
    assert args.sample_seed == 42


def test_resolve_sample_seed_idempotent_after_resolve() -> None:
    """二次调用不再覆盖已 resolved 的 seed（pause/resume 路径关键）。"""
    from runtime.training.phases.bootstrap import _resolve_sample_seed

    args = argparse.Namespace(sample_seed=0)
    _resolve_sample_seed(args)
    first = args.sample_seed
    _resolve_sample_seed(args)
    assert args.sample_seed == first


def test_resolve_sample_seed_missing_attr_treated_as_zero() -> None:
    """args 没有 sample_seed 字段 → 当成 0 抽一个具体值塞进去。"""
    from runtime.training.phases.bootstrap import _resolve_sample_seed

    args = argparse.Namespace()
    _resolve_sample_seed(args)
    assert isinstance(args.sample_seed, int)
    assert args.sample_seed > 0


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
    """返回一个轻量 namespace 暴露 PR-6 抽出的 resume_task / HTTPException 给老 tests。

    历史实现 `del sys.modules["studio.server"]` + 重新 import — 那是因为以前
    server.py 全 self-contained。PR-6 后 server.py @app 装饰器跟 api.app 共享
    同一 FastAPI 实例，反复重 import 会让装饰器对同一 app 重复注册路由（实测
    每跑一次膨胀几十条），导致 route_snapshot / route_invariants 跨测污染。

    改成只暴露 tests 直接用的 2 个名字（HTTPException + resume_task），不再
    重 import server。
    """
    try:
        from fastapi import HTTPException
        from studio.api.routers.queue.lifecycle import resume_task as _resume_task
    except ImportError:
        pytest.skip("fastapi not installed; cannot import resume endpoint")

    class _ServerShim:
        pass

    shim = _ServerShim()
    shim.HTTPException = HTTPException
    shim.resume_task = _resume_task
    return shim


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
