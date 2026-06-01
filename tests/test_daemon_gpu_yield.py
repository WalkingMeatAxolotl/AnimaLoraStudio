"""Daemon GPU 让位（commit 12）：train/reg_ai/tag/reg_build pending 时
触发 daemon unload，模型未卸载前不派 GPU 任务。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from studio import db
from studio.services.projects import jobs as project_jobs
from studio.services.inference import daemon as _daemon_mod
from studio.supervisor import Supervisor


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from studio.infrastructure import paths as _paths
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    logs = tmp_path / "logs"
    configs = tmp_path / "configs"
    tasks = tmp_path / "tasks"
    logs.mkdir()
    configs.mkdir()
    monkeypatch.setattr(_paths, "TASKS_DIR", tasks)
    monkeypatch.setattr(_paths, "LOGS_DIR", logs)
    return {"db": db_path, "logs": logs, "configs": configs, "tasks": tasks}


@pytest.fixture
def fake_daemon():
    """替换全局 daemon 实例为 MagicMock。"""
    fake = MagicMock()
    fake.is_model_loaded = False
    fake.is_busy = False
    fake.state = "stopped"
    _daemon_mod._INSTANCE = fake  # type: ignore[attr-defined]
    yield fake
    _daemon_mod._INSTANCE = None  # type: ignore[attr-defined]


@pytest.fixture
def fake_secrets(monkeypatch):
    """把 secrets.load() 替换成可调的 fake，避免读真实 secrets 文件。"""
    cfg = MagicMock()
    cfg.queue.allow_gpu_during_train = False
    monkeypatch.setattr(
        "studio.supervisor._secrets.load", lambda: cfg
    )
    return cfg


def _make_train_task(env: dict) -> int:
    cfg_path = env["configs"] / "t.yaml"
    cfg_path.write_text("epochs: 1\n", encoding="utf-8")
    with db.connection_for(env["db"]) as conn:
        return db.create_task(conn, name="t", config_name="t")


def _make_tag_job(env: dict, *, slug: str = "p") -> int:
    """造一个 tag job pending。"""
    with db.connection_for(env["db"]) as conn:
        from studio.services.projects import projects, versions
        p = projects.create_project(conn, title="P", slug=slug)
        v = versions.create_version(conn, project_id=p["id"], label="v1")
        job = project_jobs.create_job(
            conn,
            project_id=p["id"],
            version_id=v["id"],
            kind="tag",
            params={},
        )
        return int(job["id"])


# ---------- _maybe_yield_daemon 单元行为 -------------------------------------


def test_yield_no_daemon_loaded(env, fake_daemon, fake_secrets):
    fake_daemon.is_model_loaded = False
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    assert sup._maybe_yield_daemon() is False
    fake_daemon.request_unload.assert_not_called()


def test_yield_daemon_idle_but_loaded(env, fake_daemon, fake_secrets):
    """daemon 加载了模型但 idle → 触发 unload，返回 True。"""
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = False
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    assert sup._maybe_yield_daemon() is True
    fake_daemon.request_unload.assert_called_once()


def test_yield_daemon_busy_no_unload(env, fake_daemon, fake_secrets):
    """daemon 在跑 generate → 等它跑完，不强中断。"""
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = True
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    assert sup._maybe_yield_daemon() is True
    fake_daemon.request_unload.assert_not_called()


def test_yield_allow_gpu_during_train(env, fake_daemon, fake_secrets):
    """用户允许并行 → 不让位。"""
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = False
    fake_secrets.queue.allow_gpu_during_train = True
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    assert sup._maybe_yield_daemon() is False
    fake_daemon.request_unload.assert_not_called()


# ---------- _dispatch_train 集成 -------------------------------------------


def test_dispatch_train_skipped_while_daemon_loaded(env, fake_daemon, fake_secrets):
    """daemon 占 GPU 时 _dispatch_train 不应 spawn train task。"""
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = False

    spawned: list[Any] = []
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    monkey_spawn = sup._spawn_task
    sup._spawn_task = lambda slot, task: spawned.append(task)  # type: ignore

    _make_train_task(env)
    # 取 SLOT_TRAIN slot 调 dispatch_train
    train_slot = next(s for s in sup._slots if s.name == "train")
    sup._dispatch_train(train_slot)
    assert spawned == [], "train task should not be spawned while daemon holds GPU"
    fake_daemon.request_unload.assert_called_once()


def test_dispatch_train_proceeds_after_daemon_unloaded(env, fake_daemon, fake_secrets):
    """daemon unload 完成后，下一次 _dispatch_train 应该 spawn train。"""
    fake_daemon.is_model_loaded = False
    fake_daemon.is_busy = False

    spawned: list[Any] = []
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    sup._spawn_task = lambda slot, task: spawned.append(task)  # type: ignore

    _make_train_task(env)
    train_slot = next(s for s in sup._slots if s.name == "train")
    sup._dispatch_train(train_slot)
    assert len(spawned) == 1
    fake_daemon.request_unload.assert_not_called()


# ---------- _dispatch_data 集成 --------------------------------------------


def test_dispatch_data_tag_job_yields_to_daemon_load(env, fake_daemon, fake_secrets):
    """tag job pending + daemon 占 GPU + 不许并行 → 让位，不 spawn job。"""
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = False

    spawned_jobs: list[Any] = []
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    sup._spawn_job = lambda slot, job: spawned_jobs.append(job)  # type: ignore

    _make_tag_job(env)
    data_slot = next(s for s in sup._slots if s.name == "data")
    sup._dispatch_data(data_slot)
    assert spawned_jobs == []
    fake_daemon.request_unload.assert_called_once()


def test_dispatch_data_download_not_blocked_by_daemon(env, fake_daemon, fake_secrets):
    """download 是 IO-only，不应被 daemon 让位逻辑阻塞。"""
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = False

    spawned_jobs: list[Any] = []
    sup = Supervisor(
        on_event=lambda _e: None,
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )
    sup._spawn_job = lambda slot, job: spawned_jobs.append(job)  # type: ignore

    # 造 download job
    with db.connection_for(env["db"]) as conn:
        from studio.services.projects import projects, versions
        p = projects.create_project(conn, title="P2", slug="p2")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
        project_jobs.create_job(
            conn,
            project_id=p["id"],
            version_id=v["id"],
            kind="download",
            params={},
        )

    data_slot = next(s for s in sup._slots if s.name == "data")
    sup._dispatch_data(data_slot)
    assert len(spawned_jobs) == 1, "download should run regardless of daemon state"
    fake_daemon.request_unload.assert_not_called()
