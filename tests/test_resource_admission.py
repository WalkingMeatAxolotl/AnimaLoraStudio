"""R-1 资源档位准入（docs/design/queue-resource-model-0.17.md）。

覆盖三个准入漏洞的修复 + exclusive 档平级 FIFO + secrets 开关迁移：
- L1：训练运行时 generate 不再被提交给 daemon（后端守卫，原先只有前端挡）
- L2：exclusive 档数据作业（eval_samples）运行时训练不 spawn（原先仲裁单向）
- L3：eval_samples 无视 light 开关（原先与 tag 混在同一粗粒度开关下）
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from studio import db
from studio.services.inference import daemon as _daemon_mod
from studio.services.projects import jobs as project_jobs
from studio.supervisor import Supervisor
from studio.supervisor.resources import (
    RESOURCE_EXCLUSIVE,
    RESOURCE_IO,
    RESOURCE_LIGHT,
    job_resource_class,
    task_resource_class,
)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from studio.infrastructure import paths as _paths
    db_path = tmp_path / "studio.db"
    db.init_db(db_path)
    logs = tmp_path / "logs"
    configs = tmp_path / "configs"
    logs.mkdir()
    configs.mkdir()
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(_paths, "LOGS_DIR", logs)
    (configs / "fake.yaml").write_text("epochs: 1\n", encoding="utf-8")
    return {"db": db_path, "logs": logs, "configs": configs}


@pytest.fixture
def fake_daemon():
    fake = MagicMock()
    fake.is_model_loaded = False
    fake.is_busy = False
    fake.state = "stopped"
    _daemon_mod._INSTANCE = fake  # type: ignore[attr-defined]
    yield fake
    _daemon_mod._INSTANCE = None  # type: ignore[attr-defined]


@pytest.fixture
def fake_secrets(monkeypatch):
    cfg = MagicMock()
    cfg.queue.light_tasks_during_train = True  # R-1 默认
    monkeypatch.setattr("studio.supervisor._secrets.load", lambda: cfg)
    return cfg


def _make_sup(env, on_event=None) -> Supervisor:
    return Supervisor(
        on_event=on_event or (lambda _e: None),
        db_path=env["db"], logs_dir=env["logs"], configs_dir=env["configs"],
    )


def _slot(sup: Supervisor, name: str):
    return next(s for s in sup._slots if s.name == name)


def _occupy(slot, *, kind: str, job_kind: str | None = None, id_: int = 999) -> None:
    """把槽位伪装成 busy（proc.poll() 恒 None = 还在跑）。"""
    slot.proc = MagicMock(poll=lambda: None)
    slot.kind = kind
    slot.id = id_
    slot.job_kind = job_kind


def _make_task(env, *, task_type: str, created_at: float,
               with_config: bool = True) -> int:
    with db.connection_for(env["db"]) as conn:
        tid = db.create_task(conn, name=f"{task_type}-{created_at}", config_name="fake")
        fields: dict[str, Any] = {"task_type": task_type, "created_at": created_at}
        if task_type == "generate" and with_config:
            fields["config_path"] = str(env["configs"] / "g.json")
        db.update_task(conn, tid, **fields)
    return tid


def _make_job(env, *, kind: str) -> int:
    with db.connection_for(env["db"]) as conn:
        from studio.services.projects import projects
        p = projects.create_project(conn, title=f"P-{kind}", slug=f"p-{kind}")
        job = project_jobs.create_job(conn, project_id=p["id"], kind=kind, params={})
    return int(job["id"])


# ---------------------------------------------------------------------------
# 档位映射契约
# ---------------------------------------------------------------------------


def test_resource_class_mapping() -> None:
    assert task_resource_class("train") == RESOURCE_EXCLUSIVE
    assert task_resource_class("reg_ai") == RESOURCE_EXCLUSIVE
    assert task_resource_class("generate") == RESOURCE_EXCLUSIVE
    assert task_resource_class(None) == RESOURCE_EXCLUSIVE  # 老行兜底 train
    assert job_resource_class("eval_samples") == RESOURCE_EXCLUSIVE  # D-R2
    assert job_resource_class("preprocess") == RESOURCE_LIGHT  # D-R1
    for k in ("tag", "reg_build", "eval_clip", "eval_dino", "eval_tag", "eval_ccip"):
        assert job_resource_class(k) == RESOURCE_LIGHT
    assert job_resource_class("download") == RESOURCE_IO
    # 未知 kind 保守按 exclusive（绝不与训练并行）
    assert job_resource_class("future_video_thing") == RESOURCE_EXCLUSIVE


# ---------------------------------------------------------------------------
# L1：训练运行时 generate 不提交 daemon（双向）
# ---------------------------------------------------------------------------


def test_generate_not_submitted_while_train_running(env, fake_daemon, fake_secrets):
    sup = _make_sup(env)
    submitted: list[int] = []
    sup._submit_to_daemon = lambda t: submitted.append(t["id"])  # type: ignore
    _occupy(_slot(sup, "train"), kind="task")
    _make_task(env, task_type="generate", created_at=100)

    sup._tick()
    assert submitted == [], "训练运行中 generate 不得提交 daemon（L1）"

    # 训练结束（槽位释放）→ 下一 tick 提交
    _slot(sup, "train").reset()
    sup._tick()
    assert len(submitted) == 1


def test_train_not_spawned_while_daemon_generate_active(env, fake_daemon, fake_secrets):
    """反向：daemon 有 active generate → pending train 等待。"""
    sup = _make_sup(env)
    spawned: list[Any] = []
    sup._spawn_task = lambda slot, task: spawned.append(task)  # type: ignore
    sup._daemon_active_task_id = 77
    _make_task(env, task_type="train", created_at=100)

    sup._tick()
    assert spawned == []

    sup._daemon_active_task_id = None
    sup._tick()
    assert len(spawned) == 1


# ---------------------------------------------------------------------------
# L2：exclusive 数据作业运行时训练等待（仲裁不再单向）
# ---------------------------------------------------------------------------


def test_train_waits_for_running_eval_samples(env, fake_daemon, fake_secrets):
    sup = _make_sup(env)
    spawned: list[Any] = []
    sup._spawn_task = lambda slot, task: spawned.append(task)  # type: ignore
    _occupy(_slot(sup, "data"), kind="job", job_kind="eval_samples")
    _make_task(env, task_type="train", created_at=100)

    sup._tick()
    assert spawned == [], "eval_samples（底模级）运行中训练不得 spawn（L2）"

    _slot(sup, "data").reset()
    sup._tick()
    assert len(spawned) == 1


def test_train_does_not_wait_for_running_light_job(env, fake_daemon, fake_secrets):
    """light 档数据作业（tag）运行中不阻塞训练 —— 只有 exclusive 档才互斥。"""
    sup = _make_sup(env)
    spawned: list[Any] = []
    sup._spawn_task = lambda slot, task: spawned.append(task)  # type: ignore
    _occupy(_slot(sup, "data"), kind="job", job_kind="tag")
    _make_task(env, task_type="train", created_at=100)

    sup._tick()
    assert len(spawned) == 1


# ---------------------------------------------------------------------------
# L3：eval_samples 无视 light 开关；light 按开关；io 恒放行
# ---------------------------------------------------------------------------


def test_eval_samples_deferred_during_training_despite_switch(
    env, fake_daemon, fake_secrets,
):
    """训练运行 + light 开关开：tag 放行、eval_samples 推迟（L3 修复核心）。"""
    sup = _make_sup(env)
    spawned_jobs: list[Any] = []
    sup._spawn_job = lambda slot, job: spawned_jobs.append(job)  # type: ignore
    _occupy(_slot(sup, "train"), kind="task")
    _make_job(env, kind="eval_samples")
    tag_id = _make_job(env, kind="tag")

    sup._dispatch_data(_slot(sup, "data"))
    assert [j["id"] for j in spawned_jobs] == [tag_id], \
        "开关只放行 light 档；eval_samples（exclusive）必须推迟"


def test_light_deferred_when_switch_off_but_io_still_runs(
    env, fake_daemon, fake_secrets,
):
    fake_secrets.queue.light_tasks_during_train = False
    sup = _make_sup(env)
    spawned_jobs: list[Any] = []
    sup._spawn_job = lambda slot, job: spawned_jobs.append(job)  # type: ignore
    _occupy(_slot(sup, "train"), kind="task")
    _make_job(env, kind="tag")
    dl_id = _make_job(env, kind="download")

    sup._dispatch_data(_slot(sup, "data"))
    assert [j["id"] for j in spawned_jobs] == [dl_id], \
        "开关关闭时 light 推迟，io（download）仍恒放行"


def test_eval_samples_requires_daemon_lease_release(env, fake_daemon, fake_secrets):
    """eval_samples 与 train 同规格：daemon idle 常驻模型 → 先吊销租约再派。

    R-3 起 eval_samples 走 exclusive 统一 FIFO（_dispatch_exclusive_tasks），
    执行位仍是 DATA 槽。
    """
    fake_daemon.is_model_loaded = True
    fake_daemon.is_busy = False
    sup = _make_sup(env)
    spawned_jobs: list[Any] = []
    sup._spawn_job = lambda slot, job: spawned_jobs.append(job)  # type: ignore
    _make_job(env, kind="eval_samples")

    sup._dispatch_exclusive_tasks(_slot(sup, "train"))
    assert spawned_jobs == []
    fake_daemon.request_unload.assert_called_once()

    fake_daemon.is_model_loaded = False
    sup._dispatch_exclusive_tasks(_slot(sup, "train"))
    assert len(spawned_jobs) == 1
    assert spawned_jobs[0]["kind"] == "eval_samples"


def test_exclusive_fifo_eval_samples_before_train(env, fake_daemon, fake_secrets):
    """D-R3 跨类型平级：先入队的 eval_samples 先跑，后入队的 train 排队等。"""
    sup = _make_sup(env)
    spawned_jobs: list[Any] = []
    spawned_tasks: list[Any] = []
    sup._spawn_job = lambda slot, job: spawned_jobs.append(job)  # type: ignore
    sup._spawn_task = lambda slot, task: spawned_tasks.append(task)  # type: ignore
    ev = _make_job(env, kind="eval_samples")
    with db.connection_for(env["db"]) as conn:
        db.update_task(conn, ev, created_at=100.0)
    _make_task(env, task_type="train", created_at=200.0)

    sup._dispatch_exclusive_tasks(_slot(sup, "train"))
    assert [j["id"] for j in spawned_jobs] == [ev] and spawned_tasks == []

    # eval_samples 结束 → train 轮到
    with db.connection_for(env["db"]) as conn:
        db.update_task(conn, ev, status="done")
    sup._dispatch_exclusive_tasks(_slot(sup, "train"))
    assert len(spawned_tasks) == 1


# ---------------------------------------------------------------------------
# D-R3：exclusive 档平级 FIFO（train / generate 同表按入队顺序）
# ---------------------------------------------------------------------------


def test_exclusive_fifo_generate_before_train(env, fake_daemon, fake_secrets):
    """先入队的 generate 先跑；train 等它结束后（本测试直接标 done）再 spawn。"""
    sup = _make_sup(env)
    submitted: list[int] = []
    spawned: list[Any] = []
    sup._submit_to_daemon = lambda t: submitted.append(t["id"])  # type: ignore
    sup._spawn_task = lambda slot, task: spawned.append(task)  # type: ignore
    gen_id = _make_task(env, task_type="generate", created_at=100)
    _make_task(env, task_type="train", created_at=200)

    sup._dispatch_exclusive_tasks(_slot(sup, "train"))
    assert submitted == [gen_id] and spawned == [], "FIFO：先入队的 generate 先派"

    # generate 结束（本测试 stub 掉 submit，手动标 done）→ train 轮到
    with db.connection_for(env["db"]) as conn:
        db.update_task(conn, gen_id, status="done")
    sup._dispatch_exclusive_tasks(_slot(sup, "train"))
    assert len(spawned) == 1 and (spawned[0].get("task_type") or "train") == "train"


def test_exclusive_fifo_train_before_generate(env, fake_daemon, fake_secrets):
    """反向顺序：train 先入队就先跑，后点的 generate 排后面。"""
    sup = _make_sup(env)
    submitted: list[int] = []
    spawned: list[Any] = []
    sup._submit_to_daemon = lambda t: submitted.append(t["id"])  # type: ignore
    sup._spawn_task = lambda slot, task: spawned.append(task)  # type: ignore
    _make_task(env, task_type="train", created_at=100)
    _make_task(env, task_type="generate", created_at=200)

    sup._dispatch_exclusive_tasks(_slot(sup, "train"))
    assert len(spawned) == 1 and submitted == []


# ---------------------------------------------------------------------------
# secrets 开关迁移
# ---------------------------------------------------------------------------


def test_secrets_drops_legacy_allow_gpu_key() -> None:
    """老 key 一律丢弃（语义变化不迁移值），新开关默认 True。"""
    from studio.infrastructure.secrets import Secrets, _migrate_legacy_schema

    for legacy in (True, False):
        raw = _migrate_legacy_schema({"queue": {"allow_gpu_during_train": legacy}})
        assert "allow_gpu_during_train" not in raw.get("queue", {})
        s = Secrets.model_validate(raw)
        assert s.queue.light_tasks_during_train is True


def test_secrets_new_key_roundtrip() -> None:
    from studio.infrastructure.secrets import Secrets, _migrate_legacy_schema

    raw = _migrate_legacy_schema({"queue": {"light_tasks_during_train": False}})
    s = Secrets.model_validate(raw)
    assert s.queue.light_tasks_during_train is False
