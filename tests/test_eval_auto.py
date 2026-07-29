"""评估入队 —— 两个入口都建**一个** EvalSession（issue #465）。

checkpoint 采样的纯逻辑（`select_checkpoints` / `checkpoint_skip_count`）不碰 DB，单独测；
入口测的是「建了几个 task 行 / 候选是哪些 / 门控生不生效」。Session 内部的阶段编排见
test_eval_session_worker。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from studio import db, secrets
from studio.infrastructure import paths as infra_paths
from studio.services import eval_auto, eval_session
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.supervisor import Supervisor


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(infra_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(infra_paths, "EVAL_SESSIONS_DIR", tmp_path / "eval" / "sessions")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return {"db": dbfile}


def _project_version(isolated, *, validation: int = 1) -> tuple[dict[str, Any], dict[str, Any], Path]:
    with db.connection_for(isolated["db"]) as conn:
        project = projects.create_project(conn, title="Auto Eval")
        version = versions.create_version(
            conn, project_id=project["id"], label="baseline"
        )
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    train = vdir / "train" / "1_data"
    train.mkdir(parents=True, exist_ok=True)
    (train / "a.png").write_bytes(b"png")
    (train / "a.txt").write_text("solo", encoding="utf-8")
    val = vdir / "validation" / "1_data"
    val.mkdir(parents=True, exist_ok=True)
    for i in range(validation):
        (val / f"v{i}.png").write_bytes(b"png")
        (val / f"v{i}.txt").write_text("1girl, solo", encoding="utf-8")
    output = vdir / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_epoch2.safetensors").write_bytes(b"lora")
    return project, version, vdir


def _enable_validation(
    project: dict[str, Any], version: dict[str, Any], **extra: Any
) -> None:
    """开启 per-version 训练后评估。

    `extra` 透传额外训练配置键（如 eval_checkpoint_skip_count），不传就是「旧配置」
    形态 —— 只有开关、没有采样参数。
    """
    from studio.services import version_config
    version_config.write_version_config(
        project, version, {"eval_validation_enabled": True, **extra}
    )


def _candidate_paths(isolated, session: dict[str, Any]) -> set[str]:
    with db.connection_for(isolated["db"]) as conn:
        cands = eval_session.list_candidates(conn, int(session["id"]))
    return {
        str(c["checkpoint_path"]) for c in cands if c["role"] == "checkpoint"
    }


# ---------------------------------------------------------------------------
# checkpoint 采样（纯逻辑）
# ---------------------------------------------------------------------------

def _ck(kind: str, value: int) -> dict[str, Any]:
    name = "final" if kind == "final" else f"{kind}{value}"
    return {"kind": kind, "value": value, "path": f"output/model_{name}.safetensors"}


# list_lora_ckpts 的展示序：final 在前，epoch 降序。
_DISPLAY = [_ck("final", 0), *[_ck("epoch", v) for v in (10, 8, 6, 4, 2)]]


def test_select_checkpoints_skip_zero_keeps_all() -> None:
    """0 = 全评（默认）。作业膨胀已在 Session 层根治，不靠限制 checkpoint 数止血。"""
    assert eval_auto.select_checkpoints(_DISPLAY, skip_count=0) == _DISPLAY
    assert eval_auto.select_checkpoints(_DISPLAY, skip_count=-1) == _DISPLAY


def test_select_checkpoints_skip_samples_in_training_order() -> None:
    """skip=2 → 训练序 epoch2/4/6/8/10 里取 epoch2、跳 4 和 6、取 epoch8；final 始终在内。

    返回值按入参展示序，与全量评估时的顺序一致。
    """
    got = eval_auto.select_checkpoints(_DISPLAY, skip_count=2)
    assert [c["kind"] for c in got] == ["final", "epoch", "epoch"]
    assert [c["value"] for c in got] == [0, 8, 2]


def test_select_checkpoints_skip_one_takes_every_other() -> None:
    got = eval_auto.select_checkpoints(_DISPLAY, skip_count=1)
    # 训练序 2/4/6/8/10，stride 2 → epoch2、epoch6、epoch10；final 始终在内
    assert [c["value"] for c in got] == [0, 10, 6, 2]


def test_select_checkpoints_keeps_final_even_when_not_sampled() -> None:
    """final 不参与采样、始终在内 —— 用户最可能想看的就是它。"""
    no_final_in_stride = [_ck("final", 0), *[_ck("epoch", v) for v in (9, 7, 5, 3, 1)]]
    got = eval_auto.select_checkpoints(no_final_in_stride, skip_count=4)
    assert got[0]["kind"] == "final"


def test_checkpoint_skip_count_defaults_to_zero() -> None:
    """缺失 / 非法值都归一到 0（全评）。"""
    assert eval_auto.checkpoint_skip_count({}) == 0
    assert eval_auto.checkpoint_skip_count({"eval_checkpoint_skip_count": "4"}) == 4
    assert eval_auto.checkpoint_skip_count({"eval_checkpoint_skip_count": -3}) == 0
    assert eval_auto.checkpoint_skip_count({"eval_checkpoint_skip_count": "bogus"}) == 0
    assert eval_auto.checkpoint_skip_count({"eval_checkpoint_skip_count": None}) == 0


# ---------------------------------------------------------------------------
# 训练后自动评估
# ---------------------------------------------------------------------------

def test_after_training_creates_one_session_one_task(isolated) -> None:
    """#465 的核心不变量：训练后评估只产生一个用户可见 task。"""
    project, version, vdir = _project_version(isolated)
    (vdir / "output" / "model_epoch4.safetensors").write_bytes(b"lora")
    _enable_validation(project, version)
    task = {"id": 7, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        session = eval_auto.queue_training_finished_eval(conn, task, {"epoch": 4})
        eval_tasks = conn.execute(
            "SELECT * FROM tasks WHERE task_type = 'eval_session'"
        ).fetchall()

    assert session is not None
    assert session["trigger"] == "after_training"
    assert int(session["parent_task_id"]) == 7
    assert len(eval_tasks) == 1
    assert _candidate_paths(isolated, session) == {
        "output/model_epoch2.safetensors",
        "output/model_epoch4.safetensors",
    }
    # 采样参数连同来龙去脉一起冻进 plan
    assert session["plan"]["checkpoint_sampling"] == {"skip_count": 0}


def test_after_training_defaults_to_evaluating_every_checkpoint(isolated) -> None:
    """没配 skip_count（含 0.21 及以前的旧配置）→ 全评。"""
    project, version, vdir = _project_version(isolated)
    (vdir / "output" / "model_epoch4.safetensors").write_bytes(b"lora")
    (vdir / "output" / "model_final.safetensors").write_bytes(b"lora")
    _enable_validation(project, version)
    task = {"id": 11, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        session = eval_auto.queue_training_finished_eval(conn, task, {})

    assert session is not None
    assert _candidate_paths(isolated, session) == {
        "output/model_epoch2.safetensors",
        "output/model_epoch4.safetensors",
        "output/model_final.safetensors",
    }


def test_after_training_skip_count_subsets_checkpoints(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    for epoch in (4, 6, 8):
        (vdir / "output" / f"model_epoch{epoch}.safetensors").write_bytes(b"lora")
    _enable_validation(project, version, eval_checkpoint_skip_count=1)
    task = {"id": 13, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        session = eval_auto.queue_training_finished_eval(conn, task, {})

    # 训练序 epoch2/4/6/8，skip=1 → stride 2 → epoch2、epoch6（无 final）
    assert _candidate_paths(isolated, session) == {
        "output/model_epoch2.safetensors",
        "output/model_epoch6.safetensors",
    }


def test_after_training_gated_on_version_opt_in(isolated) -> None:
    project, version, _vdir = _project_version(isolated)
    task = {"id": 15, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        assert eval_auto.queue_training_finished_eval(conn, task, {}) is None
        assert conn.execute("SELECT COUNT(*) FROM eval_sessions").fetchone()[0] == 0


def test_after_training_without_checkpoints_creates_nothing(isolated) -> None:
    """output/ 空（训练早期崩了）→ 不建空 Session。"""
    project, version, vdir = _project_version(isolated)
    (vdir / "output" / "model_epoch2.safetensors").unlink()
    _enable_validation(project, version)
    task = {"id": 17, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        assert eval_auto.queue_training_finished_eval(conn, task, {}) is None
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_after_training_baseline_follows_settings(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    _enable_validation(project, version)
    secrets.update({"eval_metrics": {"eval_baseline_enabled": False}})
    task = {"id": 19, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        session = eval_auto.queue_training_finished_eval(conn, task, {})
        cands = eval_session.list_candidates(conn, int(session["id"]))

    assert all(c["role"] == "checkpoint" for c in cands)


# ---------------------------------------------------------------------------
# 手动评估
# ---------------------------------------------------------------------------

def test_manual_eval_bypasses_opt_in_and_uses_explicit_set(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    (vdir / "output" / "model_epoch4.safetensors").write_bytes(b"lora")
    # 不开 per-version 开关 —— 手动入口不看它
    task = {"id": 21, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        session = eval_auto.queue_manual_task_eval(
            conn, task, ["model_epoch4.safetensors"]
        )

    assert session is not None
    assert session["trigger"] == "manual"
    assert _candidate_paths(isolated, session) == {"output/model_epoch4.safetensors"}


def test_manual_eval_dedupes_and_rejects_paths_outside_output(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    task = {"id": 23, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        session = eval_auto.queue_manual_task_eval(conn, task, [
            "model_epoch2.safetensors",
            "model_epoch2.safetensors",            # 重复
            str(vdir / "output" / "model_epoch2.safetensors"),  # 绝对路径同一个文件
            "../../../etc/passwd",                 # 越界
            "does_not_exist.safetensors",          # 不在 list_lora_ckpts 里
        ])

    assert _candidate_paths(isolated, session) == {"output/model_epoch2.safetensors"}


def test_manual_eval_with_no_valid_checkpoint_returns_none(isolated) -> None:
    project, version, _vdir = _project_version(isolated)
    task = {"id": 25, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        assert eval_auto.queue_manual_task_eval(conn, task, ["../evil"]) is None
        assert conn.execute("SELECT COUNT(*) FROM eval_sessions").fetchone()[0] == 0


def test_manual_eval_keeps_previous_sessions(isolated) -> None:
    """A 方案：重跑不清上一轮 —— 历史 Session 全部留档。"""
    project, version, vdir = _project_version(isolated)
    task = {"id": 27, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        first = eval_auto.queue_manual_task_eval(
            conn, task, ["model_epoch2.safetensors"]
        )
        second = eval_auto.queue_manual_task_eval(
            conn, task, ["model_epoch2.safetensors"]
        )
        listed = eval_session.list_sessions(conn, parent_task_id=27)

    assert int(first["id"]) != int(second["id"])
    assert [int(s["id"]) for s in listed] == [int(second["id"]), int(first["id"])]
    # 两个 Session 各有自己的 task 行
    assert int(first["task_id"]) != int(second["task_id"])


def test_manual_eval_orders_candidates_by_display_order(isolated) -> None:
    """候选顺序不该取决于用户点选的先后。"""
    project, version, vdir = _project_version(isolated)
    for epoch in (4, 6):
        (vdir / "output" / f"model_epoch{epoch}.safetensors").write_bytes(b"lora")
    task = {"id": 29, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        session = eval_auto.queue_manual_task_eval(conn, task, [
            "model_epoch2.safetensors",
            "model_epoch6.safetensors",
            "model_epoch4.safetensors",
        ])
        cands = [
            c for c in eval_session.list_candidates(conn, int(session["id"]))
            if c["role"] == "checkpoint"
        ]

    # list_lora_ckpts 展示序：epoch 降序
    assert [c["checkpoint_path"] for c in cands] == [
        "output/model_epoch6.safetensors",
        "output/model_epoch4.safetensors",
        "output/model_epoch2.safetensors",
    ]


# ---------------------------------------------------------------------------
# 规模预估
# ---------------------------------------------------------------------------

def test_eval_scale_reports_one_task_and_stage_count(isolated) -> None:
    project, version, vdir = _project_version(isolated, validation=3)
    (vdir / "output" / "model_final.safetensors").write_bytes(b"lora")
    _enable_validation(project, version)  # 无策略键 → final

    scale = eval_auto.eval_scale(project, version, vdir)

    assert scale["checkpoints_total"] == 2  # epoch2 + final
    assert scale["checkpoints_selected"] == 2  # 默认全评
    assert scale["skip_count"] == 0
    assert scale["validation_images"] == 3
    assert scale["metric_runners"] == ["clip", "dino"]
    assert scale["candidates"] == 3  # 2 checkpoint + baseline
    assert scale["images"] == 9
    assert scale["stages"] == 3  # 出图 + clip + dino
    assert scale["tasks"] == 1


def test_eval_scale_honors_explicit_selection(isolated) -> None:
    project, version, vdir = _project_version(isolated, validation=3)
    _enable_validation(project, version)

    scale = eval_auto.eval_scale(project, version, vdir, selected_count=10)

    assert scale["checkpoints_selected"] == 10
    assert scale["candidates"] == 11  # 10 + baseline
    assert scale["images"] == 33
    assert scale["tasks"] == 1
    assert scale["skip_count"] is None  # 显式选择不走采样


def test_eval_scale_zero_selection_has_no_baseline(isolated) -> None:
    project, version, vdir = _project_version(isolated, validation=2)
    _enable_validation(project, version)

    scale = eval_auto.eval_scale(project, version, vdir, selected_count=0)

    assert scale["baseline"] is False
    assert scale["candidates"] == 0
    assert scale["images"] == 0
    assert scale["tasks"] == 1


# ---------------------------------------------------------------------------
# supervisor 接线
# ---------------------------------------------------------------------------

def test_supervisor_queues_session_after_task_done(isolated) -> None:
    project, version, vdir = _project_version(isolated)
    _enable_validation(project, version)
    events: list[dict[str, Any]] = []
    sup = Supervisor(db_path=isolated["db"], on_event=events.append)

    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(
            conn, name="train", config_name="train", task_type="train",
            project_id=int(project["id"]), version_id=int(version["id"]),
        )
        db.update_task(conn, tid, status="running")

    sup._queue_auto_eval_after_training(tid, {"epoch": 2})

    queued = [e for e in events if e["type"] == "eval_auto_after_training_queued"]
    assert len(queued) == 1
    assert queued[0]["task_id"] == tid
    assert queued[0]["session_id"]
    assert queued[0]["count"] == 2  # 1 checkpoint + baseline
    with db.connection_for(isolated["db"]) as conn:
        sessions = eval_session.list_sessions(conn, parent_task_id=tid)
    assert len(sessions) == 1
    assert int(sessions[0]["task_id"]) == int(queued[0]["eval_task_id"])


# ---------------------------------------------------------------------------
# 端到端不变量（设计稿 §0.3 第 10 项）
# ---------------------------------------------------------------------------

def test_two_hundred_checkpoints_dispatch_as_one_job(isolated) -> None:
    """#465 的核心不变量，走**完整调度循环**验证。

    旧模型：200 个 checkpoint × (1 出图 + 2 指标) + baseline = 603 个 pending 作业行，
    每个派发后建一个 `studio_data/tasks/<id>/` 只写一个 run.log；而且出图作业占 exclusive
    档彼此排队、指标作业等着各自的出图，一次评估就把队列占满。

    新模型：1 个 `eval_session` 作业，阶段编排在 worker 内部。调度器只看到一个。
    """
    from unittest.mock import MagicMock
    from studio.services.inference import daemon as _daemon_mod

    project, version, vdir = _project_version(isolated)
    out = vdir / "output"
    for epoch in range(1, 201):
        (out / f"model_epoch{epoch}.safetensors").write_bytes(b"lora")
    (out / "model_final.safetensors").write_bytes(b"lora")
    _enable_validation(project, version)
    task = {"id": 41, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        session = eval_auto.queue_training_finished_eval(conn, task, {})
        pending = conn.execute(
            "SELECT * FROM tasks WHERE status = 'pending'"
        ).fetchall()
        candidates = eval_session.list_candidates(conn, int(session["id"]))
        metric_rows = conn.execute(
            "SELECT COUNT(*) FROM eval_metric_results"
        ).fetchone()[0]

    # 202 个候选（201 checkpoint + baseline）压进**一个**待派发作业
    assert len(candidates) == 202
    assert len(pending) == 1
    assert pending[0]["task_type"] == "eval_session"
    # 指标 placeholder 按 候选 × 启用指标 布点（默认 clip_t/clip_i/dino_i）
    assert metric_rows == 202 * 3

    fake = MagicMock()
    fake.is_model_loaded = False
    fake.is_busy = False
    _daemon_mod._INSTANCE = fake  # type: ignore[attr-defined]
    try:
        sup = Supervisor(db_path=isolated["db"], on_event=lambda _e: None)
        spawned: list = []

        def fake_spawn(_slot, job) -> None:
            # 真实 _spawn_job 会把作业标 running；mock 也要做，否则第二次派发时它还是
            # pending，测不出「不重复派发」。
            spawned.append(job)
            with db.connection_for(isolated["db"]) as conn:
                project_jobs.mark_running(conn, int(job["id"]))

        sup._spawn_job = fake_spawn  # type: ignore[method-assign]
        # eval_session 是 exclusive 档：跟 train / generate 走同一条 FIFO
        # （_dispatch_exclusive_tasks），执行位落 DATA 槽。不是 _dispatch_data。
        train_slot = next(s for s in sup._slots if s.name == "train")
        sup._dispatch_exclusive_tasks(train_slot)
        sup._dispatch_exclusive_tasks(train_slot)  # 已 running → 不该再派
    finally:
        _daemon_mod._INSTANCE = None  # type: ignore[attr-defined]

    assert len(spawned) == 1
    assert spawned[0]["kind"] == "eval_session"
    assert int(spawned[0]["id"]) == int(session["task_id"])


def test_repeated_evaluations_do_not_accumulate_pending_jobs(isolated) -> None:
    """重跑不清历史（A 方案），但也不该让待派发队列越堆越长 —— 每次一个作业。"""
    project, version, vdir = _project_version(isolated)
    for epoch in (4, 6, 8):
        (vdir / "output" / f"model_epoch{epoch}.safetensors").write_bytes(b"lora")
    task = {"id": 43, "project_id": project["id"], "version_id": version["id"]}

    with db.connection_for(isolated["db"]) as conn:
        for _ in range(5):
            eval_auto.queue_manual_task_eval(
                conn, task, ["model_epoch4.safetensors", "model_epoch8.safetensors"]
            )
        sessions = eval_session.list_sessions(conn, parent_task_id=43)
        eval_tasks = conn.execute(
            "SELECT * FROM tasks WHERE task_type = 'eval_session'"
        ).fetchall()

    # 5 次评估 = 5 个 Session（历史全留）= 5 个作业行，不是 5 × 候选数 × 指标数
    assert len(sessions) == 5
    assert len(eval_tasks) == 5
