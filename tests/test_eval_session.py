"""EvalSession 数据模型 + EvalPlan 冻结（issue #465 刀 1）。

覆盖：一次评估只产生 1 个 task 行、plan 冻结验证集口径、candidates/metric placeholder
布点、状态 rollup（done / partial / failed）、报告的 baseline Δ、历史 Session 全保留。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from studio import db, secrets
from studio.infrastructure import paths as infra_paths
from studio.services import eval_session
from studio.services.projects import jobs as project_jobs, projects, versions


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


def _setup(isolated, *, validation: int = 2, ckpts=("epoch2", "final")):
    with db.connection_for(isolated["db"]) as conn:
        project = projects.create_project(conn, title="Session")
        version = versions.create_version(conn, project_id=project["id"], label="v1")
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    train = vdir / "train" / "1_data"
    train.mkdir(parents=True, exist_ok=True)
    (train / "a.png").write_bytes(b"png")
    val = vdir / "validation" / "1_data"
    val.mkdir(parents=True, exist_ok=True)
    for i in range(validation):
        (val / f"v{i}.png").write_bytes(b"png")
        (val / f"v{i}.txt").write_text("1girl, solo", encoding="utf-8")
    out = vdir / "output"
    out.mkdir(parents=True, exist_ok=True)
    for name in ckpts:
        (out / f"model_{name}.safetensors").write_bytes(b"lora")
    return project, version, vdir


def _all_ckpts(vdir: Path) -> list[dict[str, Any]]:
    return versions.list_lora_ckpts(vdir)


def test_one_session_creates_exactly_one_task(isolated) -> None:
    """核心不变量（设计稿 §0.3 第 10 项）：一次评估只产生一个用户可见 task。"""
    project, version, vdir = _setup(isolated)
    with db.connection_for(isolated["db"]) as conn:
        session = eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="manual",
        )
        rows = conn.execute("SELECT * FROM tasks").fetchall()
        task = db.get_task(conn, int(session["task_id"]))

    assert len(rows) == 1
    assert rows[0]["task_type"] == "eval_session"
    assert int(rows[0]["id"]) == int(session["task_id"])
    # worker 靠 params.session_id 找回自己要跑哪个 Session
    assert task is not None
    assert int(task["params_decoded"]["session_id"]) == int(session["id"])


def test_many_checkpoints_still_one_task(isolated) -> None:
    """200 个 checkpoint 也只有 1 个 task —— 旧模型这里会是 603 个（#465）。"""
    project, version, vdir = _setup(isolated, ckpts=())
    out = vdir / "output"
    for epoch in range(1, 201):
        (out / f"model_epoch{epoch}.safetensors").write_bytes(b"lora")
    with db.connection_for(isolated["db"]) as conn:
        eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="after_training",
        )
        task_count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        cand_count = conn.execute("SELECT COUNT(*) FROM eval_candidates").fetchone()[0]

    assert task_count == 1
    assert cand_count == 201  # 200 checkpoint + 1 baseline


def test_plan_freezes_validation_and_checkpoints(isolated) -> None:
    project, version, vdir = _setup(isolated, validation=3)
    with db.connection_for(isolated["db"]) as conn:
        session = eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="manual",
            skip_count=0,
        )
    plan = session["plan"]

    assert plan["schema_version"] == eval_session.PLAN_SCHEMA_VERSION
    assert plan["reference_manifest"]["count"] == 3
    assert plan["reference_manifest"]["digest"]
    assert len(plan["candidates"]) == 2
    assert all(c["digest"] for c in plan["candidates"])
    assert plan["checkpoint_sampling"] == {"skip_count": 0}
    # plan.json 落盘（DB plan_json 的人类可读副本）
    assert infra_paths.eval_session_plan_path(int(session["id"])).exists()


def test_plan_is_immutable_against_later_validation_changes(isolated) -> None:
    """创建后往 validation/ 加图不改这次评估的口径 —— 这是 EvalPlan 存在的意义。"""
    project, version, vdir = _setup(isolated, validation=2)
    with db.connection_for(isolated["db"]) as conn:
        session = eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="manual",
        )
        (vdir / "validation" / "1_data" / "extra.png").write_bytes(b"png")
        reread = eval_session.get_session(conn, int(session["id"]))

    assert reread is not None
    assert reread["plan"]["reference_manifest"]["count"] == 2


def test_metric_placeholders_cover_every_candidate(isolated) -> None:
    project, version, vdir = _setup(isolated)
    with db.connection_for(isolated["db"]) as conn:
        session = eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="manual",
            metric_keys=["clip_t", "clip_i", "dino_i"],
        )
        sid = int(session["id"])
        candidates = eval_session.list_candidates(conn, sid)
        results = eval_session.list_metric_results(conn, sid)

    assert [c["role"] for c in candidates] == ["checkpoint", "checkpoint", "baseline"]
    assert all(c["samples_total"] == 2 for c in candidates)
    for cand in candidates:
        keys = {r["metric_key"] for r in results[int(cand["id"])]}
        assert keys == {"clip_t", "clip_i", "dino_i"}
        assert all(r["status"] == "pending" for r in results[int(cand["id"])])


def test_baseline_can_be_disabled(isolated) -> None:
    project, version, vdir = _setup(isolated)
    with db.connection_for(isolated["db"]) as conn:
        session = eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="manual", baseline=False,
        )
        candidates = eval_session.list_candidates(conn, int(session["id"]))

    assert all(c["role"] == "checkpoint" for c in candidates)
    assert session["plan"]["baseline"]["enabled"] is False


def test_empty_checkpoint_set_is_rejected(isolated) -> None:
    project, version, vdir = _setup(isolated)
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(eval_session.EvalSessionError):
            eval_session.create_session(
                conn, project, version, vdir, checkpoints=[], trigger="manual",
            )
        # 失败不留半个 Session
        assert conn.execute("SELECT COUNT(*) FROM eval_sessions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_history_sessions_are_all_kept(isolated) -> None:
    """A 方案：每次评估一个新 Session，旧的留档（推翻 ADR 0011 Addendum §5）。"""
    project, version, vdir = _setup(isolated)
    with db.connection_for(isolated["db"]) as conn:
        first = eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="after_training",
        )
        second = eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="manual",
        )
        listed = eval_session.list_sessions(conn, version_id=int(version["id"]))
        # 第一个 Session 的行、候选、plan 文件全都还在
        first_id = int(first["id"])
        assert eval_session.get_session(conn, first_id) is not None
        assert len(eval_session.list_candidates(conn, first_id)) == 3
        assert infra_paths.eval_session_plan_path(first_id).exists()

    # 最新在前
    assert [int(s["id"]) for s in listed] == [int(second["id"]), first_id]
    # 两个 Session 各有独立的 task 行和独立目录
    assert int(first["task_id"]) != int(second["task_id"])
    assert eval_session.session_dir(first_id) != eval_session.session_dir(int(second["id"]))


def test_rollup_status_transitions() -> None:
    cands = [{"id": 1, "status": "done"}, {"id": 2, "status": "done"}]

    all_ok = {1: [{"status": "done"}], 2: [{"status": "done"}]}
    assert eval_session.rollup_status(cands, all_ok) == eval_session.STATUS_DONE

    mixed = {1: [{"status": "done"}], 2: [{"status": "failed"}]}
    assert eval_session.rollup_status(cands, mixed) == eval_session.STATUS_PARTIAL

    all_bad = {1: [{"status": "failed"}], 2: [{"status": "failed"}]}
    assert eval_session.rollup_status(cands, all_bad) == eval_session.STATUS_FAILED

    assert eval_session.rollup_status([], {}) == eval_session.STATUS_FAILED


def test_rollup_candidate_failure_counts_even_with_metric_done() -> None:
    """候选自己失败（比如出图崩了）→ 即使有指标写过值也算 bad。"""
    cands = [{"id": 1, "status": "failed"}, {"id": 2, "status": "done"}]
    results = {1: [{"status": "done"}], 2: [{"status": "done"}]}
    assert eval_session.rollup_status(cands, results) == eval_session.STATUS_PARTIAL


def test_report_computes_baseline_delta(isolated) -> None:
    project, version, vdir = _setup(isolated)
    with db.connection_for(isolated["db"]) as conn:
        session = eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="manual",
            metric_keys=["clip_i"],
        )
        sid = int(session["id"])
        cands = eval_session.list_candidates(conn, sid)
        ckpt = next(c for c in cands if c["role"] == "checkpoint")
        base = next(c for c in cands if c["role"] == "baseline")
        eval_session.set_metric_result(
            conn, int(ckpt["id"]), "clip_i", status="done", value=0.72, sample_count=2,
        )
        eval_session.set_metric_result(
            conn, int(base["id"]), "clip_i", status="done", value=0.61, sample_count=2,
        )
        report = eval_session.write_report(conn, sid)

    assert report is not None
    assert report["baseline_metrics"]["clip_i"] == pytest.approx(0.61)
    row = next(r for r in report["candidates"] if r["candidate_id"] == int(ckpt["id"]))
    assert row["metrics"]["clip_i"]["value"] == pytest.approx(0.72)
    assert row["metrics"]["clip_i"]["delta"] == pytest.approx(0.11)
    assert infra_paths.eval_session_report_path(int(session["id"])).exists()


def test_set_metric_result_upserts(isolated) -> None:
    """(candidate, metric) 上有唯一索引 —— 重跑同一指标是覆盖，不是插重复行。"""
    project, version, vdir = _setup(isolated)
    with db.connection_for(isolated["db"]) as conn:
        session = eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="manual", metric_keys=["clip_i"],
        )
        cand = eval_session.list_candidates(conn, int(session["id"]))[0]
        for value in (0.1, 0.2, 0.3):
            eval_session.set_metric_result(
                conn, int(cand["id"]), "clip_i", status="done", value=value,
            )
        rows = eval_session.list_metric_results(conn, int(session["id"]))[int(cand["id"])]

    assert len(rows) == 1
    assert rows[0]["value"] == pytest.approx(0.3)


def test_update_rejects_unknown_fields(isolated) -> None:
    project, version, vdir = _setup(isolated)
    with db.connection_for(isolated["db"]) as conn:
        session = eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="manual",
        )
        with pytest.raises(eval_session.EvalSessionError):
            eval_session.update_session(conn, int(session["id"]), bogus="x")
        cand = eval_session.list_candidates(conn, int(session["id"]))[0]
        with pytest.raises(eval_session.EvalSessionError):
            eval_session.update_candidate(conn, int(cand["id"]), bogus="x")


def test_delete_session_cascades_and_removes_dir(isolated) -> None:
    project, version, vdir = _setup(isolated)
    with db.connection_for(isolated["db"]) as conn:
        session = eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="manual",
        )
        sid = int(session["id"])
        assert eval_session.session_dir(sid).exists()

        assert eval_session.delete_session(conn, sid) is True

        assert eval_session.get_session(conn, sid) is None
        assert conn.execute(
            "SELECT COUNT(*) FROM eval_candidates WHERE session_id = ?", (sid,)
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM eval_metric_results").fetchone()[0] == 0
        assert not eval_session.session_dir(sid).exists()
        # checkpoint 是引用，删 Session 不能动它
        assert (vdir / "output" / "model_final.safetensors").exists()
        assert eval_session.delete_session(conn, sid) is False


def test_resource_summary_reports_one_task(isolated) -> None:
    """规模摘要：Session 模型下永远 1 个 task，成本用出图数 + 阶段数表达。"""
    project, version, vdir = _setup(isolated, validation=3)
    summary = eval_session.resource_summary(
        project, version, vdir, selected_count=4, metric_keys=["clip_t", "clip_i", "dino_i"],
    )

    assert summary["candidates"] == 5  # 4 + baseline
    assert summary["validation_images"] == 3
    assert summary["images"] == 15
    assert summary["metric_runners"] == ["clip", "dino"]
    assert summary["stages"] == 3  # 出图 + clip + dino
    assert summary["tasks"] == 1


def test_resource_summary_zero_selection(isolated) -> None:
    project, version, vdir = _setup(isolated)
    summary = eval_session.resource_summary(project, version, vdir, selected_count=0)
    assert summary["baseline"] is False
    assert summary["candidates"] == 0
    assert summary["images"] == 0
    assert summary["tasks"] == 1


# ---------------------------------------------------------------------------
# 与 task 终态对齐 + 重试（worker 被 kill 时 Session 会停在 running）
# ---------------------------------------------------------------------------

def _make_session(isolated):
    project, version, vdir = _setup(isolated)
    with db.connection_for(isolated["db"]) as conn:
        session = eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=_all_ckpts(vdir), trigger="manual",
        )
    return session


def _mark_running(isolated, session_id: int) -> None:
    with db.connection_for(isolated["db"]) as conn:
        eval_session.update_session(
            conn, session_id, status=eval_session.STATUS_RUNNING,
            stage=eval_session.STAGE_GENERATE,
        )
        for cand in eval_session.list_candidates(conn, session_id):
            eval_session.update_candidate(
                conn, int(cand["id"]), status=eval_session.STATUS_RUNNING
            )


def test_reconcile_pulls_session_out_of_running_when_task_failed(isolated) -> None:
    """worker 被 kill → 最后一次写没发生，Session 停在 running；靠 task 终态兜底。"""
    session = _make_session(isolated)
    sid, task_id = int(session["id"]), int(session["task_id"])
    _mark_running(isolated, sid)

    with db.connection_for(isolated["db"]) as conn:
        db.update_task(conn, task_id, status="failed", error_msg="worker killed")
        fixed = eval_session.reconcile_with_task(
            conn, eval_session.get_session(conn, sid) or {}
        )
        candidates = eval_session.list_candidates(conn, sid)
        results = eval_session.list_metric_results(conn, sid)

    assert fixed["status"] == eval_session.STATUS_FAILED
    assert fixed["stage"] is None
    assert fixed["finished_at"] is not None
    assert "worker killed" in str(fixed["error"])
    # 候选和指标一并收口，否则前端「评估中 n/m」永不消失
    assert {c["status"] for c in candidates} == {eval_session.STATUS_FAILED}
    assert all(
        r["status"] == "skipped" for rows in results.values() for r in rows
    )


def test_reconcile_maps_canceled_task_to_canceled_session(isolated) -> None:
    session = _make_session(isolated)
    sid, task_id = int(session["id"]), int(session["task_id"])
    _mark_running(isolated, sid)

    with db.connection_for(isolated["db"]) as conn:
        db.update_task(conn, task_id, status="canceled")
        fixed = eval_session.reconcile_with_task(
            conn, eval_session.get_session(conn, sid) or {}
        )
        candidates = eval_session.list_candidates(conn, sid)

    assert fixed["status"] == eval_session.STATUS_CANCELED
    assert {c["status"] for c in candidates} == {eval_session.STATUS_CANCELED}


def test_reconcile_leaves_live_session_alone(isolated) -> None:
    """task 还没到终态 → 一个字都不能改（否则会把正在跑的评估judged成失败）。"""
    session = _make_session(isolated)
    sid = int(session["id"])
    _mark_running(isolated, sid)

    with db.connection_for(isolated["db"]) as conn:
        before = eval_session.get_session(conn, sid) or {}
        after = eval_session.reconcile_with_task(conn, before)

    assert after["status"] == eval_session.STATUS_RUNNING
    assert after["stage"] == eval_session.STAGE_GENERATE
    assert after["finished_at"] is None


def test_reconcile_fails_session_when_task_row_is_gone(isolated) -> None:
    """队列历史被清掉 → 没人会再推进这个 Session，不能让它挂着 running。"""
    session = _make_session(isolated)
    sid, task_id = int(session["id"]), int(session["task_id"])
    _mark_running(isolated, sid)

    with db.connection_for(isolated["db"]) as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        fixed = eval_session.reconcile_with_task(
            conn, eval_session.get_session(conn, sid) or {}
        )

    assert fixed["status"] == eval_session.STATUS_FAILED
    assert "已删除" in str(fixed["error"])


def test_retry_queues_a_new_task_and_reopens_the_session(isolated) -> None:
    """重试 = 新建一条 task 行（每次尝试留档），Session 回到 pending 等断点续跑。"""
    session = _make_session(isolated)
    sid, first_task = int(session["id"]), int(session["task_id"])

    with db.connection_for(isolated["db"]) as conn:
        eval_session.update_session(
            conn, sid, status=eval_session.STATUS_FAILED, error="boom"
        )
        retried = eval_session.retry_session(conn, sid)
        tasks = conn.execute(
            "SELECT id FROM tasks WHERE task_type = ? ORDER BY id",
            (eval_session.TASK_TYPE,),
        ).fetchall()

    assert retried["status"] == eval_session.STATUS_PENDING
    assert retried["error"] is None
    assert retried["finished_at"] is None
    assert int(retried["task_id"]) != first_task
    assert [int(r["id"]) for r in tasks] == [first_task, int(retried["task_id"])]


def test_retry_refuses_while_still_running(isolated) -> None:
    session = _make_session(isolated)
    sid = int(session["id"])
    _mark_running(isolated, sid)

    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(eval_session.EvalSessionError):
            eval_session.retry_session(conn, sid)


def test_retry_keeps_finished_candidates_for_resume(isolated) -> None:
    """断点续跑的前提：已 done 的候选不能被重试重置（否则 200 个全部重来）。"""
    session = _make_session(isolated)
    sid = int(session["id"])
    with db.connection_for(isolated["db"]) as conn:
        first = eval_session.list_candidates(conn, sid)[0]
        eval_session.update_candidate(
            conn, int(first["id"]), status=eval_session.STATUS_DONE, run_id="run-x"
        )
        eval_session.update_session(conn, sid, status=eval_session.STATUS_FAILED)
        eval_session.retry_session(conn, sid)
        after = eval_session.list_candidates(conn, sid)

    assert after[0]["status"] == eval_session.STATUS_DONE
    assert after[0]["run_id"] == "run-x"


# ---------------------------------------------------------------------------
# 版本级手动评估（评估的对象是 checkpoint，不必存在对应的训练 task）
# ---------------------------------------------------------------------------

def test_version_scoped_manual_eval_has_no_parent_task(isolated) -> None:
    from studio.services import eval_auto

    project, version, vdir = _setup(isolated)
    paths = [c["path"] for c in _all_ckpts(vdir)]
    with db.connection_for(isolated["db"]) as conn:
        session = eval_auto.queue_manual_eval(conn, project, version, vdir, paths)

    assert session is not None
    assert session["parent_task_id"] is None
    assert session["trigger"] == "manual"
    assert int(session["version_id"]) == int(version["id"])


def test_version_scoped_session_is_listed_by_version(isolated) -> None:
    """没有 parent task 的评估必须能从 version 查到 —— 否则它就永远找不回来了。"""
    from studio.services import eval_auto

    project, version, vdir = _setup(isolated)
    paths = [c["path"] for c in _all_ckpts(vdir)]
    with db.connection_for(isolated["db"]) as conn:
        created = eval_auto.queue_manual_eval(conn, project, version, vdir, paths)
        by_version = eval_session.list_sessions(
            conn, project_id=int(project["id"]), version_id=int(version["id"])
        )
        by_task = eval_session.list_sessions(conn, parent_task_id=1)

    assert [int(s["id"]) for s in by_version] == [int(created["id"])]
    assert by_task == []
