"""EvalSession worker 编排（issue #465 刀 2）。

出图与指标算法本身由各自的测试覆盖（test_eval_samples / test_eval_clip / …），这里
只测**编排**：阶段顺序、断点续跑、失败隔离、状态 rollup、退出码。所以出图和指标都打成
假实现，不碰 GPU。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import contextlib

import pytest

from studio import db, secrets
from studio.infrastructure import paths as infra_paths
from studio.services import eval_generation, eval_samples, eval_session
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.workers import eval_session_worker as worker


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
        project = projects.create_project(conn, title="Worker")
        version = versions.create_version(conn, project_id=project["id"], label="v1")
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    (vdir / "train" / "1_data").mkdir(parents=True, exist_ok=True)
    (vdir / "train" / "1_data" / "a.png").write_bytes(b"png")
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


def _make_session(isolated, project, version, vdir, **kw) -> dict[str, Any]:
    with db.connection_for(isolated["db"]) as conn:
        return eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=versions.list_lora_ckpts(vdir),
            trigger=kw.pop("trigger", "manual"),
            **kw,
        )


def _fake_generator(run: dict[str, Any], version_dir: Path, progress) -> None:
    """写假图并逐条标 done —— 复刻真出图对 run.json 的写入，不碰模型。"""
    eval_root = (
        Path(str(run["eval_root"]))
        if run.get("storage_scope") == "task" and run.get("eval_root")
        else None
    )
    for idx, item in enumerate(run["items"]):
        progress(f"fake gen {idx}")
        run = eval_samples.mark_item_running(version_dir, run, idx, eval_root)
        path = eval_samples.sample_image_path(
            version_dir, run["run_id"], item["filename"], eval_root
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PNG")
        run = eval_samples.mark_item_done(version_dir, run, idx, eval_root)


class _NoopGenerator:
    """替掉 DaemonSampleGenerator：出图由 `_fake_generator` 顶，不许起真 daemon。

    真实现的 `__enter__` 会 spawn `runtime/anima_daemon.py` 子进程（torch import
    好几秒、READY_TIMEOUT 30s），在测试里既慢又依赖机器状态 —— CI 无 GPU、runner
    忙的时候正好是 flake 的温床。
    """

    def __init__(self, *_a, **_kw) -> None:
        pass

    def __enter__(self) -> "_NoopGenerator":
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def __call__(self, *_a, **_kw) -> None:  # 不会被调到（run_sample_job 已被换掉）
        raise AssertionError("测试不该走到真的 daemon 出图")


@contextlib.contextmanager
def _noop_shared_scorer(_progress=None):
    """替掉真的 shared_scorer：假 runner 不需要模型，也绝不能去加载。

    真实现会在这里 load CLIP / DINO / WD14；测试里必须是 no-op，否则每个用例都要
    下模型、进 CUDA（CI 无 GPU）。yield None = 让 run_fn 走它自己的 scorer 参数。
    """
    yield None


@pytest.fixture
def fake_generate(monkeypatch: pytest.MonkeyPatch):
    """让 run_sample_job 走假 generator（真签名、真 run.json 写入）。"""
    real = eval_samples.run_sample_job

    def patched(project, version, vdir, run_id, **kw):
        kw.pop("generator", None)
        return real(project, version, vdir, run_id, generator=_fake_generator, **kw)

    monkeypatch.setattr(eval_samples, "run_sample_job", patched)
    monkeypatch.setattr(
        eval_generation, "DaemonSampleGenerator", _NoopGenerator,
    )
    return patched


def _metric_states(**values: float) -> dict[str, Any]:
    return {
        "metric_states": {
            key: {
                "key": key, "status": "done", "value": value,
                "model_name": "fake-model", "sample_count": 2,
            }
            for key, value in values.items()
        }
    }


@pytest.fixture
def fake_metrics(monkeypatch: pytest.MonkeyPatch):
    """clip / dino runner 打成假实现，返回固定分数；记录被调用的 run_id。"""
    calls: dict[str, list[str]] = {"clip": [], "dino": []}

    def fake_clip(project, version, vdir, run_id, **kw):
        calls["clip"].append(run_id)
        return _metric_states(clip_t=0.05, clip_i=0.7)

    def fake_dino(project, version, vdir, run_id, **kw):
        calls["dino"].append(run_id)
        return _metric_states(dino_i=0.6)

    monkeypatch.setitem(
        worker._RUNNERS, "clip",
        (fake_clip, lambda _cfg: "fake-clip", _noop_shared_scorer)
    )
    monkeypatch.setitem(
        worker._RUNNERS, "dino",
        (fake_dino, lambda _cfg: "fake-dino", _noop_shared_scorer)
    )
    return calls


def _read(isolated, session_id: int):
    with db.connection_for(isolated["db"]) as conn:
        return (
            eval_session.get_session(conn, session_id),
            eval_session.list_candidates(conn, session_id),
            eval_session.list_metric_results(conn, session_id),
        )


# ---------------------------------------------------------------------------

def test_full_run_produces_done_session_and_report(
    isolated, fake_generate, fake_metrics
) -> None:
    project, version, vdir = _setup(isolated)
    session = _make_session(isolated, project, version, vdir)
    sid, task_id = int(session["id"]), int(session["task_id"])

    assert worker.run(task_id) == 0

    got, candidates, results = _read(isolated, sid)
    assert got["status"] == eval_session.STATUS_DONE
    assert got["stage"] is None
    assert got["started_at"] and got["finished_at"]
    # 2 checkpoint + 1 baseline，全部出图完成
    assert len(candidates) == 3
    assert all(c["status"] == eval_session.STATUS_DONE for c in candidates)
    assert all(c["samples_done"] == 2 for c in candidates)
    assert all(c["run_id"] for c in candidates)
    # 每个候选 3 个指标都有值
    for cand in candidates:
        rows = {r["metric_key"]: r for r in results[int(cand["id"])]}
        assert set(rows) == {"clip_t", "clip_i", "dino_i"}
        assert all(r["status"] == eval_session.STATUS_DONE for r in rows.values())
        assert rows["clip_i"]["value"] == pytest.approx(0.7)
        assert rows["clip_i"]["sample_count"] == 2
    assert infra_paths.eval_session_report_path(sid).exists()
    # 每个 runner 对每个候选各跑一次（批量过候选，不是每候选重启 runner）
    assert len(fake_metrics["clip"]) == 3
    assert len(fake_metrics["dino"]) == 3


def test_still_only_one_task_row_after_running(
    isolated, fake_generate, fake_metrics
) -> None:
    """#465 的核心：跑完仍然只有一个 task 行，不会派生子作业。"""
    project, version, vdir = _setup(isolated)
    session = _make_session(isolated, project, version, vdir)

    worker.run(int(session["task_id"]))

    with db.connection_for(isolated["db"]) as conn:
        rows = conn.execute("SELECT task_type FROM tasks").fetchall()
    assert [r["task_type"] for r in rows] == ["eval_session"]


def test_baseline_candidate_generates_with_lora_disabled(
    isolated, fake_generate, fake_metrics
) -> None:
    """baseline 是纯底模对照 —— 出图必须 lora_scale=0，否则 Δ 恒为 0。"""
    project, version, vdir = _setup(isolated)
    session = _make_session(isolated, project, version, vdir)
    sid = int(session["id"])

    worker.run(int(session["task_id"]))

    _got, candidates, _results = _read(isolated, sid)
    base = next(c for c in candidates if c["role"] == "baseline")
    run = eval_samples.load_run(
        vdir, str(base["run_id"]), eval_session.samples_root(sid)
    )
    assert run is not None
    assert run["baseline"] is True
    assert run["generation"]["lora_scale"] == 0.0


def test_rerun_skips_completed_candidates_and_metrics(
    isolated, fake_generate, fake_metrics
) -> None:
    """断点续跑：第二次跑不重复出图、不重算已 done 的指标。"""
    project, version, vdir = _setup(isolated)
    session = _make_session(isolated, project, version, vdir)
    task_id = int(session["task_id"])

    assert worker.run(task_id) == 0
    first_clip_calls = len(fake_metrics["clip"])
    _got, candidates_before, _ = _read(isolated, int(session["id"]))
    run_ids_before = [c["run_id"] for c in candidates_before]

    assert worker.run(task_id) == 0

    # runner 一次都没再被调用；run_id 也没换（没重新建 run）
    assert len(fake_metrics["clip"]) == first_clip_calls
    _got2, candidates_after, _ = _read(isolated, int(session["id"]))
    assert [c["run_id"] for c in candidates_after] == run_ids_before


def test_generation_marked_done_but_run_missing_is_redone(
    isolated, fake_generate, fake_metrics
) -> None:
    """DB 说 done 但 run 目录被清掉 → 必须重出图，不能光信 DB 直接进指标阶段。"""
    project, version, vdir = _setup(isolated)
    session = _make_session(isolated, project, version, vdir)
    sid = int(session["id"])
    worker.run(int(session["task_id"]))

    _got, candidates, _ = _read(isolated, sid)
    victim = candidates[0]
    samples_root = eval_session.samples_root(sid)
    eval_samples.delete_all_runs(vdir, samples_root)
    # DB 仍标 done，但所有 run 文件都没了
    assert eval_samples.load_run(vdir, str(victim["run_id"]), samples_root) is None

    assert worker.run(int(session["task_id"])) == 0

    # 重新出了图 —— 断言 run 文件回来且是 done，不断言 run_id 变了：run_id 是秒级
    # 时间戳，同一秒内重建会拿到同一个 id（旧文件已删，不触发 -2 后缀），那不是问题。
    _got2, after, results = _read(isolated, sid)
    fresh = next(c for c in after if int(c["id"]) == int(victim["id"]))
    assert fresh["status"] == eval_session.STATUS_DONE
    run = eval_samples.load_run(vdir, str(fresh["run_id"]), samples_root)
    assert run is not None
    assert run["status"] == "done"
    assert run["summary"]["done"] == 2
    # 图回来了，指标也重新算出了值
    assert all(
        r["status"] == eval_session.STATUS_DONE for r in results[int(fresh["id"])]
    )


def test_one_candidate_failing_leaves_session_partial(
    isolated, monkeypatch, fake_metrics
) -> None:
    """一个候选出图崩了，其余照跑完；整体 partial 而不是 failed，退出码仍是 0。"""
    project, version, vdir = _setup(isolated)
    session = _make_session(isolated, project, version, vdir)
    sid = int(session["id"])
    real = eval_samples.run_sample_job
    seen: list[str] = []

    def flaky(project_, version_, vdir_, run_id, **kw):
        seen.append(run_id)
        if len(seen) == 1:
            raise RuntimeError("boom: cuda oom")
        kw.pop("generator", None)
        return real(project_, version_, vdir_, run_id, generator=_fake_generator, **kw)

    monkeypatch.setattr(eval_samples, "run_sample_job", flaky)
    monkeypatch.setattr(
        eval_generation, "DaemonSampleGenerator", _NoopGenerator,
    )

    assert worker.run(int(session["task_id"])) == 0

    got, candidates, results = _read(isolated, sid)
    assert got["status"] == eval_session.STATUS_PARTIAL
    failed = [c for c in candidates if c["status"] == eval_session.STATUS_FAILED]
    assert len(failed) == 1
    assert "boom" in str(failed[0]["error"])
    # 失败候选的指标标 skipped（不是 failed —— 是没图可算，不是算错了）
    skipped = results[int(failed[0]["id"])]
    assert all(r["status"] == "skipped" for r in skipped)
    # 其余候选正常出结果
    ok = [c for c in candidates if c["status"] == eval_session.STATUS_DONE]
    assert len(ok) == 2
    assert all(
        r["status"] == eval_session.STATUS_DONE for r in results[int(ok[0]["id"])]
    )


def test_metric_failure_is_isolated_to_that_runner(
    isolated, fake_generate, monkeypatch
) -> None:
    """dino 挂了不影响 clip 的结果，整体 partial。"""
    project, version, vdir = _setup(isolated)
    session = _make_session(isolated, project, version, vdir)
    sid = int(session["id"])

    def ok_clip(project_, version_, vdir_, run_id, **kw):
        return _metric_states(clip_t=0.05, clip_i=0.7)

    def bad_dino(project_, version_, vdir_, run_id, **kw):
        raise RuntimeError("dino model missing")

    monkeypatch.setitem(worker._RUNNERS, "clip", (ok_clip, lambda _c: "fake-clip", _noop_shared_scorer))
    monkeypatch.setitem(worker._RUNNERS, "dino", (bad_dino, lambda _c: "fake-dino", _noop_shared_scorer))

    assert worker.run(int(session["task_id"])) == 0

    got, candidates, results = _read(isolated, sid)
    assert got["status"] == eval_session.STATUS_PARTIAL
    rows = {r["metric_key"]: r for r in results[int(candidates[0]["id"])]}
    assert rows["clip_i"]["status"] == eval_session.STATUS_DONE
    assert rows["dino_i"]["status"] == eval_session.STATUS_FAILED
    assert "dino model missing" in str(rows["dino_i"]["reason"])


def test_runner_not_run_status_is_recorded_verbatim(
    isolated, fake_generate, monkeypatch
) -> None:
    """runner 主动报 not_run / skipped（tag 遇自然语言 caption 等）照原状记，
    既不粉饰成 done 也不算失败。"""
    project, version, vdir = _setup(isolated)
    session = _make_session(
        isolated, project, version, vdir, metric_keys=["clip_t", "clip_i"],
    )
    sid = int(session["id"])

    def skipping_clip(project_, version_, vdir_, run_id, **kw):
        return {
            "metric_states": {
                "clip_t": {"key": "clip_t", "status": "not_run", "value": None,
                           "reason": "no caption"},
                "clip_i": {"key": "clip_i", "status": "done", "value": 0.7},
            }
        }

    monkeypatch.setitem(worker._RUNNERS, "clip", (skipping_clip, lambda _c: "m", _noop_shared_scorer))

    worker.run(int(session["task_id"]))

    _got, candidates, results = _read(isolated, sid)
    rows = {r["metric_key"]: r for r in results[int(candidates[0]["id"])]}
    assert rows["clip_t"]["status"] == "not_run"
    assert rows["clip_t"]["value"] is None
    assert rows["clip_i"]["status"] == eval_session.STATUS_DONE


def test_runner_gets_model_name_from_settings(
    isolated, fake_generate, monkeypatch
) -> None:
    """指标模型名来自全局 Settings —— 这条以前由各 metric 的 start_job 测试覆盖，
    排队路径收进 Session 后由 worker 的 model_getter 承担。故意只替换跑分函数、
    保留真实的 model_getter。"""
    project, version, vdir = _setup(isolated)
    secrets.update({"eval_metrics": {
        "clip_model_name": "/models/local-clip",
        "enabled_metrics": ["clip_t", "clip_i"],
    }})
    session = _make_session(isolated, project, version, vdir)
    seen: list[str | None] = []

    def spy_clip(project_, version_, vdir_, run_id, **kw):
        seen.append(kw.get("model_name"))
        return _metric_states(clip_t=0.05, clip_i=0.7)

    real_getter = worker._RUNNERS["clip"][1]
    monkeypatch.setitem(worker._RUNNERS, "clip", (spy_clip, real_getter, _noop_shared_scorer))

    assert worker.run(int(session["task_id"])) == 0

    assert seen and all(m == "/models/local-clip" for m in seen)
    _got, candidates, results = _read(isolated, int(session["id"]))
    rows = {r["metric_key"]: r for r in results[int(candidates[0]["id"])]}
    assert rows["clip_i"]["model_ref"] == "fake-model"  # runner 回报的名字优先


def test_unknown_runner_is_skipped_not_fatal(
    isolated, fake_generate, fake_metrics, monkeypatch
) -> None:
    """plan 里出现当前版本不认识的 runner（降级 / 老 plan）→ 跳过，不炸整个 Session。"""
    project, version, vdir = _setup(isolated)
    session = _make_session(isolated, project, version, vdir)
    sid = int(session["id"])
    with db.connection_for(isolated["db"]) as conn:
        import json
        plan = session["plan"]
        plan["metrics"]["runners"] = ["clip", "from_the_future"]
        conn.execute(
            "UPDATE eval_sessions SET plan_json = ? WHERE id = ?",
            (json.dumps(plan), sid),
        )
        conn.commit()

    assert worker.run(int(session["task_id"])) == 0
    got, _candidates, _results = _read(isolated, sid)
    assert got["status"] in (eval_session.STATUS_DONE, eval_session.STATUS_PARTIAL)


def test_missing_session_fails_cleanly(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        task_id = db.create_task(
            conn, name="eval_session", config_name="eval_session",
            task_type="eval_session", params={"session_id": 9999},
        )
    assert worker.run(task_id) == 1


def test_task_without_session_id_fails(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        task_id = db.create_task(
            conn, name="eval_session", config_name="eval_session",
            task_type="eval_session", params={},
        )
    assert worker.run(task_id) == 1


def test_all_candidates_failing_gives_failed_and_exit_1(
    isolated, monkeypatch, fake_metrics
) -> None:
    project, version, vdir = _setup(isolated)
    session = _make_session(isolated, project, version, vdir)
    sid = int(session["id"])

    def always_bad(*_a, **_kw):
        raise RuntimeError("nope")

    monkeypatch.setattr(eval_samples, "run_sample_job", always_bad)
    monkeypatch.setattr(
        eval_generation, "DaemonSampleGenerator", _NoopGenerator,
    )

    assert worker.run(int(session["task_id"])) == 1

    got, candidates, _results = _read(isolated, sid)
    assert got["status"] == eval_session.STATUS_FAILED
    assert all(c["status"] == eval_session.STATUS_FAILED for c in candidates)
