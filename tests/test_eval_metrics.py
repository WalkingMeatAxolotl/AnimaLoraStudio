"""LoRA eval metric result contract and endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studio import db, secrets, server
from studio.infrastructure import paths as infra_paths
from studio.services import eval_metrics, eval_samples, eval_session
from studio.services.projects import jobs as project_jobs, projects, versions


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(infra_paths, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(infra_paths, "EVAL_SESSIONS_DIR", tmp_path / "eval" / "sessions")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return {"db": dbfile}


@pytest.fixture
def client(isolated) -> TestClient:
    server.app.state.supervisor = None
    return TestClient(server.app)


def _new_project(isolated) -> tuple[dict[str, Any], dict[str, Any], Path]:
    with db.connection_for(isolated["db"]) as conn:
        project = projects.create_project(conn, title="Eval Metrics")
        version = versions.create_version(
            conn, project_id=project["id"], label="baseline"
        )
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    return project, version, vdir


def _seed_validation_and_ckpt(vdir: Path) -> None:
    val = vdir / "validation" / "1_data"
    val.mkdir(parents=True, exist_ok=True)
    (val / "a.png").write_bytes(b"png-a")
    (val / "a.txt").write_text("solo, red hair", encoding="utf-8")
    output = vdir / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_step100.safetensors").write_bytes(b"fake-lora")


def _sample_run(
    project: dict[str, Any],
    version: dict[str, Any],
    vdir: Path,
    *,
    eval_root: Path | None = None,
) -> dict[str, Any]:
    _seed_validation_and_ckpt(vdir)
    return eval_samples.create_run(
        project,
        version,
        vdir,
        checkpoint_path="model_step100.safetensors",
        eval_root=eval_root,
        now=2000.0,
    )


def _make(client: TestClient) -> tuple[int, int]:
    project = client.post("/api/projects", json={"title": "Eval Metrics HTTP"}).json()
    return project["id"], project["versions"][0]["id"]


def _vdir_for(pid: int, vid: int) -> tuple[dict[str, Any], dict[str, Any], Path]:
    with db.connection_for() as conn:
        project = projects.get_project(conn, pid)
        version = versions.get_version(conn, vid)
    assert project and version
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    return project, version, vdir


def test_baseline_run_sets_lora_scale_zero(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    run = eval_samples.create_run(
        project, version, vdir,
        checkpoint_path="model_step100.safetensors", baseline=True, now=2000.0,
    )
    assert run["baseline"] is True
    assert run["generation"]["lora_scale"] == 0.0


def test_list_results_attaches_baseline_delta(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    base = eval_samples.create_run(
        project, version, vdir,
        checkpoint_path="model_step100.safetensors", baseline=True, now=1000.0,
    )
    ckpt = eval_samples.create_run(
        project, version, vdir,
        checkpoint_path="model_step100.safetensors", now=2000.0,
    )
    eval_metrics.save_result(vdir, base["run_id"], {"metrics": {"clip_i": 0.60, "dino_i": 0.50}})
    eval_metrics.save_result(vdir, ckpt["run_id"], {"metrics": {"clip_i": 0.72, "dino_i": 0.50}})

    results = eval_metrics.list_results(vdir)
    base_res = next(r for r in results if r["baseline"])
    ckpt_res = next(r for r in results if not r["baseline"])
    assert "delta" not in base_res  # baseline 自己不挂 delta
    assert ckpt_res["delta"]["clip_i"] == pytest.approx(0.12)
    assert ckpt_res["delta"]["dino_i"] == pytest.approx(0.0)
    assert ckpt_res["baseline_metrics"]["clip_i"] == pytest.approx(0.60)


def test_empty_metric_result_describes_not_run_states(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)

    result = eval_metrics.load_result(vdir, run["run_id"])

    assert result is not None
    assert result["has_metrics"] is False
    assert result["status"] == "empty"
    assert result["metrics"] == {}
    assert result["summary"]["not_run"] == 8
    assert result["metric_states"]["clip_t"]["status"] == "not_run"
    assert result["metric_states"]["clip_t"]["question"]
    assert result["metric_states"]["sscd_nn"]["higher_is_better"] is False
    assert result["cache"]["embeddings_dir"] == "eval/cache/embeddings"


def test_save_metric_result_normalizes_states_and_preserves_created_at(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)

    first = eval_metrics.save_result(
        vdir,
        run["run_id"],
        {
            "metrics": {
                "clip_t": {"value": 0.31},
                "paired_cmmd2": 0.42,
            }
        },
        now=3000.0,
    )
    second = eval_metrics.save_result(
        vdir,
        run["run_id"],
        {"metric_states": {"clip_t": {"status": "failed", "error": "missing model"}}},
        now=4000.0,
    )

    assert first["has_metrics"] is True
    assert first["status"] == "partial"
    assert first["metric_states"]["clip_t"]["status"] == "done"
    assert first["metric_states"]["clip_t"]["value"] == 0.31
    assert first["metric_states"]["paired_cmmd2"]["status"] == "done"
    assert second["created_at"] == 3000.0
    assert second["updated_at"] == 4000.0
    assert second["status"] == "failed"
    assert second["metrics"]["clip_t"]["value"] == 0.31
    assert second["metrics"]["paired_cmmd2"] == 0.42
    assert second["metric_states"]["clip_t"]["error"] == "missing model"


def test_save_metric_result_can_clear_stale_values(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)
    eval_metrics.save_result(
        vdir,
        run["run_id"],
        {"metrics": {"clip_i": 0.5}},
        now=3000.0,
    )

    result = eval_metrics.save_result(
        vdir,
        run["run_id"],
        {
            "metrics": {"clip_i": None},
            "metric_states": {
                "clip_i": {
                    "status": "unavailable",
                    "value": None,
                    "reason": "no paired references",
                }
            },
        },
        now=4000.0,
    )

    assert "clip_i" not in result["metrics"]
    assert result["status"] == "partial"
    assert result["metric_states"]["clip_i"]["status"] == "unavailable"
    assert result["metric_states"]["clip_i"]["value"] is None


def test_eval_metrics_http_empty_list_and_single_run(client: TestClient) -> None:
    pid, vid = _make(client)
    project, version, vdir = _vdir_for(pid, vid)
    run = _sample_run(project, version, vdir)

    listed = client.get(f"/api/projects/{pid}/versions/{vid}/eval/metrics")
    assert listed.status_code == 200, listed.text
    listed_body = listed.json()
    assert [spec["key"] for spec in listed_body["metric_specs"]] == [
        "clip_t",
        "clip_i",
        "dino_i",
        "ccip_i",
        "tag_recall",
        "diversity",
        "sscd_nn",
        "paired_cmmd2",
    ]
    assert listed_body["results"][0]["run_id"] == run["run_id"]
    assert listed_body["results"][0]["status"] == "empty"
    assert listed_body["cache"]["embeddings_dir"] == "eval/cache/embeddings"

    got = client.get(
        f"/api/projects/{pid}/versions/{vid}/eval/samples/{run['run_id']}/metrics"
    )
    assert got.status_code == 200, got.text
    body = got.json()
    assert body["result"]["has_metrics"] is False
    assert body["result"]["sample_run"]["summary"]["total"] == 1


def test_eval_metrics_http_reads_saved_result(client: TestClient) -> None:
    pid, vid = _make(client)
    project, version, vdir = _vdir_for(pid, vid)
    run = _sample_run(project, version, vdir)
    eval_metrics.ensure_embeddings_cache_dir(vdir)
    (vdir / "eval" / "cache" / "embeddings" / "clip").mkdir(parents=True)
    (vdir / "eval" / "cache" / "embeddings" / "clip" / "real.npy").write_bytes(
        b"cache"
    )
    eval_metrics.save_result(
        vdir,
        run["run_id"],
        {"metrics": {"clip_t": 0.75}},
        now=3000.0,
    )

    got = client.get(
        f"/api/projects/{pid}/versions/{vid}/eval/samples/{run['run_id']}/metrics"
    )

    assert got.status_code == 200, got.text
    result = got.json()["result"]
    assert result["has_metrics"] is True
    assert result["status"] == "partial"
    assert result["metric_states"]["clip_t"]["value"] == 0.75
    assert result["cache"]["entries"] == [{
        "key": "clip",
        "path": "eval/cache/embeddings/clip",
        "file_count": 1,
        "size_bytes": 5,
    }]


def test_eval_metrics_can_store_results_under_task_eval_root(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    eval_root = infra_paths.task_eval_dir(42)
    run = _sample_run(project, version, vdir, eval_root=eval_root)
    (eval_metrics.ensure_embeddings_cache_dir(vdir, eval_root) / "clip").mkdir(
        parents=True
    )
    (eval_root / "cache" / "embeddings" / "clip" / "real.npy").write_bytes(b"cache")

    result = eval_metrics.save_result(
        vdir,
        run["run_id"],
        {"metrics": {"clip_t": 0.75}},
        eval_root=eval_root,
        now=3000.0,
    )

    assert result["metric_states"]["clip_t"]["value"] == 0.75
    assert result["metrics_path"].endswith(
        f"tasks/42/eval/samples/{run['run_id']}/metrics.json"
    )
    assert (eval_root / "samples" / run["run_id"] / "metrics.json").exists()
    assert not (vdir / "eval" / "samples" / run["run_id"] / "metrics.json").exists()
    assert result["cache"]["embeddings_dir"].endswith("tasks/42/eval/cache/embeddings")
    assert result["cache"]["entries"][0]["path"].endswith(
        "tasks/42/eval/cache/embeddings/clip"
    )


def test_eval_metrics_http_can_list_task_scoped_results(client: TestClient) -> None:
    pid, vid = _make(client)
    project, version, vdir = _vdir_for(pid, vid)
    task_root = infra_paths.task_eval_dir(42)
    task_run = _sample_run(project, version, vdir, eval_root=task_root)
    version_run = _sample_run(project, version, vdir)
    eval_metrics.save_result(
        vdir,
        task_run["run_id"],
        {"metrics": {"clip_t": 0.75}},
        eval_root=task_root,
    )
    eval_metrics.save_result(
        vdir,
        version_run["run_id"],
        {"metrics": {"clip_t": 0.11}},
    )

    listed = client.get(f"/api/projects/{pid}/versions/{vid}/eval/metrics?task_id=42")

    assert listed.status_code == 200, listed.text
    body = listed.json()
    assert [item["run_id"] for item in body["results"]] == [task_run["run_id"]]
    assert body["results"][0]["metric_states"]["clip_t"]["value"] == 0.75


# ---------------------------------------------------------------------------
# EvalSession 读侧（#465）—— 历史全部保留，默认给最新一次
# ---------------------------------------------------------------------------

def _seed_session(
    isolated, project, version, vdir, *, ckpts=("model_step100.safetensors",),
) -> dict[str, Any]:
    with db.connection_for(isolated["db"]) as conn:
        return eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=versions.list_lora_ckpts(vdir),
            trigger="manual", metric_keys=["clip_t", "clip_i"],
        )


def test_sessions_endpoint_lists_newest_first_with_summary(client, isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    pid, vid = project["id"], version["id"]
    first = _seed_session(isolated, project, version, vdir)
    second = _seed_session(isolated, project, version, vdir)

    r = client.get(f"/api/projects/{pid}/versions/{vid}/eval/sessions")
    assert r.status_code == 200, r.text
    sessions = r.json()["sessions"]

    assert [s["id"] for s in sessions] == [second["id"], first["id"]]
    # 列表只回摘要 —— 200 个候选的完整 plan 太大
    assert "plan" not in sessions[0] and "plan_json" not in sessions[0]
    assert sessions[0]["candidate_count"] == 2  # 1 checkpoint + baseline
    assert sessions[0]["metric_keys"] == ["clip_i", "clip_t"]
    assert sessions[0]["validation_images"] == 1  # _seed_validation_and_ckpt 建一张


def test_metrics_endpoint_defaults_to_latest_session(client, isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    pid, vid = project["id"], version["id"]
    _seed_session(isolated, project, version, vdir)
    latest = _seed_session(isolated, project, version, vdir)

    r = client.get(f"/api/projects/{pid}/versions/{vid}/eval/metrics").json()

    assert r["session"]["id"] == latest["id"]
    assert r.get("legacy") is not True
    # 结果按前端既有的 EvalMetricResult 形状返回（每个候选一条）
    assert len(r["results"]) == 2
    row = next(x for x in r["results"] if not x["baseline"])
    assert set(row["metric_states"]) == {"clip_t", "clip_i"}
    assert row["metric_states"]["clip_t"]["status"] == "pending"
    assert row["checkpoint"]["path"] == "output/model_step100.safetensors"


def test_metrics_endpoint_can_pick_a_historical_session(client, isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    pid, vid = project["id"], version["id"]
    first = _seed_session(isolated, project, version, vdir)
    _seed_session(isolated, project, version, vdir)

    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/eval/metrics?session_id={first['id']}"
    ).json()

    assert r["session"]["id"] == first["id"]


def test_metrics_endpoint_rejects_session_of_another_version(client, isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    session = _seed_session(isolated, project, version, vdir)
    with db.connection_for(isolated["db"]) as conn:
        other = versions.create_version(conn, project_id=project["id"], label="v2")

    r = client.get(
        f"/api/projects/{project['id']}/versions/{other['id']}"
        f"/eval/metrics?session_id={session['id']}"
    )
    assert r.status_code == 400, r.text


def test_metrics_endpoint_falls_back_to_legacy_files(client, isolated) -> None:
    """老项目没有任何 Session —— 存量 run.json / metrics.json 仍要读得到。"""
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    pid, vid = project["id"], version["id"]
    run = eval_samples.create_run(
        project, version, vdir, checkpoint_path="model_step100.safetensors",
    )
    eval_metrics.save_result(vdir, run["run_id"], {"metrics": {"clip_i": 0.5}})

    r = client.get(f"/api/projects/{pid}/versions/{vid}/eval/metrics").json()

    assert r["session"] is None
    assert r["legacy"] is True
    assert any(x["run_id"] == run["run_id"] for x in r["results"])


def test_delete_session_endpoint_refuses_while_running(client, isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    pid, vid = project["id"], version["id"]
    session = _seed_session(isolated, project, version, vdir)
    sid = int(session["id"])

    # pending → 拒删（要求先中断）
    assert client.delete(
        f"/api/projects/{pid}/versions/{vid}/eval/sessions/{sid}"
    ).status_code == 400

    with db.connection_for(isolated["db"]) as conn:
        eval_session.update_session(conn, sid, status="done")
    ok = client.delete(f"/api/projects/{pid}/versions/{vid}/eval/sessions/{sid}")
    assert ok.status_code == 200, ok.text

    with db.connection_for(isolated["db"]) as conn:
        assert eval_session.get_session(conn, sid) is None
    # checkpoint 是引用，删 Session 不动它
    assert (vdir / "output" / "model_step100.safetensors").exists()


def test_grid_endpoint_builds_checkpoint_by_prompt_matrix(client, isolated) -> None:
    """出图矩阵：X = 候选（baseline 在最前），Y = 验证图；cell 给 run_id + filename。"""
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    # 第二张验证图 → 矩阵两行
    val = vdir / "validation" / "1_data"
    (val / "b.png").write_bytes(b"png-b")
    (val / "b.txt").write_text("1girl, blue hair", encoding="utf-8")
    pid, vid = project["id"], version["id"]
    session = _seed_session(isolated, project, version, vdir)
    sid = int(session["id"])

    # 给两个候选各建一个 run（模拟出图阶段跑过）
    root = eval_session.samples_root(sid)
    with db.connection_for(isolated["db"]) as conn:
        cands = eval_session.list_candidates(conn, sid)
        for cand in cands:
            run = eval_samples.create_run(
                project, version, vdir,
                checkpoint_path=str(
                    eval_session.resolve_candidate_path(
                        vdir, str(cand["checkpoint_path"])
                    )
                ),
                eval_root=root,
                baseline=cand["role"] == "baseline",
            )
            eval_session.update_candidate(
                conn, int(cand["id"]), run_id=str(run["run_id"])
            )

    grid = client.get(
        f"/api/projects/{pid}/versions/{vid}/eval/sessions/{sid}/grid"
    ).json()

    # baseline 排第一列 —— 纯底模对照，测试页的 XY 没有这一列
    assert grid["columns"][0]["role"] == "baseline"
    assert grid["columns"][0]["label"] == "baseline"
    assert [c["role"] for c in grid["columns"]] == ["baseline", "checkpoint"]
    # 行来自 plan 冻结的验证集清单，带各自 prompt
    assert len(grid["rows"]) == 2
    assert {r["prompt"] for r in grid["rows"]} == {"solo, red hair", "1girl, blue hair"}
    # 每个候选 × 每行都有 cell
    for col in grid["columns"]:
        for row in grid["rows"]:
            cell = grid["cells"][f"{col['candidate_id']}:{row['index']}"]
            assert cell["filename"].endswith(".png")
            assert cell["run_id"] == col["run_id"]


def test_grid_endpoint_tolerates_candidates_without_run(client, isolated) -> None:
    """候选还没开跑（run_id 为空）→ 列在、cell 缺，不报错。"""
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    pid, vid = project["id"], version["id"]
    session = _seed_session(isolated, project, version, vdir)

    grid = client.get(
        f"/api/projects/{pid}/versions/{vid}/eval/sessions/{session['id']}/grid"
    ).json()

    assert len(grid["columns"]) == 2
    assert grid["cells"] == {}
    assert len(grid["rows"]) == 1
    assert grid["rows"][0]["prompt"] == ""  # 没有 run 就没有 prompt


def test_grid_endpoint_rejects_session_of_another_version(client, isolated) -> None:
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    session = _seed_session(isolated, project, version, vdir)
    with db.connection_for(isolated["db"]) as conn:
        other = versions.create_version(conn, project_id=project["id"], label="v2")

    r = client.get(
        f"/api/projects/{project['id']}/versions/{other['id']}"
        f"/eval/sessions/{session['id']}/grid"
    )
    assert r.status_code == 404, r.text


def test_grid_aligns_rows_by_reference_image_not_position(client, isolated) -> None:
    """按参考图路径对齐，不按数组下标 —— 否则验证集变化后会把 A 图结果贴到 B 行。"""
    project, version, vdir = _new_project(isolated)
    _seed_validation_and_ckpt(vdir)
    val = vdir / "validation" / "1_data"
    (val / "b.png").write_bytes(b"png-b")
    (val / "b.txt").write_text("second", encoding="utf-8")
    pid, vid = project["id"], version["id"]
    session = _seed_session(isolated, project, version, vdir)   # plan 冻结 a.png + b.png
    sid = int(session["id"])

    # 候选的 run 建立**之前**删掉第一张验证图 → run 只有 b.png 一项，
    # 按下标会落到第 0 行（a.png），按路径才落到第 1 行。
    (val / "a.png").unlink()
    (val / "a.txt").unlink()
    root = eval_session.samples_root(sid)
    with db.connection_for(isolated["db"]) as conn:
        cand = eval_session.list_candidates(conn, sid)[0]
        run = eval_samples.create_run(
            project, version, vdir,
            checkpoint_path=str(
                eval_session.resolve_candidate_path(vdir, str(cand["checkpoint_path"]))
            ),
            eval_root=root,
        )
        eval_session.update_candidate(conn, int(cand["id"]), run_id=str(run["run_id"]))

    grid = client.get(
        f"/api/projects/{pid}/versions/{vid}/eval/sessions/{sid}/grid"
    ).json()

    # plan 的两行都还在（冻结口径不受后来删图影响）
    assert [r["image"] for r in grid["rows"]] == [
        "validation/1_data/a.png", "validation/1_data/b.png",
    ]
    cid = int(cand["id"])
    # 结果落在 b.png 那一行（index 1），a.png 那一行是空的
    assert f"{cid}:0" not in grid["cells"]
    assert grid["cells"][f"{cid}:1"]["filename"].endswith(".png")
    assert grid["rows"][1]["prompt"] == "second"
    assert grid["rows"][0]["prompt"] == ""
