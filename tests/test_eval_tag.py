"""Tag-Recall eval metric runner（复用 WD14）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from studio import db, secrets, server
from studio.services import eval_metrics, eval_samples, eval_tag
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.supervisor import GPU_BOUND_JOB_KINDS


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return {"db": dbfile}


def _new_project(isolated) -> tuple[dict[str, Any], dict[str, Any], Path]:
    with db.connection_for(isolated["db"]) as conn:
        project = projects.create_project(conn, title="Eval Tag")
        version = versions.create_version(conn, project_id=project["id"], label="baseline")
    vdir = versions.version_dir(project["id"], project["slug"], version["label"])
    return project, version, vdir


def _seed(vdir: Path) -> None:
    val = vdir / "validation" / "1_data"
    val.mkdir(parents=True, exist_ok=True)
    (val / "a.png").write_bytes(b"png-a")
    (val / "a.txt").write_text("solo, red hair", encoding="utf-8")
    output = vdir / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_step100.safetensors").write_bytes(b"fake-lora")


def _fake_generator(run: dict[str, Any], version_dir: Path, progress) -> None:
    for idx, item in enumerate(run["items"]):
        run = eval_samples.mark_item_running(version_dir, run, idx)
        path = eval_samples.sample_image_path(version_dir, run["run_id"], item["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PNG")
        run = eval_samples.mark_item_done(version_dir, run, idx)


def _sample_run(project, version, vdir) -> dict[str, Any]:
    _seed(vdir)
    run = eval_samples.create_run(
        project, version, vdir, checkpoint_path="model_step100.safetensors", now=2000.0,
    )
    return eval_samples.run_sample_job(
        project, version, vdir, run["run_id"], generator=_fake_generator,
    )


class _FakeWD14:
    """known_tags 含 solo/red hair/blue hair；生成图回标固定 {solo, blue hair}。"""
    def is_available(self):
        return True, "ok"

    def prepare(self):
        pass

    def known_tags(self):
        return ["solo", "red hair", "blue hair", "1girl"]

    def tag(self, paths, on_progress=lambda d, t: None):
        for p in paths:
            yield {"image": p, "tags": ["solo", "blue hair"]}


# ---------------------------------------------------------------------------


def test_parse_booru_tags_normalizes_and_dedupes() -> None:
    assert eval_tag.parse_booru_tags("Solo, red_hair,  solo , ") == ["solo", "red hair"]
    assert eval_tag.parse_booru_tags("") == []
    assert eval_tag.parse_booru_tags(None) == []


def test_start_eval_tag_job_marks_tag_recall_pending(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)
    with db.connection_for(isolated["db"]) as conn:
        job, result = eval_tag.start_job(conn, project, version, vdir, run["run_id"])
    assert job["kind"] == "eval_tag"
    assert result["metric_states"]["tag_recall"]["status"] == "pending"
    assert result["metric_states"]["clip_t"]["status"] == "not_run"


def test_run_tag_job_default_scorer_computes_recall(isolated, monkeypatch) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)
    monkeypatch.setattr("studio.services.tagging.wd14.WD14Tagger", _FakeWD14)

    result = eval_tag.run_tag_job(project, version, vdir, run["run_id"])

    # prompt "solo, red hair" 词表内 = {solo, red hair}；回标 {solo, blue hair} → 命中 solo → 0.5
    assert result["metrics"]["tag_recall"] == 0.5
    state = result["metric_states"]["tag_recall"]
    assert state["status"] == "done"
    assert state["count"] == 1


def test_run_tag_job_unavailable_when_no_in_vocab_prompt(isolated, monkeypatch) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)

    class _NoVocab(_FakeWD14):
        def known_tags(self):
            return ["1girl", "blue hair"]  # 不含 solo / red hair

    monkeypatch.setattr("studio.services.tagging.wd14.WD14Tagger", _NoVocab)
    result = eval_tag.run_tag_job(project, version, vdir, run["run_id"])
    assert result["metric_states"]["tag_recall"]["status"] == "unavailable"
    assert "tag_recall" not in result["metrics"]


def test_run_tag_job_with_injected_scorer(isolated) -> None:
    project, version, vdir = _new_project(isolated)
    run = _sample_run(project, version, vdir)
    result = eval_tag.run_tag_job(
        project, version, vdir, run["run_id"],
        scorer=lambda _r, _v, _m, _p: {"tag_recall": 0.9, "tag_recall_count": 3},
    )
    assert result["metrics"]["tag_recall"] == 0.9
    assert result["metric_states"]["tag_recall"]["count"] == 3


def test_eval_tag_job_kind_gpu_bound() -> None:
    assert "eval_tag" in GPU_BOUND_JOB_KINDS
