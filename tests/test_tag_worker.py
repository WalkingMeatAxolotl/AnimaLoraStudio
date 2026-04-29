"""PP4 — tag worker: 用 mock tagger 跑通 worker 主流程，验证 caption 落盘。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio import db, project_jobs, projects, versions
from studio.services import tagger as tagger_mod
from studio.workers import tag_worker


class _FakeTagger:
    name = "wd14"
    requires_service = False

    def is_available(self):
        return True, "ok"

    def prepare(self):
        pass

    def tag(self, paths, on_progress=lambda d, t: None):
        for i, p in enumerate(paths):
            on_progress(i + 1, len(paths))
            yield {"image": p, "tags": [f"tag_{p.stem}", "common"]}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(projects, "TRASH_DIR", tmp_path / "_trash")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    # tag_worker 内部 import 用的是 studio.services.tagger.get_tagger；
    # 接受 overrides=... 的新签名（由 PP4 后续 wd14 per-job 覆盖功能引入）。
    monkeypatch.setattr(
        tagger_mod, "get_tagger", lambda name, overrides=None: _FakeTagger()
    )
    monkeypatch.setattr(
        "studio.workers.tag_worker.get_tagger",
        lambda name, overrides=None: _FakeTagger(),
    )
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
        # seed train/1_data with one image (1_data is the default folder created
        # by versions.create_version)
        train = versions.version_dir(p["id"], p["slug"], v["label"]) / "train" / "1_data"
        (train / "a.png").write_bytes(b"x")
        (train / "b.png").write_bytes(b"x")
        job = project_jobs.create_job(
            conn,
            project_id=p["id"],
            version_id=v["id"],
            kind="tag",
            params={
                "tagger": "wd14",
                "version_id": v["id"],
                "output_format": "txt",
            },
        )
    return {"db": dbfile, "p": p, "v": v, "job_id": job["id"], "train": train}


def test_run_creates_txt_captions(env) -> None:
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == "tag_a, common"
    assert (env["train"] / "b.txt").read_text(encoding="utf-8") == "tag_b, common"


def test_run_with_json_format(env, monkeypatch: pytest.MonkeyPatch) -> None:
    # 改 job 的 output_format 为 json
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, '$.output_format', 'json') "
            "WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()
    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    data = json.loads((env["train"] / "a.json").read_text(encoding="utf-8"))
    assert data["tags"] == ["tag_a", "common"]


def test_run_passes_wd14_overrides_through(env, monkeypatch) -> None:
    """worker 应把 params['wd14_overrides'] 透传到 get_tagger(overrides=...)。"""
    captured: dict = {}

    def _factory(name: str, overrides=None):
        captured["name"] = name
        captured["overrides"] = overrides
        return _FakeTagger()

    monkeypatch.setattr("studio.workers.tag_worker.get_tagger", _factory)

    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, "
            "'$.wd14_overrides', json(?)) WHERE id = ?",
            (json.dumps({"threshold_general": 0.2}), env["job_id"]),
        )
        conn.commit()

    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert captured["name"] == "wd14"
    assert captured["overrides"] == {"threshold_general": 0.2}


def test_run_unknown_job(env) -> None:
    assert tag_worker.run(99999) == 1
