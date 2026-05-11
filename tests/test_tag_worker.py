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


class _FakeLLMTagger:
    name = "llm"
    requires_service = True

    def is_available(self):
        return True, "ok"

    def prepare(self):
        pass

    def tag(self, paths, on_progress=lambda d, t: None):
        for i, p in enumerate(paths):
            on_progress(i + 1, len(paths))
            yield {
                "image": p,
                "tags": ["1girl", "watercolor", "blue background"],
                "caption": "1girl, watercolor, blue background. Soft watercolor style.",
                "caption_json": {
                    "tags": {
                        "quality": [],
                        "count": "1girl",
                        "character": "",
                        "series": "",
                        "artist": "",
                        "appearance": ["watercolor"],
                        "tags": ["soft shading"],
                        "environment": ["blue background"],
                        "nl": "Soft watercolor style.",
                    }
                },
            }


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


def test_run_passes_cltagger_overrides_through(env, monkeypatch) -> None:
    """worker 应把 params['cltagger_overrides'] 透传到 get_tagger(overrides=...)。"""
    captured: dict = {}

    def _factory(name: str, overrides=None):
        captured["name"] = name
        captured["overrides"] = overrides
        return _FakeTagger()

    monkeypatch.setattr("studio.workers.tag_worker.get_tagger", _factory)

    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, "
            "'$.tagger', 'cltagger', "
            "'$.cltagger_overrides', json(?)) WHERE id = ?",
            (json.dumps({"threshold_character": 0.55}), env["job_id"]),
        )
        conn.commit()

    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert captured["name"] == "cltagger"
    assert captured["overrides"] == {"threshold_character": 0.55}


def test_run_writes_llm_json_caption(env, monkeypatch) -> None:
    monkeypatch.setattr(
        "studio.workers.tag_worker.get_tagger",
        lambda name, overrides=None: _FakeLLMTagger(),
    )
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, "
            "'$.tagger', 'llm', '$.output_format', 'json') WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()

    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    data = json.loads((env["train"] / "a.json").read_text(encoding="utf-8"))
    assert data["ai_output"]["count"] == "1girl"
    assert data["ai_output"]["appearance"] == ["watercolor"]
    assert data["ai_output"]["environment"] == ["blue background"]
    assert data["ai_output"]["nl"] == "Soft watercolor style."
    assert not (env["train"] / "a.txt").exists()


def test_run_writes_llm_txt_caption(env, monkeypatch) -> None:
    monkeypatch.setattr(
        "studio.workers.tag_worker.get_tagger",
        lambda name, overrides=None: _FakeLLMTagger(),
    )
    with db.connection_for(env["db"]) as conn:
        conn.execute(
            "UPDATE project_jobs SET params = json_set(params, '$.tagger', 'llm') "
            "WHERE id = ?",
            (env["job_id"],),
        )
        conn.commit()

    rc = tag_worker.run(env["job_id"])
    assert rc == 0
    assert (env["train"] / "a.txt").read_text(encoding="utf-8") == (
        "1girl, watercolor, blue background. Soft watercolor style."
    )


def test_run_unknown_job(env) -> None:
    assert tag_worker.run(99999) == 1


def test_worker_imports_onnxruntime_setup_at_module_level() -> None:
    """worker 是独立 subprocess —— 必须在任何 `import onnxruntime` 之前触发
    onnxruntime_setup 的顶层 preload。回归测试：防止有人删掉那行 import 后
    Linux 上又出 CUDA EP 静默降级，UI 看不到任何信号。"""
    import re
    import sys
    src = Path(tag_worker.__file__).read_text(encoding="utf-8")
    assert "from studio.services import onnxruntime_setup" in src, (
        "tag_worker.py 顶层必须 import onnxruntime_setup 触发 preload；"
        "见 onnxruntime_setup.py 顶部 PP9.5 注释。"
    )
    # worker 文件本身不应出现 `^import onnxruntime` / `^from onnxruntime`（行首真 import）
    bad = re.findall(r"^\s*(?:import onnxruntime|from onnxruntime\b)", src, re.MULTILINE)
    assert not bad, f"worker 不应直接 import onnxruntime；命中: {bad}"
    # 模块加载本身把 onnxruntime_setup 拉进 sys.modules
    assert "studio.services.onnxruntime_setup" in sys.modules
