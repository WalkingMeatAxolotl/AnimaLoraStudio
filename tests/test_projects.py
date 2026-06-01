"""PP1 — projects.py: slug 唯一性、目录创建、硬删除。"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db
from studio.services.projects import projects


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    pdir = tmp_path / "projects"
    monkeypatch.setattr(projects, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    return {"db": dbfile, "projects_dir": pdir}


# ---------------------------------------------------------------------------
# slug
# ---------------------------------------------------------------------------


def test_slugify_basic() -> None:
    assert projects.slugify("Cosmic Kaguya") == "cosmic-kaguya"
    assert projects.slugify("  Mixed-Case 123  ") == "mixed-case-123"
    assert projects.slugify("中文名") == "project"  # 全非 ASCII → fallback
    assert projects.slugify("") == "project"
    assert projects.slugify("a/b/c") == "a-b-c"


def test_unique_slug_appends_suffix(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        a = projects.create_project(conn, title="Same")
        b = projects.create_project(conn, title="Same")
        c = projects.create_project(conn, title="Same")
    assert a["slug"] == "same"
    assert b["slug"] == "same-2"
    assert c["slug"] == "same-3"


# ---------------------------------------------------------------------------
# create / dirs / project.json
# ---------------------------------------------------------------------------


def test_create_creates_directory_layout(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title="Hello World", note="abc")
    pdir = projects.project_dir(p["id"], p["slug"])
    assert pdir.exists()
    assert (pdir / "download").is_dir()
    assert (pdir / "preprocess").is_dir()  # 预处理阶段产物目录
    assert (pdir / "versions").is_dir()
    assert (pdir / "project.json").exists()
    # ADR-0007 PR-5: project 无 stage 字段（DB 列还在但会随 v9 destructive 删）
    assert p["note"] == "abc"


def test_stats_counts_download_and_preprocess(isolated) -> None:
    """stats_for_project 同时返回 download / preprocess 图片数。"""
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title="StatTest")
    pdir = projects.project_dir(p["id"], p["slug"])

    (pdir / "download" / "a.png").write_bytes(b"x")
    (pdir / "download" / "b.jpg").write_bytes(b"x")
    (pdir / "download" / "ignore.txt").write_bytes(b"x")  # 非图
    (pdir / "preprocess" / "a.png").write_bytes(b"x")

    s = projects.stats_for_project(p)
    assert s["download_image_count"] == 2
    assert s["preprocess_image_count"] == 1


def test_create_rejects_empty_title(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(projects.ProjectError, match="title"):
            projects.create_project(conn, title="   ")


def test_update_writes_project_json(isolated) -> None:
    """ADR-0007 PR-5: stage 已删；只剩 note / title / active_version_id 可 PATCH。"""
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title="X")
        projects.update_project(conn, p["id"], note="updated")
    pdir = projects.project_dir(p["id"], p["slug"])
    text = (pdir / "project.json").read_text(encoding="utf-8")
    assert "updated" in text


# ---------------------------------------------------------------------------
# soft delete
# ---------------------------------------------------------------------------


def test_delete_removes_dir_and_row(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title="ToDel")
        pid = p["id"]
        projects.delete_project(conn, pid)
        assert projects.get_project(conn, pid) is None
    src = projects.project_dir(pid, p["slug"])
    assert not src.exists()


def test_delete_missing_raises(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(projects.ProjectError, match="不存在"):
            projects.delete_project(conn, 9999)
