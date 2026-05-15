"""PP1 — projects.py: slug 唯一性、目录创建、软删、empty_trash。"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db, projects


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    pdir = tmp_path / "projects"
    tdir = tmp_path / "_trash" / "projects"
    monkeypatch.setattr(projects, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(projects, "TRASH_DIR", tdir)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    return {"db": dbfile, "projects_dir": pdir, "trash_dir": tdir}


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
    assert p["stage"] == "created"
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


def test_preprocessing_stage_is_valid(isolated) -> None:
    """update_project 接受 stage='preprocessing'，介于 downloading 和 curating 之间。"""
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title="StageTest")
        p2 = projects.update_project(conn, p["id"], stage="preprocessing")
    assert p2["stage"] == "preprocessing"


def test_create_rejects_empty_title(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(projects.ProjectError, match="title"):
            projects.create_project(conn, title="   ")


def test_update_writes_project_json(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title="X")
        projects.update_project(conn, p["id"], note="updated", stage="curating")
    pdir = projects.project_dir(p["id"], p["slug"])
    text = (pdir / "project.json").read_text(encoding="utf-8")
    assert "updated" in text
    assert "curating" in text


def test_update_rejects_invalid_stage(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title="X")
        with pytest.raises(projects.ProjectError, match="stage"):
            projects.update_project(conn, p["id"], stage="bogus")


# ---------------------------------------------------------------------------
# soft delete
# ---------------------------------------------------------------------------


def test_soft_delete_moves_to_trash_and_removes_row(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title="ToDel")
        pid = p["id"]
        projects.soft_delete_project(conn, pid)
        assert projects.get_project(conn, pid) is None
    src = projects.project_dir(pid, p["slug"])
    assert not src.exists()
    trash_dst = isolated["trash_dir"] / f"{pid}-{p['slug']}"
    assert trash_dst.exists()


def test_empty_trash(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title="X")
        projects.soft_delete_project(conn, p["id"])
    n = projects.empty_trash()
    assert n == 1
    # 再调一次安全
    assert projects.empty_trash() == 0


def test_soft_delete_missing_raises(isolated) -> None:
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(projects.ProjectError, match="不存在"):
            projects.soft_delete_project(conn, 9999)
