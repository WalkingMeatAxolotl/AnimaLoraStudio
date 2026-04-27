"""PP1 — versions.py: label 唯一、目录树、fork、active reassign。"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db, projects, versions


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    pdir = tmp_path / "projects"
    tdir = tmp_path / "_trash" / "projects"
    monkeypatch.setattr(projects, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(projects, "TRASH_DIR", tdir)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    return {"db": dbfile}


def _new_project(isolated, title: str = "P1") -> dict:
    with db.connection_for(isolated["db"]) as conn:
        return projects.create_project(conn, title=title)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_version_builds_tree_and_activates(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
    vdir = versions.version_dir(p["id"], p["slug"], "baseline")
    assert vdir.exists()
    for sub in ("train", "reg", "output", "samples"):
        assert (vdir / sub).is_dir()
    assert (vdir / "version.json").exists()
    # 项目里第一个版本自动设为 active
    with db.connection_for(isolated["db"]) as conn:
        p2 = projects.get_project(conn, p["id"])
    assert p2 and p2["active_version_id"] == v["id"]


def test_create_version_rejects_invalid_label(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        for bad in ("has space", "../escape", "name/sub", "中文"):
            with pytest.raises(versions.VersionError, match="label"):
                versions.create_version(conn, project_id=p["id"], label=bad)


def test_create_version_rejects_duplicate_label(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.create_version(conn, project_id=p["id"], label="baseline")
        with pytest.raises(versions.VersionError, match="已存在"):
            versions.create_version(conn, project_id=p["id"], label="baseline")


# ---------------------------------------------------------------------------
# fork
# ---------------------------------------------------------------------------


def test_fork_copies_train_tree(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        src = versions.create_version(conn, project_id=p["id"], label="baseline")
    src_train = versions.version_dir(p["id"], p["slug"], "baseline") / "train"
    folder = src_train / "5_concept"
    folder.mkdir()
    (folder / "001.png").write_bytes(b"fakepng")
    (folder / "001.txt").write_text("tag1, tag2", encoding="utf-8")

    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.create_version(
            conn,
            project_id=p["id"],
            label="forked",
            fork_from_version_id=src["id"],
        )
    new_folder = (
        versions.version_dir(p["id"], p["slug"], "forked")
        / "train" / "5_concept"
    )
    assert (new_folder / "001.png").read_bytes() == b"fakepng"
    assert (new_folder / "001.txt").read_text(encoding="utf-8") == "tag1, tag2"
    assert v2["config_name"] == src["config_name"]


def test_fork_rejects_alien_source(isolated) -> None:
    a = _new_project(isolated, title="A")
    b = _new_project(isolated, title="B")
    with db.connection_for(isolated["db"]) as conn:
        src = versions.create_version(conn, project_id=a["id"], label="baseline")
        with pytest.raises(versions.VersionError, match="fork"):
            versions.create_version(
                conn,
                project_id=b["id"],
                label="x",
                fork_from_version_id=src["id"],
            )


# ---------------------------------------------------------------------------
# delete + active reassign
# ---------------------------------------------------------------------------


def test_delete_active_version_reassigns(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v1 = versions.create_version(conn, project_id=p["id"], label="v1")
        v2 = versions.create_version(conn, project_id=p["id"], label="v2")
        # active = v1（首次）；切到 v2
        versions.activate_version(conn, v2["id"])
        # 删 v2 → active 应回到 v1（剩下创建最新的）
        versions.delete_version(conn, v2["id"])
        p2 = projects.get_project(conn, p["id"])
    assert p2 and p2["active_version_id"] == v1["id"]


def test_delete_last_version_clears_active(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v1 = versions.create_version(conn, project_id=p["id"], label="only")
        versions.delete_version(conn, v1["id"])
        p2 = projects.get_project(conn, p["id"])
    assert p2 and p2["active_version_id"] is None


def test_delete_moves_dir_to_trash(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
        src = versions.version_dir(p["id"], p["slug"], "baseline")
        assert src.exists()
        versions.delete_version(conn, v["id"])
    assert not src.exists()
    trash = (
        projects.TRASH_DIR
        / f"{p['id']}-{p['slug']}"
        / "versions"
        / "baseline"
    )
    assert trash.exists()


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_for_version_counts_train_and_reg(isolated) -> None:
    p = _new_project(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v = versions.create_version(conn, project_id=p["id"], label="v1")
    vdir = versions.version_dir(p["id"], p["slug"], "v1")
    (vdir / "train" / "5_concept").mkdir(parents=True)
    (vdir / "train" / "5_concept" / "a.png").write_bytes(b"x")
    (vdir / "train" / "5_concept" / "b.png").write_bytes(b"x")
    (vdir / "reg" / "1_general").mkdir(parents=True)
    (vdir / "reg" / "1_general" / "r.png").write_bytes(b"x")
    stats = versions.stats_for_version(p, v)
    assert stats["train_image_count"] == 2
    assert stats["reg_image_count"] == 1
    assert stats["train_folders"] == [{"name": "5_concept", "image_count": 2}]
    assert stats["has_output"] is False
