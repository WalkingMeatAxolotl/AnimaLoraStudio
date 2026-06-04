"""curation 模块：folder ops + remove/has_train_images。

`copy_*` / `list_*` / 去重 等 train-scope 行为见 test_curation_train_scope.py /
test_duplicates_train_scope.py。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import db
from studio.services.dataset import curation
from studio.services.projects import projects, versions


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label="baseline")
    return {"db": dbfile, "p": p, "v": v}


def _dl(env, name: str, blob: bytes = b"img") -> Path:
    pdir = projects.project_dir(env["p"]["id"], env["p"]["slug"])
    f = pdir / "download" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(blob)
    return f


def _train_dir(env, folder: str) -> Path:
    return (
        versions.version_dir(
            env["p"]["id"], env["p"]["slug"], env["v"]["label"]
        )
        / "train"
        / folder
    )


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def test_remove_reports_missing(env) -> None:
    with db.connection_for(env["db"]) as conn:
        curation.create_folder(conn, env["p"]["id"], env["v"]["id"], "5_concept")
        r = curation.remove_from_train(
            conn, env["p"]["id"], env["v"]["id"], "5_concept", ["ghost.png"]
        )
    assert r["missing"] == ["ghost.png"]
    assert r["removed"] == []


# ---------------------------------------------------------------------------
# folder ops
# ---------------------------------------------------------------------------


def test_create_folder(env) -> None:
    with db.connection_for(env["db"]) as conn:
        curation.create_folder(conn, env["p"]["id"], env["v"]["id"], "10_x")
        with pytest.raises(curation.CurationError, match="已存在"):
            curation.create_folder(
                conn, env["p"]["id"], env["v"]["id"], "10_x"
            )


def test_rename_folder(env) -> None:
    _dl(env, "1.png")
    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"], ["1.png"], "5_concept"
        )
        curation.rename_folder(
            conn, env["p"]["id"], env["v"]["id"], "5_concept", "10_concept"
        )
    assert (_train_dir(env, "10_concept") / "1.png").exists()
    assert not _train_dir(env, "5_concept").exists()


def test_delete_folder_clears_train_copies(env) -> None:
    _dl(env, "1.png")
    with db.connection_for(env["db"]) as conn:
        curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"], ["1.png"], "5_concept"
        )
        curation.delete_folder(
            conn, env["p"]["id"], env["v"]["id"], "5_concept"
        )
    assert not _train_dir(env, "5_concept").exists()
    pdir = projects.project_dir(env["p"]["id"], env["p"]["slug"])
    assert (pdir / "download" / "1.png").exists()  # download 不动


# ---------------------------------------------------------------------------
# stage hint
# ---------------------------------------------------------------------------


def test_has_train_images_false_then_true(env) -> None:
    with db.connection_for(env["db"]) as conn:
        assert curation.has_train_images(conn, env["p"]["id"], env["v"]["id"]) is False
        _dl(env, "1.png")
        curation.copy_download_to_train(
            conn, env["p"]["id"], env["v"]["id"], ["1.png"], "5_concept"
        )
        assert curation.has_train_images(conn, env["p"]["id"], env["v"]["id"]) is True
