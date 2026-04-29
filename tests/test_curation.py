"""PP3 — curation 模块：差集 / copy / remove / folder ops。"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import curation, db, projects, versions


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(projects, "TRASH_DIR", tmp_path / "_trash")
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


def _meta(env, name: str, ext: str, content: str) -> Path:
    pdir = projects.project_dir(env["p"]["id"], env["p"]["slug"])
    f = (pdir / "download" / name).with_suffix(ext)
    f.write_text(content, encoding="utf-8")
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
# view
# ---------------------------------------------------------------------------


def _names(entries: list[dict]) -> list[str]:
    return [e["name"] for e in entries]


def test_curation_view_left_minus_right(env) -> None:
    _dl(env, "1.png")
    _dl(env, "2.png")
    _dl(env, "3.png")
    with db.connection_for(env["db"]) as conn:
        curation.copy_to_train(
            conn, env["p"]["id"], env["v"]["id"], ["1.png"], "5_concept"
        )
        view = curation.curation_view(
            conn, env["p"]["id"], env["v"]["id"]
        )
    assert _names(view["left"]) == ["2.png", "3.png"]
    # 每条记录都带 mtime（float 秒）
    assert all(isinstance(e.get("mtime"), float) for e in view["left"])
    # 默认 1_data 始终存在；这里只断言我们刚复制进去的 5_concept
    assert _names(view["right"]["5_concept"]) == ["1.png"]
    assert view["right"]["1_data"] == []
    assert view["download_total"] == 3
    assert view["train_total"] == 1
    assert set(view["folders"]) == {"1_data", "5_concept"}


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


def test_copy_skips_existing_and_reports_missing(env) -> None:
    _dl(env, "1.png")
    _dl(env, "2.png")
    with db.connection_for(env["db"]) as conn:
        curation.copy_to_train(
            conn, env["p"]["id"], env["v"]["id"], ["1.png"], "5_concept"
        )
        r = curation.copy_to_train(
            conn,
            env["p"]["id"],
            env["v"]["id"],
            ["1.png", "2.png", "ghost.png"],
            "5_concept",
        )
    assert r["copied"] == ["2.png"]
    assert r["skipped"] == ["1.png"]
    assert r["missing"] == ["ghost.png"]


def test_copy_brings_metadata(env) -> None:
    _dl(env, "1.png")
    _meta(env, "1.png", ".txt", "tag1, tag2")
    _meta(env, "1.png", ".json", '{"score": 0.9}')
    with db.connection_for(env["db"]) as conn:
        curation.copy_to_train(
            conn, env["p"]["id"], env["v"]["id"], ["1.png"], "5_concept"
        )
    folder = _train_dir(env, "5_concept")
    assert (folder / "1.png").exists()
    assert (folder / "1.txt").read_text(encoding="utf-8") == "tag1, tag2"
    assert (folder / "1.json").read_text(encoding="utf-8") == '{"score": 0.9}'


def test_copy_rejects_bad_folder_name(env) -> None:
    _dl(env, "1.png")
    with db.connection_for(env["db"]) as conn:
        for bad in ("../etc", "name with space", "5_", "name/sub"):
            with pytest.raises(curation.CurationError, match="文件夹名"):
                curation.copy_to_train(
                    conn, env["p"]["id"], env["v"]["id"], ["1.png"], bad
                )


def test_copy_rejects_bad_filename(env) -> None:
    with db.connection_for(env["db"]) as conn:
        with pytest.raises(curation.CurationError, match="文件名"):
            curation.copy_to_train(
                conn,
                env["p"]["id"],
                env["v"]["id"],
                ["../escape.png"],
                "5_concept",
            )


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def test_remove_only_deletes_train_copy(env) -> None:
    _dl(env, "1.png")
    _meta(env, "1.png", ".txt", "tag")
    with db.connection_for(env["db"]) as conn:
        curation.copy_to_train(
            conn, env["p"]["id"], env["v"]["id"], ["1.png"], "5_concept"
        )
        r = curation.remove_from_train(
            conn, env["p"]["id"], env["v"]["id"], "5_concept", ["1.png"]
        )
    assert r["removed"] == ["1.png"]
    assert not (_train_dir(env, "5_concept") / "1.png").exists()
    assert not (_train_dir(env, "5_concept") / "1.txt").exists()
    # download/ 必须还在
    pdir = projects.project_dir(env["p"]["id"], env["p"]["slug"])
    assert (pdir / "download" / "1.png").exists()
    assert (pdir / "download" / "1.txt").exists()


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
        curation.copy_to_train(
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
        curation.copy_to_train(
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
        curation.copy_to_train(
            conn, env["p"]["id"], env["v"]["id"], ["1.png"], "5_concept"
        )
        assert curation.has_train_images(conn, env["p"]["id"], env["v"]["id"]) is True
