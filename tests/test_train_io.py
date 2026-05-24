"""PP7 — 训练集导出 / 导入 round-trip + 边界。"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from studio import db, projects, versions
from studio.services import train_io


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    pdir = tmp_path / "projects"
    monkeypatch.setattr(projects, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    return {"db": dbfile, "tmp": tmp_path}


def _make_project_with_train(
    isolated, *, title: str = "Cosmic Kaguya", label: str = "v1"
) -> tuple[dict, dict, Path]:
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title=title)
        v = versions.create_version(conn, project_id=p["id"], label=label)
    train = versions.version_dir(p["id"], p["slug"], v["label"]) / "train"
    return p, v, train


def _png(content: bytes = b"fake-png") -> bytes:
    # 只是占位，IMAGE_EXTS 按后缀判定不读 magic
    return content


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_round_trip(isolated, tmp_path: Path) -> None:
    p, v, train = _make_project_with_train(isolated)
    # 默认已有 1_data；放两张 + 1 个 caption；再加一个 2_concept 子文件夹
    (train / "1_data").mkdir(parents=True, exist_ok=True)
    (train / "1_data" / "a.png").write_bytes(_png())
    (train / "1_data" / "a.txt").write_text("1girl, solo", encoding="utf-8")
    (train / "1_data" / "b.png").write_bytes(_png(b"b"))
    (train / "2_concept").mkdir()
    (train / "2_concept" / "c.png").write_bytes(_png(b"c"))
    (train / "2_concept" / "c.txt").write_text("2girls", encoding="utf-8")

    dest = tmp_path / "out.zip"
    with db.connection_for(isolated["db"]) as conn:
        result = train_io.export_train(conn, v["id"], dest)

    assert dest.exists()
    assert result["manifest"]["stats"]["image_count"] == 3
    assert result["manifest"]["stats"]["tagged_count"] == 2
    assert {c["folder"] for c in result["manifest"]["stats"]["concepts"]} == {
        "1_data",
        "2_concept",
    }

    with zipfile.ZipFile(dest) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "train/1_data/a.png" in names
        assert "train/1_data/a.txt" in names
        assert "train/1_data/b.png" in names
        assert "train/2_concept/c.png" in names

        # round-trip: 重新导入应该得到一个全新项目，stage=tagging
        new_zip = tmp_path / "round.zip"
    # 重新打开（上面 with 已关）
    import shutil

    shutil.copy(dest, new_zip)
    with db.connection_for(isolated["db"]) as conn:
        imported = train_io.import_train(conn, new_zip)

    assert imported["project"]["id"] != p["id"]
    assert imported["version"]["label"] == "v1"
    # ADR-0007 PR-5: import 不再推 stage
    assert imported["stats"]["image_count"] == 3
    assert imported["stats"]["tagged_count"] == 2

    new_train = versions.version_dir(
        imported["project"]["id"],
        imported["project"]["slug"],
        imported["version"]["label"],
    ) / "train"
    assert (new_train / "1_data" / "a.png").exists()
    assert (new_train / "1_data" / "a.txt").read_text(encoding="utf-8") == "1girl, solo"
    assert (new_train / "2_concept" / "c.png").exists()


def test_export_empty_train_raises(isolated, tmp_path: Path) -> None:
    p, v, train = _make_project_with_train(isolated)
    # 默认 1_data 是空的；不放任何图
    dest = tmp_path / "empty.zip"
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.export_train(conn, v["id"], dest)


def test_export_missing_version(isolated, tmp_path: Path) -> None:
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.export_train(conn, 9999, tmp_path / "x.zip")


# ---------------------------------------------------------------------------
# import — 边界
# ---------------------------------------------------------------------------


def _make_zip(tmp: Path, files: dict[str, bytes], manifest: dict | None = None) -> Path:
    p = tmp / f"in-{len(list(tmp.iterdir()))}.zip"
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_STORED) as zf:
        if manifest is not None:
            zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
        for name, data in files.items():
            zf.writestr(name, data)
    return p


def _basic_manifest(title: str = "Imported Project") -> dict:
    return {
        "schema_version": 1,
        "exported_at": 0,
        "source": {"title": title, "version_label": "v1", "slug": "imported-project"},
        "stats": {},
    }


def test_import_rejects_zip_slip(isolated, tmp_path: Path) -> None:
    zp = _make_zip(
        tmp_path,
        {"train/../escape.png": b"x"},
        manifest=_basic_manifest(),
    )
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, zp)


def test_import_rejects_absolute_path(isolated, tmp_path: Path) -> None:
    zp = _make_zip(
        tmp_path,
        {"/etc/evil.png": b"x"},
        manifest=_basic_manifest(),
    )
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, zp)


def test_import_rejects_non_train_prefix(isolated, tmp_path: Path) -> None:
    zp = _make_zip(
        tmp_path,
        {"download/x.png": b"x"},
        manifest=_basic_manifest(),
    )
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, zp)


def test_import_rejects_deep_nesting(isolated, tmp_path: Path) -> None:
    zp = _make_zip(
        tmp_path,
        {"train/1_data/sub/x.png": b"x"},
        manifest=_basic_manifest(),
    )
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, zp)


def test_import_rejects_missing_manifest(isolated, tmp_path: Path) -> None:
    zp = _make_zip(tmp_path, {"train/1_data/x.png": b"x"})
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, zp)


def test_import_rejects_empty(isolated, tmp_path: Path) -> None:
    zp = _make_zip(tmp_path, {}, manifest=_basic_manifest())
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, zp)


def test_import_slug_conflict_appends_suffix(isolated, tmp_path: Path) -> None:
    """同 slug 已存在 → 自动加 -imported-{ts} 后缀。"""
    # 先建一个 slug=cosmic-kaguya 的项目占位
    with db.connection_for(isolated["db"]) as conn:
        projects.create_project(conn, title="Cosmic Kaguya")  # slug=cosmic-kaguya

    zp = _make_zip(
        tmp_path,
        {"train/1_data/a.png": b"x"},
        manifest={
            "schema_version": 1,
            "exported_at": 0,
            "source": {"title": "Cosmic Kaguya", "version_label": "v1", "slug": "cosmic-kaguya"},
            "stats": {},
        },
    )
    with db.connection_for(isolated["db"]) as conn:
        result = train_io.import_train(conn, zp)
    assert result["project"]["slug"].startswith("cosmic-kaguya-imported-")


def test_import_keeps_concept_folder_names(isolated, tmp_path: Path) -> None:
    zp = _make_zip(
        tmp_path,
        {
            "train/3_chara/a.png": b"x",
            "train/3_chara/a.txt": b"tag1",
            "train/5_style/b.png": b"y",
        },
        manifest=_basic_manifest("Multi Concept"),
    )
    with db.connection_for(isolated["db"]) as conn:
        result = train_io.import_train(conn, zp)
    new_train = versions.version_dir(
        result["project"]["id"],
        result["project"]["slug"],
        result["version"]["label"],
    ) / "train"
    assert (new_train / "3_chara" / "a.png").exists()
    assert (new_train / "5_style" / "b.png").exists()
    assert result["stats"]["tagged_count"] == 1
    assert result["stats"]["image_count"] == 2


def test_import_bad_zip(isolated, tmp_path: Path) -> None:
    bad = tmp_path / "not.zip"
    bad.write_bytes(b"not a zip file at all")
    with db.connection_for(isolated["db"]) as conn, pytest.raises(train_io.TrainIOError):
        train_io.import_train(conn, bad)
