"""PP3 — curation 模块：差集 / copy / remove / folder ops。"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from studio import curation, db, projects, versions
from studio.services import duplicate_finder
from studio.services import preprocess_manifest


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


def _png(env, name: str, color: str = "#d8dde6") -> Path:
    p = _dl(env, name, blob=b"")
    img = Image.new("RGB", (96, 96), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle((12, 16, 84, 80), outline="#111111", width=4)
    draw.ellipse((34, 30, 62, 58), fill="#f0c4a0", outline="#111111", width=2)
    img.save(p)
    return p


def _different_png(env, name: str) -> Path:
    p = _dl(env, name, blob=b"")
    img = Image.new("RGB", (96, 96), "#90c8ff")
    draw = ImageDraw.Draw(img)
    draw.polygon([(16, 82), (48, 12), (82, 82)], fill="#62aa55", outline="#111111")
    img.save(p)
    return p


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
    # ADR 0004：left_source 字段已删（前端通过 projectThumbUrl 拿，后端 resolve 透明）
    assert "left_source" not in view


def _seed_processed(env, source: str, *, content: bytes = b"upscaled") -> Path:
    """模拟一张已处理：写 preprocess/{stem}.png + manifest entry。"""
    pdir = projects.project_dir(env["p"]["id"], env["p"]["slug"])
    pre = pdir / "preprocess"
    pre.mkdir(parents=True, exist_ok=True)
    product_name = Path(source).stem + ".png"
    (pre / product_name).write_bytes(content)
    preprocess_manifest.add_processed(pdir, product_name, {"source": source})
    return pre / product_name


def test_curation_view_lists_download_names_always(env) -> None:
    """ADR 0004：left 永远列 download 文件名（即便部分图已处理）。"""
    _dl(env, "1.png")
    _dl(env, "2.png")
    _seed_processed(env, "1.png")  # 1.png 已处理

    with db.connection_for(env["db"]) as conn:
        view = curation.curation_view(conn, env["p"]["id"], env["v"]["id"])
    assert _names(view["left"]) == ["1.png", "2.png"]


def test_copy_to_train_uses_processed_bytes_when_available(env) -> None:
    """已处理图 → 从 preprocess/ 拷字节，但 train/ 下保留 download 文件名。"""
    _dl(env, "1.png", blob=b"original-low-res")
    _seed_processed(env, "1.png", content=b"upscaled-hi-res")

    with db.connection_for(env["db"]) as conn:
        curation.copy_to_train(
            conn, env["p"]["id"], env["v"]["id"], ["1.png"], "5_concept"
        )
    copied = _train_dir(env, "5_concept") / "1.png"
    assert copied.read_bytes() == b"upscaled-hi-res"


def test_copy_to_train_uses_original_when_unprocessed(env) -> None:
    """未处理图 → 直接从 download/ 拷原字节。"""
    _dl(env, "1.png", blob=b"original-bytes")

    with db.connection_for(env["db"]) as conn:
        curation.copy_to_train(
            conn, env["p"]["id"], env["v"]["id"], ["1.png"], "5_concept"
        )
    copied = _train_dir(env, "5_concept") / "1.png"
    assert copied.read_bytes() == b"original-bytes"


def test_curation_view_expands_multi_crop_derivatives(env) -> None:
    """Multi-crop fan-out: download/X.png 派生 X_c0.png / X_c1.png，
    筛选 left 展开为 N 行可单独勾选，原 X.png 不再单独出现。"""
    _dl(env, "X.png", blob=b"orig")
    _dl(env, "Y.png", blob=b"orig")
    pdir = projects.project_dir(env["p"]["id"], env["p"]["slug"])
    # 模拟 multi-crop fan-out 写盘 + manifest
    (pdir / "preprocess").mkdir(parents=True, exist_ok=True)
    (pdir / "preprocess" / "X_c0.png").write_bytes(b"head")
    (pdir / "preprocess" / "X_c1.png").write_bytes(b"body")
    preprocess_manifest.replace_with_crops(
        pdir,
        source_name="X.png",
        outputs=[
            {"name": "X_c0.png", "origin": "X.png", "size": 4, "mtime": 1.0},
            {"name": "X_c1.png", "origin": "X.png", "size": 4, "mtime": 1.0},
        ],
    )
    with db.connection_for(env["db"]) as conn:
        view = curation.curation_view(conn, env["p"]["id"], env["v"]["id"])
    # X 派生为 c0 / c1；Y 未处理保持原名；X.png 自身不在 left
    assert set(_names(view["left"])) == {"X_c0.png", "X_c1.png", "Y.png"}


def test_copy_to_train_accepts_preprocess_derivative_name(env) -> None:
    """筛选 left 给的派生名（如 X_c0.png）应能直接 copy 到 train，
    bytes 来自 preprocess/，metadata 从 download/{origin} 跟着复制。"""
    _dl(env, "X.png")
    _meta(env, "X.png", ".txt", "tag for X")
    pdir = projects.project_dir(env["p"]["id"], env["p"]["slug"])
    (pdir / "preprocess").mkdir(parents=True, exist_ok=True)
    (pdir / "preprocess" / "X_c0.png").write_bytes(b"crop-0-bytes")
    (pdir / "preprocess" / "X_c1.png").write_bytes(b"crop-1-bytes")
    preprocess_manifest.replace_with_crops(
        pdir,
        source_name="X.png",
        outputs=[
            {"name": "X_c0.png", "origin": "X.png", "size": 4, "mtime": 1.0},
            {"name": "X_c1.png", "origin": "X.png", "size": 4, "mtime": 1.0},
        ],
    )
    with db.connection_for(env["db"]) as conn:
        curation.copy_to_train(
            conn, env["p"]["id"], env["v"]["id"],
            ["X_c0.png", "X_c1.png"], "5_concept",
        )
    folder = _train_dir(env, "5_concept")
    assert (folder / "X_c0.png").read_bytes() == b"crop-0-bytes"
    assert (folder / "X_c1.png").read_bytes() == b"crop-1-bytes"
    # metadata 共享原图 caption，复制到各 stem
    assert (folder / "X_c0.txt").read_text(encoding="utf-8") == "tag for X"
    assert (folder / "X_c1.txt").read_text(encoding="utf-8") == "tag for X"


def test_copy_to_train_handles_mixed_processed_unprocessed(env) -> None:
    """同批 copy：1.png 已处理 → 用 preprocess；2.png 未处理 → 用 download。"""
    _dl(env, "1.png", blob=b"orig-1")
    _dl(env, "2.png", blob=b"orig-2")
    _seed_processed(env, "1.png", content=b"upscaled-1")

    with db.connection_for(env["db"]) as conn:
        curation.copy_to_train(
            conn, env["p"]["id"], env["v"]["id"], ["1.png", "2.png"], "5_concept"
        )
    folder = _train_dir(env, "5_concept")
    assert (folder / "1.png").read_bytes() == b"upscaled-1"
    assert (folder / "2.png").read_bytes() == b"orig-2"


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


# ---------------------------------------------------------------------------
# duplicate review
# ---------------------------------------------------------------------------


def test_duplicate_scan_returns_review_groups(env) -> None:
    _png(env, "1.png")
    _png(env, "2.png")
    _different_png(env, "3.png")

    with db.connection_for(env["db"]) as conn:
        result = duplicate_finder.scan_project_duplicates(
            conn,
            env["p"]["id"],
            duplicate_finder.DuplicateOptions(
                match_scope="strict",
                hash_workers=1,
            ),
        )

    assert result["total_images"] == 3
    assert result["group_count"] == 1
    group = result["groups"][0]
    assert group["keep"] in {"1.png", "2.png"}
    assert {item["name"] for item in group["items"]} == {"1.png", "2.png"}
    assert group["best"]["match_type"] == "strict-duplicate"


def test_duplicate_apply_marks_confirmed_names_without_touching_download(env) -> None:
    _png(env, "1.png")
    _png(env, "2.png")
    _meta(env, "2.png", ".txt", "tag")

    with db.connection_for(env["db"]) as conn:
        result = duplicate_finder.apply_duplicate_removals(
            conn,
            env["p"]["id"],
            names=["2.png"],
        )
        left = curation.list_download(conn, env["p"]["id"])

    pdir = projects.project_dir(env["p"]["id"], env["p"]["slug"])
    assert result["removed"] == ["2.png"]
    assert (pdir / "download" / "1.png").exists()
    assert (pdir / "download" / "2.png").exists()
    assert (pdir / "download" / "2.txt").exists()
    assert not (pdir / "download" / "_Duplicates_Found").exists()
    assert {item["name"] for item in left} == {"1.png"}
