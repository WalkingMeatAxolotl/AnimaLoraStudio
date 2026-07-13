"""训练 mask sidecar（PR-B B1）——读写 / 变换跟随 / 删除路径联动 / bundle 往返。

设计：docs/design/preprocess-inpaint-mask-design.md §2 / §7 / §9。
mask 路径 = train/masks/{folder}/{stem}.png（恒 .png，stem 镜像图片）。
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from studio import db
from studio.domain.errors import InvalidPathError, NotFoundError, ValidationError
from studio.services.dataset import curation
from studio.services.preprocess import core as preprocess
from studio.services.preprocess import manifest as preprocess_manifest
from studio.services.preprocess import masks as train_masks
from studio.services.projects import jobs as project_jobs, projects, versions


DEFAULT_FOLDER = "1_data"


def _rel(name: str, folder: str = DEFAULT_FOLDER) -> str:
    return f"{folder}/{name}"


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    with db.connection_for(dbfile) as conn:
        p = projects.create_project(conn, title="PM")
        v = versions.create_version(conn, project_id=p["id"], label="v1")
    train_dir = preprocess.version_train_dir(p, "v1")
    (train_dir / DEFAULT_FOLDER).mkdir(parents=True, exist_ok=True)
    return {
        "db": dbfile, "project": p, "version": v,
        "train": train_dir, "sub": train_dir / DEFAULT_FOLDER,
    }


def _write_png(path: Path, size=(10, 10), color="red", mode="RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color=color).save(path, "PNG")


def _mask_bytes(size=(10, 10), value=0) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, color=value).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 路径 / CRUD
# ---------------------------------------------------------------------------


def test_mask_path_is_png_stem_mirror(isolated) -> None:
    """mask 路径恒 .png 且镜像 stem —— X.jpg 与 X.png 共享同一 mask。"""
    train = isolated["train"]
    p_jpg = train_masks.mask_path_for(train, _rel("X.jpg"))
    p_png = train_masks.mask_path_for(train, _rel("X.png"))
    assert p_jpg == p_png == train / "masks" / DEFAULT_FOLDER / "X.png"


def test_mask_save_roundtrip(isolated) -> None:
    p = isolated["project"]
    _write_png(isolated["sub"] / "X.png", (10, 10))
    res = preprocess.mask_save_train(
        p, "v1", name=_rel("X.png"), data=_mask_bytes((10, 10), 0),
    )
    assert res["name"] == _rel("X.png")
    mp = train_masks.mask_path_for(isolated["train"], _rel("X.png"))
    assert mp.is_file()
    with Image.open(mp) as im:
        assert im.mode == "L"
        assert im.getpixel((0, 0)) == 0

    assert preprocess.mask_file_train(p, "v1", name=_rel("X.png")) == mp
    assert preprocess.mask_delete_train(p, "v1", name=_rel("X.png"))["deleted"]
    assert not mp.exists()
    # 再删不报错
    assert preprocess.mask_delete_train(p, "v1", name=_rel("X.png"))["deleted"] is False


def test_mask_save_converts_rgb_to_gray(isolated) -> None:
    p = isolated["project"]
    _write_png(isolated["sub"] / "X.png", (10, 10))
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(buf, "PNG")
    preprocess.mask_save_train(p, "v1", name=_rel("X.png"), data=buf.getvalue())
    with Image.open(train_masks.mask_path_for(isolated["train"], _rel("X.png"))) as im:
        assert im.mode == "L"


def test_mask_save_size_mismatch_rejected(isolated) -> None:
    p = isolated["project"]
    _write_png(isolated["sub"] / "X.png", (10, 10))
    with pytest.raises(ValidationError):
        preprocess.mask_save_train(
            p, "v1", name=_rel("X.png"), data=_mask_bytes((11, 10)),
        )


def test_mask_save_source_missing_rejected(isolated) -> None:
    p = isolated["project"]
    with pytest.raises(NotFoundError):
        preprocess.mask_save_train(p, "v1", name=_rel("nope.png"), data=_mask_bytes())


def test_mask_save_invalid_name_rejected(isolated) -> None:
    p = isolated["project"]
    for bad in ("../X.png", "X.png", "a/b/c.png"):
        with pytest.raises(InvalidPathError):
            preprocess.mask_save_train(p, "v1", name=bad, data=_mask_bytes())


def test_workspace_reports_mask_mtime(isolated) -> None:
    p = isolated["project"]
    _write_png(isolated["sub"] / "X.png", (10, 10))
    _write_png(isolated["sub"] / "Y.png", (10, 10))
    preprocess.mask_save_train(p, "v1", name=_rel("X.png"), data=_mask_bytes((10, 10)))

    items = {it["name"]: it for it in preprocess.list_crop_workspace_train(p, "v1")}
    assert items[_rel("X.png")]["mask_mtime"] is not None
    assert items[_rel("Y.png")]["mask_mtime"] is None


# ---------------------------------------------------------------------------
# 变换跟随
# ---------------------------------------------------------------------------


def test_crop_mask_like_single_inplace(isolated) -> None:
    """N=1 crop：mask 同 box 裁剪原地覆盖（stem 不变路径不变）。"""
    train = isolated["train"]
    _write_png(isolated["sub"] / "X.png", (10, 10))
    # 上半 0 下半 255 的 mask
    m = Image.new("L", (10, 10), 255)
    m.paste(0, (0, 0, 10, 5))
    mp = train_masks.mask_path_for(train, _rel("X.png"))
    mp.parent.mkdir(parents=True, exist_ok=True)
    m.save(mp, "PNG")

    # 裁下半（y 5..10）→ mask 应全 255
    train_masks.crop_mask_like(train, _rel("X.png"), [(0, 5, 10, 10)], [_rel("X.png")])
    with Image.open(mp) as im:
        assert im.size == (10, 5)
        assert im.getpixel((0, 0)) == 255


def test_crop_mask_like_fanout_removes_source(isolated) -> None:
    train = isolated["train"]
    _write_png(isolated["sub"] / "X.png", (10, 10))
    mp = train_masks.mask_path_for(train, _rel("X.png"))
    _write_png(mp, (10, 10), color=0, mode="L")

    train_masks.crop_mask_like(
        train, _rel("X.png"),
        [(0, 0, 5, 10), (5, 0, 10, 10)],
        [_rel("X_c0.png"), _rel("X_c1.png")],
    )
    assert not mp.exists()
    for out in ("X_c0.png", "X_c1.png"):
        op = train_masks.mask_path_for(train, _rel(out))
        with Image.open(op) as im:
            assert im.size == (5, 10)


def test_crop_mask_like_noop_without_mask(isolated) -> None:
    train = isolated["train"]
    _write_png(isolated["sub"] / "X.png", (10, 10))
    train_masks.crop_mask_like(train, _rel("X.png"), [(0, 0, 5, 5)], [_rel("X.png")])
    assert not (train / "masks").exists()


def test_resize_mask_like(isolated) -> None:
    train = isolated["train"]
    _write_png(isolated["sub"] / "X.png", (10, 10))
    mp = train_masks.mask_path_for(train, _rel("X.png"))
    _write_png(mp, (10, 10), color=0, mode="L")

    train_masks.resize_mask_like(train, _rel("X.png"), (40, 40))
    with Image.open(mp) as im:
        assert im.size == (40, 40)
        assert im.getpixel((0, 0)) == 0
    # 无 mask 图 no-op 不报错
    train_masks.resize_mask_like(train, _rel("Y.png"), (40, 40))


# ---------------------------------------------------------------------------
# 删除路径联动
# ---------------------------------------------------------------------------


def test_restore_deletes_mask(isolated) -> None:
    """restore = 回 download 原点，mask 一律作废（D8）。"""
    p = isolated["project"]
    pdir = preprocess.project_root(p)
    train = isolated["train"]
    download = pdir / "download"
    _write_png(download / "X.png", (20, 20))
    _write_png(isolated["sub"] / "X.png", (10, 10))
    preprocess_manifest.train_add_processed(
        pdir, "v1", _rel("X.png"), {"origin": "X.png", "processed": True},
    )
    mp = train_masks.mask_path_for(train, _rel("X.png"))
    _write_png(mp, (10, 10), color=0, mode="L")

    res = preprocess.restore_products_train(p, "v1", [_rel("X.png")])
    assert res["restored"] == [_rel("X.png")]
    assert not mp.exists()


def test_mark_duplicate_removed_deletes_mask(isolated) -> None:
    p = isolated["project"]
    pdir = preprocess.project_root(p)
    train = isolated["train"]
    _write_png(isolated["sub"] / "X.png", (10, 10))
    mp = train_masks.mask_path_for(train, _rel("X.png"))
    _write_png(mp, (10, 10), color=0, mode="L")

    preprocess_manifest.train_mark_duplicate_removed(pdir, "v1", [_rel("X.png")])
    assert not mp.exists()


def test_remove_from_train_deletes_mask(isolated) -> None:
    p = isolated["project"]
    v = isolated["version"]
    pdir = preprocess.project_root(p)
    train = isolated["train"]
    _write_png(isolated["sub"] / "X.png", (10, 10))
    preprocess_manifest.train_add_processed(
        pdir, "v1", _rel("X.png"), {"origin": "X.png"},
    )
    mp = train_masks.mask_path_for(train, _rel("X.png"))
    _write_png(mp, (10, 10), color=0, mode="L")

    with db.connection_for(isolated["db"]) as conn:
        res = curation.remove_from_train(
            conn, p["id"], v["id"], DEFAULT_FOLDER, ["X.png"],
        )
    assert res["removed"] == ["X.png"]
    assert not mp.exists()


# ---------------------------------------------------------------------------
# 训练器 / studio 递归扫描排除
# ---------------------------------------------------------------------------


def test_trainer_scan_skips_masks_dir(isolated, tmp_path: Path) -> None:
    """runtime ImageDataset 对子目录 rglob 递归 —— masks/ 必须被排除，
    否则 mask 灰度图会被当训练样本（RESERVED_SUBDIRS）。"""
    from runtime.training.dataset import ImageDataset

    train = isolated["train"]
    _write_png(isolated["sub"] / "X.png", (64, 64))
    (isolated["sub"] / "X.txt").write_text("1girl", encoding="utf-8")
    mp = train_masks.mask_path_for(train, _rel("X.png"))
    _write_png(mp, (64, 64), color=0, mode="L")

    ds = ImageDataset(train, resolution=64)
    names = {Path(s["image"]).name for s in ds.samples}
    assert names == {"X.png"}


def test_bucket_histogram_skips_masks_dir(isolated) -> None:
    from studio.services.projects.versions import compute_bucket_histogram

    train = isolated["train"]
    _write_png(isolated["sub"] / "X.png", (64, 64))
    (isolated["sub"] / "X.txt").write_text("1girl", encoding="utf-8")
    mp = train_masks.mask_path_for(train, _rel("X.png"))
    _write_png(mp, (64, 64), color=0, mode="L")
    # masks 下就算出现带 caption 的图也不参与（保留目录整体排除）
    (train / "masks" / DEFAULT_FOLDER / "X.txt").write_text("1girl", encoding="utf-8")

    hist = compute_bucket_histogram(train, [64])
    total = sum(b["count"] for entry in hist for b in entry["buckets"])
    assert total == 1


# ---------------------------------------------------------------------------
# bundle 导出 / 导入往返
# ---------------------------------------------------------------------------


def test_bundle_masks_roundtrip(isolated, tmp_path: Path) -> None:
    from studio.services.data_io import train_io

    p = isolated["project"]
    v = isolated["version"]
    train = isolated["train"]
    _write_png(isolated["sub"] / "X.png", (10, 10))
    (isolated["sub"] / "X.txt").write_text("1girl", encoding="utf-8")
    mp = train_masks.mask_path_for(train, _rel("X.png"))
    _write_png(mp, (10, 10), color=0, mode="L")

    dest = tmp_path / "bundle.zip"
    with db.connection_for(isolated["db"]) as conn:
        res = train_io.export_bundle(
            conn, v["id"], dest,
            train_io.BundleOptions(train=True, train_captions=True, train_masks=True),
        )
    assert res["manifest"]["stats"]["train_mask_count"] == 1
    with zipfile.ZipFile(dest) as zf:
        assert f"train/masks/{DEFAULT_FOLDER}/X.png" in zf.namelist()

    with db.connection_for(isolated["db"]) as conn:
        imported = train_io.import_bundle(
            conn, dest, presets_base=tmp_path / "presets",
        )
    new_train = versions.version_dir(
        imported["project"]["id"],
        imported["project"]["slug"],
        imported["version"]["label"],
    ) / "train"
    new_mask = train_masks.mask_path_for(new_train, _rel("X.png"))
    assert new_mask.is_file()
    with Image.open(new_mask) as im:
        assert im.size == (10, 10)


def test_bundle_masks_excluded_by_default(isolated, tmp_path: Path) -> None:
    from studio.services.data_io import train_io

    v = isolated["version"]
    train = isolated["train"]
    _write_png(isolated["sub"] / "X.png", (10, 10))
    mp = train_masks.mask_path_for(train, _rel("X.png"))
    _write_png(mp, (10, 10), color=0, mode="L")

    dest = tmp_path / "bundle.zip"
    with db.connection_for(isolated["db"]) as conn:
        train_io.export_bundle(conn, v["id"], dest, train_io.BundleOptions(train=True))
    with zipfile.ZipFile(dest) as zf:
        assert not any(n.startswith("train/masks/") for n in zf.namelist())
