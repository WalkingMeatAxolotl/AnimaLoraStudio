"""preprocess_manifest: schema / load / save / resolve / migrate / restore。

ADR 0004：项目级 manifest 是预处理状态的唯一真理。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.services import preprocess_manifest as pm


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    (tmp_path / "download").mkdir()
    (tmp_path / "preprocess").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# load / save 基础
# ---------------------------------------------------------------------------


def test_load_missing_returns_empty(project_dir: Path) -> None:
    m = pm.load(project_dir)
    assert m == {"images": {}}


def test_load_corrupted_returns_empty(project_dir: Path) -> None:
    pm.manifest_path(project_dir).write_text("not json")
    m = pm.load(project_dir)
    assert m == {"images": {}}


def test_load_invalid_shape_returns_empty(project_dir: Path) -> None:
    """非 {images: {...}} 形状 → 当损坏处理。"""
    pm.manifest_path(project_dir).write_text(json.dumps(["array"]))
    assert pm.load(project_dir) == {"images": {}}


def test_atomic_write_overwrites(project_dir: Path) -> None:
    pm.add_processed(project_dir, "a.png", {"source": "a.png"})
    pm.add_processed(project_dir, "b.png", {"source": "b.png"})
    m = pm.load(project_dir)
    assert set(m["images"].keys()) == {"a.png", "b.png"}


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


def test_resolve_unknown_name_returns_download(project_dir: Path) -> None:
    """manifest 没记 → 隐式 original → download/{name}。"""
    p = pm.resolve(project_dir, "foo.jpg")
    assert p == project_dir / "download" / "foo.jpg"


def test_resolve_processed_returns_preprocess(project_dir: Path) -> None:
    pm.add_processed(project_dir, "foo.png", {"model": "X"})
    p = pm.resolve(project_dir, "foo.png")
    assert p == project_dir / "preprocess" / "foo.png"


def test_resolve_unknown_kind_returns_none(project_dir: Path) -> None:
    """未来扩展（如 deleted）：未知 kind → None，下游 skip。"""
    pm.manifest_path(project_dir).write_text(json.dumps({
        "images": {"foo.png": {"kind": "future_state"}},
    }))
    assert pm.resolve(project_dir, "foo.png") is None


# ---------------------------------------------------------------------------
# add_processed / restore
# ---------------------------------------------------------------------------


def test_add_processed_writes_new_schema_only(project_dir: Path) -> None:
    """新 schema 只写 {origin, mtime, size}，丢弃所有过程信息（model/scale/...）。

    历史 schema 字段（kind/model/scale/action/...）虽然 worker 可能透传给 meta，
    add_processed 不再持久化它们；下游 sidebar 容忍 None。
    """
    (project_dir / "preprocess" / "a.png").write_bytes(b"xxx")
    pm.add_processed(
        project_dir,
        "a.png",
        {"source": "a.jpg", "model": "4x-AnimeSharp", "scale": 4,
         "action": "upscale", "target_area": 1048576},
    )
    entry = pm.get_entry(project_dir, "a.png")
    assert entry is not None
    assert entry["origin"] == "a.jpg"
    assert "mtime" in entry
    assert entry["size"] == 3
    # 老字段不再写入
    assert "model" not in entry
    assert "scale" not in entry
    assert "kind" not in entry


def test_add_processed_overwrites(project_dir: Path) -> None:
    pm.add_processed(project_dir, "a.png", {"source": "a.jpg", "size": 1})
    pm.add_processed(project_dir, "a.png", {"source": "a.png", "size": 2})
    entry = pm.get_entry(project_dir, "a.png")
    assert entry is not None
    assert entry["origin"] == "a.png"
    assert entry["size"] == 2


def test_restore_removes_entry_and_png(project_dir: Path) -> None:
    pm.add_processed(project_dir, "a.png", {"model": "X"})
    (project_dir / "preprocess" / "a.png").write_bytes(b"upscaled")

    r = pm.restore(project_dir, ["a.png"])
    assert r == {"restored": ["a.png"], "missing": []}
    assert pm.get_entry(project_dir, "a.png") is None
    assert not (project_dir / "preprocess" / "a.png").exists()


def test_restore_missing_entry(project_dir: Path) -> None:
    r = pm.restore(project_dir, ["ghost.png"])
    assert r == {"restored": [], "missing": ["ghost.png"]}


def test_restore_self_heals_orphan_png(project_dir: Path) -> None:
    """PNG 存在但 manifest 没记（孤儿）→ 还原也把孤儿 PNG 清掉。"""
    (project_dir / "preprocess" / "orphan.png").write_bytes(b"x")
    r = pm.restore(project_dir, ["orphan.png"])
    # entry 不在 → 算 missing；但 PNG 还是被删（自愈）
    assert r["missing"] == ["orphan.png"]
    assert not (project_dir / "preprocess" / "orphan.png").exists()


# ---------------------------------------------------------------------------
# all_processed
# ---------------------------------------------------------------------------


def test_all_processed_returns_dict(project_dir: Path) -> None:
    pm.add_processed(project_dir, "a.png", {"source": "a.jpg"})
    pm.add_processed(project_dir, "b.png", {"source": "b.png"})
    out = pm.all_processed(project_dir)
    assert set(out.keys()) == {"a.png", "b.png"}
    assert out["a.png"]["origin"] == "a.jpg"


def test_all_processed_includes_legacy_entries(project_dir: Path) -> None:
    """老 schema entry（有 kind=processed + model 等字段）依然算已处理。
    新 schema entry（无 kind）也算已处理。只有显式 kind != processed 才跳过。"""
    pm.manifest_path(project_dir).write_text(json.dumps({
        "images": {
            "legacy.png": {"kind": "processed", "model": "X", "source": "legacy.jpg"},
            "new.png":    {"origin": "new.png",  "mtime": 1.0, "size": 5},
            "future.png": {"kind": "future_state"},
        },
    }))
    out = pm.all_processed(project_dir)
    assert set(out.keys()) == {"legacy.png", "new.png"}


def test_all_processed_skips_non_processed_kind(project_dir: Path) -> None:
    pm.manifest_path(project_dir).write_text(json.dumps({
        "images": {
            "a.png": {"kind": "processed", "model": "X"},
            "b.png": {"kind": "future_state"},
        },
    }))
    out = pm.all_processed(project_dir)
    assert set(out.keys()) == {"a.png"}


# ---------------------------------------------------------------------------
# 新增：replace_with_crops 多裁剪 fan-out
# ---------------------------------------------------------------------------


def test_entry_origin_prefers_origin_then_source_then_name(project_dir: Path) -> None:
    assert pm.entry_origin({"origin": "a.jpg", "source": "ignored"}, "x") == "a.jpg"
    assert pm.entry_origin({"source": "b.jpg"}, "x") == "b.jpg"
    assert pm.entry_origin({}, "fallback.png") == "fallback.png"


def test_replace_with_crops_replaces_single_entry(project_dir: Path) -> None:
    """N=1 覆盖：source 同名 entry 被替换。"""
    pm.add_processed(project_dir, "X.png", {"source": "X.png", "size": 100})
    pm.replace_with_crops(
        project_dir,
        source_name="X.png",
        outputs=[
            {"name": "X.png", "origin": "X.png", "size": 50, "mtime": 1.0},
        ],
    )
    entry = pm.get_entry(project_dir, "X.png")
    assert entry is not None
    assert entry["origin"] == "X.png"
    assert entry["size"] == 50


def test_replace_with_crops_fans_out(project_dir: Path) -> None:
    """N>1：原 entry 被 N 个 _c{n} entry 替换，origin 都指 root。"""
    pm.add_processed(project_dir, "X.png", {"source": "X.jpg", "size": 100})
    pm.replace_with_crops(
        project_dir,
        source_name="X.png",
        outputs=[
            {"name": "X_c0.png", "origin": "X.jpg", "size": 30, "mtime": 1.0},
            {"name": "X_c1.png", "origin": "X.jpg", "size": 40, "mtime": 1.0},
        ],
    )
    m = pm.load(project_dir)
    assert "X.png" not in m["images"]
    assert m["images"]["X_c0.png"]["origin"] == "X.jpg"
    assert m["images"]["X_c1.png"]["origin"] == "X.jpg"


def test_replace_with_crops_cleans_old_siblings(project_dir: Path) -> None:
    """再裁剪：之前由同一 source 派生的 entry 应当一并清除（防止幽灵残留）。"""
    # 第一次 fan-out 出 X_c0/X_c1
    pm.replace_with_crops(
        project_dir,
        source_name="X.png",
        outputs=[
            {"name": "X_c0.png", "origin": "X.jpg", "size": 1, "mtime": 1.0},
            {"name": "X_c1.png", "origin": "X.jpg", "size": 2, "mtime": 1.0},
        ],
    )
    # 现在用户对 X_c0.png 再做一次多裁剪
    pm.replace_with_crops(
        project_dir,
        source_name="X_c0.png",
        outputs=[
            {"name": "X_c0_c0.png", "origin": "X.jpg", "size": 1, "mtime": 2.0},
            {"name": "X_c0_c1.png", "origin": "X.jpg", "size": 2, "mtime": 2.0},
        ],
    )
    m = pm.load(project_dir)
    # X_c0.png 自身应被删；X_c1.png 应保留（它的 origin 不是 X_c0.png）；
    # 新两个 entry 写入
    assert "X_c0.png" not in m["images"]
    assert "X_c1.png" in m["images"]
    assert "X_c0_c0.png" in m["images"]
    assert "X_c0_c1.png" in m["images"]


def test_resolve_origin_returns_derivatives(project_dir: Path) -> None:
    """resolve_origin: 给 download/X.jpg → 返回 manifest 里 origin 匹配的所有产物。"""
    pm.replace_with_crops(
        project_dir,
        source_name="X.png",
        outputs=[
            {"name": "X_c0.png", "origin": "X.jpg", "size": 1, "mtime": 1.0},
            {"name": "X_c1.png", "origin": "X.jpg", "size": 2, "mtime": 1.0},
        ],
    )
    paths = pm.resolve_origin(project_dir, "X.jpg")
    assert sorted(p.name for p in paths) == ["X_c0.png", "X_c1.png"]
    # 不在 manifest 的 origin → 回退 download
    fallback = pm.resolve_origin(project_dir, "ghost.jpg")
    assert fallback == [project_dir / "download" / "ghost.jpg"]


# ---------------------------------------------------------------------------
# clear_all
# ---------------------------------------------------------------------------


def test_clear_all_resets(project_dir: Path) -> None:
    pm.add_processed(project_dir, "a.png", {"model": "X"})
    (project_dir / "preprocess" / "a.png").write_bytes(b"x")
    pm.clear_all(project_dir)
    assert pm.load(project_dir) == {"images": {}}
    assert not (project_dir / "preprocess" / "a.png").exists()


# ---------------------------------------------------------------------------
# Migration from legacy sidecars
# ---------------------------------------------------------------------------


def _write_legacy_sidecar(project_dir: Path, png_name: str, meta: dict) -> None:
    """模拟老项目：preprocess/{name}.png + preprocess/{name}.png.preprocess.json。"""
    pre = project_dir / "preprocess"
    pre.mkdir(parents=True, exist_ok=True)
    (pre / png_name).write_bytes(b"upscaled")
    (pre / (png_name + ".preprocess.json")).write_text(
        json.dumps(meta), encoding="utf-8"
    )


def test_ensure_manifest_creates_when_missing(project_dir: Path) -> None:
    m = pm.ensure_manifest(project_dir)
    assert m == {"images": {}}
    assert pm.manifest_path(project_dir).exists()


def test_ensure_manifest_migrates_sidecars(project_dir: Path) -> None:
    _write_legacy_sidecar(project_dir, "a.png", {
        "source": "a.jpg", "model": "RealESRGAN_x4", "scale": 4,
        "src_size": [512, 512], "dst_size": [2048, 2048],
    })
    _write_legacy_sidecar(project_dir, "b.png", {
        "source": "b.webp", "model": "RealESRGAN_x4", "scale": 4,
    })

    m = pm.ensure_manifest(project_dir)
    assert set(m["images"].keys()) == {"a.png", "b.png"}
    assert m["images"]["a.png"]["kind"] == "processed"
    assert m["images"]["a.png"]["source"] == "a.jpg"
    assert m["images"]["a.png"]["model"] == "RealESRGAN_x4"


def test_ensure_manifest_idempotent(project_dir: Path) -> None:
    """第二次调用不再扫 sidecar，返回已存在 manifest。"""
    _write_legacy_sidecar(project_dir, "a.png", {"source": "a.jpg"})
    pm.ensure_manifest(project_dir)
    # 修改 manifest（手动加一项）
    pm.add_processed(project_dir, "manual.png", {"model": "manual"})
    # 再写一个 sidecar；ensure_manifest 应该不去扫它
    _write_legacy_sidecar(project_dir, "c.png", {"source": "c.jpg"})

    m = pm.ensure_manifest(project_dir)
    assert "manual.png" in m["images"]
    assert "c.png" not in m["images"]  # 没二次迁移


def test_ensure_manifest_skips_sidecar_without_png(project_dir: Path) -> None:
    """sidecar 残留但 PNG 已删 → 不迁移（防 stale）。"""
    pre = project_dir / "preprocess"
    pre.mkdir(parents=True, exist_ok=True)
    (pre / "ghost.png.preprocess.json").write_text(
        json.dumps({"source": "ghost.jpg"}), encoding="utf-8"
    )

    m = pm.ensure_manifest(project_dir)
    assert m == {"images": {}}


def test_legacy_sidecars_not_deleted_after_migration(project_dir: Path) -> None:
    """迁移完老 sidecar 保留不删（防御性回滚）。"""
    _write_legacy_sidecar(project_dir, "a.png", {"source": "a.jpg"})
    pm.ensure_manifest(project_dir)
    assert (project_dir / "preprocess" / "a.png.preprocess.json").exists()
