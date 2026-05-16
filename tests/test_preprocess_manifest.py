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
    pm.add_processed(project_dir, "a.png", {"model": "X", "scale": 4})
    pm.add_processed(project_dir, "b.png", {"model": "Y", "scale": 2})
    m = pm.load(project_dir)
    assert set(m["images"].keys()) == {"a.png", "b.png"}
    assert m["images"]["a.png"]["model"] == "X"


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


def test_add_processed_auto_mtime(project_dir: Path) -> None:
    pm.add_processed(project_dir, "a.png", {"model": "X"})
    entry = pm.get_entry(project_dir, "a.png")
    assert entry is not None
    assert entry["kind"] == "processed"
    assert entry["model"] == "X"
    assert "mtime" in entry


def test_add_processed_overwrites(project_dir: Path) -> None:
    pm.add_processed(project_dir, "a.png", {"model": "X", "scale": 4})
    pm.add_processed(project_dir, "a.png", {"model": "Y", "scale": 2})
    entry = pm.get_entry(project_dir, "a.png")
    assert entry is not None
    assert entry["model"] == "Y"


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
    pm.add_processed(project_dir, "a.png", {"model": "X"})
    pm.add_processed(project_dir, "b.png", {"model": "Y"})
    out = pm.all_processed(project_dir)
    assert set(out.keys()) == {"a.png", "b.png"}
    assert out["a.png"]["model"] == "X"


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
