"""ADR 0010 — train-scope manifest mutation API（PR-2 step A）。

覆盖 `train_load / train_get_entry / train_all_processed / train_duplicate_removed*
/ train_add_processed / train_replace_with_crops / train_mark_duplicate_removed /
train_restore_duplicate_removed / train_restore / train_clear_all`。

老 project-scope API（`add_processed / restore / mark_duplicate_removed / etc.`）
独立测试在 `test_preprocess_manifest.py`，本文件不覆盖。

关键语义点（vs 老 API）：
- manifest 落 `versions/{label}/train/manifest.json`
- `train_restore` = 从 `download/{entry.origin}` 复制覆盖 `train/{name}`
  （老的是"删 entry"，依赖 resolver fallback）
- size 兜底 stat `train/{name}`（老的 stat preprocess/{name}）
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from studio.services.preprocess import manifest as pm


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """模拟项目目录 + 一个 version v1。"""
    (tmp_path / "download").mkdir()
    (tmp_path / "preprocess").mkdir()
    (tmp_path / "versions" / "v1" / "train").mkdir(parents=True)
    return tmp_path


def _train_path(project_dir: Path, name: str, label: str = "v1") -> Path:
    return project_dir / "versions" / label / "train" / name


def _download_path(project_dir: Path, name: str) -> Path:
    return project_dir / "download" / name


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# train_load / train_get_entry / train_all_processed
# ---------------------------------------------------------------------------


def test_load_creates_empty_via_ensure(project_dir: Path) -> None:
    """train_load 触发 fallback 重建（无 legacy → 空 manifest）。"""
    m = pm.train_load(project_dir, "v1")
    assert m == {"version": 2, "images": {}}
    # 已落盘
    assert pm.train_manifest_path(project_dir, "v1").exists()


def test_get_entry_missing_returns_none(project_dir: Path) -> None:
    assert pm.train_get_entry(project_dir, "v1", "missing.png") is None


def test_all_processed_filters_duplicate_removed(project_dir: Path) -> None:
    pm.train_add_processed(project_dir, "v1", "A.png", {"origin": "A.jpg"})
    _train_path(project_dir, "B.png").write_bytes(b"b")
    pm.train_mark_duplicate_removed(project_dir, "v1", ["B.png"])

    processed = pm.train_all_processed(project_dir, "v1")
    assert set(processed) == {"A.png"}


def test_duplicate_removed_origins_collects_all(project_dir: Path) -> None:
    _train_path(project_dir, "X.png").write_bytes(b"x")
    _train_path(project_dir, "Y.png").write_bytes(b"y")
    pm.train_mark_duplicate_removed(project_dir, "v1", ["X.png", "Y.png"])

    origins = pm.train_duplicate_removed_origins(project_dir, "v1")
    assert origins == {"X.png", "Y.png"}


# ---------------------------------------------------------------------------
# train_add_processed
# ---------------------------------------------------------------------------


def test_add_processed_stores_minimal_schema(project_dir: Path) -> None:
    """新 schema 只采纳 origin/mtime/size；过程字段全丢。"""
    _train_path(project_dir, "X.png").write_bytes(b"\x89PNG" + b"x" * 100)
    pm.train_add_processed(project_dir, "v1", "X.png", {
        "origin": "X.jpg",
        "model": "RealESRGAN_x4",  # 过程信息，应被丢
        "scale": 4,                # 过程信息
        "action": "upscale",       # 过程信息
        "src_size": [512, 512],    # 过程信息
        "mtime": 1731000000,
    })

    m = pm.train_load(project_dir, "v1")
    entry = m["images"]["X.png"]
    assert entry["origin"] == "X.jpg"
    assert entry["mtime"] == 1731000000
    assert entry["size"] == 104  # stat'd train/X.png
    # 过程字段不应进 entry
    assert "model" not in entry
    assert "scale" not in entry
    assert "action" not in entry


def test_add_processed_size_fallback_stat_train(project_dir: Path) -> None:
    """size 缺失时从 train/{name} stat（不是 preprocess/）。"""
    _train_path(project_dir, "Y.png").write_bytes(b"y" * 42)
    # preprocess/Y.png 不存在 — 防御老逻辑误用
    pm.train_add_processed(project_dir, "v1", "Y.png", {"origin": "Y.jpg"})

    entry = pm.train_get_entry(project_dir, "v1", "Y.png")
    assert entry is not None
    assert entry["size"] == 42


def test_add_processed_origin_fallback_to_name(project_dir: Path) -> None:
    """meta 没 origin/source → 用 name 自身（1:1 同名场景）。"""
    _train_path(project_dir, "Z.jpg").write_bytes(b"z")
    pm.train_add_processed(project_dir, "v1", "Z.jpg", {})

    entry = pm.train_get_entry(project_dir, "v1", "Z.jpg")
    assert entry is not None
    assert entry["origin"] == "Z.jpg"


# ---------------------------------------------------------------------------
# train_replace_with_crops
# ---------------------------------------------------------------------------


def test_replace_with_crops_removes_source_and_writes_fan_out(
    project_dir: Path,
) -> None:
    # 老 entry：X.png 是 X.jpg 的 upscale 产物
    _train_path(project_dir, "X.png").write_bytes(b"x" * 10)
    pm.train_add_processed(project_dir, "v1", "X.png", {"origin": "X.jpg"})

    # multi-crop: X.png → X_c0.png / X_c1.png
    pm.train_replace_with_crops(
        project_dir, "v1",
        source_name="X.png",
        outputs=[
            {"name": "X_c0.png", "origin": "X.jpg", "mtime": 1, "size": 100},
            {"name": "X_c1.png", "origin": "X.jpg", "mtime": 1, "size": 110},
        ],
    )

    m = pm.train_load(project_dir, "v1")
    assert "X.png" not in m["images"]
    assert m["images"]["X_c0.png"]["origin"] == "X.jpg"
    assert m["images"]["X_c1.png"]["origin"] == "X.jpg"
    assert m["images"]["X_c1.png"]["size"] == 110


def test_replace_with_crops_origin_fallback_to_source(
    project_dir: Path,
) -> None:
    """outputs 没 origin 字段时回退到 source_name。"""
    _train_path(project_dir, "raw.png").write_bytes(b"r")
    pm.train_add_processed(project_dir, "v1", "raw.png", {"origin": "raw.png"})

    pm.train_replace_with_crops(
        project_dir, "v1",
        source_name="raw.png",
        outputs=[{"name": "raw_c0.png", "size": 50}],
    )

    entry = pm.train_get_entry(project_dir, "v1", "raw_c0.png")
    assert entry is not None
    assert entry["origin"] == "raw.png"


# ---------------------------------------------------------------------------
# train_mark_duplicate_removed
# ---------------------------------------------------------------------------


def test_mark_duplicate_removed_on_existing_processed(project_dir: Path) -> None:
    _train_path(project_dir, "A.png").write_bytes(b"a")
    pm.train_add_processed(project_dir, "v1", "A.png", {"origin": "A.jpg"})

    result = pm.train_mark_duplicate_removed(project_dir, "v1", ["A.png"])
    assert result == {"removed": ["A.png"], "missing": [], "skipped": []}

    entry = pm.train_get_entry(project_dir, "v1", "A.png")
    assert entry is not None
    assert entry["kind"] == pm.DUPLICATE_REMOVED_KIND
    assert entry["origin"] == "A.jpg"
    # train/A.png 物理文件**不删**——保留作"已审核但跳过"标记
    assert _train_path(project_dir, "A.png").exists()


def test_mark_duplicate_removed_on_unrecorded_train_file(
    project_dir: Path,
) -> None:
    """name 不在 manifest 但物理在 train/ → 标记 + origin = name。"""
    _train_path(project_dir, "B.png").write_bytes(b"b" * 5)

    result = pm.train_mark_duplicate_removed(project_dir, "v1", ["B.png"])
    assert result == {"removed": ["B.png"], "missing": [], "skipped": []}

    entry = pm.train_get_entry(project_dir, "v1", "B.png")
    assert entry is not None
    assert entry["origin"] == "B.png"  # 没 manifest 时 origin = name
    assert entry["size"] == 5


def test_mark_duplicate_removed_missing_returns_missing(project_dir: Path) -> None:
    """name 不在 manifest 且 train/{name} 物理也不在 → missing。"""
    result = pm.train_mark_duplicate_removed(project_dir, "v1", ["ghost.png"])
    assert result["missing"] == ["ghost.png"]
    assert result["removed"] == []


def test_mark_duplicate_removed_already_marked_skipped(project_dir: Path) -> None:
    _train_path(project_dir, "C.png").write_bytes(b"c")
    pm.train_mark_duplicate_removed(project_dir, "v1", ["C.png"])

    result = pm.train_mark_duplicate_removed(project_dir, "v1", ["C.png"])
    assert result["skipped"] == ["C.png"]
    assert result["removed"] == []


# ---------------------------------------------------------------------------
# train_restore_duplicate_removed
# ---------------------------------------------------------------------------


def test_restore_duplicate_removed_unwinds_mark(project_dir: Path) -> None:
    _train_path(project_dir, "D.png").write_bytes(b"d")
    pm.train_mark_duplicate_removed(project_dir, "v1", ["D.png"])

    result = pm.train_restore_duplicate_removed(project_dir, "v1", ["D.png"])
    assert result == {"restored": ["D.png"], "missing": []}
    assert pm.train_get_entry(project_dir, "v1", "D.png") is None


def test_restore_duplicate_removed_missing_when_not_marked(
    project_dir: Path,
) -> None:
    _train_path(project_dir, "E.png").write_bytes(b"e")
    pm.train_add_processed(project_dir, "v1", "E.png", {"origin": "E.jpg"})

    result = pm.train_restore_duplicate_removed(project_dir, "v1", ["E.png"])
    assert result == {"restored": [], "missing": ["E.png"]}
    # processed entry 不动
    entry = pm.train_get_entry(project_dir, "v1", "E.png")
    assert entry is not None
    assert entry.get("kind", "processed") == "processed"


# ---------------------------------------------------------------------------
# train_restore — 新语义：从 download/{origin} 复制覆盖到 train/{name}
# ---------------------------------------------------------------------------


def test_restore_copies_from_download_overwriting_train(
    project_dir: Path,
) -> None:
    """已 upscale 产物 train/X.png ← 从 download/X.jpg 复制覆盖。"""
    _download_path(project_dir, "X.jpg").write_bytes(b"orig" * 10)
    _train_path(project_dir, "X.png").write_bytes(b"upscaled" * 100)
    pm.train_add_processed(project_dir, "v1", "X.png", {"origin": "X.jpg"})

    result = pm.train_restore(project_dir, "v1", ["X.png"])

    assert result == {"restored": ["X.png"], "missing": [], "no_origin": []}
    # train/X.png 内容被替换成 download 原图
    assert _train_path(project_dir, "X.png").read_bytes() == b"orig" * 10
    # manifest entry 保留 origin，mtime/size 更新到 download 文件
    entry = pm.train_get_entry(project_dir, "v1", "X.png")
    assert entry is not None
    assert entry["origin"] == "X.jpg"
    assert entry["size"] == 40


def test_restore_missing_when_no_manifest_entry(project_dir: Path) -> None:
    result = pm.train_restore(project_dir, "v1", ["ghost.png"])
    assert result == {"restored": [], "missing": ["ghost.png"], "no_origin": []}


def test_restore_no_origin_when_download_missing(project_dir: Path) -> None:
    """ADR 0010 §Restore 语义：download/{origin} 缺失 → no_origin（不是
    missing），UI 给三选项 [拖入替换 / 保留 / 移除]。"""
    _train_path(project_dir, "Y.png").write_bytes(b"y")
    pm.train_add_processed(project_dir, "v1", "Y.png", {"origin": "Y.jpg"})
    # 不创建 download/Y.jpg

    result = pm.train_restore(project_dir, "v1", ["Y.png"])

    assert result == {"restored": [], "missing": [], "no_origin": ["Y.png"]}
    # train/Y.png 内容**不动**
    assert _train_path(project_dir, "Y.png").read_bytes() == b"y"


def test_restore_mixed_results(project_dir: Path) -> None:
    """三类结果同时返回。"""
    _download_path(project_dir, "OK.jpg").write_bytes(b"ok")
    _train_path(project_dir, "OK.png").write_bytes(b"old-ok")
    _train_path(project_dir, "NoOrig.png").write_bytes(b"x")
    pm.train_add_processed(project_dir, "v1", "OK.png", {"origin": "OK.jpg"})
    pm.train_add_processed(
        project_dir, "v1", "NoOrig.png", {"origin": "missing.jpg"}
    )

    result = pm.train_restore(
        project_dir, "v1", ["OK.png", "NoOrig.png", "Ghost.png"]
    )

    assert result["restored"] == ["OK.png"]
    assert result["no_origin"] == ["NoOrig.png"]
    assert result["missing"] == ["Ghost.png"]


# ---------------------------------------------------------------------------
# train_clear_all
# ---------------------------------------------------------------------------


def test_clear_all_empties_manifest_keeps_train_files(project_dir: Path) -> None:
    """clear_all **不动** train/ 物理文件——它们是训练数据，不该被预处理清空
    操作引发删除。详 ADR 0010 §train_clear_all。"""
    _train_path(project_dir, "P.png").write_bytes(b"p")
    pm.train_add_processed(project_dir, "v1", "P.png", {"origin": "P.jpg"})

    pm.train_clear_all(project_dir, "v1")

    m = pm.train_load(project_dir, "v1")
    assert m == {"version": 2, "images": {}}
    # 物理文件保留
    assert _train_path(project_dir, "P.png").exists()


# ---------------------------------------------------------------------------
# 多 version 隔离
# ---------------------------------------------------------------------------


def test_multi_version_isolation(project_dir: Path) -> None:
    """v1 / v2 各自 manifest 独立，mutation 不串。"""
    (project_dir / "versions" / "v2" / "train").mkdir(parents=True)
    (project_dir / "versions" / "v2" / "train" / "Q.png").write_bytes(b"q")

    pm.train_add_processed(project_dir, "v1", "v1-only.png", {"origin": "x.jpg"})
    pm.train_add_processed(project_dir, "v2", "Q.png", {"origin": "Q.jpg"})

    v1 = pm.train_load(project_dir, "v1")
    v2 = pm.train_load(project_dir, "v2")
    assert set(v1["images"]) == {"v1-only.png"}
    assert set(v2["images"]) == {"Q.png"}


# ---------------------------------------------------------------------------
# 老 project-scope manifest 不动（PR-2 backward-compat 保障）
# ---------------------------------------------------------------------------


def test_train_mutations_dont_touch_project_manifest(project_dir: Path) -> None:
    """train_xxx 函数只动 train manifest；老 project-scope manifest 完全不动
    （PR-3 才删，PR-2 阶段必须保持向后兼容）。"""
    pm.manifest_path(project_dir).write_text(
        json.dumps({"images": {"legacy.png": {"origin": "legacy.jpg"}}}),
        encoding="utf-8",
    )
    legacy_mtime = pm.manifest_path(project_dir).stat().st_mtime_ns

    _train_path(project_dir, "X.png").write_bytes(b"x")
    pm.train_add_processed(project_dir, "v1", "X.png", {"origin": "X.jpg"})
    pm.train_mark_duplicate_removed(project_dir, "v1", ["X.png"])
    pm.train_clear_all(project_dir, "v1")

    # 老 manifest 文件 mtime 没变（PR-2 train_xxx 函数从不动它）
    assert pm.manifest_path(project_dir).stat().st_mtime_ns == legacy_mtime
    # 老 manifest 内容也没变
    legacy = _read(pm.manifest_path(project_dir))
    assert legacy == {"images": {"legacy.png": {"origin": "legacy.jpg"}}}
