"""落盘测试图 SQLite 索引 (`studio.services.generate_history_index`) 单测。

覆盖：
  - 首次 sync 全量建索引（single + xy），entry shape 与旧全量扫描一致
  - 增量：新文件被第二次 sync 收进；无变化时不重新解析（负缓存 / stat_key）
  - 变化：文件被覆盖（mtime/size 变）→ 重新解析出新 params
  - 删除：盘上消失 → entry 从列表消失；remove_entry 主动剔行
  - 没 anima_params 的 PNG：不入列表，且只解析一次（负缓存行）
  - limit + created_at desc 排序
  - 索引文件位置在 root 父目录的 .cache 下（测试 tmp 自动隔离）

端点层（response shape / v1→v2 迁移 / tmp 文件跳过等）由
tests/test_generate_save_metadata.py 的既有用例继续覆盖。
"""
from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path

from PIL import Image, PngImagePlugin

from studio.services import generate_history_index as ghi


def _png_with_params(params: dict | None) -> bytes:
    buf = BytesIO()
    img = Image.new("RGB", (8, 8), (0, 0, 0))
    if params is None:
        img.save(buf, format="PNG")
    else:
        info = PngImagePlugin.PngInfo()
        info.add_text("anima_params", json.dumps(params, ensure_ascii=False), zip=True)
        img.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _params(**overrides) -> dict:
    base = {"schema_version": 2, "mode": "single", "prompts": ["1girl"], "seed": 7}
    base.update(overrides)
    return base


def _write_single(root: Path, date: str, name: str, params: dict | None, *, mtime: float | None = None) -> Path:
    d = root / date / "single"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(_png_with_params(params))
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _write_xy(root: Path, date: str, folder: str, composite_params: dict, cells: list[tuple[int, int]]) -> Path:
    d = root / date / "xy" / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / ghi.XY_COMPOSITE_NAME).write_bytes(_png_with_params(composite_params))
    for xi, yi in cells:
        (d / f"cell x{xi} y{yi}.png").write_bytes(_png_with_params(None))
    return d


# ---------------------------------------------------------------- 基本 sync


def test_first_sync_builds_index_and_lists(tmp_path: Path) -> None:
    root = tmp_path / "test"
    _write_single(root, "2026-07-01", "single image 1.png", _params(seed=1))
    xy_params = _params(
        mode="xy",
        xy_draft={"x": {"axis": "cfg", "raw": "3, 5"}, "y": None},
    )
    _write_xy(root, "2026-07-02", "xy plot 1", xy_params, [(0, 0), (1, 0)])

    entries = ghi.sync_and_list(root, 100)
    assert len(entries) == 2
    by_mode = {e["mode"]: e for e in entries}
    single = by_mode["single"]
    assert single["filename"] == "single image 1.png"
    assert single["id"].startswith("disk:")
    assert single["params"]["seed"] == 1
    assert "single%20image%201.png" in single["image_url"]
    assert single["thumb_url"].endswith("?w=128")
    xy = by_mode["xy"]
    assert xy["folder"] == "xy plot 1"
    assert xy["xy_meta"]["x_values"] == ["3", "5"]
    assert [s["xy"]["xi"] for s in xy["xy_meta"]["samples"]] == [0, 1]
    # 索引文件在 root 父目录的 .cache 下
    assert ghi.index_db_path(root).is_file()
    assert ghi.index_db_path(root).parent == tmp_path / ".cache"


def test_incremental_sync_picks_up_new_files(tmp_path: Path) -> None:
    root = tmp_path / "test"
    _write_single(root, "2026-07-01", "single image 1.png", _params(seed=1))
    assert len(ghi.sync_and_list(root, 100)) == 1
    _write_single(root, "2026-07-01", "single image 2.png", _params(seed=2))
    entries = ghi.sync_and_list(root, 100)
    assert {e["filename"] for e in entries} == {"single image 1.png", "single image 2.png"}


def test_unchanged_files_are_not_reparsed(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "test"
    _write_single(root, "2026-07-01", "single image 1.png", _params(seed=1))
    ghi.sync_and_list(root, 100)

    def boom(path):
        raise AssertionError(f"should not reparse {path}")

    monkeypatch.setattr(ghi, "read_png_anima_params", boom)
    entries = ghi.sync_and_list(root, 100)
    assert len(entries) == 1  # 全部命中索引，零解析


def test_changed_file_is_reparsed(tmp_path: Path) -> None:
    root = tmp_path / "test"
    p = _write_single(root, "2026-07-01", "single image 1.png", _params(seed=1), mtime=1_000_000_000)
    assert ghi.sync_and_list(root, 100)[0]["params"]["seed"] == 1
    p.write_bytes(_png_with_params(_params(seed=99)))
    os.utime(p, (1_000_000_500, 1_000_000_500))
    assert ghi.sync_and_list(root, 100)[0]["params"]["seed"] == 99


def test_deleted_file_drops_out(tmp_path: Path) -> None:
    root = tmp_path / "test"
    p = _write_single(root, "2026-07-01", "single image 1.png", _params(seed=1))
    _write_single(root, "2026-07-01", "single image 2.png", _params(seed=2))
    assert len(ghi.sync_and_list(root, 100)) == 2
    p.unlink()
    entries = ghi.sync_and_list(root, 100)
    assert [e["filename"] for e in entries] == ["single image 2.png"]


def test_remove_entry_drops_row_without_touching_disk(tmp_path: Path) -> None:
    root = tmp_path / "test"
    _write_single(root, "2026-07-01", "single image 1.png", _params(seed=1))
    assert len(ghi.sync_and_list(root, 100)) == 1
    ghi.remove_entry(root, "2026-07-01", "single", "single image 1.png")
    # 行被剔了；但文件还在 → 下次 sync 会当新文件收回来（索引只是缓存）
    assert len(ghi.sync_and_list(root, 100)) == 1


# ---------------------------------------------------------------- 负缓存 / 排序


def test_png_without_params_excluded_and_cached(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "test"
    _write_single(root, "2026-07-01", "single image 1.png", None)
    assert ghi.sync_and_list(root, 100) == []

    def boom(path):
        raise AssertionError("negative-cache row should prevent reparse")

    monkeypatch.setattr(ghi, "read_png_anima_params", boom)
    assert ghi.sync_and_list(root, 100) == []


def test_sorted_desc_and_limit(tmp_path: Path) -> None:
    root = tmp_path / "test"
    _write_single(root, "2026-07-01", "single image 1.png", _params(seed=1), mtime=1_000_000_100)
    _write_single(root, "2026-07-01", "single image 2.png", _params(seed=2), mtime=1_000_000_300)
    _write_single(root, "2026-07-02", "single image 1.png", _params(seed=3), mtime=1_000_000_200)
    entries = ghi.sync_and_list(root, 100)
    assert [e["params"]["seed"] for e in entries] == [2, 3, 1]
    limited = ghi.sync_and_list(root, 2)
    assert [e["params"]["seed"] for e in limited] == [2, 3]


def test_tmp_and_non_png_files_ignored(tmp_path: Path) -> None:
    root = tmp_path / "test"
    d = root / "2026-07-01" / "single"
    d.mkdir(parents=True)
    (d / "x.tmp.png").write_bytes(_png_with_params(_params()))
    (d / "notes.txt").write_bytes(b"hi")
    assert ghi.sync_and_list(root, 100) == []


def test_xy_folder_without_composite_skipped(tmp_path: Path) -> None:
    root = tmp_path / "test"
    d = root / "2026-07-01" / "xy" / "xy plot 1"
    d.mkdir(parents=True)
    (d / "cell x0 y0.png").write_bytes(_png_with_params(None))
    assert ghi.sync_and_list(root, 100) == []
