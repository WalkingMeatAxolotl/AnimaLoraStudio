"""A4 — reg_dedup 模块单测：scan_for_dedup + purge_paths。

scan 用真的 PIL 图触发 duplicates 算法；purge_paths 不做 traversal 校验，
全靠调用方约定（worker 自己生成的路径必合法）。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from studio.services.reg import builder as reg_builder, dedup as reg_dedup


def _write_meta(rdir: Path, actual: int, target: int) -> None:
    meta = reg_builder.RegMeta(
        generated_at=0.0,
        based_on_version="v1",
        api_source="gelbooru",
        target_count=target,
        actual_count=actual,
        source_tags=[],
        excluded_tags=[],
        blacklist_tags=[],
        failed_tags=[],
        train_tag_distribution={},
        auto_tagged=False,
    )
    reg_builder.write_meta(rdir, meta)


def test_scan_for_dedup_empty_dir_returns_empty(tmp_path: Path) -> None:
    assert reg_dedup.scan_for_dedup(tmp_path / "noexist") == []
    empty = tmp_path / "reg"
    empty.mkdir()
    assert reg_dedup.scan_for_dedup(empty) == []


def test_scan_for_dedup_distinct_images_no_groups(tmp_path: Path) -> None:
    rdir = tmp_path / "reg" / "5_concept"
    rdir.mkdir(parents=True)
    Image.new("RGB", (64, 64), (255, 0, 0)).save(rdir / "1.png", "PNG")
    Image.new("RGB", (96, 128), (0, 255, 0)).save(rdir / "2.png", "PNG")
    assert reg_dedup.scan_for_dedup(rdir.parent) == []


def test_scan_for_dedup_identical_images_returns_n_minus_1(tmp_path: Path) -> None:
    """3 张同像素图 → 1 张保留、2 张待删。"""
    rdir = tmp_path / "reg" / "5_concept"
    rdir.mkdir(parents=True)
    img = Image.new("RGB", (128, 128), (255, 128, 64))
    img.save(rdir / "100.png", "PNG")
    img.save(rdir / "200.png", "PNG")
    img.save(rdir / "300.png", "PNG")

    to_delete = reg_dedup.scan_for_dedup(rdir.parent)
    assert len(to_delete) == 2
    # 留 1 张 + 删 2 张；不强保证留哪张，只要剩下的 stem 是合法的就行
    remaining = {p.stem for p in rdir.glob("*.png")} - {Path(r).stem for r in to_delete}
    assert len(remaining) == 1


def test_purge_paths_deletes_image_and_caption(tmp_path: Path) -> None:
    rdir = tmp_path / "reg"
    sub = rdir / "5_concept"
    sub.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(sub / "100.png", "PNG")
    (sub / "100.txt").write_text("a, b", encoding="utf-8")

    r = reg_dedup.purge_paths(rdir, ["5_concept/100.png"])
    assert r["count"] == 1
    assert r["deleted"] == ["5_concept/100.png"]
    assert not (sub / "100.png").exists()
    assert not (sub / "100.txt").exists()


def test_purge_paths_writes_deleted_ids_and_updates_meta(tmp_path: Path) -> None:
    rdir = tmp_path / "reg"
    sub = rdir / "5_concept"
    sub.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(sub / "42.png", "PNG")
    Image.new("RGB", (8, 8)).save(sub / "99.png", "PNG")
    _write_meta(rdir, actual=2, target=2)

    r = reg_dedup.purge_paths(
        rdir, ["5_concept/42.png", "5_concept/99.png"]
    )
    assert r["count"] == 2
    # .deleted_ids.json 含两个 stem
    assert reg_dedup.reg_builder.read_deleted_ids(rdir) == {"42", "99"}
    # meta.actual_count 递减到 0
    m = reg_builder.read_meta(rdir)
    assert m.actual_count == 0


def test_purge_paths_missing_file_silently_skipped(tmp_path: Path) -> None:
    rdir = tmp_path / "reg"
    rdir.mkdir()
    # 不存在 / 路径合法 → 不报错，count=0
    r = reg_dedup.purge_paths(rdir, ["5_concept/ghost.png"])
    assert r["count"] == 0
    assert r["deleted"] == []


def test_purge_paths_empty_list_noop(tmp_path: Path) -> None:
    rdir = tmp_path / "reg"
    rdir.mkdir()
    r = reg_dedup.purge_paths(rdir, [])
    assert r == {"deleted": [], "count": 0}
