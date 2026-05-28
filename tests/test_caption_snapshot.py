"""PP4 — caption_snapshot 服务（zip 备份 / 列出 / 还原 / 删除）。"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from studio.services.tagging import caption_snapshot


def _seed_train(version_dir: Path, content: dict[str, dict[str, str]]) -> None:
    """content = {folder: {filename(无扩展): caption_text}} -> 写 .png + .txt。"""
    train = version_dir / "train"
    for folder, files in content.items():
        d = train / folder
        d.mkdir(parents=True, exist_ok=True)
        for stem, tags in files.items():
            (d / f"{stem}.png").write_bytes(b"x")
            (d / f"{stem}.txt").write_text(tags, encoding="utf-8")


def test_create_lists_meta(tmp_path: Path) -> None:
    _seed_train(tmp_path, {"1_data": {"a": "x, y", "b": "z"}, "5_face": {"c": "w"}})
    meta = caption_snapshot.create_snapshot(tmp_path)
    assert meta["file_count"] == 3
    assert meta["size"] > 0
    items = caption_snapshot.list_snapshots(tmp_path)
    assert len(items) == 1
    assert items[0]["id"] == meta["id"]


def test_create_empty_train(tmp_path: Path) -> None:
    """train 目录不存在也允许（生成空 zip）。"""
    meta = caption_snapshot.create_snapshot(tmp_path)
    assert meta["file_count"] == 0


def test_restore_overwrites_current(tmp_path: Path) -> None:
    _seed_train(tmp_path, {"1_data": {"a": "old1", "b": "old2"}})
    snap = caption_snapshot.create_snapshot(tmp_path)
    # 改 caption + 加新文件
    (tmp_path / "train" / "1_data" / "a.txt").write_text("CHANGED", encoding="utf-8")
    (tmp_path / "train" / "1_data" / "extra.txt").write_text("LEAK", encoding="utf-8")

    r = caption_snapshot.restore_snapshot(tmp_path, snap["id"])
    assert r["written"] == 2
    # extra.txt 是 caption 后缀（.txt），还原应该删掉它
    assert not (tmp_path / "train" / "1_data" / "extra.txt").exists()
    assert (tmp_path / "train" / "1_data" / "a.txt").read_text(encoding="utf-8") == "old1"
    assert (tmp_path / "train" / "1_data" / "b.txt").read_text(encoding="utf-8") == "old2"
    # 图片不动
    assert (tmp_path / "train" / "1_data" / "a.png").exists()


def test_restore_does_not_touch_images(tmp_path: Path) -> None:
    _seed_train(tmp_path, {"1_data": {"a": "x"}})
    snap = caption_snapshot.create_snapshot(tmp_path)
    img = tmp_path / "train" / "1_data" / "a.png"
    img.write_bytes(b"NEWIMG")  # 模拟图片被改
    caption_snapshot.restore_snapshot(tmp_path, snap["id"])
    assert img.read_bytes() == b"NEWIMG"


def test_restore_missing_id(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        caption_snapshot.restore_snapshot(tmp_path, "9999999999")


def test_restore_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(caption_snapshot.SnapshotError):
        caption_snapshot.restore_snapshot(tmp_path, "../evil")


def test_restore_skips_unsafe_zip_entries(tmp_path: Path) -> None:
    # 手工造一个含 path-traversal 的 zip
    out_dir = caption_snapshot.snapshot_root(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    bad = out_dir / "1234567890.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../escape.txt", "haha")
        z.writestr("/abs.txt", "haha")
        z.writestr("nested/sub/deep.txt", "haha")
        z.writestr("ok/safe.txt", "fine")
    r = caption_snapshot.restore_snapshot(tmp_path, "1234567890")
    assert r["written"] == 1
    assert (tmp_path / "train" / "ok" / "safe.txt").read_text(encoding="utf-8") == "fine"
    assert "../escape.txt" in r["skipped"]
    assert "/abs.txt" in r["skipped"]
    assert "nested/sub/deep.txt" in r["skipped"]


def test_delete(tmp_path: Path) -> None:
    _seed_train(tmp_path, {"1_data": {"a": "x"}})
    snap = caption_snapshot.create_snapshot(tmp_path)
    caption_snapshot.delete_snapshot(tmp_path, snap["id"])
    assert caption_snapshot.list_snapshots(tmp_path) == []
