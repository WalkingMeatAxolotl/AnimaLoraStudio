"""本地上传 service：accept_one / accept_many。"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from studio.services import uploads


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_accept_single_jpg(tmp_path: Path) -> None:
    out = uploads.accept_one("photo.jpg", io.BytesIO(b"\xff\xd8jpgdata"), tmp_path)
    assert out.added == ["photo.jpg"]
    assert out.skipped == []
    assert (tmp_path / "photo.jpg").read_bytes() == b"\xff\xd8jpgdata"


def test_accept_png_uppercase_ext(tmp_path: Path) -> None:
    out = uploads.accept_one("a.PNG", io.BytesIO(b"png"), tmp_path)
    assert out.added == ["a.PNG"]
    assert (tmp_path / "a.PNG").exists()


def test_reject_unsupported_format(tmp_path: Path) -> None:
    out = uploads.accept_one("note.txt", io.BytesIO(b"hi"), tmp_path)
    assert out.added == []
    assert len(out.skipped) == 1
    assert out.skipped[0]["name"] == "note.txt"
    assert "格式不支持" in out.skipped[0]["reason"]


def test_skip_existing_does_not_overwrite(tmp_path: Path) -> None:
    (tmp_path / "p.png").write_bytes(b"old")
    out = uploads.accept_one("p.png", io.BytesIO(b"new"), tmp_path)
    assert out.added == []
    assert out.skipped[0]["reason"] == "已存在，跳过"
    assert (tmp_path / "p.png").read_bytes() == b"old"


def test_zip_extracts_jpg_png(tmp_path: Path) -> None:
    blob = _zip_bytes(
        {
            "a.jpg": b"AA",
            "sub/b.png": b"BB",
            "ignored.txt": b"X",
        }
    )
    out = uploads.accept_one("pack.zip", io.BytesIO(blob), tmp_path)
    assert sorted(out.added) == ["a.jpg", "b.png"]
    # txt 被跳过；子目录被拍平
    assert any("ignored.txt" in s["name"] for s in out.skipped)
    assert (tmp_path / "a.jpg").read_bytes() == b"AA"
    assert (tmp_path / "b.png").read_bytes() == b"BB"
    # 不应该创建 sub/ 子目录
    assert not (tmp_path / "sub").exists()


def test_zip_skip_dup_in_zip(tmp_path: Path) -> None:
    (tmp_path / "x.png").write_bytes(b"existing")
    blob = _zip_bytes({"x.png": b"new"})
    out = uploads.accept_one("p.zip", io.BytesIO(blob), tmp_path)
    assert out.added == []
    assert out.skipped[0]["reason"] == "已存在，跳过"
    assert (tmp_path / "x.png").read_bytes() == b"existing"


def test_corrupt_zip_skipped(tmp_path: Path) -> None:
    out = uploads.accept_one(
        "broken.zip", io.BytesIO(b"not-a-real-zip"), tmp_path
    )
    assert out.added == []
    assert out.skipped[0]["reason"] == "zip 损坏"


def test_zip_path_traversal_flattened(tmp_path: Path) -> None:
    """zip 内包含 ../ 或绝对路径段时也只取 basename，不会跳出 dest_dir。"""
    blob = _zip_bytes(
        {
            "../escape.jpg": b"E",
            "/abs/p.png": b"P",
        }
    )
    out = uploads.accept_one("evil.zip", io.BytesIO(blob), tmp_path)
    assert sorted(out.added) == ["escape.jpg", "p.png"]
    assert (tmp_path / "escape.jpg").exists()
    assert (tmp_path / "p.png").exists()
    # 不应该写到 tmp_path 之外
    assert not (tmp_path.parent / "escape.jpg").exists()


def test_accept_many_aggregates(tmp_path: Path) -> None:
    files = [
        ("a.jpg", io.BytesIO(b"A")),
        ("b.txt", io.BytesIO(b"B")),
        ("c.png", io.BytesIO(b"C")),
    ]
    out = uploads.accept_many(files, tmp_path)
    assert sorted(out.added) == ["a.jpg", "c.png"]
    assert len(out.skipped) == 1
    assert out.skipped[0]["name"] == "b.txt"


def test_empty_filename_skipped(tmp_path: Path) -> None:
    out = uploads.accept_one("", io.BytesIO(b"x"), tmp_path)
    assert out.added == []
    assert out.skipped[0]["reason"] == "文件名为空"


def test_dest_dir_created(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "download"
    out = uploads.accept_one("x.jpg", io.BytesIO(b"x"), target)
    assert out.added == ["x.jpg"]
    assert (target / "x.jpg").exists()
