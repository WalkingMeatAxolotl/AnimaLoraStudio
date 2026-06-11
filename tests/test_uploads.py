"""本地上传 service：accept_one / accept_many。"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from studio.services.dataset import uploads


def _png_bytes(size: tuple[int, int] = (4, 4), color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _jpg_bytes(size: tuple[int, int] = (4, 4), color: tuple[int, int, int] = (0, 128, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG", quality=90)
    return buf.getvalue()


def _rgba_png_bytes(size: tuple[int, int] = (4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, (10, 20, 30, 100)).save(buf, "PNG")
    return buf.getvalue()


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


def test_accepts_extended_image_formats(tmp_path: Path) -> None:
    """PP10：上传白名单与全链路 IMAGE_EXTS 对齐，webp/bmp/gif 也接受。"""
    for fname, payload in [
        ("a.webp", b"WEBP"),
        ("b.bmp", b"BMP"),
        ("c.gif", b"GIF"),
    ]:
        out = uploads.accept_one(fname, io.BytesIO(payload), tmp_path)
        assert out.added == [fname], f"{fname} 应被接受"
        assert (tmp_path / fname).read_bytes() == payload


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


# ---------------------------------------------------------------------------
# convert_to_png 模式：与 booru 下载共用的 gelbooru.convert_to_png 设置
# ---------------------------------------------------------------------------


def test_convert_jpg_renamed_to_png(tmp_path: Path) -> None:
    out = uploads.accept_one(
        "photo.jpg", io.BytesIO(_jpg_bytes()), tmp_path,
        convert_to_png=True,
    )
    assert out.added == ["photo.png"]
    assert (tmp_path / "photo.png").exists()
    # 已重编码为合法 PNG
    with Image.open(tmp_path / "photo.png") as im:
        assert im.format == "PNG"


def test_convert_same_stem_collision_gets_suffix(tmp_path: Path) -> None:
    """1.png + 1.jpg 同一次上传：第二张转完撞名 → 加 _1 后缀，避免 caption 共用。"""
    files = [
        ("1.png", io.BytesIO(_png_bytes(color=(0, 0, 0)))),
        ("1.jpg", io.BytesIO(_jpg_bytes(color=(255, 255, 255)))),
    ]
    out = uploads.accept_many(files, tmp_path, convert_to_png=True)
    assert sorted(out.added) == ["1.png", "1_1.png"]
    assert out.skipped == []
    assert (tmp_path / "1.png").exists()
    assert (tmp_path / "1_1.png").exists()


def test_convert_collision_in_zip_gets_suffix(tmp_path: Path) -> None:
    blob = _zip_bytes(
        {
            "a.png": _png_bytes(color=(10, 10, 10)),
            "a.jpg": _jpg_bytes(color=(200, 200, 200)),
        }
    )
    out = uploads.accept_one(
        "pack.zip", io.BytesIO(blob), tmp_path,
        convert_to_png=True,
    )
    assert sorted(out.added) == ["a.png", "a_1.png"]
    assert (tmp_path / "a.png").exists()
    assert (tmp_path / "a_1.png").exists()


def test_convert_collision_against_existing_file(tmp_path: Path) -> None:
    """目标目录已经有 a.png，新上传 a.jpg 在 convert 模式下落 a_1.png（不跳过）。"""
    (tmp_path / "a.png").write_bytes(_png_bytes())
    out = uploads.accept_one(
        "a.jpg", io.BytesIO(_jpg_bytes()), tmp_path,
        convert_to_png=True,
    )
    assert out.added == ["a_1.png"]
    assert (tmp_path / "a_1.png").exists()


def test_convert_corrupt_image_skipped(tmp_path: Path) -> None:
    out = uploads.accept_one(
        "broken.jpg", io.BytesIO(b"not-a-real-image"), tmp_path,
        convert_to_png=True,
    )
    assert out.added == []
    assert len(out.skipped) == 1
    assert "图片损坏" in out.skipped[0]["reason"]


def test_convert_remove_alpha_channel_flattens(tmp_path: Path) -> None:
    out = uploads.accept_one(
        "rgba.png", io.BytesIO(_rgba_png_bytes()), tmp_path,
        convert_to_png=True,
        remove_alpha_channel=True,
    )
    assert out.added == ["rgba.png"]
    with Image.open(tmp_path / "rgba.png") as im:
        assert im.mode == "RGB"  # alpha 已被白底压平


def test_convert_keeps_alpha_when_flag_off(tmp_path: Path) -> None:
    out = uploads.accept_one(
        "rgba.png", io.BytesIO(_rgba_png_bytes()), tmp_path,
        convert_to_png=True,
        remove_alpha_channel=False,
    )
    assert out.added == ["rgba.png"]
    with Image.open(tmp_path / "rgba.png") as im:
        assert im.mode == "RGBA"


def test_convert_off_preserves_raw_bytes(tmp_path: Path) -> None:
    """convert_to_png=False（默认）继续走原扩展名拷贝、目标已存在跳过。"""
    raw = b"\xff\xd8not-decoded"
    out = uploads.accept_one("photo.jpg", io.BytesIO(raw), tmp_path)
    assert out.added == ["photo.jpg"]
    assert (tmp_path / "photo.jpg").read_bytes() == raw
