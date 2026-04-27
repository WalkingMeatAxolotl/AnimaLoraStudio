"""PP2 — downloader 库化版本：mock requests，验证下载循环 + 落盘 + 取消。"""
from __future__ import annotations

import io
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from studio.services import downloader


def _png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    img = Image.new("RGB", (8, 8), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class FakeResponse:
    def __init__(self, *, json_data=None, content: bytes = b"", status: int = 200):
        self._json = json_data
        self.content = content
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, search_pages: list[list[dict]], image_bytes: bytes):
        self._pages = list(search_pages)
        self._image = image_bytes
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, auth=None, timeout=None, headers=None, stream=False):
        self.calls.append((url, dict(params or {})))
        if "/index.php" in url or "/posts.json" in url:
            page = self._pages.pop(0) if self._pages else []
            return FakeResponse(json_data={"post": page} if page is not None else {})
        # 图片下载
        return FakeResponse(content=self._image)


def _opts(tag="x", count=3, src="gelbooru") -> downloader.DownloadOptions:
    return downloader.DownloadOptions(
        tag=tag,
        count=count,
        api_source=src,
        user_id="u",
        api_key="k",
        convert_to_png=False,
        remove_alpha_channel=False,
    )


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_download_rejects_missing_credentials(tmp_path: Path) -> None:
    opts = downloader.DownloadOptions(tag="x", count=1)  # 默认 gelbooru，无 user/key
    with pytest.raises(ValueError, match="user_id"):
        downloader.download(opts, tmp_path)


def test_download_rejects_empty_tag(tmp_path: Path) -> None:
    opts = downloader.DownloadOptions(
        tag="   ", count=1, user_id="u", api_key="k"
    )
    with pytest.raises(ValueError, match="tag"):
        downloader.download(opts, tmp_path)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_download_saves_files_with_post_id_names(tmp_path: Path) -> None:
    posts = [
        {"@attributes": {"id": 11, "file_url": "http://x/11.jpg", "file_ext": "jpg"}},
        {"@attributes": {"id": 22, "file_url": "http://x/22.jpg", "file_ext": "jpg"}},
        {"@attributes": {"id": 33, "file_url": "http://x/33.jpg", "file_ext": "jpg"}},
    ]
    sess = FakeSession([posts], _png_bytes())
    n = downloader.download(
        _opts(count=3),
        tmp_path,
        on_progress=lambda _: None,
        session=sess,
        page_delay=0,
        image_delay=0,
    )
    assert n == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "11.jpg", "22.jpg", "33.jpg",
    ]


def test_download_skips_existing(tmp_path: Path) -> None:
    (tmp_path / "11.jpg").write_bytes(b"already there")
    posts = [
        {"@attributes": {"id": 11, "file_url": "http://x/11.jpg", "file_ext": "jpg"}},
        {"@attributes": {"id": 22, "file_url": "http://x/22.jpg", "file_ext": "jpg"}},
    ]
    sess = FakeSession([posts], _png_bytes())
    n = downloader.download(
        _opts(count=2),
        tmp_path,
        on_progress=lambda _: None,
        session=sess,
        page_delay=0,
        image_delay=0,
    )
    # 11 跳过，22 新增
    assert n == 1
    assert (tmp_path / "11.jpg").read_bytes() == b"already there"
    assert (tmp_path / "22.jpg").exists()


def test_download_stops_when_count_reached(tmp_path: Path) -> None:
    posts = [
        {"@attributes": {"id": str(i), "file_url": f"http://x/{i}.jpg", "file_ext": "jpg"}}
        for i in range(10)
    ]
    sess = FakeSession([posts], _png_bytes())
    n = downloader.download(
        _opts(count=2),
        tmp_path,
        on_progress=lambda _: None,
        session=sess,
        page_delay=0,
        image_delay=0,
    )
    assert n == 2
    assert len(list(tmp_path.iterdir())) == 2


def test_download_stops_when_page_returns_below_limit(tmp_path: Path) -> None:
    """gelbooru 单页 limit=100；返回少于 limit → 末页。"""
    posts = [
        {"@attributes": {"id": str(i), "file_url": f"http://x/{i}.jpg", "file_ext": "jpg"}}
        for i in range(3)
    ]
    sess = FakeSession([posts], _png_bytes())
    n = downloader.download(
        _opts(count=999),
        tmp_path,
        on_progress=lambda _: None,
        session=sess,
        page_delay=0,
        image_delay=0,
    )
    assert n == 3


def test_download_writes_booru_tags_when_save_tags(tmp_path: Path) -> None:
    posts = [
        {"@attributes": {
            "id": 11, "file_url": "http://x/11.jpg", "file_ext": "jpg",
            "tags": "1girl solo",
        }},
    ]
    sess = FakeSession([posts], _png_bytes())
    opts = _opts(count=1)
    opts.save_tags = True
    downloader.download(
        opts, tmp_path, on_progress=lambda _: None,
        session=sess, page_delay=0, image_delay=0,
    )
    assert (tmp_path / "11.booru.txt").read_text(encoding="utf-8") == "1girl solo"


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


def test_cancel_stops_mid_download(tmp_path: Path) -> None:
    posts = [
        {"@attributes": {"id": str(i), "file_url": f"http://x/{i}.jpg", "file_ext": "jpg"}}
        for i in range(10)
    ]
    sess = FakeSession([posts], _png_bytes())
    cancel = threading.Event()

    saved_calls: list[Path] = []
    def on_saved(p: Path) -> None:
        saved_calls.append(p)
        if len(saved_calls) >= 2:
            cancel.set()

    n = downloader.download(
        _opts(count=10),
        tmp_path,
        on_progress=lambda _: None,
        on_image_saved=on_saved,
        cancel_event=cancel,
        session=sess,
        page_delay=0,
        image_delay=0,
    )
    assert n == 2  # 触发 cancel 后立即返回


# ---------------------------------------------------------------------------
# convert_to_png
# ---------------------------------------------------------------------------


def test_convert_to_png_renames_to_png(tmp_path: Path) -> None:
    posts = [
        {"@attributes": {"id": 11, "file_url": "http://x/11.jpg", "file_ext": "jpg"}},
    ]
    sess = FakeSession([posts], _png_bytes())
    opts = _opts(count=1)
    opts.convert_to_png = True
    downloader.download(
        opts, tmp_path, on_progress=lambda _: None,
        session=sess, page_delay=0, image_delay=0,
    )
    assert (tmp_path / "11.png").exists()
    assert not (tmp_path / "11.jpg").exists()
