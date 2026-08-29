from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from studio.domain.errors import ValidationError
from studio.services.booru import gallery


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        danbooru=SimpleNamespace(username="dan-user", api_key="dan-key"),
        gelbooru=SimpleNamespace(user_id="gel-user", api_key="gel-key"),
        download=SimpleNamespace(
            parallel_workers=2,
            api_rate_per_sec=1.0,
            cdn_rate_per_sec=3.0,
        ),
    )


class SearchClient:
    def __init__(self, posts: list[dict]) -> None:
        self.posts = posts
        self.calls: list[tuple[str, str, dict]] = []

    def search_posts(self, source: str, query: str, **kwargs):
        self.calls.append((source, query, kwargs))
        return self.posts


class ImageResponse:
    status_code = 200

    def __init__(self, payload: bytes, *, content_type: str = "image/png") -> None:
        self.payload = payload
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(payload)),
        }
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        yield self.payload

    def close(self) -> None:
        self.closed = True


class ImageClient:
    def __init__(self, response: ImageResponse) -> None:
        self.response = response
        self.calls = 0

    @contextmanager
    def stream_get(self, *_args, **_kwargs):
        self.calls += 1
        try:
            yield self.response
        finally:
            self.response.close()


def _image_bytes(image_format: str = "PNG") -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 6), (20, 40, 60)).save(buf, image_format)
    return buf.getvalue()


def test_build_search_query_maps_source_specific_date_filters() -> None:
    assert gallery.build_search_query(
        "1girl blue_hair", "danbooru", "general",
        date(2025, 1, 2), date(2025, 2, 3),
    ) == "1girl blue_hair rating:general date:2025-01-02..2025-02-03"
    assert gallery.build_search_query(
        "cat", "gelbooru", "questionable",
        date(2024, 4, 5), date(2024, 5, 6),
    ) == "cat rating:questionable date:>=2024-04-05 date:<=2024-05-06"


def test_build_search_query_rejects_inverted_dates() -> None:
    with pytest.raises(ValidationError) as exc:
        gallery.build_search_query(
            "", "danbooru", "general", date(2025, 2, 1), date(2025, 1, 1),
        )
    assert exc.value.code == "gallery.date_range_invalid"


def test_search_gallery_uses_credentials_and_normalizes(monkeypatch) -> None:
    monkeypatch.setattr(gallery.secrets, "load", _settings)
    client = SearchClient([{
        "id": 42,
        "image_width": 1200,
        "image_height": 1800,
        "preview_file_url": "https://cdn.donmai.us/preview.jpg",
        "large_file_url": "https://cdn.donmai.us/sample.jpg",
        "file_url": "https://cdn.donmai.us/original.jpg",
        "tag_string_general": "1girl blue_hair",
        "tag_string_character": "alice",
    }])

    result = gallery.search_gallery(
        source="danbooru", query="1girl", rating="general",
        date_from=None, date_to=None, page=3, client=client,
    )

    assert result["page"] == 3
    assert result["has_more"] is False
    assert result["items"][0] == {
        "source": "danbooru",
        "post_id": "42",
        "width": 1200,
        "height": 1800,
        "tags": ["1girl", "blue_hair", "alice"],
        "thumbnail_url": "/api/gallery/image?source=danbooru&post_id=42&url=https%3A%2F%2Fcdn.donmai.us%2Fpreview.jpg",
        "image_url": "https://cdn.donmai.us/sample.jpg",
    }
    source, query, kwargs = client.calls[0]
    assert source == "danbooru"
    assert query == "1girl rating:general"
    assert kwargs["page"] == 3
    assert kwargs["limit"] == gallery.PAGE_SIZE
    assert kwargs["username"] == "dan-user"
    assert kwargs["api_key"] == "dan-key"
    assert "dan-key" not in repr(result)


def test_search_gallery_skips_non_image_posts(monkeypatch) -> None:
    monkeypatch.setattr(gallery.secrets, "load", _settings)
    client = SearchClient([{
        "id": 7,
        "image_width": 1280,
        "image_height": 720,
        "file_ext": "webm",
        "preview_file_url": "https://cdn.donmai.us/preview.jpg",
        "file_url": "https://cdn.donmai.us/video.webm",
    }])

    result = gallery.search_gallery(
        source="danbooru", query="", rating="general",
        date_from=None, date_to=None, page=1, client=client,
    )

    assert result["items"] == []


@pytest.mark.parametrize("url", [
    "https://cdn.donmai.us.evil.example/a.jpg",
    "https://user@cdn.donmai.us/a.jpg",
    "file:///etc/passwd",
    "https://cdn.donmai.us:444/a.jpg",
])
def test_validate_remote_url_rejects_ssrf_shapes(url: str) -> None:
    with pytest.raises(ValidationError) as exc:
        gallery.validate_remote_url("danbooru", url)
    assert exc.value.code == "gallery.image_url_invalid"


def test_validate_remote_url_accepts_official_cdn_hosts() -> None:
    assert gallery.validate_remote_url(
        "danbooru", "https://cdn.donmai.us/original/aa/example.jpg",
    )
    assert gallery.validate_remote_url(
        "gelbooru", "https://img3.gelbooru.com/images/aa/example.jpg",
    )


def test_fetch_cached_image_uses_post_id_across_changed_urls(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gallery, "THUMB_CACHE_DIR", tmp_path)
    monkeypatch.setattr(gallery.secrets, "load", _settings)
    monkeypatch.setattr(gallery, "_last_cleanup_at", 0.0)
    response = ImageResponse(_image_bytes())
    client = ImageClient(response)

    first = gallery.fetch_cached_image(
        "danbooru", "https://cdn.donmai.us/preview-v1.png",
        post_id="42", client=client,
    )
    second = gallery.fetch_cached_image(
        "danbooru", "https://cdn.donmai.us/preview-v2.png",
        post_id="42", client=client,
    )

    assert first == second
    assert first.name == "42.png"
    assert first.is_file()
    assert client.calls == 1
    assert response.closed is True
    assert not list(tmp_path.rglob("*.part-*"))


def test_fetch_cached_image_separates_source_and_post_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gallery, "THUMB_CACHE_DIR", tmp_path)
    monkeypatch.setattr(gallery.secrets, "load", _settings)

    paths = {
        gallery.fetch_cached_image(
            "danbooru", "https://cdn.donmai.us/a.png",
            post_id="42", client=ImageClient(ImageResponse(_image_bytes())),
        ),
        gallery.fetch_cached_image(
            "danbooru", "https://cdn.donmai.us/b.png",
            post_id="43", client=ImageClient(ImageResponse(_image_bytes())),
        ),
        gallery.fetch_cached_image(
            "gelbooru", "https://img3.gelbooru.com/a.png",
            post_id="42", client=ImageClient(ImageResponse(_image_bytes())),
        ),
    }

    assert len(paths) == 3
    assert {path.name for path in paths} == {"42.png", "43.png"}


def test_fetch_cached_image_rejects_invalid_post_id(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gallery, "THUMB_CACHE_DIR", tmp_path)

    with pytest.raises(ValidationError) as exc:
        gallery.fetch_cached_image(
            "danbooru", "https://cdn.donmai.us/a.png", post_id="../42",
        )

    assert exc.value.code == "gallery.post_id_invalid"


def test_fetch_cached_image_uses_verified_format_for_cache_suffix(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(gallery, "THUMB_CACHE_DIR", tmp_path)
    monkeypatch.setattr(gallery.secrets, "load", _settings)
    response = ImageResponse(_image_bytes("JPEG"), content_type="image/png")

    path = gallery.fetch_cached_image(
        "danbooru", "https://cdn.donmai.us/mislabeled", client=ImageClient(response),
    )

    assert path.suffix == ".jpg"


def test_fetch_cached_image_rejects_oversized_header(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(gallery, "THUMB_CACHE_DIR", tmp_path)
    monkeypatch.setattr(gallery.secrets, "load", _settings)
    response = ImageResponse(_image_bytes())
    response.headers["Content-Length"] = str(gallery.MAX_IMAGE_BYTES + 1)
    client = ImageClient(response)

    with pytest.raises(Exception) as exc:
        gallery.fetch_cached_image(
            "gelbooru", "https://img3.gelbooru.com/image.png", client=client,
        )
    assert getattr(exc.value, "code", "") == "gallery.image_too_large"
    assert not list(tmp_path.rglob("*.part-*"))


def test_tag_gallery_image_prefers_caption(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gallery.secrets, "load", _settings)
    image = tmp_path / "image.png"
    image.write_bytes(_image_bytes())
    monkeypatch.setattr(gallery, "fetch_cached_image", lambda *_a, **_k: image)

    class Tagger:
        def is_available(self):
            return True, "ok"

        def prepare(self):
            return None

        def tag(self, paths):
            assert paths == [image]
            yield {"caption": "a composed caption", "tags": ["ignored"]}

    monkeypatch.setattr(gallery, "get_tagger", lambda name: Tagger())
    assert gallery.tag_gallery_image(
        source="danbooru",
        post_id="42",
        image_url="https://cdn.donmai.us/sample.png",
        tagger_name="llm",
    ) == "a composed caption"


def test_tag_gallery_image_falls_back_to_tags(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gallery.secrets, "load", _settings)
    image = tmp_path / "image.png"
    image.write_bytes(_image_bytes())
    monkeypatch.setattr(gallery, "fetch_cached_image", lambda *_a, **_k: image)

    class Tagger:
        def is_available(self):
            return True, "ok"

        def prepare(self):
            return None

        def tag(self, _paths):
            yield {"tags": ["1girl", "blue_hair"]}

    monkeypatch.setattr(gallery, "get_tagger", lambda name: Tagger())
    assert gallery.tag_gallery_image(
        source="gelbooru",
        post_id="43",
        image_url="https://img3.gelbooru.com/sample.png",
        tagger_name="wd14",
    ) == "1girl, blue_hair"
