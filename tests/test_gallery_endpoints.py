from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from studio import server
from studio.api.routers import gallery as gallery_router


def test_gallery_search_route_forwards_normalized_filters(monkeypatch) -> None:
    seen: dict = {}

    def fake_search(**kwargs):
        seen.update(kwargs)
        return {"items": [], "page": kwargs["page"], "page_size": 30, "has_more": False}

    monkeypatch.setattr(gallery_router.gallery, "search_gallery", fake_search)
    response = TestClient(server.app).get(
        "/api/gallery/search",
        params=[
            ("source", "gelbooru"),
            ("query", "cat ears"),
            ("rating", "general"),
            ("rating", "sensitive"),
            ("date_from", "2025-01-02"),
            ("date_to", "2025-02-03"),
            ("page", "4"),
        ],
    )

    assert response.status_code == 200
    assert response.json() == {"items": [], "page": 4, "page_size": 30, "has_more": False}
    assert seen["source"] == "gelbooru"
    assert seen["query"] == "cat ears"
    assert seen["ratings"] == ["general", "sensitive"]
    assert seen["date_from"].isoformat() == "2025-01-02"
    assert seen["date_to"].isoformat() == "2025-02-03"
    assert seen["page"] == 4


def test_gallery_search_route_defaults_to_general_rating(monkeypatch) -> None:
    seen: dict = {}

    def fake_search(**kwargs):
        seen.update(kwargs)
        return {"items": [], "page": kwargs["page"], "page_size": 30, "has_more": False}

    monkeypatch.setattr(gallery_router.gallery, "search_gallery", fake_search)
    response = TestClient(server.app).get("/api/gallery/search")

    assert response.status_code == 200
    assert seen["ratings"] == ["general"]


def test_gallery_search_route_rejects_invalid_source() -> None:
    response = TestClient(server.app).get(
        "/api/gallery/search",
        params={"source": "evil", "rating": "general", "page": 1},
    )
    assert response.status_code == 422


def test_gallery_image_route_uses_cached_file(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "thumb.png"
    Image.new("RGB", (2, 2)).save(image, "PNG")
    monkeypatch.setattr(
        gallery_router.gallery,
        "fetch_cached_image",
        lambda source, url, *, post_id: image,
    )

    response = TestClient(server.app).get(
        "/api/gallery/image",
        params={
            "source": "danbooru",
            "post_id": "42",
            "url": "https://cdn.donmai.us/a.png",
        },
    )

    assert response.status_code == 200
    assert response.content == image.read_bytes()
    assert response.headers["content-type"].startswith("image/png")
    assert response.headers["x-content-type-options"] == "nosniff"


def test_gallery_image_route_requires_post_id() -> None:
    response = TestClient(server.app).get(
        "/api/gallery/image",
        params={"source": "danbooru", "url": "https://cdn.donmai.us/a.png"},
    )

    assert response.status_code == 422


def test_gallery_tag_route_returns_prompt(monkeypatch) -> None:
    seen: dict = {}

    def fake_tag(**kwargs):
        seen.update(kwargs)
        return "1girl, blue_hair"

    monkeypatch.setattr(gallery_router.gallery, "tag_gallery_image", fake_tag)
    response = TestClient(server.app).post("/api/gallery/tag", json={
        "source": "danbooru",
        "post_id": "42",
        "image_url": "https://cdn.donmai.us/a.jpg",
        "tagger": "wd14",
    })

    assert response.status_code == 200
    assert response.json() == {"prompt": "1girl, blue_hair"}
    assert seen == {
        "source": "danbooru",
        "post_id": "42",
        "image_url": "https://cdn.donmai.us/a.jpg",
        "tagger_name": "wd14",
    }
