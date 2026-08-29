"""测试页远程 Booru 画廊：查询归一化、安全图片缓存与单图打标。"""
from __future__ import annotations

import atexit
import hashlib
import logging
import os
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode, urljoin, urlparse

import requests
from PIL import Image, UnidentifiedImageError

from ...domain.errors import DomainError, ValidationError
from ...infrastructure import secrets
from ...infrastructure.paths import THUMB_CACHE_DIR
from ..tagging.base import get_tagger
from . import api as booru_api
from .pool import BooruClient, BooruPoolConfig

logger = logging.getLogger(__name__)

GallerySource = Literal["danbooru", "gelbooru"]
GalleryRating = Literal["general", "sensitive", "questionable", "explicit"]
GalleryTagger = Literal["wd14", "cltagger", "llm"]

PAGE_SIZE = 30
CACHE_TTL_SECONDS = 14 * 24 * 60 * 60
CACHE_MAX_BYTES = 512 * 1024 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGE_PIXELS = 80_000_000
_ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
_ALLOWED_IMAGE_FORMATS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "GIF": ".gif",
}
_ALLOWED_HOST_SUFFIXES: dict[str, tuple[str, ...]] = {
    "danbooru": ("donmai.us",),
    "gelbooru": ("gelbooru.com",),
}
_RATING_VALUES = {"general", "sensitive", "questionable", "explicit"}
_SOURCE_VALUES = {"danbooru", "gelbooru"}
_TAGGER_VALUES = {"wd14", "cltagger", "llm"}

_client_lock = threading.Lock()
_clients: dict[tuple[int, float, float], BooruClient] = {}
_cache_locks = [threading.Lock() for _ in range(32)]
_cleanup_lock = threading.Lock()
_last_cleanup_at = 0.0
_tag_lock = threading.Lock()


def _close_clients() -> None:
    for client in list(_clients.values()):
        client.close()


atexit.register(_close_clients)


def _shared_client() -> BooruClient:
    """按全局限流设置复用 client；同一配置跨 HTTP 请求共享 token bucket。"""
    cfg = secrets.load().download
    key = (cfg.parallel_workers, cfg.api_rate_per_sec, cfg.cdn_rate_per_sec)
    with _client_lock:
        client = _clients.get(key)
        if client is None:
            client = BooruClient(BooruPoolConfig(
                parallel_workers=cfg.parallel_workers,
                api_rate_per_sec=cfg.api_rate_per_sec,
                cdn_rate_per_sec=cfg.cdn_rate_per_sec,
            ))
            _clients[key] = client
        return client


def build_search_query(
    query: str,
    source: GallerySource,
    rating: GalleryRating,
    date_from: date | None,
    date_to: date | None,
) -> str:
    """把统一筛选条件翻译成站点 metatag，保持前端 page 从 1 开始。"""
    if source not in _SOURCE_VALUES:
        raise ValidationError(
            "Unsupported gallery source", code="gallery.source_invalid",
            details={"source": source}, http_status=400,
        )
    if rating not in _RATING_VALUES:
        raise ValidationError(
            "Unsupported gallery rating", code="gallery.rating_invalid",
            details={"rating": rating}, http_status=400,
        )
    if date_from and date_to and date_from > date_to:
        raise ValidationError(
            "Start date must not be after end date",
            code="gallery.date_range_invalid",
            details={"field": "date_from"}, http_status=400,
        )

    parts = [part for part in query.strip().split() if part]
    parts.append(f"rating:{rating}")
    if date_from or date_to:
        if source == "danbooru":
            if date_from and date_to:
                parts.append(f"date:{date_from.isoformat()}..{date_to.isoformat()}")
            elif date_from:
                parts.append(f"date:>={date_from.isoformat()}")
            else:
                parts.append(f"date:<={date_to.isoformat()}")
        else:
            if date_from:
                parts.append(f"date:>={date_from.isoformat()}")
            if date_to:
                parts.append(f"date:<={date_to.isoformat()}")
    return " ".join(parts)


def _credentials(source: GallerySource) -> dict[str, str]:
    cfg = secrets.load()
    if source == "danbooru":
        if not (cfg.danbooru.username and cfg.danbooru.api_key):
            raise ValidationError(
                "Danbooru credentials are not configured",
                code="gallery.credentials_missing",
                details={"source": source}, http_status=400,
            )
        return {"username": cfg.danbooru.username, "api_key": cfg.danbooru.api_key}
    if not (cfg.gelbooru.user_id and cfg.gelbooru.api_key):
        raise ValidationError(
            "Gelbooru credentials are not configured",
            code="gallery.credentials_missing",
            details={"source": source}, http_status=400,
        )
    return {"user_id": cfg.gelbooru.user_id, "api_key": cfg.gelbooru.api_key}


def _post_value(post: dict[str, Any], source: GallerySource, *names: str) -> Any:
    values = post.get("@attributes", {}) if source == "gelbooru" else post
    if not isinstance(values, dict):
        return None
    for name in names:
        value = values.get(name)
        if value not in (None, ""):
            return value
    return None


def _valid_remote_url(source: GallerySource, raw_url: Any) -> str | None:
    if not isinstance(raw_url, str) or not raw_url.strip():
        return None
    try:
        validate_remote_url(source, raw_url)
    except ValidationError:
        return None
    return raw_url


def _normalize_post(post: dict[str, Any], source: GallerySource) -> dict[str, Any] | None:
    post_id, file_url, file_ext, _tags_str = booru_api.post_fields(post, source)
    if file_ext.lower() not in {"jpg", "jpeg", "png", "webp", "gif"}:
        return None
    width, height = booru_api.post_dimensions(post, source)
    if not post_id or not post_id.isdigit() or not width or not height or width <= 0 or height <= 0:
        return None

    if source == "danbooru":
        thumb_raw = _post_value(post, source, "preview_file_url", "large_file_url", "file_url")
        tag_raw = _post_value(post, source, "large_file_url", "file_url", "preview_file_url")
    else:
        thumb_raw = _post_value(post, source, "preview_url", "sample_url", "file_url")
        tag_raw = _post_value(post, source, "sample_url", "file_url", "preview_url")
    thumb_url = _valid_remote_url(source, thumb_raw or file_url)
    tag_url = _valid_remote_url(source, tag_raw or file_url)
    if not thumb_url or not tag_url:
        return None

    proxy_query = urlencode({"source": source, "post_id": post_id, "url": thumb_url})
    return {
        "source": source,
        "post_id": post_id,
        "width": width,
        "height": height,
        "tags": booru_api.post_tag_list(post, source)[:200],
        "thumbnail_url": f"/api/gallery/image?{proxy_query}",
        "image_url": tag_url,
    }


def search_gallery(
    *,
    source: GallerySource,
    query: str,
    rating: GalleryRating,
    date_from: date | None,
    date_to: date | None,
    page: int,
    client: BooruClient | None = None,
) -> dict[str, Any]:
    if page < 1:
        raise ValidationError(
            "Page must be at least 1", code="gallery.page_invalid",
            details={"page": page}, http_status=400,
        )
    tags_query = build_search_query(query, source, rating, date_from, date_to)
    creds = _credentials(source)
    try:
        posts = (client or _shared_client()).search_posts(
            source, tags_query, page=page, limit=PAGE_SIZE, **creds,
        )
    except requests.RequestException as exc:
        logger.warning("gallery search failed: source=%s page=%d", source, page, exc_info=True)
        raise DomainError(
            "Gallery provider request failed", code="gallery.upstream_failed",
            details={"source": source}, http_status=502,
        ) from exc

    items = [item for post in posts if (item := _normalize_post(post, source)) is not None]
    return {
        "items": items,
        "page": page,
        "page_size": PAGE_SIZE,
        "has_more": len(posts) >= PAGE_SIZE,
    }


def validate_remote_url(source: GallerySource, raw_url: str) -> str:
    """只允许来源对应的官方域名；点边界匹配，拒绝 userinfo 和异常端口。"""
    if source not in _SOURCE_VALUES:
        raise ValidationError(
            "Unsupported gallery source", code="gallery.source_invalid",
            details={"source": source}, http_status=400,
        )
    try:
        parsed = urlparse(raw_url)
        port = parsed.port
    except ValueError as exc:
        raise ValidationError(
            "Invalid gallery image URL", code="gallery.image_url_invalid", http_status=400,
        ) from exc
    host = (parsed.hostname or "").lower().rstrip(".")
    suffixes = _ALLOWED_HOST_SUFFIXES[source]
    host_allowed = any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes)
    default_port = 443 if parsed.scheme == "https" else 80
    if (
        parsed.scheme not in {"http", "https"}
        or not host_allowed
        or parsed.username is not None
        or parsed.password is not None
        or (port is not None and port != default_port)
    ):
        raise ValidationError(
            "Invalid gallery image URL", code="gallery.image_url_invalid",
            details={"source": source}, http_status=400,
        )
    return raw_url


def _cache_root(kind: str, source: GallerySource) -> Path:
    return THUMB_CACHE_DIR / "gallery" / kind / source


def _cached_path(root: Path, key: str) -> Path | None:
    now = time.time()
    for suffix in _ALLOWED_CONTENT_TYPES.values():
        candidate = root / f"{key}{suffix}"
        try:
            stat = candidate.stat()
        except FileNotFoundError:
            continue
        if stat.st_size <= 0 or now - stat.st_mtime > CACHE_TTL_SECONDS:
            candidate.unlink(missing_ok=True)
            continue
        try:
            candidate.touch()
        except OSError:
            pass
        return candidate
    return None


def _cleanup_cache_if_needed() -> None:
    global _last_cleanup_at
    now = time.monotonic()
    with _cleanup_lock:
        if now - _last_cleanup_at < 60:
            return
        _last_cleanup_at = now
        root = THUMB_CACHE_DIR / "gallery"
        if not root.exists():
            return
        files: list[tuple[float, int, Path]] = []
        total = 0
        for path in root.rglob("*"):
            try:
                if path.is_file() and ".part-" not in path.name:
                    stat = path.stat()
                    total += stat.st_size
                    files.append((stat.st_mtime, stat.st_size, path))
            except OSError:
                continue
        if total <= CACHE_MAX_BYTES:
            return
        target = int(CACHE_MAX_BYTES * 0.9)
        for _mtime, size, path in sorted(files):
            try:
                path.unlink()
                total -= size
            except OSError:
                continue
            if total <= target:
                break


def _verify_image(path: Path) -> str:
    try:
        with Image.open(path) as image:
            suffix = _ALLOWED_IMAGE_FORMATS.get((image.format or "").upper())
            if suffix is None or image.width * image.height > MAX_IMAGE_PIXELS:
                raise DomainError(
                    "Gallery provider returned an invalid image",
                    code="gallery.image_invalid", http_status=502,
                )
            image.verify()
            return suffix
    except DomainError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise DomainError(
            "Gallery provider returned an invalid image",
            code="gallery.image_invalid", http_status=502,
        ) from exc


def _cache_key(raw_url: str, post_id: str | None) -> str:
    if post_id is None:
        # Compatibility fallback for internal callers without post metadata.
        return hashlib.sha256(raw_url.encode("utf-8")).hexdigest()
    if not post_id.isdigit() or len(post_id) > 32:
        raise ValidationError(
            "Invalid gallery post ID",
            code="gallery.post_id_invalid",
            details={"post_id": post_id}, http_status=400,
        )
    return post_id


def fetch_cached_image(
    source: GallerySource,
    raw_url: str,
    *,
    post_id: str | None = None,
    kind: Literal["thumb", "tag"] = "thumb",
    client: BooruClient | None = None,
) -> Path:
    current_url = validate_remote_url(source, raw_url)
    key = _cache_key(raw_url, post_id)
    root = _cache_root(kind, source)
    root.mkdir(parents=True, exist_ok=True)
    lock = _cache_locks[int(key[:2], 16) % len(_cache_locks)]
    with lock:
        cached = _cached_path(root, key)
        if cached is not None:
            return cached

        cfg = secrets.load()
        username = cfg.danbooru.username if source == "danbooru" else ""
        headers = booru_api._download_headers(username)  # same app identity as downloader
        headers["Accept"] = "image/avif,image/webp,image/png,image/jpeg,image/gif"
        booru_client = client or _shared_client()
        try:
            for _redirect in range(4):
                with booru_client.stream_get(
                    current_url,
                    headers=headers,
                    timeout=(10, 45),
                    stream=True,
                    allow_redirects=False,
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location", "")
                        if not location:
                            raise DomainError(
                                "Gallery image redirect is invalid",
                                code="gallery.image_fetch_failed", http_status=502,
                            )
                        current_url = validate_remote_url(source, urljoin(current_url, location))
                        continue

                    response.raise_for_status()
                    media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                    suffix = _ALLOWED_CONTENT_TYPES.get(media_type)
                    if suffix is None:
                        raise DomainError(
                            "Gallery provider returned an unsupported image type",
                            code="gallery.image_type_invalid", http_status=502,
                        )
                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        try:
                            if int(content_length) > MAX_IMAGE_BYTES:
                                raise DomainError(
                                    "Gallery image is too large", code="gallery.image_too_large",
                                    details={"max_bytes": MAX_IMAGE_BYTES}, http_status=413,
                                )
                        except ValueError:
                            pass

                    temp = root / f"{key}.part-{os.getpid()}-{threading.get_ident()}"
                    written = 0
                    try:
                        with temp.open("wb") as handle:
                            for chunk in response.iter_content(chunk_size=64 * 1024):
                                if not chunk:
                                    continue
                                written += len(chunk)
                                if written > MAX_IMAGE_BYTES:
                                    raise DomainError(
                                        "Gallery image is too large", code="gallery.image_too_large",
                                        details={"max_bytes": MAX_IMAGE_BYTES}, http_status=413,
                                    )
                                handle.write(chunk)
                        if written == 0:
                            raise DomainError(
                                "Gallery provider returned an empty image",
                                code="gallery.image_invalid", http_status=502,
                            )
                        # Trust the decoded format, not only the upstream MIME
                        # header; the suffix drives the proxy's nosniff type.
                        verified_suffix = _verify_image(temp)
                        target = root / f"{key}{verified_suffix}"
                        os.replace(temp, target)
                    finally:
                        temp.unlink(missing_ok=True)
                    _cleanup_cache_if_needed()
                    return target
            raise DomainError(
                "Gallery image redirected too many times",
                code="gallery.image_fetch_failed", http_status=502,
            )
        except DomainError:
            raise
        except requests.RequestException as exc:
            logger.warning("gallery image fetch failed: source=%s", source, exc_info=True)
            raise DomainError(
                "Gallery image request failed", code="gallery.image_fetch_failed",
                details={"source": source}, http_status=502,
            ) from exc


def tag_gallery_image(
    *,
    source: GallerySource,
    post_id: str,
    image_url: str,
    tagger_name: GalleryTagger,
    client: BooruClient | None = None,
) -> str:
    if tagger_name not in _TAGGER_VALUES:
        raise ValidationError(
            "Unsupported gallery tagger", code="gallery.tagger_invalid",
            details={"tagger": tagger_name}, http_status=400,
        )
    # 认证在下载前检查，确保画廊所有网络能力都受全局来源配置约束。
    _credentials(source)
    image_path = fetch_cached_image(
        source, image_url, post_id=post_id, kind="tag", client=client,
    )
    with _tag_lock:
        try:
            tagger = get_tagger(tagger_name)
            ok, _status = tagger.is_available()
            if not ok:
                raise ValidationError(
                    "Selected tagger is not available",
                    code="gallery.tagger_unavailable",
                    details={"tagger": tagger_name}, http_status=409,
                )
            tagger.prepare()
            result = next(iter(tagger.tag([image_path])), None)
        except DomainError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("gallery tagging failed: tagger=%s", tagger_name)
            raise DomainError(
                "Gallery image tagging failed", code="gallery.tag_failed",
                details={"tagger": tagger_name}, http_status=500,
            ) from exc
    if not result or result.get("error"):
        raise DomainError(
            "Gallery image tagging failed", code="gallery.tag_failed",
            details={"tagger": tagger_name}, http_status=500,
        )
    prompt = str(result.get("caption") or "").strip()
    if not prompt:
        prompt = ", ".join(str(tag).strip() for tag in result.get("tags", []) if str(tag).strip())
    if not prompt:
        raise DomainError(
            "Tagger returned no prompt", code="gallery.tag_empty",
            details={"tagger": tagger_name}, http_status=500,
        )
    return prompt
