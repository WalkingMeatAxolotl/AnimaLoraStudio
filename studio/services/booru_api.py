"""Booru API 共用层（PP5）。

`downloader.py`（PP2 单 tag 批量拉）与 `reg_builder.py`（PP5 多 tag 贪心
迭代）需要同一套 HTTP 原语：搜索、抽 post 字段、拉图。这里把它们集中成
不依赖 DownloadOptions / RegBuildOptions 的纯函数，避免双轨。

约定：
- `search_posts` 失败抛 `requests.RequestException`（调用方负责 catch）；
  返回值永远是 list（空 list 表示「这页没有」）。
- `download_image` 失败抛 `RuntimeError`（包装 PIL / 网络错误）；
  成功返回最终落盘路径（含 PNG 重命名后的版本）。
- 不做 retry / sleep / cancel 检测 —— 那些属于上层循环职责。
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image


DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def default_base_url(api_source: str) -> str:
    return (
        "https://gelbooru.com"
        if api_source == "gelbooru"
        else "https://danbooru.donmai.us"
    )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def search_posts(
    api_source: str,
    tags_query: str,
    *,
    page: int = 1,
    limit: int = 100,
    user_id: str = "",
    api_key: str = "",
    username: str = "",
    base_url: Optional[str] = None,
    timeout: float = 30.0,
    session: Optional[requests.Session] = None,
) -> list[dict[str, Any]]:
    """通用 booru 搜索。

    `tags_query` 是已经拼好的搜索串（多 tag 用空格分隔，含 `-排除`）。
    返回 posts list（空 list = 这页空 / 到底了）。
    """
    sess = session or requests
    url_base = base_url or default_base_url(api_source)
    if api_source == "gelbooru":
        params: dict[str, Any] = {
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": "1",
            "tags": tags_query,
            "pid": page - 1,
            "limit": min(limit, 100),
        }
        if api_key and user_id:
            params["api_key"] = api_key
            params["user_id"] = user_id
        url = f"{url_base}/index.php"
        auth = None
    else:
        params = {
            "tags": tags_query,
            "page": page,
            "limit": min(limit, 200),
        }
        url = f"{url_base}/posts.json"
        auth = (username, api_key) if username and api_key else None

    resp = sess.get(url, params=params, auth=auth, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if api_source == "gelbooru":
        if isinstance(data, dict):
            posts = data.get("post")
            if isinstance(posts, dict):
                return [posts]
            if isinstance(posts, list):
                return posts
            if "@attributes" in data:
                return [data]
            return []
        if isinstance(data, list):
            return data
        return []
    return data if isinstance(data, list) else []


# ---------------------------------------------------------------------------
# post field extraction
# ---------------------------------------------------------------------------


def post_fields(
    post: dict[str, Any], api_source: str
) -> tuple[Optional[str], Optional[str], str, Optional[str]]:
    """统一抽 (post_id, file_url, file_ext, tags_str)。

    gelbooru 的字段可能在顶层也可能在 `@attributes` 里 —— 兼容两种。
    """
    if api_source == "gelbooru" and "@attributes" in post:
        attrs = post["@attributes"]
        return (
            str(attrs.get("id")) if attrs.get("id") is not None else None,
            attrs.get("file_url"),
            str(attrs.get("file_ext", "jpg")),
            attrs.get("tags"),
        )
    return (
        str(post.get("id")) if post.get("id") is not None else None,
        post.get("file_url"),
        str(post.get("file_ext", "jpg")),
        post.get("tag_string") or post.get("tags"),
    )


def post_dimensions(
    post: dict[str, Any], api_source: str
) -> tuple[Optional[int], Optional[int]]:
    """抽 (width, height)；gelbooru 也兼容 @attributes 嵌套。"""
    if api_source == "gelbooru" and "@attributes" in post:
        attrs = post["@attributes"]
        w = attrs.get("width")
        h = attrs.get("height")
    else:
        w = post.get("image_width") or post.get("width")
        h = post.get("image_height") or post.get("height")
    try:
        return (int(w) if w is not None else None, int(h) if h is not None else None)
    except (TypeError, ValueError):
        return (None, None)


def post_tag_list(post: dict[str, Any], api_source: str) -> list[str]:
    """抽帖子的 tag 列表（小写、空格→下划线、去重保序）。

    gelbooru：`tags` 字段空格分隔字符串。
    danbooru：`tag_string_general` / `_character` / `_copyright` / `_artist` 拼。
    """
    raw: list[str] = []
    if api_source == "gelbooru":
        if "@attributes" in post:
            tags_str = post["@attributes"].get("tags", "")
        else:
            tags_str = post.get("tags", "")
        if tags_str:
            raw = [t.strip() for t in str(tags_str).split() if t.strip()]
    else:
        for cat in (
            "tag_string_general",
            "tag_string_character",
            "tag_string_copyright",
            "tag_string_artist",
        ):
            v = post.get(cat)
            if v:
                raw.extend(str(v).split())
    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        norm = t.lower().replace(" ", "_")
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


# ---------------------------------------------------------------------------
# image download
# ---------------------------------------------------------------------------


def has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "P") or "transparency" in img.info


def flatten_alpha(img: Image.Image) -> Image.Image:
    """以白底贴掉透明通道。"""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def download_image(
    url: str,
    save_path: Path,
    *,
    convert_to_png: bool,
    remove_alpha_channel: bool,
    timeout: float = 60.0,
    referer: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> Path:
    """下载单图到 save_path（或 .png 重命名后版本）；失败抛 RuntimeError。

    返回最终落盘路径。
    """
    sess = session or requests
    headers = dict(DOWNLOAD_HEADERS)
    if referer:
        headers["Referer"] = referer
    resp = sess.get(url, headers=headers, timeout=timeout, stream=True)
    resp.raise_for_status()
    raw = resp.content
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as exc:
        raise RuntimeError(f"图片损坏或无法识别: {exc}") from exc

    final = save_path
    if convert_to_png and final.suffix.lower() != ".png":
        final = final.with_suffix(".png")
    if remove_alpha_channel and has_alpha(img):
        img = flatten_alpha(img)
    if final.suffix.lower() == ".png":
        out = (
            img.convert("RGBA")
            if has_alpha(img) and not remove_alpha_channel
            else img.convert("RGB")
        )
        out.save(final, "PNG", optimize=True)
    elif final.suffix.lower() in {".jpg", ".jpeg"}:
        img.convert("RGB").save(final, "JPEG", quality=95, optimize=True)
    elif final.suffix.lower() == ".webp":
        img.save(final, "WEBP", quality=95)
    else:
        final.write_bytes(raw)
    return final
