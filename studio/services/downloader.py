"""Gelbooru / Danbooru 下载库（pp2）。

由原 `danbooru_downloader.py` 库化而来：去掉 input() / json 配置文件，全部
参数走 `DownloadOptions`；进度通过 `on_progress(line)` 推回调用方（worker
转写到日志 + bus.publish）。

设计：
- `download(opts, dest_dir, on_progress, on_image_saved, cancel_event)` 阻塞
  式下载，返回成功保存的图片数。
- 速率限制：每图 0.5s 间隔（与原脚本一致）；分页之间额外 1s。
- 失败重试 3 次（指数退避 1s/2s/4s），timeout 60s。
- 取消：`cancel_event.is_set()` 在每图 / 每分页前检测，触发后立即返回当前
  已保存数量；不抛异常。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import requests
from PIL import Image

ProgressFn = Callable[[str], None]
ImageSavedFn = Callable[[Path], None]


@dataclass
class DownloadOptions:
    tag: str
    count: int = 20
    api_source: str = "gelbooru"  # "gelbooru" | "danbooru"
    save_tags: bool = False
    convert_to_png: bool = True
    remove_alpha_channel: bool = False
    skip_existing: bool = True
    # gelbooru 凭据
    user_id: str = ""
    # danbooru 凭据
    username: str = ""
    # 通用 api key（gelbooru / danbooru 都用 .api_key）
    api_key: str = ""
    # 全局排除 tag（搜索时自动追加 -tag）；来自 secrets.download.exclude_tags
    exclude_tags: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.exclude_tags is None:
            self.exclude_tags = []

    def base_url(self) -> str:
        return (
            "https://gelbooru.com"
            if self.api_source == "gelbooru"
            else "https://danbooru.donmai.us"
        )

    def effective_tag_query(self) -> str:
        """`tag` 后面拼上 -excluded（gelbooru / danbooru 语法一致）。"""
        parts = [self.tag.strip()]
        for ex in self.exclude_tags:
            ex = ex.strip().lstrip("-")
            if ex:
                parts.append(f"-{ex}")
        return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def _search_posts(
    opts: DownloadOptions,
    page: int,
    limit: int,
    *,
    timeout: float = 30.0,
    session: Optional[requests.Session] = None,
) -> Optional[list[dict[str, Any]]]:
    """返回当前 page 的 posts 列表；HTTP 失败返回 None；空列表表示到底了。"""
    sess = session or requests
    query = opts.effective_tag_query()
    if opts.api_source == "gelbooru":
        params: dict[str, Any] = {
            "page": "dapi",
            "s": "post",
            "q": "index",
            "json": "1",
            "tags": query,
            "pid": page - 1,
            "limit": min(limit, 100),
        }
        if opts.api_key and opts.user_id:
            params["api_key"] = opts.api_key
            params["user_id"] = opts.user_id
        url = f"{opts.base_url()}/index.php"
        auth = None
    else:
        params = {
            "tags": query,
            "page": page,
            "limit": min(limit, 200),
        }
        url = f"{opts.base_url()}/posts.json"
        auth = (
            (opts.username, opts.api_key)
            if opts.username and opts.api_key
            else None
        )

    resp = sess.get(url, params=params, auth=auth, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if opts.api_source == "gelbooru":
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


def _post_fields(
    post: dict[str, Any], api_source: str
) -> tuple[Optional[str], Optional[str], str, Optional[str]]:
    """统一抽取 (post_id, file_url, file_ext, tags_str)。"""
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


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "P") or "transparency" in img.info


def _flatten_alpha(img: Image.Image) -> Image.Image:
    """以白底贴掉透明通道。"""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def _download_one(
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
    headers = dict(_DOWNLOAD_HEADERS)
    if referer:
        headers["Referer"] = referer
    resp = sess.get(url, headers=headers, timeout=timeout, stream=True)
    resp.raise_for_status()
    raw = resp.content  # 一次性读，便于后面重新解码 + 校验
    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as exc:  # 损坏 / 不可识别
        raise RuntimeError(f"图片损坏或无法识别: {exc}") from exc

    final = save_path
    if convert_to_png and final.suffix.lower() != ".png":
        final = final.with_suffix(".png")
    if remove_alpha_channel and _has_alpha(img):
        img = _flatten_alpha(img)
    if final.suffix.lower() == ".png":
        out = img.convert("RGBA") if _has_alpha(img) and not remove_alpha_channel else img.convert("RGB")
        out.save(final, "PNG", optimize=True)
    elif final.suffix.lower() in {".jpg", ".jpeg"}:
        img.convert("RGB").save(final, "JPEG", quality=95, optimize=True)
    elif final.suffix.lower() == ".webp":
        img.save(final, "WEBP", quality=95)
    else:
        # 不识别的扩展名直接落原始字节
        final.write_bytes(raw)
    return final


# ---------------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------------


def download(
    opts: DownloadOptions,
    dest_dir: Path,
    *,
    on_progress: ProgressFn = print,
    on_image_saved: Optional[ImageSavedFn] = None,
    cancel_event: Optional[threading.Event] = None,
    session: Optional[requests.Session] = None,
    page_delay: float = 1.0,
    image_delay: float = 0.5,
    max_retries: int = 3,
) -> int:
    """阻塞式下载到 dest_dir。

    返回本次新增保存的图片数（不含 skip）。中断（cancel_event 触发）时
    返回当前已保存的数量，不抛错。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if not opts.tag.strip():
        raise ValueError("tag 不能为空")
    if opts.count <= 0:
        raise ValueError("count 必须 > 0")
    if opts.api_source == "gelbooru" and not (opts.user_id and opts.api_key):
        raise ValueError(
            "gelbooru 需要 user_id + api_key（去 Settings 配置 secrets.gelbooru）"
        )

    saved = 0
    skipped = 0
    failed = 0
    page = 1
    api_limit = 100 if opts.api_source == "gelbooru" else 200

    while saved < opts.count:
        if cancel_event and cancel_event.is_set():
            on_progress("[cancel] user requested stop")
            return saved
        on_progress(f"[page {page}] fetching ...")
        try:
            posts = _search_posts(
                opts, page=page, limit=api_limit, session=session
            )
        except requests.RequestException as exc:
            on_progress(f"[err] search failed: {exc}")
            return saved
        if posts is None:
            on_progress("[err] search returned None")
            return saved
        if not posts:
            on_progress("[done] no more posts (server returned empty page)")
            break

        page_valid = 0
        for post in posts:
            if saved >= opts.count:
                break
            if cancel_event and cancel_event.is_set():
                on_progress("[cancel] user requested stop")
                return saved
            post_id, file_url, file_ext, tags_str = _post_fields(
                post, opts.api_source
            )
            if not post_id or not file_url:
                continue
            page_valid += 1

            ext = "png" if opts.convert_to_png else file_ext
            target = dest_dir / f"{post_id}.{ext}"
            if opts.skip_existing and target.exists():
                skipped += 1
                on_progress(f"[skip] {target.name} already exists")
                continue

            ok = False
            for attempt in range(1, max_retries + 1):
                try:
                    final = _download_one(
                        file_url,
                        target,
                        convert_to_png=opts.convert_to_png,
                        remove_alpha_channel=opts.remove_alpha_channel,
                        referer=opts.base_url() + "/",
                        session=session,
                    )
                    if opts.save_tags and tags_str:
                        final.with_suffix(".booru.txt").write_text(
                            str(tags_str), encoding="utf-8"
                        )
                    if on_image_saved:
                        on_image_saved(final)
                    saved += 1
                    on_progress(
                        f"[{saved}/{opts.count}] saved {final.name}"
                    )
                    ok = True
                    break
                except requests.RequestException as exc:
                    backoff = 2 ** (attempt - 1)
                    on_progress(
                        f"[retry {attempt}/{max_retries}] {target.name}: {exc}"
                    )
                    if cancel_event and cancel_event.wait(backoff):
                        on_progress("[cancel] user requested stop")
                        return saved
                except Exception as exc:
                    on_progress(f"[err] {target.name}: {exc}")
                    break
            if not ok:
                failed += 1
            if cancel_event and cancel_event.wait(image_delay):
                on_progress("[cancel] user requested stop")
                return saved

        if len(posts) < api_limit:
            on_progress(
                f"[done] page returned {len(posts)} < limit {api_limit}, "
                "reached end"
            )
            break
        if page_valid == 0:
            on_progress("[done] no valid posts on this page; stopping")
            break
        page += 1
        if cancel_event and cancel_event.wait(page_delay):
            on_progress("[cancel] user requested stop")
            return saved

    on_progress(
        f"[summary] saved={saved} skipped={skipped} failed={failed}"
    )
    return saved


def estimate(opts: DownloadOptions) -> int:
    """轻量调用 API 估算 tag（含 exclude）命中量；失败返回 -1（未知）。"""
    query = opts.effective_tag_query()
    if opts.api_source == "gelbooru":
        try:
            params: dict[str, Any] = {
                "page": "dapi",
                "s": "post",
                "q": "index",
                "json": "1",
                "tags": query,
                "pid": 0,
                "limit": 1,
            }
            if opts.api_key and opts.user_id:
                params["api_key"] = opts.api_key
                params["user_id"] = opts.user_id
            r = requests.get(
                f"{opts.base_url()}/index.php", params=params, timeout=15
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "@attributes" in data:
                return int(data["@attributes"].get("count", -1))
        except Exception:
            return -1
        return -1
    try:
        r = requests.get(
            f"{opts.base_url()}/counts/posts.json",
            params={"tags": query},
            timeout=15,
        )
        r.raise_for_status()
        return int(r.json().get("counts", {}).get("posts", -1))
    except Exception:
        return -1
