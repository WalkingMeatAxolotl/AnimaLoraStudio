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
from pathlib import Path
from typing import Callable, Optional

import requests

from . import booru_api

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
        return booru_api.default_base_url(self.api_source)

    def effective_tag_query(self) -> str:
        """`tag` 后面拼上 -excluded（gelbooru / danbooru 语法一致）。"""
        parts = [self.tag.strip()]
        for ex in self.exclude_tags:
            ex = ex.strip().lstrip("-")
            if ex:
                parts.append(f"-{ex}")
        return " ".join(p for p in parts if p)


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
            posts = booru_api.search_posts(
                opts.api_source,
                opts.effective_tag_query(),
                page=page,
                limit=api_limit,
                user_id=opts.user_id,
                api_key=opts.api_key,
                username=opts.username,
                session=session,
            )
        except requests.RequestException as exc:
            on_progress(f"[err] search failed: {exc}")
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
            post_id, file_url, file_ext, tags_str = booru_api.post_fields(
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
                    final = booru_api.download_image(
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
