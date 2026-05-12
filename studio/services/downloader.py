"""Gelbooru / Danbooru 下载库（pp2 + pp9）。

由原 `danbooru_downloader.py` 库化而来：去掉 input() / json 配置文件，全部
参数走 `DownloadOptions`；进度通过 `on_progress(line)` 推回调用方（worker
转写到日志 + bus.publish）。

PP9 改造：
- 拉图阶段并发，走 `BooruClient`（双 token bucket：API 2 / CDN 5 req/s 默认）
- 删每图 0.5s 硬 sleep，速率改由 token bucket 控
- 保留 `page_delay`（每页之间 1s 礼貌等待）

设计：
- `download(opts, dest_dir, on_progress, on_image_saved, cancel_event, client)` 阻塞
  式下载，返回成功保存的图片数。
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

from . import booru_api, booru_pool

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
    client: Optional[booru_pool.BooruClient] = None,
    page_delay: float = 1.0,
    max_retries: int = 3,
) -> int:
    """阻塞式下载到 dest_dir。

    返回本次新增保存的图片数（不含 skip）。中断（cancel_event 触发）时
    返回当前已保存的数量，不抛错。

    PP9: 拉图阶段并发（默认 4 worker）；速率由 BooruClient 的 token bucket
    控（API 2 / CDN 5 req/s）。`session=` 仍接受外部 session（旧 test 用），
    内部用 `client.search_posts/download_image` 包装。
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
    if opts.api_source == "danbooru" and not (opts.username and opts.api_key):
        # hotfix: danbooru 挂 Cloudflare 后匿名 UA 已不可靠（即使我们带应用 UA，
        # CF 仍可能随时收紧）；强制绑定账户让 UA 带 (by username)，CF 拦匿名
        # 时不会一锅端，danbooru 端也按账户配速率限制（标准 2 req/s）。
        raise ValueError(
            "danbooru 需要 username + api_key（去 Settings 配置 secrets.danbooru）"
        )

    # 没传 client 就建一个临时的（按 secrets.download.* 调速）；session 优先
    owns_client = False
    if client is None:
        try:
            from .. import secrets as _secrets
            d = _secrets.load().download
            cfg = booru_pool.BooruPoolConfig(
                parallel_workers=d.parallel_workers,
                api_rate_per_sec=d.api_rate_per_sec,
                cdn_rate_per_sec=d.cdn_rate_per_sec,
            )
        except Exception:  # noqa: BLE001
            cfg = booru_pool.BooruPoolConfig()
        client = booru_pool.BooruClient(cfg, session=session)
        owns_client = True

    try:
        return _download_with_client(
            opts, dest_dir, client,
            on_progress=on_progress,
            on_image_saved=on_image_saved,
            cancel_event=cancel_event,
            page_delay=page_delay,
            max_retries=max_retries,
        )
    finally:
        if owns_client:
            client.close()


def _download_with_client(
    opts: DownloadOptions,
    dest_dir: Path,
    client: booru_pool.BooruClient,
    *,
    on_progress: ProgressFn,
    on_image_saved: Optional[ImageSavedFn],
    cancel_event: Optional[threading.Event],
    page_delay: float,
    max_retries: int,
) -> int:
    saved = 0
    skipped = 0
    failed = 0
    page = 1
    api_limit = 100 if opts.api_source == "gelbooru" else 200
    # 跨 worker 线程共享的「已 emit」计数器，仅供 _fetch_one 实时打 [N/count] 用。
    # 与主线程的 `saved` 在正常路径会一致；retry / 失败时 emitted 不增。
    emit_lock = threading.Lock()
    emit_state = {"n": 0}

    while saved < opts.count:
        if cancel_event and cancel_event.is_set():
            on_progress("[cancel] user requested stop")
            return saved
        on_progress(f"[page {page}] fetching ...")
        try:
            posts = client.search_posts(
                opts.api_source,
                opts.effective_tag_query(),
                page=page,
                limit=api_limit,
                user_id=opts.user_id,
                api_key=opts.api_key,
                username=opts.username,
            )
        except requests.RequestException as exc:
            on_progress(f"[err] search failed: {exc}")
            return saved
        if not posts:
            on_progress("[done] no more posts (server returned empty page)")
            break

        # 收集本页所有「待下载」候选；并发拉图
        candidates: list[tuple[str, str, str, Optional[str], Path]] = []
        page_valid = 0
        for post in posts:
            if saved + len(candidates) >= opts.count:
                break
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
            candidates.append((post_id, file_url, file_ext, tags_str, target))

        # 拉图函数（含失败重试）；每张独立调度到 worker pool
        referer = opts.base_url() + "/"

        def _fetch_one(item: tuple[str, str, str, Optional[str], Path]) -> Path:
            post_id, file_url, _file_ext, _tags, target = item
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("canceled")
                try:
                    final = client.download_image(
                        file_url,
                        target,
                        convert_to_png=opts.convert_to_png,
                        remove_alpha_channel=opts.remove_alpha_channel,
                        referer=referer,
                        username=opts.username,
                    )
                    # 实时进度（worker 线程内）：每张拉完立即 emit，让 LogTailer
                    # 把 SSE 推到前端。否则 parallel_download 会一直阻塞，整段
                    # 下载阶段没有日志，前端看上去像「卡住」。
                    with emit_lock:
                        emit_state["n"] += 1
                        n = emit_state["n"]
                    on_progress(f"[{n}/{opts.count}] saved {final.name}")
                    return final
                except requests.RequestException as exc:
                    last_exc = exc
                    backoff = 2 ** (attempt - 1)
                    on_progress(
                        f"[retry {attempt}/{max_retries}] {target.name}: {exc}"
                    )
                    if cancel_event and cancel_event.wait(backoff):
                        raise RuntimeError("canceled") from exc
            raise RuntimeError(f"max_retries exceeded: {last_exc}")

        results = client.parallel_download(
            candidates, _fetch_one, cancel_event=cancel_event
        )

        for (post_id, _url, _ext, tags_str, target), final, exc in results:
            if cancel_event and cancel_event.is_set():
                on_progress("[cancel] user requested stop")
                return saved
            if exc is not None:
                # 取消引发的 RuntimeError("canceled") 在 _fetch_one 里抛出，
                # 不当成「下载失败」report，否则用户取消会看到一堆 [err]。
                if isinstance(exc, RuntimeError) and "canceled" in str(exc):
                    continue
                on_progress(f"[err] {target.name}: {exc}")
                failed += 1
                continue
            assert isinstance(final, Path)
            if opts.save_tags and tags_str:
                final.with_suffix(".booru.txt").write_text(
                    str(tags_str), encoding="utf-8"
                )
            if on_image_saved:
                on_image_saved(final)
            saved += 1
            # 进度行已在 _fetch_one 内实时 emit，这里只做账面计数 / 提前 break。
            if saved >= opts.count:
                break

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
