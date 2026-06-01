"""PP9 — Booru API 统一池子：Session keepalive + 双 token bucket + 并发拉图 + 429 退避。

为什么要池子：
- `downloader.py` 现状是完全同步串行 + 每图强制 0.5s sleep，云上比家庭宽带慢 5-10x
- API 限速主要落在 `gelbooru.com`，CDN 拉图域名 (`img*.gelbooru.com`) 限频要松得多
- 多个 task（download + reg_build）同时跑时各自不知道对方进度，容易撞速率墙

设计：
- 双 token bucket：API host 2 req/s + CDN host 5 req/s（按 URL netloc 区分）
- ThreadPoolExecutor 默认 4 worker（与 CDN 桶匹配）
- 收到 429/503 → sticky backoff 60s + 速率减半，永久到 client 销毁（保守）
- `requests.Session` 复用 TCP/TLS（HTTP keepalive 单这一项就快 2x）

公开接口与 `booru_api.py` 平行：`search_posts` / `download_image` / `parallel_download`。
原始 `booru_api.search_posts(...)` 仍可单独调用（测试 / 旧代码兼容），池子只是上层薄壳。
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
from urllib.parse import urlparse

import requests

from . import api as booru_api
from ..proxy_manager import patch_requests_session

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# host 分类
# ---------------------------------------------------------------------------


# CDN 主机匹配（gelbooru / danbooru 都把图托管在子域）：
# gelbooru: img3.gelbooru.com / video-cdn3.gelbooru.com / ...
# danbooru: cdn.donmai.us / sample.donmai.us / ...
_CDN_HOST_HINTS = ("cdn", "img", "video-cdn", "raikou", "sample")


def is_cdn_host(url: str) -> bool:
    """按 netloc 判定是否是 CDN（图床）；其它按 API host 计。"""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return any(h in host for h in _CDN_HOST_HINTS)


# ---------------------------------------------------------------------------
# token bucket
# ---------------------------------------------------------------------------


class TokenBucket:
    """简单时间窗 token bucket：每秒最多 `rate` 个 token，跨线程安全。

    `acquire()` 阻塞直到拿到 token；不实现 burst（连续请求自然等齐）。
    """

    def __init__(self, rate_per_sec: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec 必须 > 0")
        self._interval = 1.0 / rate_per_sec
        self._next_time = 0.0
        self._lock = threading.Lock()

    @property
    def interval(self) -> float:
        return self._interval

    def set_rate(self, rate_per_sec: float) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec 必须 > 0")
        with self._lock:
            self._interval = 1.0 / rate_per_sec

    def acquire(self) -> None:
        """阻塞到下一个 token 可用。"""
        with self._lock:
            now = time.monotonic()
            wait = self._next_time - now
            if wait > 0:
                # 持锁 sleep —— 简单粗暴，rate 不大时 contention 可接受
                time.sleep(wait)
                now = time.monotonic()
            self._next_time = max(now, self._next_time) + self._interval


# ---------------------------------------------------------------------------
# config + client
# ---------------------------------------------------------------------------


@dataclass
class BooruPoolConfig:
    parallel_workers: int = 4
    api_rate_per_sec: float = 2.0
    cdn_rate_per_sec: float = 5.0
    backoff_on_429: float = 60.0


class BooruClient:
    """Session + ThreadPoolExecutor + 双 token bucket + 429 sticky 退避。

    使用模式：
        with BooruClient() as client:
            posts = client.search_posts("gelbooru", "1girl", api_key=..., user_id=...)
            results = client.parallel_download(items, lambda item: client.download_image(...))

    线程安全；同实例可被 downloader / reg_builder 同时调用。
    """

    def __init__(
        self,
        cfg: Optional[BooruPoolConfig] = None,
        *,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.cfg = cfg or BooruPoolConfig()
        self._session = session or requests.Session()
        patch_requests_session(self._session)
        logger.info(f"BooruClient session proxies: {self._session.proxies}")
        self._owns_session = session is None
        self._api_bucket = TokenBucket(self.cfg.api_rate_per_sec)
        self._cdn_bucket = TokenBucket(self.cfg.cdn_rate_per_sec)
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, self.cfg.parallel_workers),
            thread_name_prefix="booru-pool",
        )
        # 429 sticky 退避状态
        self._lock = threading.Lock()
        self._backoff_until: float = 0.0
        self._rate_halved = False

    # -------------------- lifecycle --------------------

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        if self._owns_session:
            try:
                self._session.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "BooruClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -------------------- 429 sticky 状态 --------------------

    def _wait_if_backoff(self) -> None:
        """若上次撞 429，整 client 等到 backoff_until 之后才发新请求。"""
        with self._lock:
            until = self._backoff_until
        wait = until - time.monotonic()
        if wait > 0:
            logger.warning("[booru_pool] 429 backoff，等 %.1fs", wait)
            time.sleep(wait)

    def _trigger_backoff(self, status_code: int) -> None:
        """收到 429/503 → 全 client backoff 60s + 永久减半速率。"""
        with self._lock:
            self._backoff_until = time.monotonic() + self.cfg.backoff_on_429
            if not self._rate_halved:
                self._rate_halved = True
                self.cfg.api_rate_per_sec /= 2
                self.cfg.cdn_rate_per_sec /= 2
                self._api_bucket.set_rate(self.cfg.api_rate_per_sec)
                self._cdn_bucket.set_rate(self.cfg.cdn_rate_per_sec)
                logger.warning(
                    "[booru_pool] 收到 %d，速率永久减半（API %.2f / CDN %.2f req/s）",
                    status_code,
                    self.cfg.api_rate_per_sec,
                    self.cfg.cdn_rate_per_sec,
                )

    def _check_response(self, resp: requests.Response) -> None:
        """429/503 触发 sticky 退避；其他 4xx/5xx 不动池子（让上层抛错）。"""
        if resp.status_code in (429, 503):
            self._trigger_backoff(resp.status_code)

    # -------------------- 公开 API --------------------

    def search_posts(self, api_source: str, tags_query: str, **kw: Any) -> list[dict[str, Any]]:
        """走 API bucket。kw 透传给 booru_api.search_posts。"""
        self._wait_if_backoff()
        self._api_bucket.acquire()
        kw.setdefault("session", self._session)
        try:
            return booru_api.search_posts(api_source, tags_query, **kw)
        except requests.HTTPError as exc:
            if exc.response is not None:
                self._check_response(exc.response)
            raise

    def download_image(self, url: str, save_path: Path, **kw: Any) -> Path:
        """走 CDN bucket。kw 透传给 booru_api.download_image。"""
        self._wait_if_backoff()
        self._cdn_bucket.acquire()
        kw.setdefault("session", self._session)
        try:
            return booru_api.download_image(url, save_path, **kw)
        except requests.HTTPError as exc:
            if exc.response is not None:
                self._check_response(exc.response)
            raise

    def parallel_download(
        self,
        items: list[T],
        fn: Callable[[T], Any],
        *,
        cancel_event: Optional[threading.Event] = None,
    ) -> list[tuple[T, Any, Optional[Exception]]]:
        """并发跑 fn(item) 跨 items；保持原顺序返回 [(item, result, exc)]。

        异常不冒泡（每个 item 独立报错），由调用方按需处理。
        cancel_event 在 submit 前 + result 后双检；触发后取消未跑的 future。
        """
        if not items:
            return []
        results: list[tuple[T, Any, Optional[Exception]]] = []
        futures = []
        # submit 前批量检 cancel
        for it in items:
            if cancel_event is not None and cancel_event.is_set():
                break
            futures.append((it, self._executor.submit(fn, it)))
        for it, fut in futures:
            if cancel_event is not None and cancel_event.is_set():
                fut.cancel()
                results.append((it, None, RuntimeError("canceled")))
                continue
            try:
                r = fut.result()
                results.append((it, r, None))
            except Exception as exc:  # noqa: BLE001
                results.append((it, None, exc))
        return results
