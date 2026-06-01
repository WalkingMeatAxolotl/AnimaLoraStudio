"""PP9 — BooruClient 池子：token bucket 分离 / 并发 worker / 429 退避。

不真发 HTTP；用 monkeypatch 替 booru_api.search_posts / download_image 验证
路由（API/CDN 桶）+ 计时 + 429 自适应。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from studio.services.booru import api as booru_api, pool as booru_pool


# ---------------------------------------------------------------------------
# host 分类
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,is_cdn",
    [
        ("https://img3.gelbooru.com/images/abc.png", True),
        ("https://video-cdn3.gelbooru.com/x.mp4", True),
        ("https://cdn.donmai.us/original/a/b.jpg", True),
        ("https://gelbooru.com/index.php", False),
        ("https://danbooru.donmai.us/posts.json", False),
    ],
)
def test_is_cdn_host_classifies_correctly(url: str, is_cdn: bool) -> None:
    assert booru_pool.is_cdn_host(url) is is_cdn


# ---------------------------------------------------------------------------
# TokenBucket
# ---------------------------------------------------------------------------


def test_token_bucket_enforces_minimum_interval() -> None:
    """rate=10/s → 连续两次 acquire 间隔 ≥ 0.1s。"""
    bucket = booru_pool.TokenBucket(rate_per_sec=10.0)
    bucket.acquire()  # 第一次零等待（next_time=0）
    t0 = time.monotonic()
    bucket.acquire()
    bucket.acquire()
    elapsed = time.monotonic() - t0
    # 两次后续 acquire 各等 0.1s
    assert elapsed >= 0.18, f"expected >= 0.18s, got {elapsed:.3f}s"


def test_token_bucket_set_rate_takes_effect() -> None:
    bucket = booru_pool.TokenBucket(rate_per_sec=100.0)
    bucket.set_rate(5.0)
    assert bucket.interval == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# BooruClient
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_client(monkeypatch: pytest.MonkeyPatch):
    """高速桶 + mock booru_api，避免真等真发请求。"""
    cfg = booru_pool.BooruPoolConfig(
        parallel_workers=4,
        api_rate_per_sec=100.0,
        cdn_rate_per_sec=100.0,
        backoff_on_429=0.1,  # 测试加速
    )
    return booru_pool.BooruClient(cfg)


def test_search_uses_api_bucket(monkeypatch: pytest.MonkeyPatch, fast_client) -> None:
    """search_posts 调底层 booru_api.search_posts；走 API bucket。"""
    seen = []

    def fake_search(api_source, query, **kw):
        seen.append((api_source, query, kw.get("page", 1)))
        return [{"id": 1, "file_url": "https://img/a.jpg"}]

    monkeypatch.setattr(booru_api, "search_posts", fake_search)
    out = fast_client.search_posts("gelbooru", "1girl", page=2, user_id="u", api_key="k")
    assert out == [{"id": 1, "file_url": "https://img/a.jpg"}]
    assert seen == [("gelbooru", "1girl", 2)]
    fast_client.close()


def test_download_uses_cdn_bucket(
    monkeypatch: pytest.MonkeyPatch, fast_client, tmp_path: Path
) -> None:
    seen: list[str] = []

    def fake_download(url, save_path, **kw):
        seen.append(url)
        save_path.write_bytes(b"x")
        return save_path

    monkeypatch.setattr(booru_api, "download_image", fake_download)
    out = fast_client.download_image(
        "https://img3.gelbooru.com/x.png",
        tmp_path / "x.png",
        convert_to_png=False,
        remove_alpha_channel=False,
    )
    assert out.exists()
    assert seen == ["https://img3.gelbooru.com/x.png"]
    fast_client.close()


def test_buckets_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """API rate=1/s + CDN rate=100/s → 100 次拉图 + 1 次 search 总耗时 < 1s（API 没拖慢 CDN）。"""
    cfg = booru_pool.BooruPoolConfig(
        parallel_workers=4, api_rate_per_sec=1.0, cdn_rate_per_sec=100.0
    )
    client = booru_pool.BooruClient(cfg)

    monkeypatch.setattr(booru_api, "search_posts", lambda *a, **k: [])
    monkeypatch.setattr(booru_api, "download_image", lambda u, p, **k: p.write_bytes(b"x") or p)

    t0 = time.monotonic()
    # 1 个 search（吃掉 API token）+ 5 个并发拉图
    client.search_posts("gelbooru", "x", user_id="u", api_key="k")
    items = list(range(5))
    paths = [Path(f"/tmp/dummy_{i}.png") for i in items]
    # 但是 path 用 tmp 目录避免真写到 /tmp
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        paths = [Path(td) / f"x{i}.png" for i in items]
        client.parallel_download(
            list(zip(items, paths)),
            lambda pair: client.download_image(
                f"https://img3.gelbooru.com/{pair[0]}.png",
                pair[1],
                convert_to_png=False,
                remove_alpha_channel=False,
            ),
        )
    elapsed = time.monotonic() - t0
    # API 1/s 限速对 search 单次没影响（第一次零等待）；CDN 100/s 几乎不限
    assert elapsed < 1.0, f"耗时 {elapsed:.2f}s，桶可能未分离"
    client.close()


def test_429_triggers_backoff_and_halves_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = booru_pool.BooruPoolConfig(
        parallel_workers=2,
        api_rate_per_sec=10.0,
        cdn_rate_per_sec=10.0,
        backoff_on_429=0.05,
    )
    client = booru_pool.BooruClient(cfg)

    # 模拟一次 search 抛 HTTPError(status=429)
    resp = MagicMock()
    resp.status_code = 429
    err = requests.HTTPError("429"); err.response = resp

    def fake_search(*_a, **_k):
        raise err

    monkeypatch.setattr(booru_api, "search_posts", fake_search)
    with pytest.raises(requests.HTTPError):
        client.search_posts("gelbooru", "x", user_id="u", api_key="k")
    # 速率应已减半
    assert client.cfg.api_rate_per_sec == pytest.approx(5.0)
    assert client.cfg.cdn_rate_per_sec == pytest.approx(5.0)
    # 第二次 429 不应再次减半
    with pytest.raises(requests.HTTPError):
        client.search_posts("gelbooru", "x", user_id="u", api_key="k")
    assert client.cfg.api_rate_per_sec == pytest.approx(5.0)
    client.close()


def test_parallel_download_respects_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    """parallel_workers=2 → 同时活动的 fn 调用数 ≤ 2。"""
    cfg = booru_pool.BooruPoolConfig(
        parallel_workers=2, api_rate_per_sec=100.0, cdn_rate_per_sec=100.0
    )
    client = booru_pool.BooruClient(cfg)

    active = 0
    max_active = 0
    lock = threading.Lock()

    def slow_fn(_item):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return "ok"

    out = client.parallel_download(list(range(10)), slow_fn)
    assert len(out) == 10
    assert all(r[2] is None for r in out)  # 没有异常
    assert max_active <= 2
    client.close()


def test_parallel_download_cancel_event(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = booru_pool.BooruPoolConfig(
        parallel_workers=2, api_rate_per_sec=100.0, cdn_rate_per_sec=100.0
    )
    client = booru_pool.BooruClient(cfg)

    cancel = threading.Event()

    def fn(item):
        if item == 1:
            cancel.set()
        time.sleep(0.02)
        return item

    out = client.parallel_download(list(range(10)), fn, cancel_event=cancel)
    # 至少前几个会跑；后面被 cancel 拦下
    canceled = [r for r in out if isinstance(r[2], RuntimeError)]
    assert len(canceled) > 0
    client.close()


def test_context_manager_closes_resources() -> None:
    cfg = booru_pool.BooruPoolConfig(parallel_workers=2)
    with booru_pool.BooruClient(cfg) as client:
        assert client._executor is not None
    # 关闭后调用 close 应幂等
    client.close()


def test_external_session_not_owned() -> None:
    """传入外部 session 时，client.close 不应关掉它。"""
    sess = requests.Session()
    closed = []

    real_close = sess.close

    def track_close():
        closed.append(True)
        real_close()

    sess.close = track_close  # type: ignore[method-assign]
    client = booru_pool.BooruClient(session=sess)
    client.close()
    assert closed == [], "外部 session 不应被关闭"
    sess.close = real_close  # type: ignore[method-assign]
