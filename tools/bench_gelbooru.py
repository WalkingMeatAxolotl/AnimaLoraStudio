"""三步定位 gelbooru 下载速度瓶颈：网络 / 上游限速 / Studio 代码。

跑法（在仓库根目录）：
    venv/bin/python scripts/bench_gelbooru.py
    # Windows: venv\\Scripts\\python.exe scripts\\bench_gelbooru.py

凭证自动从 studio_data/secrets.json 读；找不到时从环境变量 GELBOORU_USER_ID /
GELBOORU_API_KEY 读。日志同时打到 stdout 和 bench_gelbooru.log。
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOG_PATH = REPO_ROOT / "bench_gelbooru.log"
SAMPLE_SIZE = 20  # 拉多少张图做 serial / parallel 对比
TAGS = "1girl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("bench")

HEADERS = {
    "Referer": "https://gelbooru.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def load_credentials() -> tuple[str, str]:
    secrets_path = REPO_ROOT / "studio_data" / "secrets.json"
    if secrets_path.exists():
        try:
            data = json.loads(secrets_path.read_text(encoding="utf-8"))
            g = data.get("gelbooru") or {}
            uid, key = g.get("user_id", ""), g.get("api_key", "")
            if uid and key:
                log.info("凭证从 studio_data/secrets.json 读取")
                return uid, key
        except Exception as exc:  # noqa: BLE001
            log.warning("secrets.json 读失败：%s", exc)
    uid = os.environ.get("GELBOORU_USER_ID", "")
    key = os.environ.get("GELBOORU_API_KEY", "")
    if uid and key:
        log.info("凭证从环境变量读取")
        return uid, key
    log.error(
        "找不到 gelbooru 凭证。请确认 studio_data/secrets.json 已配 user_id/api_key，"
        "或导出 GELBOORU_USER_ID / GELBOORU_API_KEY 环境变量。"
    )
    sys.exit(1)


def fetch_post_urls(user_id: str, api_key: str, limit: int) -> list[str]:
    log.info("拉 %d 张 post 元数据 ...", limit)
    r = requests.get(
        "https://gelbooru.com/index.php",
        params={
            "page": "dapi", "s": "post", "q": "index", "json": "1",
            "tags": TAGS, "pid": 0, "limit": limit,
            "user_id": user_id, "api_key": api_key,
        },
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    posts = r.json().get("post") or []
    urls = [p["file_url"] for p in posts if p.get("file_url")]
    log.info("拿到 %d 个 file_url", len(urls))
    if not urls:
        log.error("一个 URL 都没拿到，凭证或 tag 有问题，终止")
        sys.exit(1)
    return urls


def ping_host(host: str, port: int = 443, attempts: int = 5) -> None:
    """TCP connect RTT —— 不靠 ICMP（云机房常 block ping），看 TCP 握手时间。"""
    samples: list[float] = []
    for _ in range(attempts):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=10):
                samples.append((time.perf_counter() - t0) * 1000)
        except OSError as exc:
            log.warning("  TCP connect %s 失败：%s", host, exc)
    if samples:
        log.info(
            "  TCP RTT %s:%d → min=%.0fms median=%.0fms max=%.0fms (n=%d)",
            host, port, min(samples), statistics.median(samples), max(samples), len(samples),
        )


# ---------------------------------------------------------------------------
# ① 单图原始网络（无 Session、无并发，最接近 curl）
# ---------------------------------------------------------------------------


def step1_raw_network(urls: list[str]) -> None:
    log.info("=" * 70)
    log.info("① 单图原始网络速度（每图新建连接，不复用 session）")
    log.info("=" * 70)

    api_host = "gelbooru.com"
    cdn_host = urlparse(urls[0]).hostname or ""
    log.info("API host: %s", api_host)
    log.info("CDN host: %s", cdn_host)
    ping_host(api_host)
    if cdn_host and cdn_host != api_host:
        ping_host(cdn_host)

    # 测前 3 张图，看 speed = bytes / total_time
    speeds: list[float] = []
    for i, u in enumerate(urls[:3]):
        t0 = time.perf_counter()
        try:
            resp = requests.get(u, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            content = resp.content
        except Exception as exc:  # noqa: BLE001
            log.warning("  图 %d 失败：%s", i + 1, exc)
            continue
        elapsed = time.perf_counter() - t0
        size = len(content)
        speed = size / elapsed if elapsed > 0 else 0
        speeds.append(speed)
        log.info(
            "  图 %d  size=%.2fMB  time=%.2fs  speed=%.2f MB/s  (%s)",
            i + 1, size / 1e6, elapsed, speed / 1e6, urlparse(u).hostname,
        )
    if speeds:
        log.info("  平均单图速度 %.2f MB/s", statistics.mean(speeds) / 1e6)


# ---------------------------------------------------------------------------
# ② 串行 vs 并发（裸 requests，绕开 Studio）
# ---------------------------------------------------------------------------


def step2_serial_vs_parallel(urls: list[str]) -> tuple[float, float]:
    log.info("=" * 70)
    log.info("② 裸 requests 串行 vs 并发对比 (n=%d)", len(urls))
    log.info("=" * 70)

    sess1 = requests.Session()
    t0 = time.perf_counter()
    total = 0
    fail = 0
    for u in urls:
        try:
            r = sess1.get(u, headers=HEADERS, timeout=60)
            r.raise_for_status()
            total += len(r.content)
        except Exception as exc:  # noqa: BLE001
            fail += 1
            log.warning("  serial 失败：%s", exc)
    serial_t = time.perf_counter() - t0
    sess1.close()
    log.info(
        "  SERIAL   : %d imgs / %d fail in %.1fs  (%.2f MB/s, %.2f img/s)",
        len(urls) - fail, fail, serial_t,
        total / serial_t / 1e6 if serial_t else 0,
        (len(urls) - fail) / serial_t if serial_t else 0,
    )

    sess2 = requests.Session()

    def fetch(u: str) -> int:
        r = sess2.get(u, headers=HEADERS, timeout=60)
        r.raise_for_status()
        return len(r.content)

    t0 = time.perf_counter()
    total = 0
    fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(fetch, u) for u in urls]
        for f in concurrent.futures.as_completed(futs):
            try:
                total += f.result()
            except Exception as exc:  # noqa: BLE001
                fail += 1
                log.warning("  parallel 失败：%s", exc)
    para_t = time.perf_counter() - t0
    sess2.close()
    log.info(
        "  PARALLEL4: %d imgs / %d fail in %.1fs  (%.2f MB/s, %.2f img/s)",
        len(urls) - fail, fail, para_t,
        total / para_t / 1e6 if para_t else 0,
        (len(urls) - fail) / para_t if para_t else 0,
    )
    if para_t > 0:
        log.info("  speedup  : %.2fx", serial_t / para_t)
    return serial_t, para_t


# ---------------------------------------------------------------------------
# ③ Studio 实际代码（PP9 BooruClient）
# ---------------------------------------------------------------------------


def step3_studio_client(urls: list[str]) -> None:
    log.info("=" * 70)
    log.info("③ Studio BooruClient (PP9 真实代码路径)")
    log.info("=" * 70)
    try:
        from studio.services.booru.pool import BooruClient, BooruPoolConfig
    except Exception as exc:  # noqa: BLE001
        log.error("import studio.services.booru_pool 失败：%s", exc)
        return

    cfg = BooruPoolConfig(parallel_workers=4, api_rate_per_sec=2.0, cdn_rate_per_sec=5.0)
    log.info(
        "  cfg: workers=%d api_rate=%.1f/s cdn_rate=%.1f/s",
        cfg.parallel_workers, cfg.api_rate_per_sec, cfg.cdn_rate_per_sec,
    )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        items = [(i, u, td_path / f"{i}.bin") for i, u in enumerate(urls)]
        with BooruClient(cfg) as client:
            t0 = time.perf_counter()

            def dl(item: tuple[int, str, Path]) -> Path:
                _, u, p = item
                return client.download_image(
                    u, p,
                    convert_to_png=False,
                    remove_alpha_channel=False,
                    referer="https://gelbooru.com/",
                )

            results = client.parallel_download(items, dl)
            elapsed = time.perf_counter() - t0

        ok = sum(1 for _, _, exc in results if exc is None)
        fail = len(results) - ok
        total = sum(p.stat().st_size for _, _, p in items if p.exists())
        log.info(
            "  PP9 BooruClient: %d ok / %d fail in %.1fs  (%.2f MB/s, %.2f img/s)",
            ok, fail, elapsed,
            total / elapsed / 1e6 if elapsed else 0,
            ok / elapsed if elapsed else 0,
        )
        for _, _, exc in results[:3]:
            if exc is not None:
                log.warning("  示例错误：%s", exc)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("bench_gelbooru.py — 日志写入 %s", LOG_PATH)
    log.info("Python: %s", sys.version.split()[0])
    log.info("requests: %s", requests.__version__)

    user_id, api_key = load_credentials()
    urls = fetch_post_urls(user_id, api_key, SAMPLE_SIZE)

    step1_raw_network(urls)
    step2_serial_vs_parallel(urls)
    step3_studio_client(urls)

    log.info("=" * 70)
    log.info("完成。把 %s 整份贴回来即可。", LOG_PATH.name)


if __name__ == "__main__":
    main()
