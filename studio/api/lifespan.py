"""FastAPI lifespan + import-time 副作用迁移（PR-5 从 server.py 抽出）。

PR-5 关键改动：把 `ensure_dirs()` + `db.init_db()` 从 server.py 顶层
（import-time 副作用）移到 lifespan startup —— 这样 `from studio.server
import app` 不再触发文件系统初始化，便于测试 / 工具 import 而不写盘。

启动阶段：
    1. 装 Windows ProactorEventLoop ConnectionResetError 静音 filter
    2. ensure_dirs() + db.init_db()  ← PR-5 新位置
    3. 清扫遗留 generate tempdir（防 supervisor crash 后泄漏）
    4. 后台下载 TAEFlux（中间步预览，~1.6MB，不阻塞 server）
    5. event bus 绑定 loop + 配 SSE 连接回调
    6. Supervisor 启动 + 写入 app.state.supervisor
    7. SystemStatsSampler 启动

关闭阶段：
    1. 取消挂着的 SSE disconnect timer
    2. SystemStatsSampler.stop
    3. Supervisor.stop（含 daemon stop + 子进程 graceful terminate）
    4. generate_cache.clear_all（释放图缓存内存）
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI

from .. import db
from ..infrastructure.event_bus import bus
from ..infrastructure.logging import setup_logging
from ..paths import ensure_dirs
from ..supervisor import Supervisor

logger = logging.getLogger(__name__)


def _install_proactor_disconnect_filter(loop: asyncio.AbstractEventLoop) -> None:
    """吞 Windows + asyncio Proactor 的 cosmetic ConnectionResetError 噪声。

    Python asyncio 在 Windows 上有 [bpo-44291](https://github.com/python/cpython/issues/87691) 类问题：
    远端 TCP 强制断开（用户关 tab / 刷新 / SSE 重连，WinError 10054 / 10053）
    时 `_ProactorBasePipeTransport._call_connection_lost` 走 `socket.shutdown()`
    抛 `ConnectionResetError` / `ConnectionAbortedError`，但 callback 内部
    没 catch 这两个 expected error，asyncio 默认 handler 打 traceback 到
    stderr。server 完全没事，只是日志被刷一行无意义 stack。

    精确过滤：只在 exception 是 ConnectionResetError / ConnectionAbortedError
    且 handle repr 含 `_call_connection_lost` 时静默吞掉；其它 asyncio
    异常仍交给 default handler。仅 Windows 装；其它平台用 SelectorEventLoop
    没这个 bug。
    """
    if os.name != "nt":
        return

    def _filter(loop_: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
            handle = context.get("handle")
            if handle and "_call_connection_lost" in repr(handle):
                return
        loop_.default_exception_handler(context)

    loop.set_exception_handler(_filter)


@asynccontextmanager
async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
    """启动绑定 event bus 到当前 loop 并起 supervisor；关闭时停 supervisor。"""
    # PR-1 C4: 统一日志体系入口 (ADR-0009)。第一行调，让 ensure_dirs / db.init_db
    # 自己 emit 的 log 也能进 studio.log。setup_logging 自身 mkdir LOGS_DIR
    # 不需要等 ensure_dirs。env ANIMA_LOGGING_NO_BOOTSTRAP=1 时 noop（测试态）。
    setup_logging("webui", level=os.environ.get("ANIMA_LOG_LEVEL", "INFO"))

    # 装 Windows ProactorEventLoop 的 ConnectionResetError 过滤器（详见 helper docstring）
    _install_proactor_disconnect_filter(asyncio.get_running_loop())

    # PR-5：从 server.py 顶层搬来的 import-time 副作用 —— 现在跟随 app 启动
    # 才落盘，便于测试 / 工具 import 而不写文件系统。
    ensure_dirs()
    db.init_db()

    # 测试出图 tempdir 遗留清扫（防 supervisor crash 泄漏 anima_gen_* 目录）
    from ..services.inference.core import cleanup_stale_generate_tempdirs
    from ..services.inference import cache as generate_cache
    from ..services import models as _md
    from ..services import system_stats
    cleanup_stale_generate_tempdirs()

    # TAEFlux（中间步预览）后台下载：跟 server 一起启动；下载失败不阻塞 server。
    # 如果已下载则 noop；下载期间用户能正常用其他功能，预览功能等下载完才生效。
    def _bg_download_taeflux() -> None:
        try:
            if _md.taeflux_available():
                return
            logger.info("background-downloading TAEFlux (~1.6MB)…")
            ok = _md.download_taeflux(on_log=lambda m: logger.info("[taeflux] %s", m))
            if not ok:
                logger.warning("taeflux background download failed; preview disabled until manual install")
        except Exception:
            logger.exception("taeflux background download crashed")
    threading.Thread(target=_bg_download_taeflux, name="taeflux-bg-download", daemon=True).start()

    bus.attach_loop(asyncio.get_running_loop())

    # commit 11：SSE 客户端断连 + 30s 缓冲后清 generate cache。
    # 防刷新/短抖动：用户重连（_on_first_subscribe）取消计时器。
    _disconnect_timer: dict[str, Optional[threading.Timer]] = {"t": None}

    def _on_last_unsubscribe() -> None:
        # 已有 timer 不重置（多个客户端各自 unsubscribe 时，最后一个才是关键）
        if _disconnect_timer["t"] is not None:
            return
        timer = threading.Timer(30.0, _flush_cache)
        timer.daemon = True
        _disconnect_timer["t"] = timer
        timer.start()

    def _on_first_subscribe() -> None:
        timer = _disconnect_timer.get("t")
        if timer is not None:
            timer.cancel()
            _disconnect_timer["t"] = None

    def _flush_cache() -> None:
        n = generate_cache.total_count()
        if n:
            generate_cache.clear_all()
            logger.info("flushed generate cache (%d images) after SSE idle", n)
        _disconnect_timer["t"] = None

    bus.set_connection_callbacks(
        on_first_subscribe=_on_first_subscribe,
        on_last_unsubscribe=_on_last_unsubscribe,
    )

    sup = Supervisor(on_event=bus.publish)
    sup.start()
    app_.state.supervisor = sup

    # PR #37: system stats SSE — 后台 sampler 每 2.5s 采集 + bus.publish。前端
    # 只 mount 时 GET 一次冷启动，避免 cloud 部署被每客户端独立轮询污染。
    def _publish_system_stats(payload: dict[str, Any]) -> None:
        bus.publish({"type": "system_stats_updated", "payload": payload})

    sys_sampler = system_stats.SystemStatsSampler(_publish_system_stats)
    sys_sampler.start()
    app_.state.system_stats_sampler = sys_sampler

    try:
        yield
    finally:
        # 取消可能挂着的 disconnect timer，shutdown 阶段不需要再延迟
        timer = _disconnect_timer.get("t")
        if timer is not None:
            timer.cancel()
        sys_sampler.stop()
        sup.stop()
        # commit 11：lifespan shutdown 清掉所有图 cache（释放内存 + 干净退出）
        generate_cache.clear_all()
