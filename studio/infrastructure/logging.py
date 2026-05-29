"""统一日志体系入口（PR-1 C2 骨架；PR-1 C3 扩 setup_logging 完整实现）。

本模块是 ADR-0009 的落地代码。当前 C2 阶段只暴露：
    make_studio_log_handler() -> logging.Handler   创建写 studio.log 的 handler 实例

不在 import time 装任何 handler / excepthook（防破 import smoke test）；
不调 setup_logging（不存在）；不动 root logger 状态。

后续 commit 扩充：
    C3  setup_logging / JsonLineFormatter / HumanConsoleFormatter / silence list / uvicorn 接管
    C4  api/lifespan + cli + workers/_base 三处入口接入
    C5  ContextVar trace_id + Filter
    C6  跨进程 env 透传 + db.tasks.request_trace_id
    C7  workers/supervisor 收尾

Windows 跨进程文件锁：用 concurrent-log-handler 的 ConcurrentRotatingFileHandler；
如果该包未安装（极少见，可能云端镜像缺）fallback stdlib RotatingFileHandler。
单写场景（C2 阶段仅 webui 进程写）两者行为一致；多进程同写到 C7+ 才开始真正受益。
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from .paths import LOGS_DIR

try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler as _RotatingHandler
except ImportError:
    _RotatingHandler = logging.handlers.RotatingFileHandler

STUDIO_LOG_NAME = "studio.log"
STUDIO_LOG_MAX_BYTES = 50 * 1024 * 1024
STUDIO_LOG_BACKUP_COUNT = 5


def make_studio_log_handler(log_dir: Path | None = None) -> logging.Handler:
    """创建写 studio.log 的 rotating handler（不装到 root；caller 决定）。

    C2 阶段的 formatter 是简单 `%(asctime)s [%(levelname)s] %(name)s: %(message)s`，
    给 dev 立刻能 grep；C3 改成 JsonLineFormatter 跨字段结构化。

    Args:
        log_dir: 落盘目录；默认 paths.LOGS_DIR。caller 通常不传（测试用 tmp_path）。
    """
    target = (log_dir or LOGS_DIR)
    target.mkdir(parents=True, exist_ok=True)
    handler = _RotatingHandler(
        str(target / STUDIO_LOG_NAME),
        maxBytes=STUDIO_LOG_MAX_BYTES,
        backupCount=STUDIO_LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    return handler
