"""PR-1 C2 — infrastructure/logging.py 骨架最小测试。

只覆盖 C2 暴露的 make_studio_log_handler；不测 setup_logging（C3 才有）。
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def test_import_logging_module_does_not_touch_root(tmp_path: Path) -> None:
    """import studio.infrastructure.logging 不能装 handler / 改 excepthook（破 import_smoke）。"""
    root_handlers_before = len(logging.getLogger().handlers)
    excepthook_before = sys.excepthook

    import studio.infrastructure.logging as _slog  # noqa: F401

    assert len(logging.getLogger().handlers) == root_handlers_before, (
        "import 不能给 root logger 加 handler；C2 骨架阶段必须 inert"
    )
    assert sys.excepthook is excepthook_before, (
        "import 不能动 sys.excepthook；C3 setup_logging() explicit 调用才装"
    )


def test_make_studio_log_handler_creates_file_on_first_emit(tmp_path: Path) -> None:
    from studio.infrastructure.logging import (
        STUDIO_LOG_BACKUP_COUNT,
        STUDIO_LOG_MAX_BYTES,
        STUDIO_LOG_NAME,
        make_studio_log_handler,
    )

    h = make_studio_log_handler(log_dir=tmp_path)
    try:
        log_file = tmp_path / STUDIO_LOG_NAME
        # delay=True：文件首次 emit 才创建
        assert not log_file.exists(), "delay=True 时未 emit 不应创建文件"

        lg = logging.getLogger("studio.test_pr1_c2")
        lg.setLevel(logging.INFO)
        lg.addHandler(h)
        try:
            lg.info("c2 skeleton smoke")
            h.flush()
        finally:
            lg.removeHandler(h)

        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "[INFO] studio.test_pr1_c2: c2 skeleton smoke" in content, (
            f"format 应含 level + logger name + msg；实际: {content!r}"
        )

        # 锁默认配置常量值（C3 不能随意调）
        assert STUDIO_LOG_NAME == "studio.log"
        assert STUDIO_LOG_MAX_BYTES == 50 * 1024 * 1024
        assert STUDIO_LOG_BACKUP_COUNT == 5
    finally:
        h.close()


def test_make_studio_log_handler_uses_concurrent_handler_if_available() -> None:
    """有 concurrent-log-handler 时用它（Windows 跨进程文件锁支持）；
    无则 fallback stdlib RotatingFileHandler — 两条路径都不能 crash。"""
    from studio.infrastructure.logging import _RotatingHandler

    # 验证 type 是 RotatingFileHandler 子类或本身（满足任一即可）
    import logging.handlers as _h
    assert (
        _RotatingHandler is _h.RotatingFileHandler
        or issubclass(_RotatingHandler, _h.BaseRotatingHandler)
    ), f"_RotatingHandler 必须是 (Concurrent)RotatingFileHandler；得到 {_RotatingHandler!r}"
