"""PR-LOG-1 安全网 — 锁现有 logger 行为基线（防 PR-LOG-2 装 root config 破 caplog）。

现状（B 审计 §1.1）：
  - 21 文件 `getLogger(__name__)`，0 处 `basicConfig` / FileHandler / Formatter
  - root logger 默认 WARNING 阈值 + 默认 StreamHandler
  - pytest caplog 依赖 propagate=True

PR-LOG-2 引入 `infrastructure/logging.py:setup_logging` 后必须：
  - 保 propagate=True（caplog 仍可用）
  - 测试 fixture 不触发 setup_logging（除非显式测它）
  - studio.* logger 仍能被 caplog 捕

本测试在 PR-LOG-2 前后**都应通过**。
"""
from __future__ import annotations

import logging

import pytest


def test_studio_loggers_propagate_to_root() -> None:
    """所有 studio.* logger 必须 propagate=True，否则 caplog 接不到。"""
    import studio  # noqa: F401  确保 studio 子模块加载触发 logger 创建
    from studio import supervisor as _sup  # noqa: F401
    from studio.services import system_stats as _ss  # noqa: F401

    failed = []
    for name, lg in logging.Logger.manager.loggerDict.items():
        if not name.startswith("studio."):
            continue
        if isinstance(lg, logging.PlaceHolder):
            continue
        if not lg.propagate:
            failed.append(name)
    assert not failed, f"以下 studio.* logger propagate=False，caplog 会接不到: {failed}"


def test_logger_call_via_caplog_works(caplog: pytest.LogCaptureFixture) -> None:
    """模拟现有代码风格：模块级 logger 调用能被 caplog 捕。"""
    logger = logging.getLogger("studio.test_log_baseline")
    with caplog.at_level(logging.INFO, logger="studio.test_log_baseline"):
        logger.info("baseline info message")
        logger.warning("baseline warning message")
        try:
            raise ValueError("baseline exc")
        except ValueError:
            logger.exception("baseline exception with stack")

    messages = [r.getMessage() for r in caplog.records]
    assert "baseline info message" in messages
    assert "baseline warning message" in messages
    assert "baseline exception with stack" in messages
    # exception 路径必须带 exc_info
    exc_records = [r for r in caplog.records if r.exc_info]
    assert len(exc_records) >= 1, "logger.exception 必须带 exc_info（栈信息）"


def test_logger_getlogger_name_uses_dunder_name() -> None:
    """grep 验证：所有 21 处 `getLogger(__name__)` 都用 __name__ 不用 custom string。

    这是未来 setup_logging 一刀覆盖 `studio.*` 的前提。
    """
    from pathlib import Path
    import re

    studio_root = Path(__file__).resolve().parent.parent / "studio"
    bad = []
    pattern = re.compile(r"logging\.getLogger\(([^)]+)\)")
    for py in studio_root.rglob("*.py"):
        if "web/" in str(py).replace("\\", "/"):
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in pattern.finditer(text):
            arg = m.group(1).strip()
            if arg not in ("__name__", "", "f\"{__name__}\""):
                # 允许 logging.getLogger("uvicorn") / logging.getLogger("uvicorn.access") 给第三方 logger silence 用
                if arg.startswith('"uvicorn') or arg.startswith("'uvicorn"):
                    continue
                bad.append(f"{py.relative_to(studio_root)}: getLogger({arg})")
    assert not bad, (
        "studio/ 内 getLogger 应该统一用 __name__；下列违反必须修或加 silence list 注释:\n"
        + "\n".join(bad)
    )
