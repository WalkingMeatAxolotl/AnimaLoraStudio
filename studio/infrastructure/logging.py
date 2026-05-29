"""统一日志体系入口（ADR-0009）。

C2：骨架（make_studio_log_handler）
C3：完整 setup_logging + JsonLineFormatter + HumanConsoleFormatter + silence list
    + uvicorn handler 接管 + utf8 reconfigure（本文件）
C5：ContextVar trace_id + Filter（待加）
C6：跨进程 env / db.tasks.request_trace_id（待加）

使用规则（ADR-0009 §未来开发指南简版）：
    每个进程入口（webui / cli / worker）开头调一次 `setup_logging(process=...)`。
    业务代码模块顶 `logger = logging.getLogger(__name__)`，调 `.info/.warning/.exception`。
    不要在 import time 调 setup_logging（破 import smoke test）。

`process` 标识规范：
    "webui"                 webui server
    "cli:<subcmd>"          CLI run / dev / build / test
    "worker:<kind>/<id>"    worker subprocess（kind=download/tag/preprocess/reg_build）
    "client"                前端上报（C 系列 PR-3 才用到）
"""
from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import logging.handlers
import os
import sys
import traceback as _traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from .paths import LOGS_DIR

try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler as _RotatingHandler
except ImportError:
    _RotatingHandler = logging.handlers.RotatingFileHandler

STUDIO_LOG_NAME = "studio.log"
STUDIO_LOG_MAX_BYTES = 50 * 1024 * 1024
STUDIO_LOG_BACKUP_COUNT = 5

# trace_id 跨进程 / 跨 surface 名约定 (ADR-0009 §3.1)
TRACE_HEADER = "X-Trace-Id"           # HTTP 双向 header
TRACE_ENV = "ANIMA_TRACE_ID"          # 子进程 env (C6 supervisor 注入)
PROCESS_ENV = "ANIMA_PROCESS_NAME"    # 子进程预设 process 名

# ContextVar — 同进程跨 async/thread 自动传播 (C5)
# C6 / C7 后会用 job_id / task_id
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "studio_trace_id", default=None
)
_job_id_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "studio_job_id", default=None
)
_task_id_var: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "studio_task_id", default=None
)

# 第三方库 logger 静音 list — root level=INFO 时这些库会大量出 INFO 噪音。
# silence list 必须显式，避免漏一条让 stderr 爆 10×（B audit N.1 / A round2 §5.2）。
# 在 setup_logging 末尾应用。
_NOISY_LOGGERS = (
    "asyncio",
    "urllib3",
    "httpcore",
    "httpx",
    "PIL",
    "matplotlib",
    "modelscope",
    "huggingface_hub",
    "filelock",
)

# ── trace_id 公开 API ────────────────────────────────────────────────────


def new_trace_id() -> str:
    """生成 24 字符 hex trace_id（uuid4 hex[:24]）。

    24 字符：(a) 用户截图末 8 字符易复制；(b) 跟 supervisor 后台 spawn 的
    `bg-{new_trace_id()}` (28 字符) 区分清晰；(c) 比 ULID 26 简单不引依赖。
    """
    return uuid.uuid4().hex[:24]


def bind_trace_id(trace_id: str) -> contextvars.Token:
    """设当前 ContextVar trace_id，返 token 用于 reset。

    用法（middleware / worker bootstrap）：
        token = bind_trace_id(tid)
        try:
            ...do work...
        finally:
            reset_trace_id(token)
    """
    return _trace_id_var.set(trace_id)


def reset_trace_id(token: contextvars.Token) -> None:
    _trace_id_var.reset(token)


def get_trace_id() -> Optional[str]:
    """当前 ContextVar trace_id；未 bind 返 None。"""
    return _trace_id_var.get()


def bind_job_id(job_id: int) -> contextvars.Token:
    return _job_id_var.set(job_id)


def bind_task_id(task_id: int) -> contextvars.Token:
    return _task_id_var.set(task_id)


def get_job_id() -> Optional[int]:
    return _job_id_var.get()


def get_task_id() -> Optional[int]:
    return _task_id_var.get()


# ── ContextFilter ────────────────────────────────────────────────────────


class ContextFilter(logging.Filter):
    """读 ContextVar 注入到 LogRecord，让 JsonLineFormatter 直接拿。

    用 Filter 不用 LogRecord factory：factory 改全局，filter 装到 root 就行，
    回滚干净（C5 PR 撤掉 filter 即恢复，不动 LogRecord 类）。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = _trace_id_var.get()
        if not hasattr(record, "job_id"):
            jid = _job_id_var.get()
            if jid is not None:
                record.job_id = jid
        if not hasattr(record, "task_id"):
            tid = _task_id_var.get()
            if tid is not None:
                record.task_id = tid
        return True


# 模块级 sentinel — 同一 process 名重复调 setup_logging 不重复装 handler。
# supervisor 重启 / 测试 reload / worker 进程入口被双调时受益。
_CONFIGURED_PROCESSES: set[str] = set()


# ── Formatter ─────────────────────────────────────────────────────────────


class JsonLineFormatter(logging.Formatter):
    """JSON line 输出，10 固定字段（ADR-0009 §2.3）。

    顶层字段：ts / level / process / pid / trace_id / logger / msg / job_id / task_id / exc。
    可选嵌套：extra（logger.x(..., extra={...}) 传入的 user fields）。
    """

    def __init__(self, process: str) -> None:
        super().__init__()
        self._process = process

    def format(self, record: logging.LogRecord) -> str:
        # stdlib formatTime 用 time.strftime 不支持 %f；用 datetime 自己拼 ms
        ts = (
            _dt.datetime.fromtimestamp(record.created, tz=_dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )
        out: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "process": self._process,
            "pid": record.process,
            "trace_id": getattr(record, "trace_id", None),  # C5 Filter 会注入
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # 可选 contextvar 字段
        for k in ("job_id", "task_id"):
            v = getattr(record, k, None)
            if v is not None:
                out[k] = v
        # exc_info → 结构化
        if record.exc_info:
            etype, evalue, etb = record.exc_info
            out["exc"] = {
                "type": etype.__name__ if etype else "Unknown",
                "message": str(evalue),
                "traceback": "".join(_traceback.format_exception(etype, evalue, etb)),
            }
        # extra（用户 logger.x(..., extra={"k": "v"}) 进 record.__dict__）
        # 区分原生字段 + 我们注入的 + 用户 extra
        _builtin = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
            "message", "asctime", "trace_id", "job_id", "task_id",
        }
        extra = {k: v for k, v in record.__dict__.items() if k not in _builtin}
        if extra:
            out["extra"] = extra
        return json.dumps(out, ensure_ascii=False, default=str)


class HumanConsoleFormatter(logging.Formatter):
    """人读 console format，给 CLI / dev terminal。

    `2026-05-28 14:32:18.453 INFO  studio.api.routers.queue: queued task=42`
    """

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# ── Public API ────────────────────────────────────────────────────────────


def reconfigure_console_utf8() -> None:
    """Windows 控制台默认 cp932/cp936，写中文 / emoji 会 UnicodeEncodeError。
    强制 stdout/stderr 用 UTF-8 + replace 模式，让 logger 永远不抛。

    setup_logging 内首步调用，覆盖 webui / cli / worker 所有进程入口；
    workers/_base.py:reconfigure_console_utf8 (B audit 4.6 — 4 worker 缺一半) 由此兜底。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass


def make_studio_log_handler(
    log_dir: Path | None = None,
    *,
    process: str = "webui",
    formatter: logging.Formatter | None = None,
) -> logging.Handler:
    """创建写 studio.log 的 rotating handler（不装到 root；caller 决定）。

    Args:
        log_dir: 落盘目录；默认 paths.LOGS_DIR
        process: process 标识写进 JsonLineFormatter；默认 "webui"
        formatter: 自定义 formatter；默认 JsonLineFormatter(process)
    """
    target = log_dir or LOGS_DIR
    target.mkdir(parents=True, exist_ok=True)
    handler = _RotatingHandler(
        str(target / STUDIO_LOG_NAME),
        maxBytes=STUDIO_LOG_MAX_BYTES,
        backupCount=STUDIO_LOG_BACKUP_COUNT,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(formatter or JsonLineFormatter(process))
    return handler


def setup_logging(
    process: str,
    *,
    log_dir: Path | None = None,
    level: str = "INFO",
    console: str | bool = "auto",
    file: bool = True,
    extra_handlers: list[logging.Handler] | None = None,
) -> None:
    """安装全局 logging 配置。每个进程入口调一次（webui / cli / worker）。

    幂等：同一 `process` 名重复调 noop（防 supervisor 重启 / 测试 reload 累加 handler）。

    行为：
      1. reconfigure_console_utf8（Windows 编码兜底）
      2. 清 root logger 现有 handler（防累加）
      3. （可选）装 RotatingFileHandler 到 {log_dir}/studio.log（JSON line, 50MB×5）
      4. 装 console handler（auto/json/bool 三态，详 below）
      5. 装 extra_handlers（worker 用来塞 jobs/<id>.log handler）
      6. 静音第三方库（asyncio/urllib3/PIL/...）→ WARNING
      7. 接管 uvicorn.access / uvicorn.error logger（走我们的 root handler 而非 uvicorn 默认）
      8. 装 sys.excepthook → logger.critical 记未捕获异常
      9. 装 threading.excepthook → 同上（仅 Python 3.8+）

    Args:
        process: 进程标识（"webui" / "cli:run" / "worker:tag/42" / ...）
        log_dir: studio.log 落盘目录；默认读 env ANIMA_LOG_DIR，否则 paths.LOGS_DIR
        level: root logger level（"DEBUG" / "INFO" / "WARNING" / "ERROR"）
        console: "auto" = stderr isatty → Human, 否则 JSON；
                 "json" 强制 JSON；True 强制 Human；False 不装 console
        file: 是否装 file handler 到 studio.log；webui 默认 True；
              worker / cli 应传 False（worker 走 stdout 进 supervisor 重定向单写；
              CLI 5s 短命周期落盘价值低）。0.13.x ADR-0009 §还的债"演进双写"后 worker 改 True
        extra_handlers: 额外装到 root 的 handler（worker 用来塞 jobs/<id>.log）
    """
    # pytest 全局 fixture 设 ANIMA_LOGGING_NO_BOOTSTRAP=1 让业务代码 bootstrap
    # 调用全部 noop，避免污染 caplog / 反复装 handler；测 setup_logging 本身的
    # tests/test_logging_setup.py 自己 monkeypatch.delenv 解除。
    if os.environ.get("ANIMA_LOGGING_NO_BOOTSTRAP"):
        return
    if process in _CONFIGURED_PROCESSES:
        return
    _CONFIGURED_PROCESSES.add(process)

    reconfigure_console_utf8()

    root = logging.getLogger()
    # 清掉默认 / 累加的 handler（pytest fixture 反复调时关键）
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 1. 文件 handler — JSON line 到 studio.log（worker / cli 不装）
    if file:
        effective_log_dir = log_dir or _resolve_log_dir()
        root.addHandler(make_studio_log_handler(log_dir=effective_log_dir, process=process))

    # 2. console handler
    console_handler = _build_console_handler(console, process)
    if console_handler is not None:
        root.addHandler(console_handler)

    # ContextFilter — 装到每个 handler 而非 logger。
    # stdlib Logger.filter 只在 Logger.handle 顶层调一次；子 logger propagate
    # 到 root 后**不**调 root.filter，只调 root.handlers[*].emit。所以 filter
    # 必须装 handler 上，确保所有 record 经过 handler 时都有 ContextVar 注入。
    _ctx_filter = ContextFilter()
    for h in root.handlers:
        h.addFilter(_ctx_filter)

    # 3. extra handlers（worker job log 等）— 同样装 ContextFilter
    for h in extra_handlers or ():
        h.addFilter(ContextFilter())
        root.addHandler(h)

    # 4. 第三方库静音（防 root=INFO 后 stderr 噪音爆 10×）
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    # 5. uvicorn logger 接管 — 让 access log 跟业务 log 同格式同 trace_id
    #    (B audit 跨问题 C / A round2 §4.1 盲点 2)
    for uname in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        u = logging.getLogger(uname)
        u.handlers = []        # 清掉 uvicorn 自带 StreamHandler
        u.propagate = True     # 让 root handler 接管

    # 6. 未捕获异常路由进 logger
    _install_excepthooks()


def _resolve_log_dir() -> Path:
    """优先 ANIMA_LOG_DIR env，否则 paths.LOGS_DIR。

    pytest conftest 设 ANIMA_LOG_DIR=tmp_path_factory 隔离测试不写 repo studio_data/。
    """
    env = os.environ.get("ANIMA_LOG_DIR", "").strip()
    if env:
        return Path(env)
    return LOGS_DIR


def _build_console_handler(console: str | bool, process: str) -> logging.Handler | None:
    if console is False:
        return None
    if console == "auto":
        use_json = not sys.stderr.isatty()
    elif console == "json":
        use_json = True
    else:  # True 或其它
        use_json = False
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonLineFormatter(process) if use_json else HumanConsoleFormatter())
    return h


_EXCEPTHOOKS_INSTALLED = False


def _install_excepthooks() -> None:
    """注入 sys.excepthook + threading.excepthook → logger.critical。

    幂等（多次 setup_logging 调用只装一次）。
    """
    global _EXCEPTHOOKS_INSTALLED
    if _EXCEPTHOOKS_INSTALLED:
        return
    _EXCEPTHOOKS_INSTALLED = True

    _orig_excepthook = sys.excepthook

    def _excepthook(etype, evalue, etb):
        # KeyboardInterrupt 仍走默认，避免吞 Ctrl+C
        if issubclass(etype, KeyboardInterrupt):
            _orig_excepthook(etype, evalue, etb)
            return
        logging.getLogger("studio.unhandled").critical(
            "unhandled exception in main thread", exc_info=(etype, evalue, etb)
        )

    sys.excepthook = _excepthook

    # Python 3.8+ 有 threading.excepthook
    try:
        import threading
        _orig_thread_hook = threading.excepthook

        def _thread_excepthook(args: Any) -> None:
            logging.getLogger("studio.unhandled").critical(
                "unhandled exception in thread %s", args.thread.name if args.thread else "?",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
        threading.excepthook = _thread_excepthook
    except (AttributeError, ImportError):
        pass


def _reset_for_tests() -> None:
    """测试钩子：清 sentinel + 清 ContextVar，让多个测试可独立 setup_logging。

    生产代码不应该调。fixture 用：
        from studio.infrastructure.logging import _reset_for_tests
        _reset_for_tests()
    """
    global _EXCEPTHOOKS_INSTALLED
    _CONFIGURED_PROCESSES.clear()
    _EXCEPTHOOKS_INSTALLED = False
    # ContextVar 在测试间 leaky（同 thread / async ctx），显式清掉
    _trace_id_var.set(None)
    _job_id_var.set(None)
    _task_id_var.set(None)
    # 不还原 sys.excepthook（测试间也不应该依赖那个）


# 编码兜底是无条件常量，导出给 workers/_base.py 旧 import 兼容
__all__ = [
    "STUDIO_LOG_NAME", "STUDIO_LOG_MAX_BYTES", "STUDIO_LOG_BACKUP_COUNT",
    "TRACE_HEADER", "TRACE_ENV", "PROCESS_ENV",
    "JsonLineFormatter", "HumanConsoleFormatter", "ContextFilter",
    "make_studio_log_handler", "setup_logging", "reconfigure_console_utf8",
    "new_trace_id", "bind_trace_id", "reset_trace_id", "get_trace_id",
    "bind_job_id", "bind_task_id", "get_job_id", "get_task_id",
    "_reset_for_tests",
]
