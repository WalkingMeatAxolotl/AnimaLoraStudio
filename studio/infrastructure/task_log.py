"""级别化的任务日志通道（设计稿 tmp/log-text-audit/leveling-rules.md R7）。

worker 与 services 之间传递的进度回调从裸 ``Callable[[str], None]`` 收编为
本模块的对象：旧签名 ``fn(line)`` 仍然兼容（落 INFO），新代码用
``.debug/.info/.warning/.error`` 把级别交给底下的 logging 体系——级别到
着色、显示过滤、``error_msg`` ERROR 块提取都由行契约接管，不再靠
``[error]`` / ``⚠`` 之类的伪级别前缀表达严重度。
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class TaskLogLike(Protocol):
    """进度/日志回调的协议类型。

    ``__call__(line)`` 与历史 ``Callable[[str], None]`` 签名兼容（语义 INFO）；
    级别方法与 ``logging.Logger`` 同形（``msg, *args`` 惰性格式化）。
    """

    def __call__(self, line: str) -> None: ...

    def debug(self, msg: str, *args: Any) -> None: ...

    def info(self, msg: str, *args: Any) -> None: ...

    def warning(self, msg: str, *args: Any, exc_info: bool = False) -> None: ...

    def error(self, msg: str, *args: Any, exc_info: bool = False) -> None: ...


class TaskLog:
    """标准实现：级别原样转发给一个 ``logging.Logger``。

    worker 进程里它落 stderr → run.log，行契约（级别/续行/着色）由
    ``setup_logging`` 的 formatter 保证。
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def __call__(self, line: str) -> None:
        self._logger.info(line)

    def debug(self, msg: str, *args: Any) -> None:
        self._logger.debug(msg, *args)

    def info(self, msg: str, *args: Any) -> None:
        self._logger.info(msg, *args)

    def warning(self, msg: str, *args: Any, exc_info: bool = False) -> None:
        self._logger.warning(msg, *args, exc_info=exc_info)

    def error(self, msg: str, *args: Any, exc_info: bool = False) -> None:
        self._logger.error(msg, *args, exc_info=exc_info)


class CallbackTaskLog:
    """把任意 line-callback 适配成 :class:`TaskLogLike`。

    级别信息在回调侧丢失（回调只收文本），仅用于旧接口过渡与静默场景；
    exc_info 无处附着，忽略。
    """

    def __init__(self, fn: Callable[[str], None]) -> None:
        self._fn = fn

    def _emit(self, msg: str, args: tuple[Any, ...]) -> None:
        self._fn(msg % args if args else msg)

    def __call__(self, line: str) -> None:
        self._fn(line)

    def debug(self, msg: str, *args: Any) -> None:
        self._emit(msg, args)

    def info(self, msg: str, *args: Any) -> None:
        self._emit(msg, args)

    def warning(self, msg: str, *args: Any, exc_info: bool = False) -> None:
        self._emit(msg, args)

    def error(self, msg: str, *args: Any, exc_info: bool = False) -> None:
        self._emit(msg, args)


#: 丢弃一切输出的空实现（库函数默认值用，替代 ``lambda _l: None``）。
NULL_LOG = CallbackTaskLog(lambda _l: None)
