"""日志风暴节流小工具（tmp/log-text-audit/verdicts-b-subprocess.md §4 方案 A/B/C-2）。

子进程域里「循环内每张图/每个参数张量打一条」的告警行，正常路径 0 条、
故障路径直接是数据集规模（1000 张图 = 1000 条 WARNING），会把 run.log
冲成噪音。三个类分别对应设计稿的三种收法：

- :class:`RepeatThrottle`（方案 A）——同因重复告警：首条全文 WARNING（可带
  traceback），2..N 条降 DEBUG，任务收尾 ``drain()`` 补一条计数汇总。
- :class:`ProgressThrottle`（方案 B）——进度行：每 ``max(1, total // 100)``
  条或每 1.5s 一条，首末强制发。
- :class:`BackoffThrottle`（方案 C-2）——自治指数退避：第 1、10、100… 次
  报，其余静默，不需要外部收尾配合。

三个类都只依赖 ``TaskLogLike`` 的 ``debug/warning`` 形状（``logging.Logger``
天然满足），因此 runtime 入口、utils、workers 可以共用。
"""
from __future__ import annotations

import time
from typing import Any, Optional, Protocol


class _LogLike(Protocol):
    def debug(self, msg: str, *args: Any) -> None: ...

    def warning(self, msg: str, *args: Any, exc_info: bool = False) -> None: ...


class RepeatThrottle:
    """方案 A：首条全文 + 中间 DEBUG + 收尾计数汇总。

    ``key`` 是「同一个原因」的分组键（例如 ``"no_caption"``），同 key 的第 2
    条起只落 DEBUG。``summary`` 是收尾汇总的惰性格式串：带 ``first`` 样例时
    按 ``(count, first)`` 两个位置参数渲染，否则只 ``(count,)``。

    汇总只在 count >= 2 时发——只出现一次的告警，首条就是全部信息。
    """

    def __init__(self, log: _LogLike) -> None:
        self._log = log
        self._counts: dict[str, int] = {}
        self._summaries: dict[str, str] = {}
        self._firsts: dict[str, Optional[Any]] = {}

    def hit(
        self,
        key: str,
        summary: str,
        msg: str,
        *args: Any,
        first: Optional[Any] = None,
        exc_info: bool = False,
    ) -> None:
        n = self._counts.get(key, 0) + 1
        self._counts[key] = n
        if n == 1:
            self._summaries[key] = summary
            self._firsts[key] = first
            self._log.warning(msg, *args, exc_info=exc_info)
        else:
            self._log.debug(msg, *args)

    def count(self, key: str) -> int:
        return self._counts.get(key, 0)

    def drain(self) -> None:
        """任务收尾调用一次（放 ``finally`` 里），发汇总并清空状态。"""
        for key, n in self._counts.items():
            if n < 2:
                continue
            first = self._firsts.get(key)
            if first is None:
                self._log.warning(self._summaries[key], n)
            else:
                self._log.warning(self._summaries[key], n, first)
        self._counts.clear()
        self._summaries.clear()
        self._firsts.clear()


class ProgressThrottle:
    """方案 B：进度行按间隔收（D3 豁免级别不豁免量）。

    ``should_emit(done)`` 为 True 时调用方才发那条进度 INFO。首条（done<=1）
    与末条（done>=total）强制发，中间每 ``max(1, total // 100)`` 条或每
    ``min_interval`` 秒一条。
    """

    def __init__(self, total: int, min_interval: float = 1.5) -> None:
        self._total = max(int(total or 0), 0)
        self._step = max(1, self._total // 100)
        self._interval = min_interval
        self._last = 0.0

    def should_emit(self, done: int) -> bool:
        now = time.monotonic()
        if done <= 1 or (self._total and done >= self._total):
            self._last = now
            return True
        if done % self._step == 0 or (now - self._last) >= self._interval:
            self._last = now
            return True
        return False


class BackoffThrottle:
    """方案 C-2：指数退避，无需收尾配合。

    ``tick()`` 返回 ``(count, kind)``，``kind`` ∈ ``{"first", "milestone", ""}``：
    第 1 次 ``first``（调用方发全文 WARNING），第 10/100/1000… 次
    ``milestone``（发累计汇总 WARNING），其余空串（调用方只 DEBUG）。
    """

    def __init__(self, factor: int = 10) -> None:
        self._factor = max(2, int(factor))
        self._count = 0
        self._next = self._factor

    def tick(self) -> tuple[int, str]:
        self._count += 1
        if self._count == 1:
            return self._count, "first"
        if self._count >= self._next:
            self._next *= self._factor
            return self._count, "milestone"
        return self._count, ""

    @property
    def count(self) -> int:
        return self._count
