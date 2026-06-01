"""测试出图的 server 进程内存缓存（commit 10 + LRU 兜底 commit 11）。

设计：
  - daemon 把 PNG bytes 通过 stdout JSON 推回（base64 编码）
  - InferenceDaemon reader 解码后调 cache_image(task_id, filename, bytes)
  - HTTP `GET /api/generate/{tid}/sample/{fn}` 从这里取，不走磁盘
  - 关 server / 重启 → 内存自动没；强杀也不残留

清理触发器（commit 11）：
  1. LRU：上限 200 张 / 500MB（取小先触发，按写入访问 order；read 也算）
  2. SSE 客户端断连 + 30s 缓冲（server.py lifespan 内挂 timer）
  3. lifespan shutdown → clear_all
  4. supervisor 主动 drop_task（task 失败/取消等场景，仍保留接口）
"""
from __future__ import annotations

import collections
import threading
from typing import Optional

# 默认上限：200 张 OR 500MB（取小先触发）。可通过 configure() 调。
DEFAULT_MAX_COUNT = 200
DEFAULT_MAX_BYTES = 500 * 1024 * 1024

# (task_id, filename) → PNG bytes；OrderedDict 维护访问顺序（最旧在头部）
_CACHE: "collections.OrderedDict[tuple[int, str], bytes]" = collections.OrderedDict()
_LOCK = threading.RLock()
_BYTES_TOTAL = 0  # 当前缓存总字节
_max_count = DEFAULT_MAX_COUNT
_max_bytes = DEFAULT_MAX_BYTES


def configure(*, max_count: Optional[int] = None, max_bytes: Optional[int] = None) -> None:
    """动态调上限（启动时或测试用）。改完立刻 enforce。"""
    global _max_count, _max_bytes
    with _LOCK:
        if max_count is not None:
            _max_count = int(max_count)
        if max_bytes is not None:
            _max_bytes = int(max_bytes)
        _enforce_limits_locked()


def cache_image(task_id: int, filename: str, data: bytes) -> None:
    """daemon image_done 时调用。同 task_id+filename 重复则覆盖；触发 LRU。"""
    global _BYTES_TOTAL
    with _LOCK:
        key = (task_id, filename)
        if key in _CACHE:
            _BYTES_TOTAL -= len(_CACHE[key])
            del _CACHE[key]
        _CACHE[key] = data
        _BYTES_TOTAL += len(data)
        _enforce_limits_locked()


def get_image(task_id: int, filename: str) -> Optional[bytes]:
    """HTTP 拉图。命中返回 bytes 并 move_to_end（LRU 看作最近使用）。"""
    with _LOCK:
        key = (task_id, filename)
        if key not in _CACHE:
            return None
        _CACHE.move_to_end(key)
        return _CACHE[key]


def list_filenames(task_id: int) -> list[str]:
    """列出该 task 当前在 cache 里的全部 filename（按字母序）。"""
    with _LOCK:
        return sorted(fn for (tid, fn) in _CACHE if tid == task_id)


def drop_task(task_id: int) -> int:
    """删该 task 的全部 cache 条目；返回删了多少条。"""
    global _BYTES_TOTAL
    with _LOCK:
        keys = [k for k in _CACHE if k[0] == task_id]
        for k in keys:
            _BYTES_TOTAL -= len(_CACHE[k])
            del _CACHE[k]
        return len(keys)


def total_count() -> int:
    """当前 cache 里图片数量（不含 task 数）。"""
    with _LOCK:
        return len(_CACHE)


def total_bytes() -> int:
    """当前 cache 占字节数。"""
    with _LOCK:
        return _BYTES_TOTAL


def clear_all() -> None:
    """server lifespan shutdown 调；测试也用。"""
    global _BYTES_TOTAL
    with _LOCK:
        _CACHE.clear()
        _BYTES_TOTAL = 0


def _enforce_limits_locked() -> None:
    """剔最旧条目直到满足 max_count 和 max_bytes。调用方持锁。"""
    global _BYTES_TOTAL
    while _CACHE and (len(_CACHE) > _max_count or _BYTES_TOTAL > _max_bytes):
        _, data = _CACHE.popitem(last=False)  # 最旧
        _BYTES_TOTAL -= len(data)
