"""测试出图的 server 进程内存缓存（commit 10）。

设计：
  - daemon 把 PNG bytes 通过 stdout JSON 推回（base64 编码）
  - InferenceDaemon reader 解码后调 cache_image(task_id, filename, bytes)
  - HTTP `GET /api/generate/{tid}/sample/{fn}` 从这里取，不走磁盘
  - 关 server / 重启 → 内存自动没；强杀也不残留

LRU / 客户端断连清理 / lifespan 钩子在 commit 11 加。本 commit 只提供
基础读写接口和按 task_id 批量删（让 supervisor 在 task 终止时主动清）。
"""
from __future__ import annotations

import threading
from typing import Optional

# (task_id, filename) → PNG bytes
_CACHE: dict[tuple[int, str], bytes] = {}
_LOCK = threading.RLock()


def cache_image(task_id: int, filename: str, data: bytes) -> None:
    """daemon image_done 时调用。同 task_id+filename 重复则覆盖。"""
    with _LOCK:
        _CACHE[(task_id, filename)] = data


def get_image(task_id: int, filename: str) -> Optional[bytes]:
    """HTTP 拉图调用。命中返回 bytes，未命中返回 None（→ 404）。"""
    with _LOCK:
        return _CACHE.get((task_id, filename))


def list_filenames(task_id: int) -> list[str]:
    """列出该 task 当前在 cache 里的全部 filename（按字母序）。"""
    with _LOCK:
        return sorted(fn for (tid, fn) in _CACHE if tid == task_id)


def drop_task(task_id: int) -> int:
    """删该 task 的全部 cache 条目；返回删了多少条。"""
    with _LOCK:
        keys = [k for k in _CACHE if k[0] == task_id]
        for k in keys:
            del _CACHE[k]
        return len(keys)


def total_count() -> int:
    """当前 cache 里图片数量（不含 task 数）。"""
    with _LOCK:
        return len(_CACHE)


def total_bytes() -> int:
    """当前 cache 占字节数（commit 11 LRU 用）。"""
    with _LOCK:
        return sum(len(v) for v in _CACHE.values())


def clear_all() -> None:
    """server lifespan shutdown 调；测试也用。"""
    with _LOCK:
        _CACHE.clear()
