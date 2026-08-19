"""任务日志读取（PR-6 commit 1 从 server.py 抽出）。

2 route（docs/design/logging-target-state.md §3.4）：
    GET /api/logs/{task_id}        分页读 tasks/<id>/run.log（老 task fallback
                                   LOGS_DIR/<id>.log），去掉 worker EVENT 行。
                                   三种模式：tail=N（默认，末 N 行）/
                                   before=<offset>&limit=N（往前翻）/
                                   after=<offset>&limit=N（断线补拉）
    GET /api/logs/{task_id}/raw    原始文件下载（诊断包 / 下载按钮）

行 offset = 该行在文件里的起始字节偏移；page.end_offset = 最后一行结束后的偏移
（= 断线补拉的 after 游标，与 SSE task_log_appended.end_offset 同一坐标系）。
按**字节**切行、逐行 clean_log_line 解码，与 LogTailer 给 SSE 的文本一致。
末尾没有换行的半行（子进程写到一半）不返回，游标停在它前面。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from ...domain.errors import NotFoundError, ValidationError
from ...infrastructure.log_tail import clean_log_line
from ...paths import LOGS_DIR, task_log_path

router = APIRouter()

_EVENT_PREFIX = b"__EVENT__:"
_READ_CHUNK = 64 * 1024
DEFAULT_TAIL_LINES = 500
MAX_PAGE_LINES = 5000


def _resolve_log_path(task_id: int) -> Path | None:
    """新 task 走 tasks/<id>/run.log；老 task 在 studio_data/logs/<id>.log，
    不写迁移脚本（DB 里也没记 log 路径，看哪个存在），按存在性 fallback。"""
    p = task_log_path(task_id)
    if p.exists():
        return p
    p = LOGS_DIR / f"{task_id}.log"
    return p if p.exists() else None


def read_task_log(task_id: int) -> str:
    """task 全量日志文本（剥掉 __EVENT__: 协议行）；不存在返回 ""。

    只给整段回放的老调用方（training.py 的 reg 先验回放端点）用；前端日志
    视图走分页的 read_task_log_page。
    """
    p = _resolve_log_path(task_id)
    if p is None:
        return ""
    raw = p.read_text(encoding="utf-8", errors="replace")
    return "".join(
        ln for ln in raw.splitlines(keepends=True)
        if not ln.startswith("__EVENT__:")
    )


def _empty_page(task_id: int) -> dict[str, Any]:
    return {
        "task_id": task_id, "lines": [], "start_offset": 0, "end_offset": 0,
        "size": 0, "has_more_before": False,
    }


def _split_complete_lines(buf: bytes, base: int) -> tuple[list[tuple[int, bytes]], int]:
    """buf 从文件偏移 base 起；返回 [(line_start_offset, line_bytes)] 与「完整行
    结束后的偏移」。末尾无换行的半行不算。"""
    out: list[tuple[int, bytes]] = []
    pos = base
    end = base
    for part in buf.split(b"\n")[:-1]:  # 最后一段是半行（或空串），丢
        out.append((pos, part))
        pos += len(part) + 1
        end = pos
    return out, end


def read_task_log_page(
    task_id: int,
    *,
    tail: int | None = None,
    before: int | None = None,
    after: int | None = None,
    limit: int = DEFAULT_TAIL_LINES,
) -> dict[str, Any]:
    """分页读 run.log。tail / before / after 三选一（都不给 = tail）。"""
    p = _resolve_log_path(task_id)
    if p is None:
        return _empty_page(task_id)
    limit = max(1, min(int(limit), MAX_PAGE_LINES))
    size = p.stat().st_size

    if after is not None:
        # 断线补拉：从 after 往后最多 limit 行；半行不返回
        start = max(0, min(int(after), size))
        with open(p, "rb") as f:
            f.seek(start)
            buf = f.read()
        raw_lines, end = _split_complete_lines(buf, start)
        kept = [(off, b) for off, b in raw_lines if not b.startswith(_EVENT_PREFIX)]
        if len(kept) > limit:
            kept = kept[:limit]
            last_off, last_b = kept[-1]
            end = last_off + len(last_b) + 1  # 游标停在最后一条返回行之后
        lines = [{"offset": off, "text": clean_log_line(b)} for off, b in kept]
        return {
            "task_id": task_id, "lines": lines,
            "start_offset": start, "end_offset": end, "size": size,
            "has_more_before": start > 0,
        }

    # tail / before：从 upper 往前按块读，直到凑够 limit 行（剥掉 EVENT 行后）或到文件头
    upper = size if before is None else max(0, min(int(before), size))
    lower = upper
    collected: list[tuple[int, bytes]] = []
    while lower > 0 and len(collected) < limit:
        new_lower = max(0, lower - _READ_CHUNK)
        with open(p, "rb") as f:
            f.seek(new_lower)
            buf = f.read(upper - new_lower)
        # 窗口不从 0 起时第一段可能是半行：丢掉，它属于更早的一块（下一轮的
        # upper 退到它的换行之后，届时它作为完整行被收进来）。整块没有换行
        # （单行 > 64KB）时整块都是半行，upper 不动、只把 lower 再往前推。
        parts = buf.split(b"\n")
        if new_lower == 0:
            first_partial = 0
        elif len(parts) > 1:
            first_partial = len(parts[0]) + 1
        else:
            first_partial = len(buf)
        window_lines, _ = _split_complete_lines(buf[first_partial:], new_lower + first_partial)
        # 窗口末尾落在文件末尾且最后一段无换行 → 半行已被 _split_complete_lines 丢掉
        collected = [(o, b) for o, b in window_lines if not b.startswith(_EVENT_PREFIX)] + collected
        lower = new_lower
        upper = new_lower + first_partial
    if len(collected) > limit:
        collected = collected[-limit:]
    lines = [{"offset": off, "text": clean_log_line(b)} for off, b in collected]
    start_offset = collected[0][0] if collected else (size if before is None else upper)
    if collected:
        last_off, last_b = collected[-1]
        end_offset = last_off + len(last_b) + 1
    else:
        end_offset = start_offset
    return {
        "task_id": task_id, "lines": lines,
        "start_offset": start_offset, "end_offset": end_offset, "size": size,
        "has_more_before": start_offset > 0,
    }


@router.get("/api/logs/{task_id}")
def get_log(
    task_id: int,
    tail: int | None = Query(None, ge=1),
    before: int | None = Query(None, ge=0),
    after: int | None = Query(None, ge=0),
    limit: int | None = Query(None, ge=1),
) -> dict[str, Any]:
    if sum(x is not None for x in (tail, before, after)) > 1:
        raise ValidationError(
            "tail / before / after are mutually exclusive", code="log.query_conflict",
        )
    lim = limit if limit is not None else (tail if tail is not None else DEFAULT_TAIL_LINES)
    return read_task_log_page(task_id, tail=tail, before=before, after=after, limit=lim)


@router.get("/api/logs/{task_id}/raw")
def get_log_raw(task_id: int) -> FileResponse:
    p = _resolve_log_path(task_id)
    if p is None:
        raise NotFoundError("Log not found", code="log.not_found", details={"task_id": task_id})
    return FileResponse(
        path=str(p), media_type="text/plain; charset=utf-8",
        filename=f"task-{task_id}.log",
    )
