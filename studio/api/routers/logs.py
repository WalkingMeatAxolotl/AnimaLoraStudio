"""任务日志读取（PR-6 commit 1 从 server.py 抽出）。

1 route：
    GET /api/logs/{task_id}    读 LOGS_DIR/{task_id}.log，去掉 worker EVENT 行
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ...paths import LOGS_DIR

router = APIRouter()


@router.get("/api/logs/{task_id}")
def get_log(task_id: int) -> dict[str, Any]:
    p = LOGS_DIR / f"{task_id}.log"
    if not p.exists():
        return {"task_id": task_id, "content": "", "size": 0}
    raw = p.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in raw.splitlines(keepends=True) if not ln.startswith("__EVENT__:")]
    text = "".join(lines)
    return {"task_id": task_id, "content": text, "size": len(text.encode("utf-8"))}
