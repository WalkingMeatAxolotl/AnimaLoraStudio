"""任务日志读取（PR-6 commit 1 从 server.py 抽出）。

1 route：
    GET /api/logs/{task_id}    读 tasks/<id>/run.log（老 task fallback
                               LOGS_DIR/<id>.log），去掉 worker EVENT 行
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from ...paths import LOGS_DIR, task_log_path

router = APIRouter()


def read_task_log(task_id: int) -> str:
    """task 全量日志文本（剥掉 __EVENT__: 协议行）；不存在返回 ""。

    新 task 走 tasks/<id>/run.log；老 task 在 studio_data/logs/<id>.log，
    不写迁移脚本（DB 里也没记 log 路径，看哪个存在），按存在性 fallback。
    其他 router（training.py 的 reg 先验回放端点）也复用这个 helper。
    """
    p = task_log_path(task_id)
    if not p.exists():
        p = LOGS_DIR / f"{task_id}.log"
    if not p.exists():
        return ""
    raw = p.read_text(encoding="utf-8", errors="replace")
    return "".join(
        ln for ln in raw.splitlines(keepends=True)
        if not ln.startswith("__EVENT__:")
    )


@router.get("/api/logs/{task_id}")
def get_log(task_id: int) -> dict[str, Any]:
    text = read_task_log(task_id)
    return {"task_id": task_id, "content": text, "size": len(text.encode("utf-8"))}
