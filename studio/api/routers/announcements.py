"""公告栏数据端点（Phase 1）。设计见 docs/todo/announcement-center.md。

1 route：
    GET /api/announcements    docs/announcements/ 解析出的结构化双语 post 列表
                              （pin 优先 → date 降序）；read 状态在前端 localStorage。
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter

from ...services import announcements as announcements_svc

router = APIRouter()


@router.get("/api/announcements")
def list_announcements() -> dict[str, Any]:
    """返回全部公告 post（双语）。前端按当前语言取 title/body，缺 en → zh。"""
    return {"posts": [asdict(p) for p in announcements_svc.list_posts()]}
