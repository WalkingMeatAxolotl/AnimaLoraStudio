"""/api/queue 请求 BaseModel（PR-6 commit 6 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class EnqueueRequest(BaseModel):
    config_name: str
    name: Optional[str] = None
    priority: int = 0
    # 0.17 P-B：计划开始时间（unix 秒）。给了 → task 建成 scheduled，到点由
    # supervisor 提升为 pending；不给 → 立即 pending（原行为）。
    scheduled_at: Optional[float] = None


class ScheduleTrainingRequest(BaseModel):
    """POST /api/projects/{pid}/versions/{vid}/queue 可选 body（0.17 P-B）。"""
    scheduled_at: Optional[float] = None


class ReorderRequest(BaseModel):
    ordered_ids: list[int]


class ImportRequest(BaseModel):
    payload: dict[str, Any]


class ExportOutputsBody(BaseModel):
    files: Optional[list[str]] = None


class DeleteOutputsBody(BaseModel):
    files: list[str]
