"""/api/queue 请求 BaseModel（PR-6 commit 6 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class EnqueueRequest(BaseModel):
    config_name: str
    name: Optional[str] = None
    priority: int = 0


class ReorderRequest(BaseModel):
    ordered_ids: list[int]


class ImportRequest(BaseModel):
    payload: dict[str, Any]


class ExportOutputsBody(BaseModel):
    files: Optional[list[str]] = None
