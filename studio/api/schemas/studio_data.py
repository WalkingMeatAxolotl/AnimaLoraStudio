"""studio_data 迁移请求模型。"""
from __future__ import annotations

from pydantic import BaseModel


class StudioDataMigrateRequest(BaseModel):
    """迁移目标目录（绝对路径；空目录或不存在）。"""
    target: str
