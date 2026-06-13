"""studio_data 迁移请求模型。"""
from __future__ import annotations

from pydantic import BaseModel


class StudioDataMigrateRequest(BaseModel):
    """迁移目标父目录（绝对路径；数据落 `target/studio_data/`，目标不要求为空）。"""
    target: str
