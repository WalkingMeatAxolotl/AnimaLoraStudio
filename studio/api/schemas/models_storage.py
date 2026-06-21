"""模型根目录迁移请求模型。"""
from __future__ import annotations

from pydantic import BaseModel


class ModelsRootMigrateRequest(BaseModel):
    """迁移目标父目录（绝对路径；数据落 `target/models/`，目标不要求为空）。"""
    target: str
