"""模型根目录迁移请求模型。"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class ModelsRootMigrateRequest(BaseModel):
    """迁移目标父目录（绝对路径；数据落 `target/models/`，目标不要求为空）。

    on_conflict：`target/models` 已有数据时的合并策略。不传 → 后端返回 409
    `models_root.target_conflict`（前端据此弹「跳过/覆盖/取消」三选后重发）；
    "skip" → 只复制目标缺少的文件；"overwrite" → 同名文件以当前根的副本覆盖。
    """
    target: str
    on_conflict: Optional[Literal["skip", "overwrite"]] = None
