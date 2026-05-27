"""LoRA 加载参数（被 GenerateConfig 等共用）。

注意：不使用 `from __future__ import annotations`——Pydantic v2 + Python 3.12+
在延迟求值模式下会将 typing._SpecialForm 当成 schema key，触发 AttributeError。
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LoraEntry(BaseModel):
    """单个 LoRA 的加载参数。Generate / API 共享，避免 server.py 私有定义。"""

    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="LoRA safetensors 绝对路径")
    scale: float = Field(1.0, description="贡献权重（multiplier），多 LoRA 各自独立")
    # 关联到的 version（picker 选的；外部文件无）；前端用 vid 拉 ckpt 列表
    project_id: Optional[int] = Field(None, ge=1)
    version_id: Optional[int] = Field(None, ge=1)
