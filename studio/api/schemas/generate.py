"""/api/generate 请求 BaseModel（PR-6 commit 5 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, model_validator

from ...domain import AttentionBackend, LoraEntry, XYMatrixSpec
from ...domain.migrations import migrate_legacy_attention


class GenerateRequest(BaseModel):
    prompts: list[str] = ["newest, safe, 1girl, masterpiece, best quality"]
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg_scale: float = 4.0
    sampler_name: str = "er_sde"
    scheduler: str = "simple"
    count: int = 1
    seed: int = 0
    lora_configs: list[LoraEntry] = []
    mixed_precision: str = "bf16"
    # commit C：attention_backend 默认从 secrets.generate.attention_backend 读，
    # 前端 Generate 页不再发这个字段；保留 Optional 兼容老客户端 / 临时覆盖。
    attention_backend: Optional[AttentionBackend] = None
    # XY 矩阵：None=单图模式；设值时 schema 强制 prompts 单条 + count=1
    xy_matrix: Optional[XYMatrixSpec] = None

    # 兼容老前端送 xformers / flash_attn 双 bool（自动映射成 attention_backend）
    @model_validator(mode="before")
    @classmethod
    def _migrate_attention(cls, data: Any) -> Any:
        return migrate_legacy_attention(data)
