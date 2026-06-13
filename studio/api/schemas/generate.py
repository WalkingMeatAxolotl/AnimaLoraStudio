"""/api/generate 请求 BaseModel（PR-6 commit 5 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel

from ...domain import AttentionBackend, LoraEntry, XYMatrixSpec


class GenerateRequest(BaseModel):
    prompts: list[str] = ["newest, safe, 1girl, masterpiece, best quality"]
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg_scale: float = 4.0
    sampler_name: Literal["er_sde", "dpmpp_3m_sde"] = "er_sde"
    scheduler: Literal["simple", "sgm_uniform"] = "simple"
    count: int = 1
    seed: int = 0
    lora_configs: list[LoraEntry] = []
    mixed_precision: str = "bf16"
    # commit C：attention_backend 默认从 secrets.generate.attention_backend 读，
    # 前端 Generate 页不再发这个字段；保留 Optional 兼容老客户端 / 临时覆盖。
    attention_backend: Optional[AttentionBackend] = None
    # XY 矩阵：None=单图模式；设值时 schema 强制 prompts 单条 + count=1
    xy_matrix: Optional[XYMatrixSpec] = None
    # 前端构造的 GenerateParamsSnapshot dict（prefs 视图：含 prompts/loras/
    # xy_draft/dataset_pick 等），server 不解释结构、原样透传到 daemon →
    # image_done 时塞进加密 cache payload header。/api/generate/cache/index
    # 时返还前端，作为 CacheEntry.params 回填用。save_test_images=true 走
    # 落盘分支也共用这份 snapshot 写入 PNG anima_params metadata。
    params_snapshot: Optional[dict[str, Any]] = None
