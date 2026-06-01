"""先验生成 schema —— 对应 runtime/anima_reg_ai.py 的 JSON 配置。

设计来自 DreamBooth prior preservation：训练损失同时见到「LoRA 学到的样子」和
「base 模型本来的样子」，让 LoRA 只学差异。**不带 LoRA** —— 出现 LoRA
反而会把要保留的 prior 给覆盖了。

注意：不使用 `from __future__ import annotations`——Pydantic v2 + Python 3.12+
在延迟求值模式下会将 typing._SpecialForm 当成 schema key，触发 AttributeError。
"""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import AttentionBackend


class RegAiConfig(BaseModel):
    """先验生成的 JSON 配置（对应 runtime/anima_reg_ai.py）。

    设计来自 DreamBooth prior preservation：训练损失同时见到「LoRA 学到的样子」和
    「base 模型本来的样子」，让 LoRA 只学差异。**不带 LoRA** —— 出现 LoRA
    反而会把要保留的 prior 给覆盖了。
    """

    model_config = ConfigDict(extra="forbid")

    # 模型路径（服务端从 secrets 填充）
    transformer_path: str = Field("")
    vae_path: str = Field("")
    text_encoder_path: str = Field("")
    t5_tokenizer_path: str = Field("")

    # 数据目录（服务端填充）
    train_dir: str = Field("")
    reg_dir: str = Field("")

    # 生成控制
    excluded_tags: list[str] = Field(
        default_factory=list,
        description="排除的 tag（不参与 prompt 拼接）",
    )
    negative_prompt: str = Field("")
    width: int = Field(1024, ge=256, le=4096)
    height: int = Field(1024, ge=256, le=4096)
    steps: int = Field(25, ge=1, le=150)
    cfg_scale: float = Field(4.0, ge=0.0, le=20.0)
    sampler_name: str = Field("er_sde")
    scheduler: str = Field("simple")
    seed: int = Field(0, description="随机种子（0=随机）")
    incremental: bool = Field(
        False,
        description="补足模式：跳过 reg 子文件夹中已有以 train_stem 开头的图（重启续跑用）",
    )
    mixed_precision: str = Field("bf16")
    attention_backend: AttentionBackend = Field(
        "flash_attn",
        description="Attention backend：none（SDPA）/ xformers / flash_attn",
    )
