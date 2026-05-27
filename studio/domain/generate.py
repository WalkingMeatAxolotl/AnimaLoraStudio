"""测试出图 schema —— 对应 runtime/anima_generate.py 的 JSON 配置。

LoRA 加载走 inference_core.apply_loras —— 每份 LoRA 独立 inject，
rank/alpha 从 ss_network_args 读，正确合并多 LoRA。

注意：不使用 `from __future__ import annotations`——Pydantic v2 + Python 3.12+
在延迟求值模式下会将 typing._SpecialForm 当成 schema key，触发 AttributeError。
"""
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import AttentionBackend
from .lora import LoraEntry
from .migrations import migrate_legacy_attention
from .xy_matrix import XYMatrixSpec, _check_axis_values


class GenerateConfig(BaseModel):
    """测试出图任务参数。对应 runtime/anima_generate.py 的 JSON 配置。

    LoRA 加载走 inference_core.apply_loras —— 每份 LoRA 独立 inject，
    rank/alpha 从 ss_network_args 读，正确合并多 LoRA。
    """

    model_config = ConfigDict(extra="forbid")

    # 模型路径（服务端从 secrets 填充）
    transformer_path: str = Field("models/diffusion_models/anima-base-v1.0.safetensors")
    vae_path: str = Field("models/vae/qwen_image_vae.safetensors")
    text_encoder_path: str = Field("models/text_encoders")
    t5_tokenizer_path: str = Field("models/t5_tokenizer")

    # 生成参数
    prompts: list[str] = Field(
        default_factory=lambda: ["newest, safe, 1girl, masterpiece, best quality"],
        description="正向提示词列表（每条 prompt 生成 count 张）",
    )
    negative_prompt: str = Field("")
    width: int = Field(1024, ge=256, le=4096)
    height: int = Field(1024, ge=256, le=4096)
    steps: int = Field(25, ge=1, le=150)
    cfg_scale: float = Field(4.0, ge=0.0, le=20.0)
    sampler_name: str = Field("er_sde")
    scheduler: str = Field("simple")
    count: int = Field(1, ge=1, le=32, description="每个 prompt 生成张数")
    seed: int = Field(0, description="随机种子（0=随机）")

    # LoRA（多 LoRA 独立 inject + multiplier=scale 控贡献权重）
    lora_configs: list[LoraEntry] = Field(
        default_factory=list,
        description="LoRA 列表（每份独立 inject，multiplier=scale）",
    )

    # XY 矩阵（None=普通单图模式；设了就 anima_generate.py 走 XY 循环分支）
    xy_matrix: Optional[XYMatrixSpec] = Field(
        None,
        description="XY 矩阵参数；设值时 prompts 限单条、count=1（避免排列爆炸）",
    )

    # 运行时
    output_dir: str = Field("", description="输出目录（服务端填 tempdir，task 结束清）")
    mixed_precision: str = Field("bf16")
    attention_backend: AttentionBackend = Field(
        "flash_attn",
        description="Attention backend：none（SDPA）/ xformers / flash_attn",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_attention(cls, data: Any) -> Any:
        return migrate_legacy_attention(data)

    @model_validator(mode="after")
    def _validate_xy(self) -> "GenerateConfig":
        """XY 与 prompts/count 互斥；axis lora_index 必须指向已存在的 lora_configs。"""
        if self.xy_matrix is None:
            return self
        if len(self.prompts) > 1:
            raise ValueError("xy_matrix 与多 prompt 互斥（排列爆炸）—— 单 prompt 时才能开 XY")
        if self.count != 1:
            raise ValueError("xy_matrix 与 count>1 互斥 —— XY 模式下每个 (x,y) 出 1 张")
        for label, axis in (("x", self.xy_matrix.x), ("y", self.xy_matrix.y)):
            if axis is None:
                continue
            _check_axis_values(axis)
            if axis.lora_index is not None and axis.lora_index >= len(self.lora_configs):
                raise ValueError(
                    f"xy_matrix.{label}.lora_index={axis.lora_index} 越界（仅 "
                    f"{len(self.lora_configs)} 个 lora_configs）"
                )
        return self
