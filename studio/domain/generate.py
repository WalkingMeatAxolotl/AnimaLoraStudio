"""测试出图 schema —— 对应 runtime/anima_generate.py 的 JSON 配置。

LoRA 加载走 inference_core.apply_loras —— 每份 LoRA 独立 inject，
rank/alpha 从 ss_network_args 读，正确合并多 LoRA。

注意：不使用 `from __future__ import annotations`——Pydantic v2 + Python 3.12+
在延迟求值模式下会将 typing._SpecialForm 当成 schema key，触发 AttributeError。
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import FAMILY_SAMPLING, AttentionBackend
from .lora import LoraEntry
from .xy_matrix import XYMatrixSpec, _check_axis_values


def validate_sampling_for_family(
    family: str, sampler_name: str, scheduler: str,
) -> None:
    """generate / reg_ai 的 sampler 按族严格校验（多模型 P4-4）。

    与训练 config 不同，这两份 config 是每任务临时 JSON——没有 legacy 语料，
    不需要 grandfather：越族值直接报错 + 可操作文案（两族 sample_image 都在
    入口对白名单外的值 raise，这里提前到 422）。
    """
    allowed = FAMILY_SAMPLING.get(family)
    if allowed is None:
        return  # 未知族由 model_family 的 Literal 校验报错
    for label, value, kind in (
        ("sampler_name", sampler_name, "samplers"),
        ("scheduler", scheduler, "schedulers"),
    ):
        if value not in allowed[kind]:
            raise ValueError(
                f"{label}='{value}' 不适用于 model_family='{family}'"
                f"（该族可用：{'、'.join(allowed[kind])}）"
            )


class GenerateConfig(BaseModel):
    """测试出图任务参数。对应 runtime/anima_generate.py 的 JSON 配置。

    LoRA 加载走 inference_core.apply_loras —— 每份 LoRA 独立 inject，
    rank/alpha 从 ss_network_args 读，正确合并多 LoRA。
    """

    model_config = ConfigDict(extra="forbid")

    # 模型族（服务端从请求填充；daemon 按此派发加载与采样栈）
    model_family: Literal["anima", "krea2"] = Field("anima")
    # 蒸馏推理底模（Krea2 Turbo：8 步 / guidance 0 / mu 固定 1.15）。
    # 服务端按 catalog variant purpose 检测官方 Turbo 路径后注入
    distilled: bool = Field(False)

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
    sampler_name: Literal["er_sde", "dpmpp_3m_sde", "euler"] = Field("er_sde")
    scheduler: Literal["simple", "sgm_uniform", "krea2_shift"] = Field("simple")
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
    vae_precision: Literal["bf16", "fp32"] = Field(
        "bf16",
        description="VAE decode 精度：bf16 对齐 ComfyUI 现代 GPU 默认；fp32 全精度（decode 前会 offload 腾显存）",
    )
    vae_tiling: Literal["auto", "on", "off"] = Field(
        "auto",
        description="VAE 分块 decode：auto=可用显存紧张时自动分块（推荐）；on=始终分块（省显存、慢约 30%）；"
                    "off=整图，仅真正 OOM 时回退。fp32 / 高分辨率在大显存卡上整图 decode 会逼近占满显存、"
                    "触发系统内存回退导致单次 decode 卡上百秒，auto 可避免",
    )
    attention_backend: AttentionBackend = Field(
        "flash_attn",
        description="Attention backend：none（SDPA）/ xformers / flash_attn",
    )
    vram_policy: Literal["auto", "save_vram", "performance"] = Field(
        "auto",
        description="显存策略（krea2 生效）：auto=按空闲显存决定文本编码器与 DiT 是否让位（推荐）；"
                    "save_vram=强制顺序化，峰值最低（16GB 卡可跑 fp8）、每图多几秒 CPU↔GPU 搬运；"
                    "performance=全部常驻显存，峰值最高、零搬运",
    )

    @model_validator(mode="after")
    def _validate_sampler_family(self) -> "GenerateConfig":
        validate_sampling_for_family(
            self.model_family, self.sampler_name, self.scheduler)
        return self

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
