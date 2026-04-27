"""训练参数 schema —— pydantic v2 单一权威源。

后续：
    - argparse 由 studio.argparse_bridge 反向生成（P2-B）
    - 前端表单读取 /api/schema 自动渲染
    - YAML 配置用 TrainingConfig.model_validate(yaml_dict) 校验

约定：每个字段通过 `json_schema_extra={"group", "control", "show_when"?}`
携带 UI 元信息。前端按 `group` 分区，按 `show_when` 做条件显示。
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def _meta(group: str, control: str = "auto", **extra: Any) -> dict[str, Any]:
    return {"group": group, "control": control, **extra}


class TrainingConfig(BaseModel):
    """与 config/train_template.yaml 对齐的完整训练参数。

    `extra="forbid"`：未知键直接报错，避免拼写错误悄悄失效。
    """

    model_config = ConfigDict(extra="forbid")

    # ---------------------------------------------------------------- 模型路径
    transformer_path: str = Field(
        "models/transformers/anima-preview.safetensors",
        description="主 transformer 权重 (.safetensors)",
        json_schema_extra=_meta("model", "path"),
    )
    vae_path: str = Field(
        "models/vae/qwen_image_vae.safetensors",
        description="VAE 权重",
        json_schema_extra=_meta("model", "path"),
    )
    text_encoder_path: str = Field(
        "models/text_encoders",
        description="Qwen 文本编码器目录",
        json_schema_extra=_meta("model", "path"),
    )
    t5_tokenizer_path: str = Field(
        "models/t5_tokenizer",
        description="T5 tokenizer 目录",
        json_schema_extra=_meta("model", "path"),
    )

    # ----------------------------------------------------------------- 数据集
    data_dir: str = Field(
        "./dataset",
        description="数据集目录（支持 Kohya 风格 N_xxx 子目录设定 repeat）",
        json_schema_extra=_meta("dataset", "path"),
    )
    resolution: int = Field(
        1024, ge=256, le=4096,
        description="训练分辨率",
        json_schema_extra=_meta("dataset"),
    )
    reg_data_dir: Optional[str] = Field(
        None,
        description="正则集目录（可选，防过拟合）",
        json_schema_extra=_meta("dataset", "path"),
    )
    reg_caption: Optional[str] = Field(
        None,
        description="正则集统一 caption（留空则使用各图自带的 .txt/.json）",
        json_schema_extra=_meta("dataset"),
    )
    reg_weight: float = Field(
        1.0, ge=0.0, le=1.0,
        description="正则集 loss 权重",
        json_schema_extra=_meta("dataset"),
    )

    # -------------------------------------------------------------- Caption
    shuffle_caption: bool = Field(
        True,
        description="启用标签打乱（JSON 模式分类内打乱，TXT 模式全部打乱）",
        json_schema_extra=_meta("caption"),
    )
    keep_tokens: int = Field(
        0, ge=0,
        description="保护前 N 个标签不打乱（仅 TXT 模式）",
        json_schema_extra=_meta("caption"),
    )
    flip_augment: bool = Field(
        True,
        description="水平翻转增强",
        json_schema_extra=_meta("caption"),
    )
    tag_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="标签随机丢弃概率",
        json_schema_extra=_meta("caption"),
    )
    prefer_json: bool = Field(
        True,
        description="优先使用 JSON 标签文件（推荐，支持分类 shuffle）",
        json_schema_extra=_meta("caption"),
    )
    cache_latents: bool = Field(
        True,
        description="缓存 VAE latent 加速训练",
        json_schema_extra=_meta("caption"),
    )

    # ------------------------------------------------------------------- LoRA
    lora_type: Literal["lora", "lokr"] = Field(
        "lokr",
        description="适配器类型",
        json_schema_extra=_meta("lora"),
    )
    lora_rank: int = Field(
        32, ge=4, le=256,
        description="LoRA rank（推荐 8/16/32/64）",
        json_schema_extra=_meta("lora"),
    )
    lora_alpha: float = Field(
        32.0, ge=0.0,
        description="LoRA alpha（通常与 rank 相同）",
        json_schema_extra=_meta("lora"),
    )
    lokr_factor: int = Field(
        8, ge=2,
        description="LoKr 分解因子（仅 lora_type=lokr）",
        json_schema_extra=_meta("lora", show_when="lora_type==lokr"),
    )

    # ------------------------------------------------------------------ 训练
    epochs: int = Field(
        10, ge=1,
        description="训练轮数",
        json_schema_extra=_meta("training"),
    )
    max_steps: int = Field(
        0, ge=0,
        description="最大步数（0=不限）",
        json_schema_extra=_meta("training"),
    )
    batch_size: int = Field(
        1, ge=1,
        description="批次大小",
        json_schema_extra=_meta("training"),
    )
    grad_accum: int = Field(
        4, ge=1,
        description="梯度累积步数（有效 batch = batch_size × grad_accum）",
        json_schema_extra=_meta("training"),
    )
    learning_rate: float = Field(
        1e-4, gt=0.0,
        description="学习率（Prodigy 必须为 1.0）",
        json_schema_extra=_meta("training"),
    )
    lr_scheduler: Literal["none", "cosine", "cosine_with_restart"] = Field(
        "none",
        description="学习率调度",
        json_schema_extra=_meta("training"),
    )
    lr_scheduler_t0: int = Field(
        500, ge=1,
        description="cosine_with_restart 首次周期",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_with_restart"),
    )
    lr_scheduler_t_mult: float = Field(
        2.0, ge=1.0,
        description="cosine_with_restart 周期倍数",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_with_restart"),
    )
    lr_scheduler_eta_min: float = Field(
        1e-6, ge=0.0,
        description="最小学习率",
        json_schema_extra=_meta("training", show_when="lr_scheduler!=none"),
    )
    optimizer_type: Literal["adamw", "prodigy"] = Field(
        "adamw",
        description="优化器（prodigy 需 pip install prodigyopt）",
        json_schema_extra=_meta("training"),
    )
    prodigy_d_coef: float = Field(
        1.0, ge=0.1, le=10.0,
        description="Prodigy d 缩放系数（小数据集 0.5，过拟合 2.0）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy"),
    )
    prodigy_safeguard_warmup: bool = Field(
        True,
        description="Prodigy warmup 期间保护 d 增长",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy"),
    )
    weight_decay: float = Field(
        0.0, ge=0.0,
        description="权重衰减（0=禁用）",
        json_schema_extra=_meta("training"),
    )
    grad_clip_max_norm: float = Field(
        0.0, ge=0.0,
        description="梯度裁剪最大范数（0=禁用）",
        json_schema_extra=_meta("training"),
    )
    mixed_precision: Literal["bf16", "fp16", "no"] = Field(
        "bf16",
        description="混合精度",
        json_schema_extra=_meta("training"),
    )
    grad_checkpoint: bool = Field(
        True,
        description="梯度检查点（省显存）",
        json_schema_extra=_meta("training"),
    )
    xformers: bool = Field(
        False,
        description="xformers attention（5090 推荐 false）",
        json_schema_extra=_meta("training"),
    )
    num_workers: int = Field(
        0, ge=0,
        description="数据加载线程（Windows 必须 0）",
        json_schema_extra=_meta("training"),
    )

    # ---------------------------------------------------------------- 输出/保存
    output_dir: str = Field(
        "./output",
        description="输出目录",
        json_schema_extra=_meta("output", "path"),
    )
    output_name: str = Field(
        "anima_lora",
        description="输出文件名前缀",
        json_schema_extra=_meta("output"),
    )
    save_every: int = Field(
        0, ge=0,
        description="每 N epoch 保存（0=禁用）",
        json_schema_extra=_meta("output"),
    )
    save_every_steps: int = Field(
        500, ge=0,
        description="每 N step 保存（推荐）",
        json_schema_extra=_meta("output"),
    )
    save_state_every: int = Field(
        1000, ge=0,
        description="每 N step 保存完整训练状态（断点续训）",
        json_schema_extra=_meta("output"),
    )
    seed: int = Field(
        42,
        description="随机种子",
        json_schema_extra=_meta("output"),
    )
    resume_lora: Optional[str] = Field(
        None,
        description="从已有 LoRA 继续训练（仅加载权重）",
        json_schema_extra=_meta("output", "path"),
    )
    resume_state: Optional[str] = Field(
        None,
        description="从训练状态恢复（完整断点续训）",
        json_schema_extra=_meta("output", "path"),
    )

    # -------------------------------------------------------------------- 采样
    sample_every: int = Field(
        5, ge=0,
        description="每 N epoch 采样（0=禁用）",
        json_schema_extra=_meta("sample"),
    )
    sample_steps: int = Field(
        0, ge=0,
        description="每 N step 采样（0=禁用）",
        json_schema_extra=_meta("sample"),
    )
    sample_infer_steps: int = Field(
        25, ge=1,
        description="推理步数",
        json_schema_extra=_meta("sample"),
    )
    sample_cfg_scale: float = Field(
        4.0, ge=0.0,
        description="CFG Scale",
        json_schema_extra=_meta("sample"),
    )
    sample_sampler_name: str = Field(
        "er_sde",
        description="采样器",
        json_schema_extra=_meta("sample"),
    )
    sample_scheduler: str = Field(
        "simple",
        description="调度器",
        json_schema_extra=_meta("sample"),
    )
    sample_width: int = Field(
        0, ge=0,
        description="采样宽度（0=跟随 resolution）",
        json_schema_extra=_meta("sample"),
    )
    sample_height: int = Field(
        0, ge=0,
        description="采样高度（0=跟随 resolution）",
        json_schema_extra=_meta("sample"),
    )
    sample_seed: int = Field(
        0,
        description="采样种子（0=随机）",
        json_schema_extra=_meta("sample"),
    )
    sample_negative_prompt: str = Field(
        "",
        description="负面提示词",
        json_schema_extra=_meta("sample", "textarea"),
    )
    sample_prompt: str = Field(
        "newest, safe, 1girl, masterpiece, best quality",
        description="单 prompt 模式",
        json_schema_extra=_meta("sample", "textarea"),
    )
    sample_prompts: list[str] = Field(
        default_factory=list,
        description="多 prompt 轮换（优先于 sample_prompt）",
        json_schema_extra=_meta("sample", "string-list"),
    )

    # ---------------------------------------------------------------- 监控/进度
    loss_curve_steps: int = Field(
        100, ge=10,
        description="终端 loss 曲线显示最近 N 步",
        json_schema_extra=_meta("monitor"),
    )
    no_progress: bool = Field(
        False,
        description="禁用进度条",
        json_schema_extra=_meta("monitor"),
    )
    log_every: int = Field(
        10, ge=1,
        description="日志输出间隔（步）",
        json_schema_extra=_meta("monitor"),
    )
    no_monitor: bool = Field(
        False,
        description="禁用内置 Web 监控（Studio 模式应设 true）",
        json_schema_extra=_meta("monitor"),
    )
    monitor_host: str = Field(
        "127.0.0.1",
        description="监控绑定地址",
        json_schema_extra=_meta("monitor"),
    )
    monitor_port: int = Field(
        8765, ge=1, le=65535,
        description="监控端口",
        json_schema_extra=_meta("monitor"),
    )
    no_browser: bool = Field(
        False,
        description="不自动打开浏览器",
        json_schema_extra=_meta("monitor"),
    )


# ---------------------------------------------------------------------------
# 分组顺序（前端按这个顺序渲染区块）
# ---------------------------------------------------------------------------

GROUP_ORDER: list[tuple[str, str]] = [
    ("model", "模型路径"),
    ("dataset", "数据集"),
    ("caption", "Caption 处理"),
    ("lora", "LoRA / LoKr"),
    ("training", "训练参数"),
    ("output", "输出与保存"),
    ("sample", "采样"),
    ("monitor", "监控与进度"),
]
