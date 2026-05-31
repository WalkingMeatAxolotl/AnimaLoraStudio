"""主训练配置 schema —— pydantic v2 单一权威源。

与 config/train_template.yaml 对齐的完整训练参数。

后续：
    - argparse 由 studio.argparse_bridge 反向生成（P2-B）
    - 前端表单读取 /api/schema 自动渲染
    - YAML 配置用 TrainingConfig.model_validate(yaml_dict) 校验

约定：每个字段通过 `json_schema_extra={"group", "control", "show_when"?}`
携带 UI 元信息。前端按 `group` 分区，按 `show_when` 做条件显示。

注意：不使用 `from __future__ import annotations`——Pydantic v2 + Python 3.12+
在延迟求值模式下会将 typing._SpecialForm 当成 schema key，触发 AttributeError。
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import AttentionBackend, _meta
from .migrations import (
    migrate_legacy_attention,
    migrate_legacy_save_keys,
    migrate_noise_enhancement_type,
)


class TrainingConfig(BaseModel):
    """与 config/train_template.yaml 对齐的完整训练参数。

    `extra="forbid"`：未知键直接报错，避免拼写错误悄悄失效。
    """

    model_config = ConfigDict(extra="forbid")

    # ---------------------------------------------------------------- 模型路径
    # 这些路径在 Studio 创建 version 时会被替换成 **绝对路径**（基于
    # secrets.models.root + secrets.models.selected_anima），用户在 yaml /
    # Train 页看到的总是无歧义的绝对路径，不用考虑相对路径锚定。
    # 这里的默认值仅 fallback：裸 CLI 跑训练 + yaml 完全没填时，按 repo
    # 相对路径解析（与历史行为一致）。
    transformer_path: str = Field(
        "models/diffusion_models/anima-base-v1.0.safetensors",
        description="主 transformer 权重 (.safetensors)",
        json_schema_extra=_meta("model", "path", cli_alias="--transformer"),
    )
    vae_path: str = Field(
        "models/vae/qwen_image_vae.safetensors",
        description="VAE 权重",
        json_schema_extra=_meta("model", "path", cli_alias="--vae"),
    )
    text_encoder_path: str = Field(
        "models/text_encoders",
        description="Qwen 文本编码器目录",
        json_schema_extra=_meta("model", "path", cli_alias="--qwen"),
    )
    t5_tokenizer_path: str = Field(
        "models/t5_tokenizer",
        description="T5 tokenizer 目录",
        json_schema_extra=_meta("model", "path", cli_alias="--t5-tokenizer"),
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
        json_schema_extra=_meta("system"),
    )

    # ------------------------------------------------------------------- LoRA
    lora_type: Literal["lora", "lokr", "loha"] = Field(
        "lokr",
        description="适配器算法（lora/lokr/loha）",
        json_schema_extra=_meta("lora"),
    )
    lora_rank: int = Field(
        32, ge=4,
        description="rank（推荐 8/16/32/64；LoKr 可设足够大触发 full dimension）",
        json_schema_extra=_meta("lora"),
    )
    lora_alpha: float = Field(
        32.0, ge=0.0,
        description="alpha（通常与 rank 相同；rs_lora 开启时常设为 √rank）",
        json_schema_extra=_meta("lora"),
    )
    lokr_factor: int = Field(
        8, ge=2,
        description="LoKr 分解因子（仅 lora_type=lokr）",
        json_schema_extra=_meta("lora", show_when="lora_type==lokr"),
    )
    lora_dora: bool = Field(
        False,
        description="DoRA：方向/幅度分离训练，社区共识收敛更稳",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_rs: bool = Field(
        False,
        description="rs-LoRA：scale=α/√r 而非 α/r，高 rank（>32）训练更稳",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="LoRA 输入 dropout（0 关闭）",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_rank_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="rank 维 dropout（防过拟合，对小数据集效果好）",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_module_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="层级 stochastic depth（整层级别随机跳过）",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_reg_dims: Optional[dict[str, int]] = Field(
        None,
        description="分层 rank：正则表达式 → rank 的字典，按模块名正则全匹配覆盖默认 rank（如 {\"lora_unet_.*double.*\": 16}）",
        examples=[{"lora_unet_.*double.*": 16}],
        json_schema_extra=_meta("lora", "code", advanced=True),
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
    grad_checkpoint: bool = Field(
        True,
        description="梯度检查点（省显存，约增加 1/3 计算量）",
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
        json_schema_extra=_meta(
            "training",
            cli_alias="--lr",
            disable_when="optimizer_type==prodigy||optimizer_type==prodigy_plus_schedulefree",
            disable_value=1.0,
            disable_hint="Prodigy 接管学习率",
        ),
    )
    lr_scheduler: Literal["none", "cosine", "cosine_with_restart"] = Field(
        "none",
        description="学习率调度（none = 常数；Prodigy / PPSF 固定为 none）",
        json_schema_extra=_meta(
            "training",
            disable_when="optimizer_type==prodigy||optimizer_type==prodigy_plus_schedulefree",
            disable_value="none",
            disable_hint="Prodigy 固定为常数学习率",
        ),
    )
    lr_scheduler_t0: int = Field(
        500, ge=1,
        description="cosine_with_restart 首次周期",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_with_restart", advanced=True),
    )
    lr_scheduler_t_mult: float = Field(
        2.0, ge=1.0,
        description="cosine_with_restart 周期倍数",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_with_restart", advanced=True),
    )
    lr_scheduler_eta_min: float = Field(
        1e-6, ge=0.0,
        description="最小学习率",
        json_schema_extra=_meta("training", show_when="lr_scheduler!=none", advanced=True),
    )
    optimizer_type: Literal["adamw", "prodigy", "prodigy_plus_schedulefree"] = Field(
        "adamw",
        description="优化器（prodigy_plus_schedulefree 是 DiT LoRA 训练推荐，averaged weights 解决 Prodigy 的风格突变 ep 问题）",
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
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy", advanced=True),
    )
    # ---------------- ProdigyPlusScheduleFree (PPSF) 专属字段 ----------------
    # 选 PPSF 时 lr_scheduler 必须为 none（Schedule-Free 不需要 scheduler，
    # 启动期校验会 fatal）。lr 强制 1.0（工厂内部覆盖）。
    ppsf_d_coef: float = Field(
        1.0, ge=0.1, le=10.0,
        description="PPSF d 缩放系数（小数据集 0.5，过拟合 2.0）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree"),
    )
    ppsf_prodigy_steps: int = Field(
        0, ge=0,
        description="PPSF 在第 N 步后冻结 d（0=训练全程不冻结，建议为总步数 1/4 到 1/2）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_beta1: float = Field(
        0.9, ge=0.0, le=1.0,
        description="PPSF Adam β1（PPSF 默认 0.9）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_beta2: float = Field(
        0.99, ge=0.0, le=1.0,
        description="PPSF Adam β2（PPSF 默认 0.99，比 AdamW 默认 0.999 更适合小 epoch）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_split_groups: bool = Field(
        True,
        description="PPSF 按 param group 分别估计 d（推荐开启）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_split_groups_mean: bool = Field(
        False,
        description="PPSF split_groups 启用时取各组 d 均值（LoRA 多 param group 建议关闭）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_use_speed: bool = Field(
        False,
        description="PPSF 加速模式（实验性，可能引入不稳定）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_fused_back_pass: bool = Field(
        False,
        description="PPSF 与 fused backward 集成（显存吃紧时开，可显著降显存）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_use_stableadamw: bool = Field(
        True,
        description="PPSF 用 stable AdamW 归一化（推荐开启）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    weight_decay: float = Field(
        0.0, ge=0.0,
        description="权重衰减（0=禁用）",
        json_schema_extra=_meta("training", advanced=True),
    )
    kv_trim: bool = Field(
        False,
        description="【性能】Cross-attention KV trim：按实际 token 数裁到最近 bucket（64/128/256/512），减少 padding 计算量",
        json_schema_extra=_meta("system", advanced=True),
    )
    noise_enhancement_type: Literal["none", "offset", "pyramid"] = Field(
        "none",
        description="【噪声增强】二选一：offset 给噪声加常数低频偏移（亮暗对比）；pyramid 多尺度叠加低频（构图/光照）。kohya 上游同样禁止同开。",
        json_schema_extra=_meta("noise_schedule", advanced=True),
    )
    noise_offset: float = Field(
        0.0, ge=0.0, le=0.2,
        description="【噪声增强】低频偏移强度，缓解亮度均值偏差（推荐 0.05-0.1）",
        json_schema_extra=_meta("noise_schedule", show_when="noise_enhancement_type==offset", advanced=True),
    )
    pyramid_noise_iters: int = Field(
        0, ge=0, le=6,
        description="【噪声增强】多尺度噪声叠加层数（2-3 帮助全局光照构图学习）",
        json_schema_extra=_meta("noise_schedule", show_when="noise_enhancement_type==pyramid", advanced=True),
    )
    pyramid_noise_discount: float = Field(
        0.35, ge=0.1, le=0.9,
        description="【噪声增强】金字塔每层衰减系数",
        json_schema_extra=_meta("noise_schedule", show_when="noise_enhancement_type==pyramid", advanced=True),
    )
    timestep_sampling: Literal[
        "logit_normal",
        "uniform",
        "logit_normal_low",
        "mode",
        "mixed_uniform_low",
        "mixed_uniform_logit",
    ] = Field(
        "logit_normal",
        description="【时间步采样】分布（logit_normal 为 SD3/Anima 默认；mixed_* 模式混合 uniform 与偏置端，按 timestep_mix_low_prob 控制比例）",
        json_schema_extra=_meta(
            "noise_schedule",
            alt_description="【时间步采样】分布；InfoNoise 启用时作为热身期 baseline，正式阶段由自适应 CDF 接管",
            alt_description_when="infonoise_enabled==true",
            disable_when="infonoise_enabled==true",
            disable_hint="InfoNoise 接管时间步采样",
            advanced=True,
        ),
    )
    timestep_shift: float = Field(
        3.0, ge=0.1, le=10.0,
        description="【时间步采样】logit-normal / mode shift（>1 偏向高噪声端，<1 偏向细节端）",
        json_schema_extra=_meta(
            "noise_schedule",
            show_when="timestep_sampling!=uniform",
            alt_description="【InfoNoise 热身期】InfoNoise 开启时作为热身阶段的 baseline shift，正式阶段由自适应 CDF 接管",
            alt_description_when="infonoise_enabled==true",
            disable_when="infonoise_enabled==true",
            disable_hint="InfoNoise 接管时间步采样",
            advanced=True,
        ),
    )
    timestep_mix_low_prob: float = Field(
        0.0, ge=0.0, le=1.0,
        description="【时间步采样】mixed_* 模式下走偏置端的样本比例（0 = 全 uniform；典型 0.15-0.30；仅 mixed_uniform_low / mixed_uniform_logit 生效，其他 mode 忽略）",
        json_schema_extra=_meta(
            "noise_schedule",
            show_when="timestep_sampling!=uniform",
            alt_description="【InfoNoise 热身期】InfoNoise 开启 + mixed_* baseline 时，热身阶段混合比例；正式阶段由自适应 CDF 接管",
            alt_description_when="infonoise_enabled==true",
            advanced=True,
        ),
    )
    timestep_schedule_shift: float = Field(
        1.0, ge=0.1, le=10.0,
        description="【时间步采样】采样后对 t 做的额外 σ schedule 偏移（1.0 = 无偏移；与 timestep_shift 不同：后者作用于 logit-normal 内部，前者作用于最终 t）",
        json_schema_extra=_meta(
            "noise_schedule",
            alt_description="【InfoNoise 热身期】InfoNoise 开启时仅热身期生效；正式阶段由自适应 CDF 接管",
            alt_description_when="infonoise_enabled==true",
            advanced=True,
        ),
    )
    infonoise_enabled: bool = Field(
        False,
        description="【InfoNoise】启用自适应时间步采样（基于 I-MMSE 信息量，自动聚焦有效训练区间）",
        json_schema_extra=_meta("noise_schedule", advanced=True),
    )
    infonoise_K: int = Field(
        64, ge=16, le=256,
        description="【InfoNoise】log-σ bin 数量",
        json_schema_extra=_meta("noise_schedule", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_N_warm: int = Field(
        0, ge=0,
        description="【InfoNoise】热身步数（0 = 自动，取总步数的 1/5，最少 200 步）",
        json_schema_extra=_meta("noise_schedule", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_M: int = Field(
        100, ge=10,
        description="【InfoNoise】schedule 刷新周期（每 M 步重算一次采样分布）",
        json_schema_extra=_meta("noise_schedule", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_B: int = Field(
        256, ge=32,
        description="【InfoNoise】每 bin 的 FIFO buffer 容量",
        json_schema_extra=_meta("noise_schedule", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_beta: float = Field(
        0.9, ge=0.1, le=0.999,
        description="【InfoNoise】EMA 新值权重（论文 β 乘新值，0.9 即新值占 0.9 权重；FIFO 已做一轮平均故 β 偏高合理）",
        json_schema_extra=_meta("noise_schedule", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_N_min: int = Field(
        50, ge=1,
        description="【InfoNoise】触发刷新所需的每 bin 最小样本数",
        json_schema_extra=_meta("noise_schedule", show_when="infonoise_enabled==true", advanced=True),
    )
    loss_type: Literal["mse", "huber"] = Field(
        "mse",
        description="【损失函数】训练 loss 类型（mse 默认；huber 对 outlier 鲁棒）",
        json_schema_extra=_meta("noise_schedule"),
    )
    huber_c: float = Field(
        0.15, ge=0.01, le=5.0,
        description="【Huber loss】delta 系数（典型 0.1–0.3；控制 quad/linear 转折点，|x|<δ 走二次，|x|≥δ 走线性）",
        json_schema_extra=_meta("noise_schedule", show_when="loss_type==huber", advanced=True),
    )
    loss_weighting: Literal["none", "min_snr", "detail_inv_t", "cosmap"] = Field(
        "none",
        description="【损失加权】方案（min_snr 推荐；detail_inv_t 细节强化；cosmap SD3 风格）",
        json_schema_extra=_meta("noise_schedule"),
    )
    min_snr_gamma: float = Field(
        5.0, ge=0.1, le=20.0,
        description="【损失加权】Min-SNR gamma 值（仅 loss_weighting=min_snr）",
        json_schema_extra=_meta("noise_schedule", show_when="loss_weighting==min_snr", advanced=True),
    )
    weight_cap_ratio: float = Field(
        0.0, ge=0.0, le=50.0,
        description="【损失加权】Batch 内权重 max/min 比上限（0=禁用；小 batch+Prodigy 建议 5）",
        json_schema_extra=_meta("noise_schedule", show_when="loss_weighting!=none", advanced=True),
    )
    detail_inv_t_min: float = Field(
        1.0, ge=1.0, le=20.0,
        description="【损失加权】detail_inv_t 权重下限（默认 1.0；升至 1.5 可让高 t 步也被略微加权；<1.0 因 1/t>1 恒成立故无效）",
        json_schema_extra=_meta("noise_schedule", show_when="loss_weighting==detail_inv_t", advanced=True),
    )
    detail_inv_t_max: float = Field(
        5.0, ge=0.1, le=50.0,
        description="【损失加权】detail_inv_t 权重上限（默认 5.0；雾蒙蒙画风建议降到 3，激进细节可升到 8）",
        json_schema_extra=_meta("noise_schedule", show_when="loss_weighting==detail_inv_t", advanced=True),
    )
    grad_clip_max_norm: float = Field(
        1.0, ge=0.0,
        description="梯度裁剪最大范数：当本步所有可训练参数的梯度全局范数超过该值时按比例缩到该值，防止单步极端梯度把模型推飞；默认 1.0 适合绝大多数场景，bf16+DoRA/LoKr 不稳可降到 0.5，0=禁用",
        json_schema_extra=_meta("training", advanced=True),
    )

    mixed_precision: Literal["bf16", "fp16", "no"] = Field(
        "bf16",
        description="混合精度",
        json_schema_extra=_meta("system"),
    )

    attention_backend: AttentionBackend = Field(
        "flash_attn",
        description="Attention backend：none（PyTorch SDPA）/ xformers / Flash Attention（5090 推荐 flash_attn）",
        json_schema_extra=_meta("system"),
    )
    num_workers: int = Field(
        0, ge=0,
        description="数据加载线程（Windows 必须 0）",
        json_schema_extra=_meta("system", advanced=True),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_attention(cls, data: Any) -> Any:
        return migrate_legacy_attention(data)

    @model_validator(mode="before")
    @classmethod
    def _migrate_save_keys(cls, data: Any) -> Any:
        return migrate_legacy_save_keys(data)

    @model_validator(mode="before")
    @classmethod
    def _migrate_noise_enhancement(cls, data: Any) -> Any:
        return migrate_noise_enhancement_type(data)

    @model_validator(mode="after")
    def _validate_prodigy_scheduler(self) -> "TrainingConfig":
        """Prodigy 系列固定使用常数学习率，外部 scheduler 统一拦截。"""
        if self.optimizer_type in {"prodigy", "prodigy_plus_schedulefree"} and self.lr_scheduler != "none":
            raise ValueError(
                f"optimizer_type={self.optimizer_type} requires lr_scheduler=none "
                "(Prodigy 系列固定使用常数学习率)."
            )
        return self

    @model_validator(mode="after")
    def _validate_detail_inv_t_range(self) -> "TrainingConfig":
        """detail_inv_t 加权曲线的 min 必须 <= max；fail-fast 取代历史的静默 swap。"""
        if self.detail_inv_t_min > self.detail_inv_t_max:
            raise ValueError(
                f"detail_inv_t_min ({self.detail_inv_t_min}) 不能大于 "
                f"detail_inv_t_max ({self.detail_inv_t_max})。"
            )
        return self

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
    save_every_epochs: int = Field(
        2, ge=0,
        description="每 N epoch 保存（0=禁用）",
        json_schema_extra=_meta("output"),
    )
    save_every_steps: int = Field(
        0, ge=0,
        description="每 N step 保存（0=禁用）",
        json_schema_extra=_meta("output"),
    )
    save_state_every_epochs: int = Field(
        0, ge=0,
        description="每 N epoch 保存完整训练状态（断点续训，0=禁用）",
        json_schema_extra=_meta("output"),
    )
    save_state_every_steps: int = Field(
        0, ge=0,
        description="每 N step 保存完整训练状态（断点续训，0=禁用）",
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
        2, ge=0,
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
    trigger_word: str = Field(
        "",
        description="触发词（version 级，由 Step 4 Tagging 页面写入；空串=不启用）。"
                    "训练时 bootstrap_phase 会自动 prepend 到 sample_prompt / "
                    "sample_prompts，确保采样图能反映 LoRA 是否激活。",
        json_schema_extra=_meta("sample", hidden=True),
    )

    # ---------------------------------------------------------------- 监控/进度
    # 这一组对 Studio 用户全部隐藏（hidden=True）—— Studio 跑训练用 subprocess 把
    # stdout 重定向到 task log（非 tty），这些「终端体验」字段对 web 用户没意义；
    # monitor 页用的是 monitor_state.json，跟这些值零相关。
    # 字段保留在 schema 是为了：(1) 旧 project yaml 里写过的值不丢；(2) 裸 CLI 用户
    # 仍可在 yaml 手动覆盖。BaseConfig.extra="forbid" 也要求字段定义存在。
    loss_curve_steps: int = Field(
        100, ge=10,
        description="终端 rich live 曲线宽度（仅 CLI 终端，不影响 Studio 监控页）",
        json_schema_extra=_meta("monitor", hidden=True),
    )
    # 默认 True：Studio 起 subprocess stdout 是 pipe 不是 tty，rich 在非 tty 下仍会
    # 刷屏式打 progress 行，让 task log 巨大且难读；走 plain log_every 节流分支更干净。
    # 裸 CLI 用户想看 rich 进度条可以 yaml 显式 `no_progress: false` 覆盖。
    no_progress: bool = Field(
        True,
        description="禁用终端 rich 进度条与曲线（CLI / log file 场景）",
        json_schema_extra=_meta("monitor", hidden=True),
    )
    log_every: int = Field(
        10, ge=1,
        description="终端日志输出间隔（仅在禁用 rich 进度条时生效）",
        json_schema_extra=_meta("monitor", hidden=True),
    )
    # PP6.1：以下字段保留是为了不破坏既有 yaml；HTTP monitor server 已退役，
    # 这些值不再生效。Studio 前端通过 /api/state?task_id= 读 monitor_state.json，
    # 路径由 --monitor-state-file（CLI-only）决定。
    no_monitor: bool = Field(
        True,
        description="(已废弃) 内置 Web monitor server 已删除；保留字段兼容旧 yaml",
        json_schema_extra=_meta("monitor", hidden=True),
    )
    monitor_host: str = Field(
        "127.0.0.1",
        description="(已废弃) 旧 monitor server 绑定地址；当前忽略",
        json_schema_extra=_meta("monitor", hidden=True),
    )
    monitor_port: int = Field(
        8765, ge=1, le=65535,
        description="(已废弃) 旧 monitor server 端口；当前忽略",
        json_schema_extra=_meta("monitor", hidden=True),
    )
    no_browser: bool = Field(
        True,
        description="(已废弃) 旧 monitor server 自动开浏览器；当前忽略",
        json_schema_extra=_meta("monitor", hidden=True),
    )
