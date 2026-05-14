"""训练参数 schema —— pydantic v2 单一权威源。

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


def _meta(group: str, control: str = "auto", **extra: Any) -> dict[str, Any]:
    return {"group": group, "control": control, **extra}


# ---------------------------------------------------------------------------
# attention backend 字段：xformers / flash_attn / 无 三选一（替代原本两个 bool）
# ---------------------------------------------------------------------------

AttentionBackend = Literal["none", "xformers", "flash_attn"]


def migrate_legacy_attention(data: Any) -> Any:
    """把老 cfg 的 `xformers` / `flash_attn` 双 bool 映射成 `attention_backend`。

    Idempotent：已有 attention_backend 就剥掉老字段；只有老字段时按下面规则映射；
    都没有则保持空（让 schema default 生效）。

    映射规则（与原代码 `use_flash = flash_attn and not xformers` 一致 — xformers 优先）：
        xformers=true  → "xformers"
        xformers=false, flash_attn=true → "flash_attn"
        xformers=false, flash_attn=false → "none"

    在两个地方调用：
      1. schema model_validator(mode='before')（pydantic 校验前先洗）—— server
         构造 cfg / 前端送老字段都兼容
      2. runtime/anima_train.py apply_yaml_config 之前显式调一次 —— 子进程读老
         yaml 时 argparse_bridge.merge_yaml_into_namespace 不走 pydantic validator,
         需要这层兜底
    """
    if not isinstance(data, dict):
        return data
    for key in (
        "wandb_enabled",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "wandb_mode",
        "wandb_log_samples",
    ):
        data.pop(key, None)
    if "attention_backend" in data:
        data.pop("xformers", None)
        data.pop("flash_attn", None)
        return data
    has_legacy = "xformers" in data or "flash_attn" in data
    if not has_legacy:
        return data
    xf = bool(data.pop("xformers", False))
    fa = bool(data.pop("flash_attn", True))
    if xf:
        data["attention_backend"] = "xformers"
    elif fa:
        data["attention_backend"] = "flash_attn"
    else:
        data["attention_backend"] = "none"
    return data


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
        json_schema_extra=_meta("caption"),
    )

    # ------------------------------------------------------------------- LoRA
    lora_type: Literal["lora", "lokr", "loha"] = Field(
        "lokr",
        description="适配器算法（lora/lokr/loha）",
        json_schema_extra=_meta("lora"),
    )
    lora_rank: int = Field(
        32, ge=4, le=256,
        description="rank（推荐 8/16/32/64）",
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
        json_schema_extra=_meta("lora"),
    )
    lora_rs: bool = Field(
        False,
        description="rs-LoRA：scale=α/√r 而非 α/r，高 rank（>32）训练更稳",
        json_schema_extra=_meta("lora"),
    )
    lora_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="LoRA 输入 dropout（0 关闭）",
        json_schema_extra=_meta("lora"),
    )
    lora_rank_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="rank 维 dropout（防过拟合，对小数据集效果好）",
        json_schema_extra=_meta("lora"),
    )
    lora_module_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="层级 stochastic depth（整层级别随机跳过）",
        json_schema_extra=_meta("lora"),
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
        json_schema_extra=_meta("training", cli_alias="--lr"),
    )
    lr_scheduler: Literal["none", "cosine", "cosine_with_restart"] = Field(
        "none",
        description="学习率调度（选 prodigy_plus_schedulefree 时必须 none）",
        json_schema_extra=_meta(
            "training",
            disable_when="optimizer_type==prodigy_plus_schedulefree",
            disable_hint="Schedule-Free 自带调度",
        ),
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
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy"),
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
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree"),
    )
    ppsf_beta1: float = Field(
        0.9, ge=0.0, le=1.0,
        description="PPSF Adam β1（PPSF 默认 0.9）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree"),
    )
    ppsf_beta2: float = Field(
        0.99, ge=0.0, le=1.0,
        description="PPSF Adam β2（PPSF 默认 0.99，比 AdamW 默认 0.999 更适合小 epoch）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree"),
    )
    ppsf_split_groups: bool = Field(
        True,
        description="PPSF 按 param group 分别估计 d（推荐开启）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree"),
    )
    ppsf_split_groups_mean: bool = Field(
        False,
        description="PPSF split_groups 启用时取各组 d 均值（LoRA 多 param group 建议关闭）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree"),
    )
    ppsf_use_speed: bool = Field(
        False,
        description="PPSF 加速模式（实验性，可能引入不稳定）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree"),
    )
    ppsf_fused_back_pass: bool = Field(
        False,
        description="PPSF 与 fused backward 集成（显存吃紧时开，可显著降显存）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree"),
    )
    ppsf_use_stableadamw: bool = Field(
        True,
        description="PPSF 用 stable AdamW 归一化（推荐开启）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree"),
    )
    weight_decay: float = Field(
        0.0, ge=0.0,
        description="权重衰减（0=禁用）",
        json_schema_extra=_meta("training"),
    )
    kv_trim: bool = Field(
        False,
        description="【性能】Cross-attention KV trim：按实际 token 数裁到最近 bucket（64/128/256/512），减少 padding 计算量",
        json_schema_extra=_meta("training"),
    )
    noise_offset: float = Field(
        0.0, ge=0.0, le=0.2,
        description="【噪声增强】低频偏移强度，缓解亮度均值偏差（0=禁用，推荐 0.05-0.1）",
        json_schema_extra=_meta("training"),
    )
    pyramid_noise_iters: int = Field(
        0, ge=0, le=6,
        description="【噪声增强】多尺度噪声叠加层数（0=禁用；2-3 帮助全局光照构图学习）",
        json_schema_extra=_meta("training"),
    )
    pyramid_noise_discount: float = Field(
        0.35, ge=0.1, le=0.9,
        description="【噪声增强】金字塔每层衰减系数（仅 pyramid_noise_iters > 0）",
        json_schema_extra=_meta("training", show_when="pyramid_noise_iters!=0"),
    )
    timestep_sampling: Literal["logit_normal", "uniform", "logit_normal_low", "mode"] = Field(
        "logit_normal",
        description="【时间步采样】分布（logit_normal 为 SD3/Anima 默认）",
        json_schema_extra=_meta("training"),
    )
    timestep_shift: float = Field(
        3.0, ge=0.1, le=10.0,
        description="【时间步采样】logit-normal / mode shift（>1 偏向高噪声端，<1 偏向细节端）",
        json_schema_extra=_meta("training", show_when="timestep_sampling!=uniform"),
    )
    loss_weighting: Literal["none", "min_snr", "detail_inv_t", "cosmap"] = Field(
        "none",
        description="【损失加权】方案（min_snr 推荐；detail_inv_t 细节强化；cosmap SD3 风格）",
        json_schema_extra=_meta("training"),
    )
    min_snr_gamma: float = Field(
        5.0, ge=0.1, le=20.0,
        description="【损失加权】Min-SNR gamma 值（仅 loss_weighting=min_snr）",
        json_schema_extra=_meta("training", show_when="loss_weighting==min_snr"),
    )
    weight_cap_ratio: float = Field(
        0.0, ge=0.0, le=50.0,
        description="【损失加权】Batch 内权重 max/min 比上限（0=禁用；小 batch+Prodigy 建议 5）",
        json_schema_extra=_meta("training", show_when="loss_weighting!=none"),
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
    attention_backend: AttentionBackend = Field(
        "flash_attn",
        description="Attention backend：none（PyTorch SDPA）/ xformers / Flash Attention（5090 推荐 flash_attn）",
        json_schema_extra=_meta("training"),
    )
    num_workers: int = Field(
        0, ge=0,
        description="数据加载线程（Windows 必须 0）",
        json_schema_extra=_meta("training"),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_attention(cls, data: Any) -> Any:
        return migrate_legacy_attention(data)

    @model_validator(mode="after")
    def _validate_ppsf_scheduler(self) -> "TrainingConfig":
        """ProdigyPlusScheduleFree 内置 Schedule-Free，外面再叠 scheduler 会破坏
        averaged weights 的收敛保证。UI/CLI/YAML 三个入口在这里统一拦下来。"""
        if self.optimizer_type == "prodigy_plus_schedulefree" and self.lr_scheduler != "none":
            raise ValueError(
                "optimizer_type=prodigy_plus_schedulefree requires lr_scheduler=none "
                "(Schedule-Free 不需要 scheduler；强行叠加会破坏 averaged weights)."
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


# ---------------------------------------------------------------------------
# 测试出图（独立工具页，多 LoRA + multi-prompt）—— 对应 runtime/anima_generate.py
# ---------------------------------------------------------------------------


class LoraEntry(BaseModel):
    """单个 LoRA 的加载参数。Generate / API 共享，避免 server.py 私有定义。"""

    model_config = ConfigDict(extra="forbid")
    path: str = Field(..., description="LoRA safetensors 绝对路径")
    scale: float = Field(1.0, description="贡献权重（multiplier），多 LoRA 各自独立")
    # 关联到的 version（picker 选的；外部文件无）；前端用 vid 拉 ckpt 列表
    project_id: Optional[int] = Field(None, ge=1)
    version_id: Optional[int] = Field(None, ge=1)


# ---------------------------------------------------------------------------
# XY 矩阵：轴枚举 + axis spec + matrix spec
# ---------------------------------------------------------------------------
#
# 设计：单 task 内循环全图（一次 model load 摊销 ~30s 启动成本）。前端拿到
# samples[].xy={xi,yi,xv,yv} 元数据按 (yi, xi) 排成 grid 渲染。
#
# 轴值类型按 axis 枚举派生：
#   lora_scale / cfg_scale → float
#   steps / seed           → int
#   sampler_name           → str
#
# v1 暂不支持 lora_path 轴：AnimaLycorisAdapter 没有 unhook 接口，切换 LoRA
# 文件需要 unload 旧 forward hook，留 v2。lora_scale 走 mutate
# adapter.network.multiplier 的轻量路径，无 re-inject 成本。

XYAxisType = Literal[
    "lora_scale",   # 改 lora_configs[lora_index].scale 的多个值
    "steps",        # 不同采样步数
    "cfg_scale",    # 不同 CFG
    "lora_ckpt",    # 同一 LoRA 训练过程的不同 step/epoch ckpt（找过拟合拐点）
]


class XYAxisSpec(BaseModel):
    """单轴定义：axis 枚举 + values 列表 + (lora_axis 时) lora_index。"""

    model_config = ConfigDict(extra="forbid")
    axis: XYAxisType = Field(..., description="轴绑定的字段")
    values: list[Any] = Field(..., min_length=1, description="此轴扫描的值列表")
    lora_index: Optional[int] = Field(
        None, ge=0,
        description="axis=lora_scale / lora_ckpt 时指定改 lora_configs 哪一项",
    )


class XYMatrixSpec(BaseModel):
    """XY 矩阵：x 轴必填，y 可选（None = 单轴 N×1 退化成一行）。"""

    model_config = ConfigDict(extra="forbid")
    x: XYAxisSpec
    y: Optional[XYAxisSpec] = None


def _check_axis_values(axis: XYAxisSpec) -> None:
    """按 axis 枚举校验 values 类型（浮点 / 整数 / 字符串）。"""
    int_axes = {"steps"}
    float_axes = {"lora_scale", "cfg_scale"}
    str_axes = {"lora_ckpt"}  # ckpt 路径列表
    needs_lora_index = {"lora_scale", "lora_ckpt"}

    if axis.axis in int_axes:
        for v in axis.values:
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"axis={axis.axis} values 必须为 int，收到 {type(v).__name__}")
    elif axis.axis in float_axes:
        for v in axis.values:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"axis={axis.axis} values 必须为 number，收到 {type(v).__name__}")
    elif axis.axis in str_axes:
        for v in axis.values:
            if not isinstance(v, str):
                raise ValueError(f"axis={axis.axis} values 必须为 str，收到 {type(v).__name__}")

    if axis.axis in needs_lora_index and axis.lora_index is None:
        raise ValueError(f"axis={axis.axis} 必须指定 lora_index（绑定到 lora_configs 哪一项）")
    if axis.axis not in needs_lora_index and axis.lora_index is not None:
        raise ValueError(f"axis={axis.axis} 不允许设 lora_index（仅 lora_scale / lora_ckpt 可设）")


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


# ---------------------------------------------------------------------------
# 先验生成（base 模型对每张训练图反向出对照图作正则集）—— 对应 runtime/anima_reg_ai.py
# ---------------------------------------------------------------------------


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

    @model_validator(mode="before")
    @classmethod
    def _migrate_attention(cls, data: Any) -> Any:
        return migrate_legacy_attention(data)


# ---------------------------------------------------------------------------
# 分组顺序（前端按这个顺序渲染区块）
# ---------------------------------------------------------------------------

# 每组：(key, label, default_collapsed)。default_collapsed=True 让前端 SchemaForm
# 初始默认折叠（用户能手动展开）。模型路径 readonly 显示「自动 · 全局设置」徽章
# （跟 output_dir 等项目特定字段同款），不再折叠。
GROUP_ORDER: list[tuple[str, str, bool]] = [
    ("model", "模型路径", False),
    ("dataset", "数据集", False),
    ("caption", "Caption 处理", False),
    ("lora", "LoRA / LoKr", False),
    ("training", "训练参数", False),
    ("output", "输出与保存", False),
    ("sample", "采样", False),
    ("monitor", "监控与进度", False),
]
