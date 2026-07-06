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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .common import AttentionBackend, _meta
from .migrations import migrate_legacy_save_keys, migrate_noise_enhancement_type


class TrainingConfig(BaseModel):
    """与 config/train_template.yaml 对齐的完整训练参数。

    `extra="ignore"`：云端/旧版预设里多出的键静默丢弃。
    """

    model_config = ConfigDict(extra="ignore")

    # ---------------------------------------------------------------- 模型路径
    # 这些路径在 Studio 创建 version 时会被替换成 **绝对路径**（基于
    # secrets.models.root + secrets.models.selected_anima），用户在 yaml /
    # Train 页看到的总是无歧义的绝对路径，不用考虑相对路径锚定。
    # 这里的默认值仅 fallback：裸 CLI 跑训练 + yaml 完全没填时，按 repo
    # 相对路径解析（与历史行为一致）。
    transformer_path: str = Field(
        "models/diffusion_models/anima-base-v1.0.safetensors",
        description="主扩散模型权重（.safetensors）",
        json_schema_extra=_meta("model", "path", cli_alias="--transformer"),
    )
    vae_path: str = Field(
        "models/vae/qwen_image_vae.safetensors",
        description="VAE 权重（.safetensors）",
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
    resolution: list[int] = Field(
        default=[1024],
        description="训练分辨率。可填多个（逗号分隔，如 512, 768, 1024）——无分辨率前缀的文件夹里每张图会在每个分辨率各训一遍；单个值即传统单分辨率训练",
        json_schema_extra=_meta("dataset"),
    )
    aspect_ratio_limit: float = Field(
        2.0, ge=1.0, le=4.0,
        description="桶的最大长宽比上限。2.0 = 最宽 2:1、最高 1:2。值越大越长/越扁的图按原比例训练，但极端比例的桶图片更少、短边有效分辨率更低",
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
        description="正则集 loss 相对训练集的权重；1.0 = 等权重，调低削弱 reg 影响",
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
        description="保护前 N 个标签不参与打乱与 dropout（仅 TXT 模式；JSON 模式由"
                    " meta.trigger 与固定字段承担同类角色）",
        json_schema_extra=_meta("caption"),
    )
    flip_augment: bool = Field(
        True,
        description="水平翻转增强",
        json_schema_extra=_meta("caption"),
    )
    tag_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="训练时每个标签的随机丢弃概率（0 = 关闭）；非 0 帮助泛化、减弱单标签依赖",
        json_schema_extra=_meta("caption"),
    )
    prefer_json: bool = Field(
        True,
        description="优先使用 JSON 标签文件（推荐，支持分类 shuffle）",
        json_schema_extra=_meta("caption"),
    )
    caption_comfy_encoding: bool = Field(
        True,
        description="标签按 ComfyUI 方式编码文本（与测试出图、采样预览同一条编码链路，"
                    "推荐保持开启）；关闭走旧版逐 tag 编码，用于新旧编码 A/B 对比，"
                    "或继续用旧编码训练的断点状态",
        json_schema_extra=_meta("caption", advanced=True),
    )
    cache_latents: bool = Field(
        True,
        description="缓存 VAE latent 加速训练",
        json_schema_extra=_meta("system"),
    )
    vae_cache_batch_size: int = Field(
        0, ge=0,
        description="VAE latent 缓存编码批次大小；0=跟随训练 batch size，显存不足时设为 1 逐张编码",
        json_schema_extra=_meta("system", advanced=True),
    )
    vae_tiling: Literal["auto", "on", "off"] = Field(
        "auto",
        description="VAE 分块 decode：auto=可用显存紧张时自动分块（推荐）；on=始终分块（省显存、慢约 30%）；"
                    "off=整图，仅真正 OOM 时回退。大显存卡整图 decode 接近占满显存时会触发系统内存回退、"
                    "单次 decode 从不到 1 秒退化到上百秒，auto 可避免",
        json_schema_extra=_meta("system", advanced=True),
    )

    # --------------------------------------------------- NaViT / Patch-n-Pack
    # 块对角打包训练（Phase 2 数据层）。以下字段全部默认关闭 / 行为中立：
    # navit_packing=False 时走原有 ARB 分桶路径，字节等价。
    navit_packing: bool = Field(
        False,
        description="启用 NaViT / Patch-n-Pack 块对角打包：按 token 预算把多张不同尺寸的图"
                    "拼进一个训练序列（零 padding），替代 ARB 固定桶分批。需配合 cache_latents",
        json_schema_extra=_meta("system", advanced=True),
    )
    navit_token_budget: int = Field(
        16384, ge=1,
        description="单个打包序列的 token 预算上限（所有图 token 数之和 ≤ 此值）",
        json_schema_extra=_meta("system", advanced=True),
    )
    navit_max_images_per_pack: int = Field(
        0, ge=0,
        description="每个包最多图片数（0=不限，仅受 token 预算约束）",
        json_schema_extra=_meta("system", advanced=True),
    )
    navit_text_trim_padding: bool = Field(
        False,
        description="cross-attn 文本截断 padding（数据层标志，仅 navit_packing 时生效）",
        json_schema_extra=_meta("system", advanced=True),
    )
    navit_pack_strategy: Literal["next_fit", "ffd"] = Field(
        "next_fit",
        description="打包策略：next_fit=顺序贪心（快、包较松）；ffd=First-Fit-Decreasing 窗口化（包更紧、更少步）",
        json_schema_extra=_meta("system", advanced=True),
    )
    navit_pack_ffd_window: int = Field(
        256, ge=0,
        description="FFD 窗口大小（0=全局 FFD，每 epoch 包固定；>0=窗口内 FFD + 跨 epoch reshuffle）",
        json_schema_extra=_meta("system", advanced=True),
    )
    navit_drop_last: bool = Field(
        False,
        description="丢弃最后一个未填满的包（对齐 step 计数；False=保留短包）",
        json_schema_extra=_meta("system", advanced=True),
    )
    navit_native_resolution: bool = Field(
        False,
        description="原生定尺寸：按源图尺寸 floor 对齐到 16px（VAE 8×patch 2），不 resize/crop",
        json_schema_extra=_meta("system", advanced=True),
    )
    navit_multiscale: bool = Field(
        False,
        description="多尺度阶梯：为每张大图额外生成降采样副本（≤指定 token 档），拼入打包序列",
        json_schema_extra=_meta("system", advanced=True),
    )
    navit_multiscale_token_ladder: str = Field(
        "4096",
        description="多尺度 token 档位（逗号分隔，如 \"4096,16384\"），每张大于该档的图生成一份缩放副本",
        json_schema_extra=_meta("system", advanced=True),
    )
    navit_multiscale_loss_weight: float = Field(
        1.0, ge=0.0,
        description="多尺度副本的 loss 权重（1.0=等权；<1 削弱副本影响）",
        json_schema_extra=_meta("system", advanced=True),
    )
    cache_encode_tiled: bool = Field(
        False,
        description="缓存编码分块：超大图按 tile_px 分块 VAE encode + latent 羽化拼接（峰值显存 ∝ 单块像素）",
        json_schema_extra=_meta("system", advanced=True),
    )
    cache_encode_tile_px: int = Field(
        1024, ge=64,
        description="分块 encode 的像素块边长（须为 VAE 下采样 8 的整倍数）",
        json_schema_extra=_meta("system", advanced=True),
    )
    cache_encode_tile_overlap: int = Field(
        128, ge=0,
        description="分块重叠像素（羽化接缝宽度；0=无重叠硬接缝）",
        json_schema_extra=_meta("system", advanced=True),
    )
    cache_encode_max_pixels: int = Field(
        0, ge=0,
        description="单次 encode 总像素上限（含翻转份）；0=用内置保守默认 4M。也是分块触发阈值",
        json_schema_extra=_meta("system", advanced=True),
    )

    # ------------------------------------------------------------------- LoRA
    lora_type: Literal["lora", "lokr", "loha", "ortho", "tlora"] = Field(
        "lokr",
        description="适配器算法。lokr：Kronecker 分解，参数最省（默认）；lora：经典低秩，通用；loha：Hadamard 积，表达力较高但参数较多；ortho：正交参数化，可训练参数极少、防过拟合，适合小数据集人物/主体；tlora：噪声越高 rank 越小，专为单图/极少图主体定制防过拟合",
        json_schema_extra=_meta("lora"),
    )
    lora_rank: int = Field(
        32, ge=4,
        description="rank：越大表达力越强，参数量与显存上升、易过拟合；越小越省但易欠拟合。常用 8/16/32/64",
        json_schema_extra=_meta("lora"),
    )
    lora_alpha: float = Field(
        32.0, ge=0.0,
        description="alpha：LoRA 缩放系数，越大 LoRA 效果越强。通常等于 rank；启用 rs_lora 时常设为 √rank",
        json_schema_extra=_meta("lora"),
    )
    lokr_factor: int = Field(
        8, ge=2,
        description="LoKr 矩阵分解因子：越大压缩越强、参数越少；越小参数越多。默认 8 适合大多数场景",
        json_schema_extra=_meta("lora", show_when="lora_type==lokr"),
    )
    tlora_min_rank: int = Field(
        1, ge=1,
        description="T-LoRA 高噪声时保留的最小 active rank（与论文 ControlGenAI/T-LoRA 对齐，默认 1）",
        json_schema_extra=_meta("lora", show_when="lora_type==tlora", advanced=True),
    )
    tlora_alpha_rank_scale: float = Field(
        1.0, ge=0.0,
        description=(
            "T-LoRA 幂次缩放（对齐官方 SDXL `alpha_rank_scale`）：1.0=线性 schedule；"
            ">1 越偏向低噪声端才开高 rank；<1 越早开高 rank。"
            "公式 r=(1-t)^α·(rank-min_rank)+min_rank，t 为 noise level (0=clean, 1=noisy)"
        ),
        json_schema_extra=_meta("lora", show_when="lora_type==tlora", advanced=True),
    )
    tlora_use_ortho: bool = Field(
        True,
        description="T-LoRA 专属：叠加 OrthoLoRA 正交参数化（论文完整配方，默认开启）；关闭时使用普通 T-LoRA",
        json_schema_extra=_meta("lora", show_when="lora_type==tlora", advanced=True),
    )
    lora_dora: bool = Field(
        False,
        description="DoRA：分解权重为方向 + 幅度独立训练；收敛通常更稳，显存略增",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_rs: bool = Field(
        False,
        description="rs-LoRA：scale=α/√r 而非 α/r，高 rank（>32）训练更稳",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="LoRA 输入特征的随机丢弃概率：越大正则化越强、收敛越慢；0 = 关闭",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_rank_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="LoRA 内部 rank 维度的随机丢弃概率（每步随机激活部分 rank）：越大正则化越强；0 = 关闭",
        json_schema_extra=_meta("lora", advanced=True),
    )
    lora_module_dropout: float = Field(
        0.0, ge=0.0, le=1.0,
        description="整个 LoRA 模块的随机跳过概率（stochastic depth）：每步以此概率完全不应用此模块；越大正则化越强；0 = 关闭",
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
        description="学习率。Automagic 作为初始每参数学习率，推荐 1e-6（切换 optimizer 到 automagic 时会自动改写）；Lion 推荐为 AdamW lr / 3；Prodigy / PPSF 必须 1.0",
        json_schema_extra=_meta(
            "training",
            cli_alias="--lr",
            disable_when="optimizer_type==prodigy||optimizer_type==prodigy_plus_schedulefree",
            disable_value=1.0,
            disable_hint="此优化器自动管理学习率",
        ),
    )
    lr_scheduler: Literal["none", "cosine", "cosine_with_restart", "cosine_with_warmup"] = Field(
        "none",
        description="学习率调度（none = 常数；Prodigy / PPSF / Automagic / SOAP-SF 固定为 none）",
        json_schema_extra=_meta(
            "training",
            disable_when="optimizer_type==automagic||optimizer_type==prodigy||optimizer_type==prodigy_plus_schedulefree||optimizer_type==soap_sf",
            disable_value="none",
            disable_hint="自适应 / Schedule-Free 优化器固定为常数学习率",
        ),
    )
    lr_scheduler_t0: int = Field(
        500, ge=1,
        description="cosine_with_restart 首次重启周期（单位：step）",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_with_restart", advanced=True),
    )
    lr_scheduler_t_mult: float = Field(
        2.0, ge=1.0,
        description="cosine_with_restart 每次重启后周期相对上轮的倍数（>1 周期递增）",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_with_restart", advanced=True),
    )
    lr_scheduler_eta_min: float = Field(
        1e-6, ge=0.0,
        description="学习率衰减下限：cosine 调度到此值后不再下降；通常远小于初始 lr（如初始 1e-4 配 1e-6）",
        json_schema_extra=_meta("training", show_when="lr_scheduler!=none", advanced=True),
    )
    lr_scheduler_warmup_steps: int = Field(
        100, ge=0,
        description="cosine_with_warmup 预热步数",
        json_schema_extra=_meta("training", show_when="lr_scheduler==cosine_with_warmup", advanced=True),
    )
    optimizer_type: Literal["adamw", "automagic", "came", "lion", "prodigy", "prodigy_plus_schedulefree", "soap", "soap_sf"] = Field(
        "adamw",
        description="优化器。adamw 标准基线；automagic 自适应每参数 lr（推荐 lr=1e-6）；came 置信度引导 + 分解二阶矩（state 显存低于 AdamW，lr 同 AdamW 量级）；lion 显存约 AdamW 一半（推荐 lr=AdamW lr / 3）；prodigy / prodigy_plus_schedulefree 自适应估 lr（lr 填 1.0）；soap Adam-in-Shampoo-eigenbasis 二阶预条件（拟合更快，lr 同 AdamW 量级）；soap_sf SOAP + Schedule-Free（lr_scheduler 固定 none）",
        json_schema_extra=_meta("training"),
    )
    prodigy_d_coef: float = Field(
        1.0, ge=0.1, le=10.0,
        description="Prodigy 估出的 d 整体缩放系数；越大有效 lr 越大。欠拟合时调高（2.0+），过拟合 / 小数据集时调低（0.5）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy"),
    )
    prodigy_safeguard_warmup: bool = Field(
        True,
        description="Prodigy warmup 期间防止 d 被初期高梯度推高；默认开启更稳",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy", advanced=True),
    )
    # ---------------------------- CAME 专属字段 ----------------------------
    # CAME = Confidence-guided Adaptive Memory Efficient（Luo et al. 2023,
    # arxiv 2307.02047）。lr 用 AdamW 量级真实值，可配常规 lr_scheduler。
    came_beta1: float = Field(
        0.9, ge=0.0, lt=1.0,
        description="CAME β1（update 动量 EMA 衰减）",
        json_schema_extra=_meta("training", show_when="optimizer_type==came", advanced=True),
    )
    came_beta2: float = Field(
        0.999, ge=0.0, lt=1.0,
        description="CAME β2（分解二阶矩 EMA 衰减）",
        json_schema_extra=_meta("training", show_when="optimizer_type==came", advanced=True),
    )
    came_beta3: float = Field(
        0.9999, ge=0.0, lt=1.0,
        description="CAME β3（置信度 instability EMA 衰减；越大置信度估计越平滑）",
        json_schema_extra=_meta("training", show_when="optimizer_type==came", advanced=True),
    )
    came_eps1: float = Field(
        1e-30, gt=0.0,
        description="CAME eps1（二阶矩正则项，防除零）",
        json_schema_extra=_meta("training", show_when="optimizer_type==came", advanced=True),
    )
    came_eps2: float = Field(
        1e-16, gt=0.0,
        description="CAME eps2（instability 下限正则；调大会压平逐坐标置信度差异并整体缩小步长）",
        json_schema_extra=_meta("training", show_when="optimizer_type==came", advanced=True),
    )
    came_clip_threshold: float = Field(
        1.0, gt=0.0,
        description="CAME update RMS 裁剪阈值（同 Adafactor 的 d）",
        json_schema_extra=_meta("training", show_when="optimizer_type==came", advanced=True),
    )
    lion_beta1: float = Field(
        0.9, ge=0.0, lt=1.0,
        description="Lion β1（动量插值系数）",
        json_schema_extra=_meta("training", show_when="optimizer_type==lion", advanced=True),
    )
    lion_beta2: float = Field(
        0.99, ge=0.0, lt=1.0,
        description="Lion β2（动量累计系数）",
        json_schema_extra=_meta("training", show_when="optimizer_type==lion", advanced=True),
    )
    automagic_variant: Literal["v1", "v2"] = Field(
        "v1",
        description="v1: per-element lr mask（经典，推荐）；v2: per-param scalar lr + fused backward（实验性，省显存；与 grad_accum / grad_clip / fp16 不兼容）",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic"),
    )
    automagic_min_lr: float = Field(
        1e-7, ge=0.0,
        description="Automagic 每参数学习率下限",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic", advanced=True),
    )
    automagic_max_lr: float = Field(
        1e-3, gt=0.0,
        description="Automagic 每参数学习率上限",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic", advanced=True),
    )
    automagic_lr_bump: float = Field(
        1e-6, ge=0.0,
        description="Automagic 同向/反向更新时调整每参数学习率的步幅",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic", advanced=True),
    )
    automagic_beta2: float = Field(
        0.999, ge=0.0, lt=1.0,
        description="Automagic 二阶矩 β2",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic", advanced=True),
    )
    automagic_eps: float = Field(
        1e-30, gt=0.0,
        description="Automagic 数值稳定项",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic", advanced=True),
    )
    automagic_clip_threshold: float = Field(
        1.0, gt=0.0,
        description="Automagic update RMS 裁剪阈值",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic", advanced=True),
    )
    automagic_agreement_threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description="v2 符号一致率阈值：超过此比例认为方向一致 → 涨 lr",
        json_schema_extra=_meta("training", show_when="optimizer_type==automagic&&automagic_variant==v2", advanced=True),
    )
    # ---------------- ProdigyPlusScheduleFree (PPSF) 专属字段 ----------------
    # 选 PPSF 时 lr_scheduler 必须为 none（Schedule-Free 不需要 scheduler，
    # 启动期校验会 fatal）。lr 强制 1.0（工厂内部覆盖）。
    ppsf_d_coef: float = Field(
        1.0, ge=0.1, le=10.0,
        description="PPSF 估出的 d 整体缩放系数；越大有效 lr 越大。欠拟合时调高（2.0+），过拟合 / 小数据集时调低（0.5）",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree"),
    )
    ppsf_prodigy_steps: int = Field(
        0, ge=0,
        description="PPSF 在第 N 步后冻结 d 估计；0 = 全程持续估计。建议总步数 1/4 ~ 1/2 让后期 lr 稳定",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_beta1: float = Field(
        0.9, ge=0.0, le=1.0,
        description="PPSF 一阶动量衰减率 (β1)：默认 0.9；越大平滑越强、响应越慢",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_beta2: float = Field(
        0.99, ge=0.0, le=1.0,
        description="PPSF 二阶动量衰减率 (β2)：默认 0.99；越大梯度方差估计越平滑、响应越慢",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    ppsf_split_groups: bool = Field(
        True,
        description="PPSF 按 param group 分别估计 d（LoRA 多组参数时让每组用各自适合的 lr）；默认开启",
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
        description="PPSF 启用 stable AdamW 风格归一化，防止单步梯度尺度异常；默认开启",
        json_schema_extra=_meta("training", show_when="optimizer_type==prodigy_plus_schedulefree", advanced=True),
    )
    # ------------------------- SOAP / SOAP-SF 专属字段 -------------------------
    # SOAP = Adam in the Shampoo eigenbasis（Vyas et al. 2024, arxiv 2409.11321）。
    # soap_sf = SOAP + Schedule-Free（arxiv 2405.15682）；选 soap_sf 时 lr_scheduler
    # 必须 none（启动期校验会 fatal），lr 用 AdamW 量级（不像 Prodigy 填 1.0）。
    soap_beta1: float = Field(
        0.95, ge=0.0, lt=1.0,
        description="SOAP β1。soap：Adam 一阶动量衰减；soap_sf：Schedule-Free 的 z↔x 插值权重（不是动量）。soap_sf 常用 0.9",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap||optimizer_type==soap_sf", advanced=True),
    )
    soap_beta2: float = Field(
        0.95, ge=0.0, lt=1.0,
        description="SOAP β2（二阶矩 / eigenbasis 协方差衰减）",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap||optimizer_type==soap_sf", advanced=True),
    )
    soap_precondition_frequency: int = Field(
        10, ge=1,
        description="每 N 步刷新一次 Shampoo 特征基：越大越省算力、特征基越旧。典型 5-20",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap||optimizer_type==soap_sf", advanced=True),
    )
    soap_max_precond_dim: int = Field(
        10000, ge=1,
        description="逐维阈值：某轴维度 ≤ 此值才建满秩二阶预条件，> 此值该轴退化为 Adam。设大（10000）让大特征维也做二阶=提速主来源；设小=SOAP-lite 省显存",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap||optimizer_type==soap_sf", advanced=True),
    )
    soap_shampoo_beta: float = Field(
        -1.0, le=1.0,
        description="Shampoo 协方差 EMA 衰减；< 0 时复用 β2（推荐）",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap||optimizer_type==soap_sf", advanced=True),
    )
    soap_precond_in_state: bool = Field(
        True,
        description="是否把可重算的 Shampoo 矩阵（GG/Q）存进 ckpt。False=ckpt 更小、resume 时冷重建特征基（从零训练不 resume 时零代价）",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap||optimizer_type==soap_sf", advanced=True),
    )
    soap_sf_weight_lr_power: float = Field(
        2.0, ge=0.0,
        description="Schedule-Free Polyak 权重里 lr 的幂；越大越偏向 lr 大的步",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap_sf", advanced=True),
    )
    soap_sf_r: float = Field(
        0.0, ge=0.0,
        description="Schedule-Free Polyak 权重里 step index 的幂（0=均匀平均；越大越偏向后期 iterate，短训练 x 追 z 更快）",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap_sf", advanced=True),
    )
    soap_sf_warmup_steps: int = Field(
        0, ge=0,
        description="Schedule-Free 线性 lr warmup 步数；SF 一般不需要，几步可稳定早期预条件估计",
        json_schema_extra=_meta("training", show_when="optimizer_type==soap_sf", advanced=True),
    )
    weight_decay: float = Field(
        0.0, ge=0.0,
        description="权重衰减：越大对权重的抑制越强、缓解过拟合；0 = 关闭，常用 0.001-0.1，过大会破坏训练",
        json_schema_extra=_meta("training", advanced=True),
    )
    kv_trim: bool = Field(
        False,
        description="Cross-attention KV trim：按实际 token 数裁到最近 bucket（64/128/256/512），减少 padding 计算量",
        json_schema_extra=_meta("system", advanced=True),
    )
    noise_enhancement_type: Literal["none", "offset", "pyramid"] = Field(
        "none",
        description="噪声增强机制（默认 none）。offset 在噪声上加 per-sample DC 偏置；pyramid 在多个尺度叠加低频噪声。两者机制不同，但都改变低频成分，互斥防双倍叠加。LoRA 训练默认保持 none",
        json_schema_extra=_meta(
            "noise_augmentation",
            advanced=True,
            disable_when="infonoise_enabled==true",
            disable_hint="InfoNoise 启用时禁用噪声增强（schema 互斥）",
        ),
    )
    noise_offset: float = Field(
        0.0, ge=0.0, le=0.2,
        description="DC 偏置强度（0-0.2，0=关闭）。让噪声 mean 偏离 0，让模型有机会学习生成极端亮度场景（pure black / pure white / 强对比）。典型范围 0.05-0.1；0.05 以下噪声场跟 baseline 几乎一样，超过 0.1 起点 loss 会显著偏高",
        json_schema_extra=_meta("noise_augmentation", show_when="noise_enhancement_type==offset", advanced=True),
    )
    pyramid_noise_iters: int = Field(
        0, ge=0, le=6,
        description="金字塔噪声层数（0-6，0=关闭）。每层在 spatial // 2^(k+1) 尺度注入。实际效果强度由 pyramid_noise_discount 决定 —— iters 单独决定覆盖的频段范围，discount 低时层数多少差异很小",
        json_schema_extra=_meta("noise_augmentation", show_when="noise_enhancement_type==pyramid", advanced=True),
    )
    pyramid_noise_discount: float = Field(
        0.5, ge=0.1, le=0.9,
        description="每层相对衰减系数（0.1-0.9）。控制低频强度的核心参数：anima 实现把整体噪声 std 归一化到 1，所以 discount 决定低频占比。0.1-0.4 归一化后噪声接近标准高斯，等价于关闭；0.5-0.7 显著改变低频结构",
        json_schema_extra=_meta("noise_augmentation", show_when="noise_enhancement_type==pyramid", advanced=True),
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
        description="采样分布。logit_normal 偏中段（SD3/Anima 默认）；uniform 等概率；mode 单峰偏移；mixed_* 混合 uniform 与偏置端（比例由 timestep_mix_low_prob 控制）",
        json_schema_extra=_meta(
            "timestep_sampling",
            alt_description="【时间步采样】InfoNoise 启用时作为热身期 baseline，正式阶段由自适应 CDF 接管；Leap 启用时 leap 路径恒用 U(0,1)，本字段仅作用于 (1-leap_ratio) 比例的标准 step",
            alt_description_when="infonoise_enabled==true||leap_enabled==true",
            advanced=True,
        ),
    )
    timestep_shift: float = Field(
        3.0, ge=0.1, le=10.0,
        description="logit-normal / mode 内部的分布偏移：>1 偏向高噪声端（粗结构），<1 偏向低噪声端（细节）",
        json_schema_extra=_meta(
            "timestep_sampling",
            show_when="timestep_sampling!=uniform",
            alt_description="InfoNoise 开启时作为热身阶段的 baseline shift，正式阶段由自适应 CDF 接管；Leap 启用时 leap 路径恒用 U(0,1)，本字段仅作用于 (1-leap_ratio) 比例的标准 step",
            alt_description_when="infonoise_enabled==true||leap_enabled==true",
            advanced=True,
        ),
    )
    timestep_mix_low_prob: float = Field(
        0.0, ge=0.0, le=1.0,
        description="mixed_* 模式下走偏置端的样本比例：0 = 全 uniform；典型 0.15-0.30",
        json_schema_extra=_meta(
            "timestep_sampling",
            show_when="timestep_sampling!=uniform",
            alt_description="InfoNoise 开启 + mixed_* baseline 时，热身阶段混合比例，正式阶段由自适应 CDF 接管；Leap 启用时 leap 路径恒用 U(0,1)，本字段仅作用于 (1-leap_ratio) 比例的标准 step",
            alt_description_when="infonoise_enabled==true||leap_enabled==true",
            advanced=True,
        ),
    )
    timestep_schedule_shift: float = Field(
        1.0, ge=0.1, le=10.0,
        description="采样后对 t 做的额外 σ schedule 偏移：1.0 = 无偏移；越大整体偏向高噪声端。与 timestep_shift 区别：作用于最终 t 而非 logit-normal 内部",
        json_schema_extra=_meta(
            "timestep_sampling",
            alt_description="InfoNoise 开启时仅热身期生效，正式阶段由自适应 CDF 接管；Leap 启用时 leap 路径恒用 U(0,1)，本字段仅作用于 (1-leap_ratio) 比例的标准 step",
            alt_description_when="infonoise_enabled==true||leap_enabled==true",
            advanced=True,
            disable_when="infonoise_enabled==true",
            disable_hint="InfoNoise 启用时禁用 schedule shift（schema 互斥，仅 1.0 兼容）",
        ),
    )
    infonoise_enabled: bool = Field(
        False,
        description="【InfoNoise】启用自适应时间步采样：训练中根据信息量自动调整 t 分布，聚焦更有效的训练区间",
        json_schema_extra=_meta(
            "timestep_sampling",
            advanced=True,
            disable_when=(
                "noise_enhancement_type!=none"
                "||loss_weighting!=none"
                "||loss_type==huber"
                "||timestep_schedule_shift!=1"
                "||leap_enabled==true"
            ),
            disable_hint="互斥字段（noise_enhancement / loss_weighting / loss_type / schedule_shift / leap）非默认时不可启用（schema 互斥）",
        ),
    )
    infonoise_K: int = Field(
        64, ge=16, le=256,
        description="【InfoNoise】log-σ 分箱数量（16-256）：越大分辨率越高但每箱样本越稀疏",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_N_warm: int = Field(
        0, ge=0,
        description="【InfoNoise】热身步数：0 = 自动取总步数的 1/5（最少 200 步）",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_M: int = Field(
        100, ge=10,
        description="【InfoNoise】采样分布刷新周期：每 M 步重算一次。越大计算开销越小、分布更新越滞后",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_B: int = Field(
        256, ge=32,
        description="【InfoNoise】每 bin 的 FIFO buffer 容量：越大平均越稳但响应越慢",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_beta: float = Field(
        0.9, ge=0.1, le=0.999,
        description="【InfoNoise】EMA 新值权重（论文 β 乘新值，非标准 EMA 方向）：0.9 表示新值占 90%；越大对最新分布响应越快",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_N_min: int = Field(
        50, ge=1,
        description="【InfoNoise】刷新触发条件：每个 bin 至少需要的样本数才会重算分布（必须 ≤ infonoise_B）",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    infonoise_gate_pivot_c: float = Field(
        0.15, ge=0.0, le=10.0,
        description="【InfoNoise】gate 函数 pivot c：默认 0.15（论文 §5 CIFAR 报告值，跨数据集鲁棒）；设 0 走自适应选取（论文 Eq 87 字面实现）；其他正数为自定义 c。多数情况保持默认",
        json_schema_extra=_meta("timestep_sampling", show_when="infonoise_enabled==true", advanced=True),
    )
    loss_type: Literal["mse", "huber"] = Field(
        "mse",
        description="训练 loss 类型。mse 经典；huber 对 outlier 鲁棒（在 |x|<δ 时用二次，|x|≥δ 时用线性）",
        json_schema_extra=_meta(
            "loss",
            disable_when="infonoise_enabled==true||leap_enabled==true",
            disable_hint="InfoNoise / Leap 启用时禁用 loss 类型切换（schema 互斥，仅 mse 兼容）",
        ),
    )
    huber_c: float = Field(
        0.15, ge=0.01, le=5.0,
        description="【Huber loss】delta 系数（控制二次/线性转折点）：越大越接近 MSE，越小越宽容 outlier。典型 0.1-0.3",
        json_schema_extra=_meta("loss", show_when="loss_type==huber", advanced=True),
    )
    loss_weighting: Literal["none", "min_snr", "detail_inv_t", "cosmap"] = Field(
        "none",
        description="loss 加权方案：none 不加权；min_snr 抑制极端时步的权重；detail_inv_t 强化低 t 细节；cosmap 用 SD3 cosine 映射",
        json_schema_extra=_meta(
            "loss",
            disable_when="infonoise_enabled==true||leap_enabled==true",
            disable_hint="InfoNoise / Leap 启用时禁用 loss 加权（schema 互斥，仅 none 兼容）",
        ),
    )
    min_snr_gamma: float = Field(
        5.0, ge=0.1, le=20.0,
        description="Min-SNR 阈值：高 SNR 简单步（低 t 端）的权重压制阈值。默认 5.0；越小压制越强",
        json_schema_extra=_meta("loss", show_when="loss_weighting==min_snr", advanced=True),
    )
    weight_cap_ratio: float = Field(
        0.0, ge=0.0, le=50.0,
        description="Batch 内权重 max/min 比上限：限制极端权重影响。0 = 禁用；小 batch + Prodigy 建议 5",
        json_schema_extra=_meta("loss", show_when="loss_weighting!=none", advanced=True),
    )
    detail_inv_t_min: float = Field(
        1.0, ge=1.0, le=20.0,
        description="detail_inv_t 权重下限。默认 1.0；升至 1.5 让高 t 步也略微加权（<1.0 因 1/t≥1 恒成立故无效）",
        json_schema_extra=_meta("loss", show_when="loss_weighting==detail_inv_t", advanced=True),
    )
    detail_inv_t_max: float = Field(
        5.0, ge=0.1, le=50.0,
        description="detail_inv_t 权重上限。默认 5.0；降低（如 3）减弱细节强化，提高（如 8）激进强化细节",
        json_schema_extra=_meta("loss", show_when="loss_weighting==detail_inv_t", advanced=True),
    )
    leap_enabled: bool = Field(
        False,
        description="【LeapAlign 自蒸馏】启用两步跳跃自蒸馏（去奖励模型版）：每步用真实 latent 当 x0，per-sample 采两个时刻 (k>j) 做两步跳跃，loss=MSE(两步预测的 x̂0, 真实 x0)。本质是 shortcut/consistency 式自蒸馏。开销：leap_ratio=1.0 时每步 2 次前向 ≈ 2× 算力 + activation 显存接近 2×（两次前向都带 grad）。与 InfoNoise / loss_weighting / loss_type=huber 互斥",
        json_schema_extra=_meta(
            "loss",
            advanced=True,
            disable_when="infonoise_enabled==true||loss_weighting!=none||loss_type==huber",
            disable_hint="互斥字段（InfoNoise / loss_weighting / loss_type=huber）非默认时不可启用（schema 互斥，与对侧形成对称锁）",
        ),
    )
    leap_ratio: float = Field(
        0.6, ge=0.0, le=1.0,
        description="【LeapAlign 混合训练】每步按此概率走 leap 自蒸馏、其余走传统 rectified flow：1.0 纯 leap（管全局结构）；0.0 纯传统（管细节锐度）；0.6 大头吃 leap 全局对齐、留点传统精修兜住细节。两股梯度叠在同一组 LoRA 权重上各取所长",
        json_schema_extra=_meta("loss", show_when="leap_enabled==true", advanced=True),
    )
    leap_variant: Literal["original", "sparse", "bridge", "lagrange"] = Field(
        "original",
        description="【LeapAlign/FlowBP】轨迹自蒸馏变体（统一形式：解析构造轨迹点+沿轨迹积分 x̂0+MSE(x̂0,真实x0)）：original=两步跳+straight-through connector（K=2，1 雅可比，行为同历史版，默认）；sparse=K 点 Euler 重放纯直接项求和（FlowBP-Sparse，零 connector/零雅可比，K 点稠密监督，最稳，代价 K× 前向+K× 显存，K 由 leap_activation_k 控）；bridge=两步跳+Euler 重构 connector（FlowBP-Bridge，无 straight-through 偏差）；lagrange=两段跳每段三点 Lagrange/Simpson 积分（FlowBP-Lagrange，6× 前向，单段积分误差 O(Δt²)→O(Δt⁵)，论文 §A.2）。注：自蒸馏下真值是解析直线插值点、无 rollout 噪声，connector 残差被釜底抽薪，故 bridge/lagrange 相比 original 增益收窄，sparse 是唯一结构性差异",
        json_schema_extra=_meta("loss", show_when="leap_enabled==true", advanced=True),
    )
    leap_activation_k: int = Field(
        3, ge=2, le=8,
        description="【FlowBP-Sparse】激活集大小 K：沿 (0,1) 分层抖动采 K 个降序时刻做 Euler 重放，K 点全带梯度。K 直接决定显存/算力（K× 前向+K× activation 显存）与监督稠密度。3 是显存与稠密度的平衡点（比 original 的 2× 略重）；消费级小显存可设 2（退化到 original 同档显存）；4+ 监督更密但 12G 卡可能吃紧。仅 sparse 变体生效",
        json_schema_extra=_meta("loss", show_when="leap_enabled==true&&leap_variant==sparse", advanced=True),
    )
    leap_nested_grad_coe: float = Field(
        0.3, ge=0.0, le=1.0,
        description="【LeapAlign】梯度折扣 α（论文 Eq 9）：缩放第二跳对 x_j 的嵌套梯度。0=砍掉嵌套梯度（最省显存），1=不折扣（梯度最完整但易爆）。论文最优 0.3。对 original/bridge/lagrange 生效；sparse 零 connector/零雅可比不使用此参数",
        json_schema_extra=_meta("loss", show_when="leap_enabled==true&&leap_variant!=sparse", advanced=True),
    )
    leap_min_gap: float = Field(
        0.1, ge=0.01, le=0.9,
        description="【LeapAlign】两个采样时刻 (k,j) 的最小间隔：越大跳跃跨度越大、自蒸馏越激进但误差累积越多。典型 0.1-0.3。仅 original/bridge/lagrange 生效；sparse 的激活集用分层抖动铺满 (0,1)，不用此字段",
        json_schema_extra=_meta("loss", show_when="leap_enabled==true&&leap_variant!=sparse", advanced=True),
    )
    leap_traj_sim_weighting: bool = Field(
        False,
        description="【LeapAlign】轨迹相似度加权（论文 Eq 12）：跳跃越贴近真实路径权重越高，抑制大跨度跳跃的离谱预测主导 loss。默认关",
        json_schema_extra=_meta("loss", show_when="leap_enabled==true", advanced=True),
    )
    leap_traj_sim_min: float = Field(
        0.1, ge=1e-4,
        description="【LeapAlign】轨迹相似度加权下限 τ：防止近乎相同的跳跃对被 1/d 过度放大。越小越激进。典型 0.05-0.2",
        json_schema_extra=_meta("loss", show_when="leap_traj_sim_weighting==true", advanced=True),
    )

    # ----------------------------------------------------------- SRA v2 表征对齐
    sra_enabled: bool = Field(
        False,
        description="【SRA v2 表征对齐】启用 VAE Self-Representation Alignment：将中间 transformer block 的 hidden state 对齐到 clean VAE latent，加速收敛并正则化表征。仅增加 ~4% GFLOPs（一个轻量 MLP），训练完自动丢弃",
        json_schema_extra=_meta("loss", advanced=True),
    )
    sra_block: int = Field(
        4, ge=1, le=35,
        description="【SRA v2】从哪一层 block 取中间表征做对齐（0-indexed）。论文建议浅层效果最好",
        json_schema_extra=_meta("loss", show_when="sra_enabled==true", advanced=True),
    )
    sra_weight: float = Field(
        0.2, ge=0.0,
        description="【SRA v2】对齐 loss 权重 λ：align_loss 乘以此值后加到总 loss。trainer 默认 0.2，过大会导致异常",
        json_schema_extra=_meta("loss", show_when="sra_enabled==true", advanced=True),
    )
    sra_normalize: bool = Field(
        True,
        description="【SRA v2】对 projected/target 各自做 per-sample z-score 标准化后再算 smooth-L1（论文 cosine 消融的同族思路：幅度无关、只对齐结构）。原论文用 SD-VAE（latent ~单位尺度）从零训练故不归一化；本项目视频 VAE latent 尺度不同 + LoRA 微调，关闭会导致 align loss 比 denoise 高几个量级并很快崩坏。建议保持开启",
        json_schema_extra=_meta("loss", show_when="sra_enabled==true", advanced=True),
    )
    sra_decay_type: Literal["none", "linear", "cosine", "jump"] = Field(
        "linear",
        description="【SRA v2】权重衰减方式：none 全程固定；linear 从起点线性降到 0；cosine 从起点余弦降到 0；jump 到起点直接关掉。实际权重 = sra_weight × 衰减系数",
        json_schema_extra=_meta("loss", show_when="sra_enabled==true", advanced=True),
    )
    sra_decay_start_ratio: float = Field(
        0.2, ge=0.0, le=1.0,
        description="【SRA v2】衰减起点（训练总步数比例）。linear/cosine 在此之前保持满权重；jump 在此比例直接从 sra_weight 跳到 0",
        json_schema_extra=_meta("loss", show_when="sra_enabled==true&&sra_decay_type!=none", advanced=True),
    )
    sra_decay_end_ratio: float = Field(
        0.3, ge=0.0, le=1.0,
        description="【SRA v2】衰减终点（训练总步数比例）。linear/cosine 到此比例降为 0；jump 不使用此字段",
        json_schema_extra=_meta("loss", show_when="sra_enabled==true&&sra_decay_type!=none&&sra_decay_type!=jump", advanced=True),
    )

    grad_clip_max_norm: float = Field(
        1.0, ge=0.0,
        description="梯度裁剪最大范数：当本步所有可训练参数的梯度全局范数超过该值时按比例缩到该值，防止单步极端梯度把模型推飞；默认 1.0 适合绝大多数场景，bf16+DoRA/LoKr 不稳可降到 0.5，0=禁用",
        json_schema_extra=_meta("training", advanced=True),
    )

    mixed_precision: Literal["bf16", "fp16", "no"] = Field(
        "bf16",
        description="训练精度。bf16 推荐（与 fp32 同动态范围、稳定）；fp16 同显存但动态范围小、梯度易溢出；no 用 fp32 最稳但显存翻倍",
        json_schema_extra=_meta("system"),
    )

    attention_backend: AttentionBackend = Field(
        "flash_attn",
        description="Attention 后端。none = PyTorch SDPA 默认；xformers 显存更省；flash_attn 最快（需 Ampere+ GPU 支持）",
        json_schema_extra=_meta("system"),
    )
    num_workers: int = Field(
        0, ge=0,
        description="数据加载并行线程数；越大加载越快但内存占用上升。Windows 必须填 0",
        json_schema_extra=_meta("system", advanced=True),
    )

    @field_validator("resolution", mode="before")
    @classmethod
    def _normalize_resolution(cls, v: Any) -> list[int]:
        """标量 / 列表 / 旧 config 标量 → 归一成 list[int]。

        各值 snap 到最近的 64 倍数（half-up，与前端 `Math.round` + dataset 的
        `_parse_folder_meta` 一致，避免偏心桶）+ clamp 到 [256, 4096]，再**去重保序**
        （否则 `[1000, 1024]` snap 后都成 1024 → 该档被 fan-out 训两遍 + 直方图双计数）。
        """
        if v is None:
            return [1024]
        if isinstance(v, (int, float, str)):
            v = [v]
        out: list[int] = []
        seen: set[int] = set()
        for x in v:
            n = int((float(x) + 32) // 64) * 64  # round-half-up to /64
            n = max(256, min(4096, n))
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out or [1024]

    @model_validator(mode="before")
    @classmethod
    def _migrate_save_keys(cls, data: Any) -> Any:
        return migrate_legacy_save_keys(data)

    @model_validator(mode="before")
    @classmethod
    def _migrate_noise_enhancement(cls, data: Any) -> Any:
        return migrate_noise_enhancement_type(data)

    @model_validator(mode="before")
    @classmethod
    def _coerce_sample_sampler_scheduler(cls, data: Any) -> Any:
        """sampler/scheduler 收紧为 Literal 前是自由文本，旧 preset / 旧版本
        config 可能存了其他值（如 euler，当年走 inline Euler 兜底）。统一
        归并到默认值而不是让整个 config 加载失败。"""
        if isinstance(data, dict):
            if data.get("sample_sampler_name") not in (None, "er_sde", "dpmpp_3m_sde"):
                data["sample_sampler_name"] = "er_sde"
            if data.get("sample_scheduler") not in (None, "simple", "sgm_uniform"):
                data["sample_scheduler"] = "simple"
        return data

    @model_validator(mode="after")
    def _validate_prodigy_scheduler(self) -> "TrainingConfig":
        """Prodigy / Automagic / Schedule-Free 系列固定使用常数学习率，外部 scheduler 统一拦截。"""
        if self.optimizer_type in {"automagic", "prodigy", "prodigy_plus_schedulefree", "soap_sf"} and self.lr_scheduler != "none":
            raise ValueError(
                f"optimizer_type={self.optimizer_type} requires lr_scheduler=none "
                "(自适应优化器固定使用常数学习率)."
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

    @model_validator(mode="after")
    def _validate_sra_decay_range(self) -> "TrainingConfig":
        """SRA linear/cosine 衰减需要 start <= end；jump 只读 start。"""
        if self.sra_decay_type in {"linear", "cosine"} and self.sra_decay_start_ratio > self.sra_decay_end_ratio:
            raise ValueError(
                f"sra_decay_start_ratio ({self.sra_decay_start_ratio}) 不能大于 "
                f"sra_decay_end_ratio ({self.sra_decay_end_ratio})。"
            )
        return self

    @model_validator(mode="after")
    def _validate_infonoise_loss_weighting_exclusive(self) -> "TrainingConfig":
        """InfoNoise 与 loss_weighting 互斥：两者都在做 schedule 重塑，叠加会互相消磨。

        InfoNoise 用未加权 MSE 估各噪声区间的信息量（论文 entropy rate 推导的必要前提，
        见 arxiv 2602.18647 §3.1）；loss_weighting 改实际优化目标。同开时 InfoNoise 学
        到的分布跟用户配的 loss_weighting 方向冲突（如 detail_inv_t 抬高低 t 权重 vs
        InfoNoise 默认压低低 t 采样）。强制二选一避免 silent 不一致。
        """
        if self.infonoise_enabled and self.loss_weighting != "none":
            raise ValueError(
                f"infonoise_enabled=true 与 loss_weighting={self.loss_weighting!r} 互斥："
                "两个机制都在做 schedule 重塑（前者自适应 resample，后者手工 reweight）。"
                "请二选一：(a) 关闭 InfoNoise 走传统 loss_weighting 路径；"
                "或 (b) 设 loss_weighting=none 走 InfoNoise 自适应路径。"
            )
        return self

    @model_validator(mode="after")
    def _validate_infonoise_n_min_le_b(self) -> "TrainingConfig":
        """N_min > B 会让自适应分布永远学不出来（FIFO 容量不够触发刷新）。"""
        if self.infonoise_enabled and self.infonoise_N_min > self.infonoise_B:
            raise ValueError(
                f"infonoise_N_min ({self.infonoise_N_min}) 不能大于 "
                f"infonoise_B ({self.infonoise_B})：超出会让自适应分布永远学不出来。"
            )
        return self

    @model_validator(mode="after")
    def _validate_infonoise_schedule_shift_exclusive(self) -> "TrainingConfig":
        """InfoNoise 与 timestep_schedule_shift 互斥：InfoNoise CDF 接管后 shift 静默失效。

        timestep_schedule_shift 仅在 sample_t 的 baseline 路径生效；InfoNoise sample()
        走 CDF 路径直接返回 t，不再应用 shift。同开时用户期望的"全程偏移"会在 warmup
        结束后悄悄消失。强制二选一，避免 silent 行为切换。
        """
        if self.infonoise_enabled and self.timestep_schedule_shift != 1.0:
            raise ValueError(
                f"infonoise_enabled=true 与 timestep_schedule_shift={self.timestep_schedule_shift} 互斥："
                "InfoNoise 自适应 CDF 接管后 schedule_shift 不再生效，会在 warmup 结束时"
                "悄悄切换行为。请二选一：(a) 关闭 InfoNoise 保留 schedule_shift；"
                "或 (b) 设 timestep_schedule_shift=1.0 走 InfoNoise 自适应路径。"
            )
        return self

    @model_validator(mode="after")
    def _validate_infonoise_loss_type_exclusive(self) -> "TrainingConfig":
        """InfoNoise 与 loss_type=huber 互斥：huber 削峰让 InfoNoise 推 mass 进死循环。

        InfoNoise 用 raw MSE（不削峰）估各噪声区间的信息量，huber 让模型对 outlier
        不学。某区间 outlier 多时，InfoNoise 看到 raw MSE 仍高 → 推 mass 过去 → huber
        让模型仍然不学那里 → raw MSE 仍高 → InfoNoise 继续推 mass 过去（反馈环）。
        """
        if self.infonoise_enabled and self.loss_type == "huber":
            raise ValueError(
                "infonoise_enabled=true 与 loss_type=huber 互斥：huber 对 outlier 鲁棒"
                "（不学），但 InfoNoise 用 raw MSE 看到 outlier 区间高损失会持续把采样推过去，"
                "形成 mass 集中在不学的区间的反馈环。请二选一：(a) 关闭 InfoNoise 保留 huber；"
                "或 (b) 设 loss_type=mse 走 InfoNoise 自适应路径。"
            )
        return self

    @model_validator(mode="after")
    def _validate_infonoise_noise_enhancement_exclusive(self) -> "TrainingConfig":
        """InfoNoise 与 noise_enhancement_type 互斥：噪声增强改变 noise 形状会让 InfoNoise
        schedule 偏离论文最优。

        InfoNoise 论文 I-MMSE 推导假设标准高斯 noise；offset 加 DC 偏置、pyramid 加多尺度
        低频成分都改变 noise 频谱，让 InfoNoise 学到的不再是 clean entropy rate profile。
        """
        if self.infonoise_enabled and self.noise_enhancement_type != "none":
            raise ValueError(
                f"infonoise_enabled=true 与 noise_enhancement_type={self.noise_enhancement_type!r} 互斥："
                "噪声增强会改变 noise 形状，让 InfoNoise 学到的 schedule 偏离论文最优"
                "（I-MMSE 推导假设标准高斯 noise）。请二选一：(a) 关闭 InfoNoise 保留噪声增强；"
                "或 (b) 设 noise_enhancement_type=none 走 InfoNoise 自适应路径。"
            )
        return self

    @model_validator(mode="after")
    def _validate_leap_exclusive(self) -> "TrainingConfig":
        """LeapAlign/FlowBP 自蒸馏与 InfoNoise / loss_weighting / loss_type=huber 互斥。

        leap 路径（任意 variant）每步 per-sample 采多个时刻沿代理轨迹积分（original/bridge/
        lagrange 采两个时刻 (k,j)，sparse 采 K 个时刻），不存在单一 t，且 loss 在 leap.py
        里写死 MSE(x̂0, x0)：
        - InfoNoise：用单一 t 的 raw MSE 学 I-MMSE schedule，多 timestep 无从 record；
          且 leap 的 loss 是 x̂0 自蒸馏 MSE，不是 v 预测 MSE，语义也不匹配。
        - loss_weighting：min_snr / detail_inv_t / cosmap 全都按单一 t 算 SNR 权重，
          多 timestep 没有定义；leap 自带 traj_sim_weighting 做轨迹质量加权。
        - loss_type=huber：leap.py 直接走 (x̂0-x0)**2 内联 MSE，绕过 ctx.loss_fn，开了
          huber 会被静默无视。
        故 leap 路径在 loop.py 里有意跳过这三个机制，这里强制配置层面关闭，避免用户
        以为开了却被静默忽略。三条互斥对全部四个 variant 一致成立。
        """
        if self.leap_enabled:
            if self.infonoise_enabled:
                raise ValueError(
                    "leap_enabled=true 与 infonoise_enabled=true 互斥：leap 每步沿代理轨迹采"
                    "多个时刻积分，没有 InfoNoise 学 I-MMSE 所需的单一 t，且 loss 是 x̂0 "
                    "自蒸馏而非 v 预测 MSE。请二选一：(a) 关闭 leap 用 InfoNoise；"
                    "或 (b) 设 infonoise_enabled=false 走 leap 自蒸馏。"
                )
            if self.loss_weighting != "none":
                raise ValueError(
                    f"leap_enabled=true 与 loss_weighting={self.loss_weighting!r} 互斥：loss 加权"
                    "按单一 t 算 SNR 权重，leap 的多 timestep 无从定义；leap 用 "
                    "leap_traj_sim_weighting 做轨迹质量加权。请二选一：(a) 关闭 leap；"
                    "或 (b) 设 loss_weighting=none 走 leap 自蒸馏。"
                )
            if self.loss_type == "huber":
                raise ValueError(
                    "leap_enabled=true 与 loss_type=huber 互斥：leap.py 写死 MSE(x̂0, x0) "
                    "内联自蒸馏目标，绕过 ctx.loss_fn 分发，huber 会被静默忽略。"
                    "请二选一：(a) 关闭 leap；或 (b) 设 loss_type=mse 走 leap 自蒸馏。"
                )
        return self

    @model_validator(mode="after")
    def _validate_navit_exclusive(self) -> "TrainingConfig":
        """NaViT / Patch-n-Pack 打包训练的配置校验与互斥。

        NaViT 改变了 batch 语义（一包异构图、逐图 t），许多按"批量网格 + 逐 batch
        单 timestep"假设写的特性会语义错位。v1 的策略是 fail-fast：同时开 → 启动即
        报错，提示显式关掉，避免"开着却悄悄不生效"的隐性行为改变。

        互斥项（详见 docs/navit-packing.md 第 3 节）：
        - leap_enabled：leap 假设批量网格 + 共享 noise，与逐图打包不兼容。
        - infonoise_enabled：InfoNoise 的 I-MMSE record 需走标准路径的 per-sample
          MSE；navit 路径的 per-image loss 语义不同，v1 未适配。
        - lora_type=tlora：T-LoRA 的 rank mask 按 batch 维匹配 timestep，navit 是
          B=1 打包序列 + 逐图 t，维度不匹配。
        - sra_enabled：SRA v2 表征对齐按批量网格中间特征 tap 对齐，navit 是 B=1
          打包序列 + 逐图，per-image 适配 v1 未做（loop 路径已守卫跳过，这里 fail-fast）。

        前置要求：
        - cache_latents：navit 打包按 latent token 数预算分包，需要预编码缓存。
        - navit_token_budget > 0：必须显式设置（按显存定）。
        - navit_multiscale → navit_native_resolution：副本经原生定尺寸路径生成。
        """
        if not self.navit_packing:
            return self

        _conflicts = []
        if self.leap_enabled:
            _conflicts.append("leap_enabled")
        if self.infonoise_enabled:
            _conflicts.append("infonoise_enabled")
        if self.lora_type == "tlora":
            _conflicts.append("lora_type=tlora")
        if self.sra_enabled:
            _conflicts.append("sra_enabled")
        if _conflicts:
            raise ValueError(
                "navit_packing(v1) 暂不支持与以下特性同时开启："
                + ", ".join(_conflicts)
                + "。请关掉它们，或暂不使用 NaViT 打包。"
            )

        if not self.cache_latents:
            raise ValueError(
                "navit_packing 需要 cache_latents=true（打包按 latent token 数预算分包，"
                "需要预编码缓存）。"
            )
        if self.navit_token_budget <= 0:
            raise ValueError(
                "navit_packing 需要显式设置 navit_token_budget（>0，按显存定，"
                "见 docs/navit-packing.md 显存对照表）。"
            )
        if self.navit_multiscale and not self.navit_native_resolution:
            raise ValueError(
                "navit_multiscale 需要 navit_native_resolution=true"
                "（副本经原生定尺寸路径生成；ARB 桶定尺寸暂不支持）。"
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
        description="训练随机种子",
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
    sample_sampler_name: Literal["er_sde", "dpmpp_3m_sde"] = Field(
        "er_sde",
        description="采样器。er_sde 默认；dpmpp_3m_sde 与 ComfyUI 同款"
                    "（BrownianTree 噪声，需要 torchsde）",
        json_schema_extra=_meta("sample"),
    )
    sample_scheduler: Literal["simple", "sgm_uniform"] = Field(
        "simple",
        description="调度器。simple 默认；sgm_uniform 为 ComfyUI 的 SGM 均匀切分",
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
        description="单 prompt 模式：训练中所有采样图共用此 prompt（设置 sample_prompts 时被忽略）",
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

    # ----------------------------------------------------------- 训练后指标评估
    eval_validation_enabled: bool = Field(
        False,
        description="训练结束后用验证集（held-out）出图并计算 CLIP-T / CLIP-I / DINO-I 指标。"
                    "验证集是从训练集划出、不参与训练的图，放在与 train/ 同级的 validation/。",
        json_schema_extra=_meta("eval_validation"),
    )
    eval_validation_split_ratio: float = Field(
        0.0, ge=0.0, le=1.0,
        description="训练开始前从数据集随机划入验证集的比例（0–1，0=不自动划分）。"
                    "推荐约 0.1。按比例补足：验证集已达到该比例就不再划分；数据集很小时取整后可能为 0。",
        json_schema_extra=_meta("eval_validation", show_when="eval_validation_enabled==true"),
    )
    eval_validation_split_seed: int = Field(
        0, ge=0,
        description="验证集随机划分的种子，固定后划分结果可复现。",
        json_schema_extra=_meta("eval_validation", show_when="eval_validation_enabled==true"),
    )

    # WandB：0.18 起 per-config 覆盖块整体移除 —— wandb 属于账号/工作流级配置,
    # 不随项目变化;api_key/entity/base_url 写进 yaml 会随预设分享/bundle 导出/
    # 任务快照明文外泄。全局配置在 Settings(secrets.WandBConfig),经 supervisor
    # 注入 WANDB_* 环境变量到训练进程,secrets 不落盘任何 yaml。
    # 老 yaml 里的 wandb_* 键由 _tolerant_validate 当未知字段丢弃(dropped_fields
    # 提示),argparse bridge 对未知键直接跳过。

    # ---------------------------------------------------------------- 监控/进度
    # 这一组对 Studio 用户全部隐藏（hidden=True）—— Studio 跑训练用 subprocess 把
    # stdout 重定向到 task log（非 tty），这些「终端体验」字段对 web 用户没意义；
    # monitor 页用的是 monitor_state.json，跟这些值零相关。
    # 字段保留在 schema 只为裸 CLI 用户仍可在 yaml 手动覆盖；等于默认值时
    # config_prune 不落盘（hidden 裁剪），非默认覆盖照常保留。
    # 旧的 HTTP monitor server 字段（no_monitor / monitor_host / monitor_port /
    # no_browser）已随 server 一并删除，见 migrations.RETIRED_MONITOR_KEYS。
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
