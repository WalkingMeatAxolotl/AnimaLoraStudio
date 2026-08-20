"""models_phase: paths + family-owned weights + LoRA injection.

Cached-varlen families may defer their large DiT until dataset captions have
been cached and the text encoder released. ``finish(ctx)`` closes that deferred
half immediately after ``text_cache`` and before optimizer construction.
"""

from __future__ import annotations

import logging
from pathlib import Path

from studio.infrastructure.log_messages import msg
from training.context import TrainingContext
from training.families import resolve_family
from training.families.anima import ANIMA_SPEC as _ANIMA_SPEC
from training.sysmem import log_vram
from training.model_loading import (
    find_diffusion_pipe_root,
    resolve_path_best_effort,
)


logger = logging.getLogger(__name__)

#: 日志里 lora_type 的规范大小写。裸 ``.upper()`` 会打出「LOKR」这种不成词的
#: 拼写；schema 的 Literal 集合是权威来源，新增变体在此补一行。
_LORA_TYPE_LABELS = {
    "lora": "LoRA",
    "lokr": "LoKr",
    "loha": "LoHa",
    "ortho": "Ortho",
    "tlora": "T-LoRA",
}


def _resolve_paths(ctx: TrainingContext) -> None:
    args = ctx.args
    ctx.repo_root = find_diffusion_pipe_root()
    logger.debug("model_code: path=%s", ctx.repo_root)

    phases_dir = Path(__file__).resolve().parent
    training_dir = phases_dir.parent
    runtime_dir = training_dir.parent
    bases = [
        Path.cwd(),
        ctx.config_dir,
        ctx.config_dir.parent if ctx.config_dir else None,
        runtime_dir,
        runtime_dir.parent,
        ctx.repo_root,
        ctx.repo_root.parent,
    ]
    args.transformer_path = resolve_path_best_effort(args.transformer_path, bases)
    args.vae_path = resolve_path_best_effort(args.vae_path, bases)
    args.text_encoder_path = resolve_path_best_effort(args.text_encoder_path, bases)
    args.t5_tokenizer_path = resolve_path_best_effort(args.t5_tokenizer_path, bases)
    args.data_dir = resolve_path_best_effort(args.data_dir, bases)
    reg_data_dir = getattr(args, "reg_data_dir", "") or ""
    if reg_data_dir:
        args.reg_data_dir = resolve_path_best_effort(reg_data_dir, bases)


def _swap_vram_discount(ctx: TrainingContext) -> float:
    """开 block swap 时不会进显存的权重**比例**（预算折扣，见 check_load_budget）。

    比例而非字节：fp8 与 bf16 文件大小差一倍，按字节折扣会在 fp8 场景折扣穿。
    族未实现估算就返回 0（护栏退化成保守，不会误放行）。
    """
    blocks_to_swap = int(getattr(ctx.args, "blocks_to_swap", 0) or 0)
    if blocks_to_swap <= 0:
        return 0.0
    ratio_fn = getattr(ctx.family, "swapped_param_ratio", None)
    if ratio_fn is None:
        return 0.0
    try:
        # checkpoint_path：anima 靠它区分 28/36 层版本；krea2 结构唯一、忽略
        return float(ratio_fn(
            blocks_to_swap,
            checkpoint_path=str(getattr(ctx.args, "transformer_path", "") or ""),
        ))
    except Exception:  # noqa: BLE001
        return 0.0


def _load_dit(ctx: TrainingContext) -> None:
    args = ctx.args
    backend = getattr(args, "attention_backend", "flash_attn")
    if backend == "none":
        logger.info(msg("train.attention_sdpa"))
    logger.info(msg("train.loading_transformer"))
    extra = {}
    blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
    if blocks_to_swap > 0:
        # 能力位在 schema 侧已用 cap_gate 门控；裸 CLI / 旧 yaml 仍可能带上，
        # 这里 fail-fast 而非静默忽略（否则用户以为省了显存其实没有）
        if "block_swap" not in ctx.family.spec.capabilities:
            raise RuntimeError(
                f"model_family='{ctx.family.spec.family_id}' 不支持 block swap，"
                f"但 blocks_to_swap={blocks_to_swap}。请置 0。"
            )
        # 换出层由 loader 直接落 CPU pinned，不经过显存（12/16GB 目标的前提）
        extra["blocks_to_swap"] = blocks_to_swap
        logger.debug(
            "block_swap: planned swap_blocks=%d (tail blocks stay in pinned memory, "
            "never loaded to VRAM)", blocks_to_swap,
        )
    ctx.model = ctx.family.load_dit(
        args.transformer_path,
        ctx.device,
        ctx.dtype,
        attention_backend=backend,
        repo_root=ctx.repo_root,
        **extra,
    )
    # 大权重 mmap 缓存页归还系统（13-26GB；真机换页卡死案例，训练同样受益）
    from training.sysmem import trim_working_set

    trim_working_set()
    log_vram("train.vram_stage_transformer_loaded", ctx.device)


def _log_train_start_vram(ctx: TrainingContext) -> None:
    """训练循环开始前的显存基线 —— 判断 blocks_to_swap 实际效果的读数点。"""
    swap = getattr(ctx, "block_swap", None)
    if swap is not None:
        logger.info(msg(
            "train.block_swap_active",
            n=swap.num_swap, total=swap.total,
            pinned=f"{swap.pinned_bytes / 1024**3:.2f}",
        ))
    log_vram("train.vram_stage_train_start", ctx.device)


def _load_vae(ctx: TrainingContext) -> None:
    args = ctx.args
    logger.info(msg("train.loading_vae"))
    ctx.vae = ctx.family.load_vae(
        args.vae_path,
        ctx.device,
        ctx.vae_dtype,
        tiling=getattr(args, "vae_tiling", "auto"),
    )


def _load_text(ctx: TrainingContext) -> None:
    args = ctx.args
    logger.info(msg("train.loading_text_encoder"))
    ctx.text_stack = ctx.family.load_text(
        args.text_encoder_path,
        ctx.device,
        ctx.dtype,
        t5_tokenizer_path=args.t5_tokenizer_path,
        cache_enabled=bool(getattr(args, "text_encoder_cache", True)),
    )


def _setup_block_swap(ctx: TrainingContext) -> None:
    """构造并挂载 block swap（docs/design/block-swap.md 刀 2）。

    **必须在 LoRA 注入之后**：LyCORIS ``apply_to()`` 会读基权重的 shape 建适配器，
    此时换出层的权重是 loader 落下的 CPU pinned 张量（shape/dtype 完好，只是不在
    显存），注入正常；反过来若先 attach 再注入也可行，但没有理由把顺序搞复杂。

    挂载走 hook（``attach()``），不改模型 forward 循环 —— krea2 的循环在
    parity 敏感的 ``modeling/`` 内，anima 的手工展开循环也不必动。前向 +
    反向四个钩子缺一不可（反向必须自己取回权重，重算不触发 forward_hook，
    见 doc §9.10 与 tests/test_block_swap_grad_fidelity.py）。
    """
    blocks_to_swap = int(getattr(ctx.args, "blocks_to_swap", 0) or 0)
    if blocks_to_swap <= 0:
        return
    from training.block_swap import PinnedBlockSwap

    # clamp 到总层数：blocks_to_swap 是跨族/跨版本共享的设置值（krea2 28 层、
    # anima 28/36 层），超界按全量换出处理而非 fail（loader 侧同口径）
    ctx.block_swap = PinnedBlockSwap(
        ctx.model.blocks, min(blocks_to_swap, len(ctx.model.blocks)), ctx.device,
    )
    ctx.block_swap.attach()
    log_vram("train.vram_stage_block_swap_ready", ctx.device)


def _inject_adapter(ctx: TrainingContext) -> None:
    args = ctx.args
    lora_type = str(args.lora_type)
    logger.info(msg(
        "train.injecting_lora",
        lora_type=_LORA_TYPE_LABELS.get(lora_type, lora_type),
    ))
    from training.adapters import build_adapter

    ctx.injector = build_adapter(args, preset=ctx.family.lora_preset())
    ctx.injector.metadata_extra = ctx.family.lora_metadata()
    ctx.injector.inject(ctx.model)

    if getattr(args, "resume_lora", "") and Path(args.resume_lora).exists():
        lora_family = _read_lora_family(args.resume_lora)
        if lora_family != ctx.family.spec.family_id:
            raise RuntimeError(
                f"resume_lora 跨模型族被拒绝：{args.resume_lora} 属于 '{lora_family}'，"
                f"当前 model_family='{ctx.family.spec.family_id}'"
            )
        ctx.injector.load(args.resume_lora)
        logger.info(msg("train.resume_from_lora", path=args.resume_lora))

    _setup_block_swap(ctx)

    if getattr(args, "sra_enabled", False):
        from training.families.anima.sra_align import SRAAligner

        model_channels = ctx.model.model_channels
        block_idx = int(getattr(args, "sra_block", 4))
        num_blocks = len(ctx.model.blocks)
        if block_idx >= num_blocks:
            logger.warning(
                "sra_block=%d is beyond the model block count (%d): clamped to %d",
                block_idx,
                num_blocks,
                num_blocks - 1,
            )
            block_idx = num_blocks - 1
        ctx.sra_aligner = SRAAligner(
            model=ctx.model,
            block_idx=block_idx,
            patch_spatial=ctx.model.patch_spatial,
            patch_temporal=ctx.model.patch_temporal,
            model_channels=model_channels,
            vae_channels=_ANIMA_SPEC.latent.channels,
            device=ctx.device,
            dtype=ctx.dtype,
            normalize=bool(getattr(args, "sra_normalize", True)),
        )


def _defer_dit_for_text_cache(ctx: TrainingContext) -> bool:
    return (
        ctx.family.spec.text.strategy == "cached_varlen"
        and bool(getattr(ctx.args, "text_encoder_cache", True))
    )


def _validate_fp8_base(ctx: TrainingContext) -> None:
    """fp8 底模（fp8_base 训练）的组合校验——fail-fast 于任何大加载之前。

    探测只读 safetensors header（毫秒级），非 fp8 底模零开销直通。目前只有
    krea2 loader 接受 fp8 checkpoint（Anima loader 自行拒绝），但探测本身
    族无关。两条硬约束：

    - grad_checkpoint 必须开：fp8 的显存收益依赖重算段释放逐层 dequant 的
      临时权重；不开则 autograd 全量驻留 264 层 bf16 副本，占用反超 bf16。
    - DoRA 不支持：lycoris weight_decompose 初始化读底模权重数值（范数），
      fp8 直接 cast 缺 scale 校正，数值不正确（与推理/merge 拒绝口径一致）。
    """
    from training.families.krea2.loader import checkpoint_contains_fp8

    args = ctx.args
    if not checkpoint_contains_fp8(getattr(args, "transformer_path", "") or ""):
        return
    problems = []
    if not bool(getattr(args, "grad_checkpoint", True)):
        problems.append(
            "grad_checkpoint=false：fp8 底模的逐层 dequant 临时权重会被 "
            "autograd 全量驻留，显存占用反超 bf16。请开启梯度检查点。"
        )
    if bool(getattr(args, "lora_dora", False)):
        problems.append(
            "lora_dora=true：DoRA 初始化读取底模权重数值，fp8 存储下数值"
            "不正确。请关闭 DoRA 或改用 bf16 底模。"
        )
    if problems:
        raise RuntimeError(
            "fp8 底模与当前配置不兼容：\n- " + "\n- ".join(problems)
        )
    logger.info(msg("train.fp8_base_detected"))


def run(ctx: TrainingContext) -> None:
    """Resolve paths and load either the complete stack or the cache-first half."""
    from training.sysmem import check_load_budget, guard_enabled_from_env

    if ctx.family is None:
        ctx.family = resolve_family(ctx.args)
    _resolve_paths(ctx)
    _validate_fp8_base(ctx)

    if _defer_dit_for_text_cache(ctx):
        logger.info(msg("train.text_cache_order"))
        # 分段预算：本段只加载 VAE + TE（DiT 由 finish 段单独预算）。
        # 开关来自 设置 → 训练 → 训练参数（supervisor 经 env 注入，默认开）。
        check_load_budget(
            guard_enabled_from_env(),
            weight_paths=[getattr(ctx.args, "vae_path", ""),
                          getattr(ctx.args, "text_encoder_path", "")],
            stage="训练模型加载（VAE/文本编码器）",
            settings_hint="设置 → 训练 → 训练参数",
        )
        _load_vae(ctx)
        _load_text(ctx)
        return

    # Preserve the historical Anima order. Storage-free Krea2 deliberately keeps
    # the DiT resident while its text encoder is loaded for per-batch encoding.
    check_load_budget(
        guard_enabled_from_env(),
        weight_paths=[
            getattr(ctx.args, "transformer_path", ""),
            getattr(ctx.args, "vae_path", ""),
            getattr(ctx.args, "text_encoder_path", ""),
        ],
        stage="训练模型加载",
        vram_discount_ratio=_swap_vram_discount(ctx),
        settings_hint="设置 → 训练 → 训练参数",
    )
    _load_dit(ctx)
    _load_vae(ctx)
    _load_text(ctx)
    _inject_adapter(ctx)
    _log_train_start_vram(ctx)


def finish(ctx: TrainingContext) -> None:
    """Load/inject a DiT deferred by cached text preparation; otherwise no-op."""
    if ctx.model is not None:
        return
    from training.sysmem import check_load_budget, guard_enabled_from_env

    logger.info(msg("train.text_cache_done_loading"))
    check_load_budget(
        guard_enabled_from_env(),
        weight_paths=[getattr(ctx.args, "transformer_path", "")],
        stage="训练模型加载（Transformer）",
        vram_discount_ratio=_swap_vram_discount(ctx),
        settings_hint="设置 → 训练 → 训练参数",
    )
    _load_dit(ctx)
    _inject_adapter(ctx)
    _log_train_start_vram(ctx)


def _read_lora_family(path) -> str:
    """Read artifact family; legacy unmarked safetensors grandfather to Anima."""
    import json

    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            meta = handle.metadata() or {}
        args = json.loads(meta.get("ss_network_args") or "{}")
        return str(args.get("model_family") or "anima")
    except Exception:
        return "anima"
