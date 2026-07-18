"""models_phase: paths + family-owned weights + LoRA injection.

Cached-varlen families may defer their large DiT until dataset captions have
been cached and the text encoder released. ``finish(ctx)`` closes that deferred
half immediately after ``text_cache`` and before optimizer construction.
"""

from __future__ import annotations

import logging
from pathlib import Path

from training.context import TrainingContext
from training.families import resolve_family
from training.families.anima import ANIMA_SPEC as _ANIMA_SPEC
from training.model_loading import (
    find_diffusion_pipe_root,
    resolve_path_best_effort,
)


logger = logging.getLogger(__name__)


def _resolve_paths(ctx: TrainingContext) -> None:
    args = ctx.args
    ctx.repo_root = find_diffusion_pipe_root()
    logger.info("模型代码路径: %s", ctx.repo_root)

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


def _load_dit(ctx: TrainingContext) -> None:
    args = ctx.args
    backend = getattr(args, "attention_backend", "flash_attn")
    if backend == "none":
        logger.info(
            "attention_backend=none，flash_attn / xformers 都不启用，走 PyTorch SDPA"
        )
    logger.info("加载 Transformer...")
    ctx.model = ctx.family.load_dit(
        args.transformer_path,
        ctx.device,
        ctx.dtype,
        attention_backend=backend,
        repo_root=ctx.repo_root,
    )
    # 大权重 mmap 缓存页归还系统（13-26GB；真机换页卡死案例，训练同样受益）
    from training.sysmem import trim_working_set

    trim_working_set()


def _load_vae(ctx: TrainingContext) -> None:
    args = ctx.args
    logger.info("加载 VAE...")
    ctx.vae = ctx.family.load_vae(
        args.vae_path,
        ctx.device,
        ctx.vae_dtype,
        tiling=getattr(args, "vae_tiling", "auto"),
    )


def _load_text(ctx: TrainingContext) -> None:
    args = ctx.args
    logger.info("加载文本编码器...")
    ctx.text_stack = ctx.family.load_text(
        args.text_encoder_path,
        ctx.device,
        ctx.dtype,
        t5_tokenizer_path=args.t5_tokenizer_path,
        cache_enabled=bool(getattr(args, "text_encoder_cache", True)),
    )


def _inject_adapter(ctx: TrainingContext) -> None:
    args = ctx.args
    logger.info("注入 %s...", args.lora_type.upper())
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
        logger.info("将从已有 LoRA 继续训练: %s", args.resume_lora)

    if getattr(args, "sra_enabled", False):
        from training.families.anima.sra_align import SRAAligner

        model_channels = ctx.model.model_channels
        block_idx = int(getattr(args, "sra_block", 4))
        num_blocks = len(ctx.model.blocks)
        if block_idx >= num_blocks:
            logger.warning(
                "sra_block=%d >= 模型 blocks 数(%d)，clamp 到 %d",
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
    logger.info(
        "检测到 fp8 底模：以 fp8_base 语义训练（权重常驻 fp8，前向逐层 dequant）"
    )


def run(ctx: TrainingContext) -> None:
    """Resolve paths and load either the complete stack or the cache-first half."""
    from training.sysmem import check_load_budget

    if ctx.family is None:
        ctx.family = resolve_family(ctx.args)
    _resolve_paths(ctx)
    _validate_fp8_base(ctx)

    if _defer_dit_for_text_cache(ctx):
        logger.info(
            "文本缓存已开启：先加载 VAE/Qwen3-VL，缓存并释放 TE 后再加载 Transformer"
        )
        # 分段预算：本段只加载 VAE + TE（DiT 由 finish 段单独预算）。
        # 训练侧常开（多小时任务，中途换页卡死代价远高于一条报错）。
        check_load_budget(
            True,
            weight_paths=[getattr(ctx.args, "vae_path", ""),
                          getattr(ctx.args, "text_encoder_path", "")],
            stage="训练模型加载（VAE/文本编码器）",
        )
        _load_vae(ctx)
        _load_text(ctx)
        return

    # Preserve the historical Anima order. Storage-free Krea2 deliberately keeps
    # the DiT resident while its text encoder is loaded for per-batch encoding.
    check_load_budget(
        True,
        weight_paths=[
            getattr(ctx.args, "transformer_path", ""),
            getattr(ctx.args, "vae_path", ""),
            getattr(ctx.args, "text_encoder_path", ""),
        ],
        stage="训练模型加载",
    )
    _load_dit(ctx)
    _load_vae(ctx)
    _load_text(ctx)
    _inject_adapter(ctx)


def finish(ctx: TrainingContext) -> None:
    """Load/inject a DiT deferred by cached text preparation; otherwise no-op."""
    if ctx.model is not None:
        return
    from training.sysmem import check_load_budget

    logger.info("文本缓存完成且文本编码器已释放；继续加载 Transformer...")
    check_load_budget(
        True,
        weight_paths=[getattr(ctx.args, "transformer_path", "")],
        stage="训练模型加载（Transformer）",
    )
    _load_dit(ctx)
    _inject_adapter(ctx)


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
