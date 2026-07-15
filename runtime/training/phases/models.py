"""models_phase：path resolve + transformer/vae/text_encoders + LoRA inject。

抽自 main() L187-255（ADR 0003 PR-B）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from training.context import TrainingContext
from training.families.anima import ANIMA_SPEC as _ANIMA_SPEC
from training.families import resolve_family
from training.model_loading import (
    find_diffusion_pipe_root,
    resolve_path_best_effort,
)


logger = logging.getLogger(__name__)


def run(ctx: TrainingContext) -> None:
    """
    - find_diffusion_pipe_root + path resolution（args 路径多 base 兜底）
    - 按 attention_backend 加载 transformer + xformers / flash_attn / sdpa
    - 加载 vae + text encoders
    - 注入 LoRA + 可选 resume_lora
    """
    args = ctx.args
    if ctx.family is None:
        ctx.family = resolve_family(args)

    # 查找模型代码
    ctx.repo_root = find_diffusion_pipe_root()
    logger.info(f"模型代码路径: {ctx.repo_root}")

    # 解析路径：相对路径优先按 config 位置 / AnimaLoraToolkit 目录解析
    # 注：原 main() 用 Path(__file__).resolve().parent 拿 runtime/；本模块在
    # runtime/training/phases/ 下，往上两级才是 runtime/，三级是 repo_root。
    phases_dir = Path(__file__).resolve().parent           # runtime/training/phases
    training_dir = phases_dir.parent                        # runtime/training
    runtime_dir = training_dir.parent                       # runtime
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

    # 经 ModelFamily 派发（D8'）；attention backend 决策收进 family.load_dit
    backend = getattr(args, "attention_backend", "flash_attn")
    if backend == "none":
        logger.info("attention_backend=none，flash_attn / xformers 都不启用，走 PyTorch SDPA")

    logger.info("加载 Transformer...")
    ctx.model = ctx.family.load_dit(
        args.transformer_path, ctx.device, ctx.dtype,
        attention_backend=backend, repo_root=ctx.repo_root,
    )

    logger.info("加载 VAE...")
    ctx.vae = ctx.family.load_vae(args.vae_path, ctx.device, ctx.vae_dtype,
                                  tiling=getattr(args, "vae_tiling", "auto"))

    logger.info("加载文本编码器...")
    ctx.qwen_model, ctx.qwen_tok, ctx.t5_tok = ctx.family.load_text(
        args.text_encoder_path, ctx.device, ctx.dtype,
        t5_tokenizer_path=args.t5_tokenizer_path,
    )

    # 注入 LoRA — PR-C 通过 adapters/ plugin registry 派发，加新变体见
    # training/adapters/__init__.py docstring
    logger.info(f"注入 {args.lora_type.upper()}...")
    from training.adapters import build_adapter
    ctx.injector = build_adapter(args)
    ctx.injector.metadata_extra = ctx.family.lora_metadata()  # D13：产物族标记
    ctx.injector.inject(ctx.model)

    # 从已有 LoRA 继续训练（D13：跨族 fail-fast；无标记存量产物 grandfather 为 anima）
    if getattr(args, "resume_lora", "") and Path(args.resume_lora).exists():
        _lora_family = _read_lora_family(args.resume_lora)
        if _lora_family != ctx.family.spec.family_id:
            raise RuntimeError(
                f"resume_lora 跨模型族被拒绝：{args.resume_lora} 属于 '{_lora_family}'，"
                f"当前 model_family='{ctx.family.spec.family_id}'"
            )
        ctx.injector.load(args.resume_lora)
        logger.info(f"将从已有 LoRA 继续训练: {args.resume_lora}")

    # SRA v2 表征对齐（LoRA 注入后构造）
    if getattr(args, "sra_enabled", False):
        from training.families.anima.sra_align import SRAAligner
        model_channels = ctx.model.model_channels
        block_idx = int(getattr(args, "sra_block", 4))
        num_blocks = len(ctx.model.blocks)
        if block_idx >= num_blocks:
            logger.warning(f"sra_block={block_idx} >= 模型 blocks 数({num_blocks})，clamp 到 {num_blocks - 1}")
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


def _read_lora_family(path) -> str:
    """从 safetensors metadata 读产物所属族；无标记 grandfather 为 anima（D13）。"""
    import json

    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        args = json.loads(meta.get("ss_network_args") or "{}")
        return str(args.get("model_family") or "anima")
    except Exception:
        return "anima"
