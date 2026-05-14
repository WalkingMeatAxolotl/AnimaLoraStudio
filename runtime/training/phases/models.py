"""models_phase：path resolve + transformer/vae/text_encoders + LoRA inject。

抽自 main() L187-255（ADR 0003 PR-B）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from training.context import TrainingContext
from training.model_loading import (
    enable_xformers,
    find_diffusion_pipe_root,
    resolve_path_best_effort,
)
from training.models import load_anima_model, load_text_encoders, load_vae


logger = logging.getLogger(__name__)


def run(ctx: TrainingContext) -> None:
    """
    - find_diffusion_pipe_root + path resolution（args 路径多 base 兜底）
    - 按 attention_backend 加载 transformer + xformers / flash_attn / sdpa
    - 加载 vae + text encoders
    - 注入 LoRA + 可选 resume_lora
    """
    args = ctx.args

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

    # 按 attention_backend 决策：xformers / flash_attn / none。
    # load_anima_model 内部按 flash_attn 参数设 flash_attn 全局开关；
    # xformers 是 model 层面的额外注入（与 flash_attn 互斥）。
    backend = getattr(args, "attention_backend", "flash_attn")
    use_flash = (backend == "flash_attn")

    # 加载模型
    logger.info("加载 Transformer...")
    ctx.model = load_anima_model(
        args.transformer_path, ctx.device, ctx.dtype, ctx.repo_root, flash_attn=use_flash,
    )

    if backend == "xformers":
        enable_xformers(ctx.model)
    elif backend == "none":
        logger.info("attention_backend=none，flash_attn / xformers 都不启用，走 PyTorch SDPA")

    logger.info("加载 VAE...")
    ctx.vae = load_vae(args.vae_path, ctx.device, ctx.dtype, ctx.repo_root)

    logger.info("加载文本编码器...")
    ctx.qwen_model, ctx.qwen_tok, ctx.t5_tok = load_text_encoders(
        args.text_encoder_path, args.t5_tokenizer_path, ctx.device, ctx.dtype,
    )

    # 注入 LoRA — PR-C 通过 adapters/ plugin registry 派发，加新变体见
    # training/adapters/__init__.py docstring
    logger.info(f"注入 {args.lora_type.upper()}...")
    from training.adapters import build_adapter
    ctx.injector = build_adapter(args)
    ctx.injector.inject(ctx.model)

    # 从已有 LoRA 继续训练
    if getattr(args, "resume_lora", "") and Path(args.resume_lora).exists():
        ctx.injector.load(args.resume_lora)
        logger.info(f"将从已有 LoRA 继续训练: {args.resume_lora}")
