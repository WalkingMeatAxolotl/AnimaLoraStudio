"""Automagic optimizer build wrapper (v1 + v2 via automagic_variant field)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build(args, params, lr: float, weight_decay: float):
    variant = (getattr(args, "automagic_variant", "v1") or "v1").lower()
    common = dict(
        min_lr=float(getattr(args, "automagic_min_lr", 1e-7)),
        max_lr=float(getattr(args, "automagic_max_lr", 1e-3)),
        lr_bump=float(getattr(args, "automagic_lr_bump", 1e-6)),
        eps=float(getattr(args, "automagic_eps", 1e-30)),
        clip_threshold=float(getattr(args, "automagic_clip_threshold", 1.0)),
        beta2=float(getattr(args, "automagic_beta2", 0.999)),
    )
    if variant == "v2":
        from utils.optimizer_utils import create_automagic_v2
        return create_automagic_v2(
            params=params, lr=lr, weight_decay=weight_decay,
            agreement_threshold=float(getattr(args, "automagic_agreement_threshold", 0.5)),
            **common,
        )
    from utils.optimizer_utils import create_automagic
    return create_automagic(params=params, lr=lr, weight_decay=weight_decay, **common)


def validate(args) -> None:
    scheduler = (getattr(args, "lr_scheduler", "none") or "none").lower()
    if scheduler != "none":
        raise ValueError(
            "Automagic manages learning rates internally and requires lr_scheduler=none"
        )

    variant = (getattr(args, "automagic_variant", "v1") or "v1").lower()
    if variant == "v2":
        if getattr(args, "mixed_precision", "bf16") == "fp16":
            raise ValueError(
                "Automagic v2 不兼容 fp16 混合精度"
                "（fused backward hook 绕过 GradScaler，fp16 需要 scaler 防溢出）"
            )
        grad_accum = int(getattr(args, "grad_accum", 1) or 1)
        if grad_accum > 1:
            raise ValueError(
                "Automagic v2 (fused backward) 与梯度累积不兼容：参数在每个 "
                "micro-batch backward 时就地更新并清掉 grad，grad_accum>1 的累积"
                "语义会被静默破坏。请改用 automagic_variant=v1 或 grad_accum=1。"
            )
        grad_clip = float(getattr(args, "grad_clip_max_norm", 0) or 0)
        if grad_clip > 0:
            logger.warning(
                "Automagic v2 使用 fused backward hook，grad_clip=%.2g 将无效"
                "（参数在 backward 中已更新）", grad_clip
            )
    else:
        grad_clip = float(getattr(args, "grad_clip_max_norm", 0) or 0)
        if grad_clip > 0:
            logger.warning(
                "Automagic v1 内置 RMS clip (clip_threshold)；外部 grad_clip=%.2g "
                "会双重裁剪梯度", grad_clip
            )
