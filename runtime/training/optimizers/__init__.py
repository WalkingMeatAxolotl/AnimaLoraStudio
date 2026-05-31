"""Optimizer plugin registry（ADR 0003 PR-C）。

加新优化器（Lion / CAME / Schedule-Free AdamW）的步骤：
1. 写 training/optimizers/{variant}.py 含 `build(args, params, lr, weight_decay)`，
   可选 `validate(args)` 启动期检查
2. 本文件 BUILDERS / VALIDATORS 字典加一行
3. studio/schema.py 的 optimizer_type Literal 加枚举值 + 该 variant 专属字段
4. requirements.txt 加依赖（如有）

详见 ADR 0003 "Case 6: Lion / CAME / Schedule-Free AdamW"。
"""

from __future__ import annotations

from typing import Callable, Optional

from training.optimizers import adamw, automagic, lion, prodigy, prodigy_plus_schedulefree

__all__ = ["BUILDERS", "VALIDATORS", "build_optimizer", "validate_optimizer",
           "validate_schema_consistency"]


BUILDERS: dict[str, Callable] = {
    "adamw": adamw.build,
    "automagic": automagic.build,
    "lion": lion.build,
    "prodigy": prodigy.build,
    "prodigy_plus_schedulefree": prodigy_plus_schedulefree.build,
}

# 启动期校验函数（None / 未注册 = 跳过）
VALIDATORS: dict[str, Callable[[object], None]] = {
    "automagic": automagic.validate,
    "prodigy_plus_schedulefree": prodigy_plus_schedulefree.validate,
}


def build_optimizer(args, params, lr: float, weight_decay: float):
    """按 args.optimizer_type 派发。"""
    optimizer_type = (getattr(args, "optimizer_type", "adamw") or "adamw").lower()
    if optimizer_type not in BUILDERS:
        raise ValueError(
            f"未知 optimizer_type={optimizer_type!r}；已注册: {sorted(BUILDERS)}"
        )
    return BUILDERS[optimizer_type](args, params, lr, weight_decay)


def validate_optimizer(args) -> None:
    """跑 optimizer 专属启动期兼容检查（如 PPSF 要求 lr_scheduler=none）。"""
    optimizer_type = (getattr(args, "optimizer_type", "adamw") or "adamw").lower()
    validator = VALIDATORS.get(optimizer_type)
    if validator:
        validator(args)


def validate_schema_consistency() -> None:
    from studio.schema import TrainingConfig

    field = TrainingConfig.model_fields["optimizer_type"]
    schema_options = set(field.annotation.__args__)
    registered = set(BUILDERS)
    if schema_options != registered:
        raise RuntimeError(
            f"optimizer 注册与 schema 不同步（PR-C registry）：\n"
            f"  schema 有但未注册: {schema_options - registered}\n"
            f"  注册但 schema 没列: {registered - schema_options}"
        )
