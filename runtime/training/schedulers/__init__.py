"""LR scheduler plugin registry（ADR 0003 PR-C）。

加新 scheduler（warmup_cosine / one_cycle / polynomial）的步骤：
1. 写 training/schedulers/{variant}.py 含 `build(args, optimizer, total_steps)`
2. 本文件 BUILDERS 字典加一行
3. studio/schema.py 的 lr_scheduler Literal 加枚举值 + 该 variant 专属字段

详见 ADR 0003 "Case 7: warmup_cosine"。

特殊键 "none" 不开文件：build_scheduler 直接返回 None。
"""

from __future__ import annotations

from typing import Callable, Optional

from training.schedulers import cosine, cosine_with_restart, cosine_with_warmup

__all__ = ["BUILDERS", "build_scheduler", "validate_schema_consistency"]


# "none" 不在 BUILDERS —— build_scheduler 显式判一下返回 None。
BUILDERS: dict[str, Callable] = {
    "cosine": cosine.build,
    "cosine_with_restart": cosine_with_restart.build,
    "cosine_with_warmup": cosine_with_warmup.build,
}

# schema 允许 "none" 但 BUILDERS 不收录；validate_schema_consistency 据此放行
SCHEMA_ONLY_OPTIONS = {"none"}


def build_scheduler(args, optimizer, total_steps: Optional[int]):
    """按 args.lr_scheduler 派发；"none" 或未配置返回 None。"""
    lr_sched = (getattr(args, "lr_scheduler", "none") or "none").lower()
    if lr_sched == "none":
        return None
    if lr_sched not in BUILDERS:
        raise ValueError(
            f"未知 lr_scheduler={lr_sched!r}；已注册: {sorted(BUILDERS)} + 'none'"
        )
    return BUILDERS[lr_sched](args, optimizer, total_steps)


def validate_schema_consistency() -> None:
    from studio.schema import TrainingConfig

    field = TrainingConfig.model_fields["lr_scheduler"]
    schema_options = set(field.annotation.__args__) - SCHEMA_ONLY_OPTIONS
    registered = set(BUILDERS)
    if schema_options != registered:
        raise RuntimeError(
            f"scheduler 注册与 schema 不同步（PR-C registry）：\n"
            f"  schema 有但未注册: {schema_options - registered}\n"
            f"  注册但 schema 没列: {registered - schema_options}\n"
            f"  （schema-only 跳过校验: {SCHEMA_ONLY_OPTIONS}）"
        )
