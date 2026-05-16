"""训练 loss plugin registry（ADR 0003 PR-C 模式）。

`build_loss(args) -> LossProtocol` 按 args.loss_type 派发到具体 loss：
- mse    — F.mse_loss 包装（默认，与历史行为字节级一致）
- huber  — Huber loss with constant/snr/sigma delta schedule

加新 loss 步骤：
1. 写 training/losses/{name}.py 含 `build(args) -> LossProtocol`
2. 本文件 BUILDERS 字典加一行
3. studio/schema.py 的 `loss_type: Literal[...]` 加值 + 该 loss 专属字段
4. 完。phases/optimizer.py / loop.py / TrainingContext 0 改动。

删 loss：逆操作，3 步删完；`validate_schema_consistency()` 保不漏。
"""

from __future__ import annotations

from typing import Callable

from training.losses import huber, mse
from training.losses.protocol import LossProtocol

__all__ = ["LossProtocol", "BUILDERS", "build_loss", "validate_schema_consistency"]


# 单一 truth source：所有 loss 工厂的注册表
BUILDERS: dict[str, Callable[..., LossProtocol]] = {
    "mse": mse.build,
    "huber": huber.build,
}


def build_loss(args) -> LossProtocol:
    """按 args.loss_type 派发到对应 build()。"""
    loss_type = (getattr(args, "loss_type", "mse") or "mse").lower()
    if loss_type not in BUILDERS:
        raise ValueError(
            f"未知 loss_type={loss_type!r}；已注册: {sorted(BUILDERS)}"
        )
    return BUILDERS[loss_type](args)


def validate_schema_consistency() -> None:
    """启动期校验：TrainingConfig.loss_type Literal 集合 == BUILDERS keys。

    失配通常意味着加了新 loss 但漏改了一处（schema 或 registry）。早 fail 早修。
    """
    from studio.schema import TrainingConfig

    field = TrainingConfig.model_fields["loss_type"]
    schema_options = set(field.annotation.__args__)
    registered = set(BUILDERS)
    if schema_options != registered:
        raise RuntimeError(
            f"loss 注册与 schema 不同步：\n"
            f"  schema 有但未注册: {schema_options - registered}\n"
            f"  注册但 schema 没列: {registered - schema_options}"
        )
