"""LoRA adapter plugin registry（ADR 0003 PR-C）。

加新变体的步骤（参考 ADR 0003 Case 2-5 落地案例）：
1. 写 training/adapters/{variant}.py 含 `build(args) -> AdapterProtocol`
2. 本文件 BUILDERS 字典加一行
3. studio/schema.py 的 `lora_type: Literal[...]` 加一个枚举值 + 该变体专属字段
4. 完。phases/models.py / loop.py / main() 0 改动。

删变体：相反顺序，3 步删完。
"""

from __future__ import annotations

from typing import Callable

from training.adapters import lycoris, ortho, tlora
from training.adapters.protocol import AdapterProtocol, StepContext

__all__ = ["AdapterProtocol", "StepContext", "BUILDERS", "build_adapter",
           "validate_schema_consistency"]


# 单一 truth source：所有 adapter 工厂的注册表
BUILDERS: dict[str, Callable[..., AdapterProtocol]] = {
    "lokr": lycoris.build,
    "loha": lycoris.build,
    "lora": lycoris.build,
    "ortho": ortho.build,
    "tlora": tlora.build,
}


def build_adapter(args) -> AdapterProtocol:
    """按 args.lora_type 派发到对应 build()。"""
    lora_type = args.lora_type
    if lora_type not in BUILDERS:
        raise ValueError(
            f"未知 lora_type={lora_type!r}；已注册: {sorted(BUILDERS)}"
        )
    return BUILDERS[lora_type](args)


def validate_schema_consistency() -> None:
    """启动期校验：TrainingConfig.lora_type Literal 集合 == BUILDERS keys。

    失配通常意味着加了新变体但漏改了一处（schema 或 registry）。早 fail 早修。
    """
    from studio.schema import TrainingConfig

    field = TrainingConfig.model_fields["lora_type"]
    schema_options = set(field.annotation.__args__)
    registered = set(BUILDERS)
    if schema_options != registered:
        raise RuntimeError(
            f"adapter 注册与 schema 不同步（PR-C registry）：\n"
            f"  schema 有但未注册: {schema_options - registered}\n"
            f"  注册但 schema 没列: {registered - schema_options}"
        )
