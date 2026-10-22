"""Riemannion optimizer build wrapper（POC，arxiv 2507.12142）。

固定秩流形上的 Muon：只优化 lora_type=riemann adapter 标记的参数，其余参数走
同一 optimizer 内的辅助 AdamW。lr 直接作用于 ΔW（每步谱范数步长 ≈ lr），与
AdamW 在 LoRA 因子上的 lr 不同口径。实现在 utils/riemannion_optimizer.py。
"""

from __future__ import annotations


def validate(args) -> None:
    """启动期约束：必须配 riemann adapter；POC 不支持 block swap。"""
    lora_type = getattr(args, "lora_type", None)
    if lora_type != "riemann":
        raise ValueError(f"optimizer_type=riemannion 需要 lora_type=riemann（当前 {lora_type!r}）")
    if int(getattr(args, "blocks_to_swap", 0) or 0) > 0:
        raise ValueError("optimizer_type=riemannion 暂不支持 block swap（blocks_to_swap 须为 0）")


def build(args, params, lr: float, weight_decay: float):
    import logging

    from studio.infrastructure.log_messages import msg
    from utils.riemannion_optimizer import RiemannionWithAuxAdam

    momentum = float(getattr(args, "riemannion_momentum", 0.9))
    optimizer = RiemannionWithAuxAdam(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    logging.getLogger(__name__).info(msg(
        "optim.created", name="Riemannion",
        params=f"lr={lr} momentum={momentum} weight_decay={weight_decay}",
    ))
    return optimizer
