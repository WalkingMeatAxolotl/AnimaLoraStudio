"""SOAP optimizer build wrapper（ADR 0003 PR-C）。

SOAP = Adam in the Shampoo eigenbasis（Vyas et al., 2024, arxiv 2409.11321）。
普通（非 schedule-free）变体：可配 lr_scheduler，无 train()/eval() 切换。
schedule-free 变体见 soap_sf.py。
"""

from __future__ import annotations


def build(args, params, lr: float, weight_decay: float):
    """实例化 SOAP，读 soap_* 参数。"""
    from utils.optimizer_utils import create_optimizer

    return create_optimizer(
        optimizer_type="soap",
        params=params,
        learning_rate=lr,
        weight_decay=weight_decay,
        betas=(
            float(getattr(args, "soap_beta1", 0.95)),
            float(getattr(args, "soap_beta2", 0.95)),
        ),
        shampoo_beta=float(getattr(args, "soap_shampoo_beta", -1.0)),
        precondition_frequency=int(getattr(args, "soap_precondition_frequency", 10)),
        max_precond_dim=int(getattr(args, "soap_max_precond_dim", 10000)),
        precond_in_state=bool(getattr(args, "soap_precond_in_state", True)),
    )
