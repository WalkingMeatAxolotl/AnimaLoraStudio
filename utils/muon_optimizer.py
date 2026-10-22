"""Muon + 辅助 AdamW 单一优化器。

Muon = MomentUm Orthogonalized by Newton-Schulz（Keller Jordan, 2024,
https://kellerjordan.github.io/posts/muon/）。对 2D 权重做「动量 → Newton-Schulz
正交化 → 按形状缩放」的更新，相当于谱范数约束下的最速下降。
lr 形状缩放的 match_rms_adamw 口径来自 Moonshot（Liu et al., 2025,
"Muon is Scalable for LLM Training", arxiv 2502.16982）。

为什么不直接用 ``torch.optim.Muon``：它构造期要求**所有**参数都是 2D，而训练里
会混进 1D 参数（DoRA magnitude / SRA projection bias / Ortho-Hydra 标量等）。
这里在同一个 Optimizer 内按参数形状分派：真矩阵走 Muon，其余走 AdamW（见 ``is_muon_param``）。
单一 optimizer 对象让 lr_scheduler / resume / grad_clip / GradScaler 无需改动。

Muon 分支的数学与 ``torch.optim.Muon``（torch 2.11）逐项对齐（动量 lerp 写法、
Nesterov、quintic 系数、weight decay 用未缩放 lr），由 tests/test_muon_optimizer.py
对拍锁定。差异只有一处：同形状矩阵堆叠成 batch 一起做 Newton-Schulz——LoRA 有
数百个同形状小矩阵，逐个迭代会被 kernel launch 开销主导。
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["MuonWithAuxAdam", "zeropower_via_newtonschulz5", "adjust_lr_ratio", "is_muon_param"]

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_EPS = 1e-7
ADJUST_LR_FNS = ("original", "match_rms_adamw")


def zeropower_via_newtonschulz5(g: Tensor, steps: int = 5, eps: float = NS_EPS) -> Tensor:
    """把 ``g``（[..., m, n]）的奇异值近似推到 1，保留奇异向量（≈ U Vᵀ）。

    支持前置 batch 维；每个矩阵独立归一化。全程 bf16，结果奇异值约落在 [0.7, 1.2]，
    并非精确 1（系数刻意追求零点斜率，见 Keller 博客）。
    """
    if g.ndim < 2:
        raise ValueError(f"Newton-Schulz needs a matrix, got shape {tuple(g.shape)}")
    a, b, c = NS_COEFFICIENTS
    batch_shape = g.shape[:-2]
    x = g.bfloat16().reshape(-1, *g.shape[-2:])
    transposed = g.size(-2) > g.size(-1)
    if transposed:
        x = x.mT
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        # 与 torch.optim.Muon 相同的融合乘加（addmm → baddbmm），bf16 舍入路径一致
        gram = x @ x.mT
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, gram_update, x, beta=a)
    if transposed:
        x = x.mT
    return x.reshape(*batch_shape, *x.shape[-2:])


def adjust_lr_ratio(adjust_lr_fn: str, rows: int, cols: int) -> float:
    """按矩阵形状缩放 lr。

    - original（Keller）：sqrt(max(1, rows/cols))
    - match_rms_adamw（Moonshot）：0.2·sqrt(max(rows, cols))，让每元素更新 RMS ≈ 0.2，
      与 AdamW 典型更新同量级，可沿用 AdamW 的 lr
    """
    if adjust_lr_fn == "original":
        return math.sqrt(max(1.0, rows / cols))
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, cols))
    raise ValueError(f"Unknown adjust_lr_fn={adjust_lr_fn!r}; choose from {ADJUST_LR_FNS}")


def is_muon_param(p: Tensor) -> bool:
    """真矩阵才走 Muon：ndim>=2 且摊平后两个维度都 >1。

    形如 (1, r) / (r, 1) 的「矩阵」（Ortho-Hydra 的 lambda_layer 等）本质是向量：
    正交化后只剩一个单位方向，等价于 sign 式更新，与 1D 参数一样交给 AdamW。
    """
    if p.ndim < 2:
        return False
    rows = p.shape[0]
    cols = p.numel() // rows if rows else 0
    return rows > 1 and cols > 1


def _as_matrix(t: Tensor) -> Tensor:
    """ndim>2（卷积核）摊平成 [out, rest]；2D 原样返回。"""
    return t if t.ndim == 2 else t.reshape(t.shape[0], -1)


class MuonWithAuxAdam(Optimizer):
    """真矩阵参数走 Muon，其余（1D / 向量形状）走 AdamW；两者共享 lr / weight_decay / scheduler。

    构造时把每个传入的 param group 按维度拆成 Muon / AdamW 两组（空组丢弃），
    group 上的 ``use_muon`` 标记分派。所以 ``param_groups`` 数可能多于传入数，
    lr_scheduler 对每组独立缩放，行为不受影响。
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adjust_lr_fn: str = "match_rms_adamw",
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 1 <= ns_steps < 100:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")
        if adjust_lr_fn not in ADJUST_LR_FNS:
            raise ValueError(f"Unknown adjust_lr_fn={adjust_lr_fn!r}; choose from {ADJUST_LR_FNS}")

        defaults = dict(
            lr=lr, weight_decay=weight_decay,
            momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
            adam_betas=tuple(adam_betas), adam_eps=adam_eps,
            use_muon=True,
        )
        super().__init__(self._split_groups(params), defaults)

    @staticmethod
    def _split_groups(params: Iterable) -> list[dict]:
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        groups = params if isinstance(params[0], dict) else [{"params": params}]
        split: list[dict] = []
        for group in groups:
            group_params = list(group["params"])
            extra = {k: v for k, v in group.items() if k != "params"}
            muon = [p for p in group_params if is_muon_param(p)]
            adam = [p for p in group_params if not is_muon_param(p)]
            if muon:
                split.append({**extra, "params": muon, "use_muon": True})
            if adam:
                split.append({**extra, "params": adam, "use_muon": False})
        return split

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adam_step(group)
        return loss

    def _muon_step(self, group: dict) -> None:
        lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
        buckets: dict[tuple, list[tuple[Tensor, Tensor]]] = defaultdict(list)
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse or torch.is_complex(p):
                raise RuntimeError("Muon does not support sparse or complex parameters")
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad)
            buf = state["momentum_buffer"]
            buf.lerp_(p.grad, 1 - mom)
            update = p.grad.lerp(buf, mom) if group["nesterov"] else buf
            update = _as_matrix(update)
            buckets[(tuple(update.shape), update.device)].append((p, update))

        for (shape, _device), items in buckets.items():
            ortho = zeropower_via_newtonschulz5(
                torch.stack([u for _, u in items]), steps=group["ns_steps"],
            )
            adjusted_lr = lr * adjust_lr_ratio(group["adjust_lr_fn"], shape[0], shape[1])
            for (p, _), o in zip(items, ortho):
                p.mul_(1 - lr * wd)
                p.add_(o.view_as(p).to(p.dtype), alpha=-adjusted_lr)

    def _adam_step(self, group: dict) -> None:
        lr, wd = group["lr"], group["weight_decay"]
        beta1, beta2 = group["adam_betas"]
        eps = group["adam_eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            step = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.lerp_(p.grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            bias1 = 1 - beta1 ** step
            bias2 = 1 - beta2 ** step
            denom = (exp_avg_sq / bias2).sqrt_().add_(eps)
            p.mul_(1 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-lr / bias1)
