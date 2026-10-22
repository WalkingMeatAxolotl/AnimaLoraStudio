"""Riemannion：固定秩流形上的 Muon（POC）。

论文：Bogachev et al., "LoRA meets Riemannion: Muon Optimizer for
Parametrization-independent Low-Rank Adapters", arxiv 2507.12142（Algorithm 1/2/4）。
未发现官方代码；按论文 Algorithm 4 实现，以下两处为本实现的取舍（论文 HTML 版
公式排版有残缺，无法逐字核对）：

- weight decay 用解耦形式 X ← X·(1−ηγ)，与 AdamW / Muon 一致
- 动量缓存正交化**之前**的 heavy-ball 方向（与 Muon 的 momentum buffer 语义一致）

记号（单层，W 为 out×in = m×n，秩 r）：
    X = A_L · Bᵀ            ΔW；A_L 列正交（m×r），B 为 n×r
    B_R                     col(B) 的正交基（n×r）
    ξ = Ȧ · B_Rᵀ + A_L · Ḃᵀ  切向量（Ȧ ⊥ A_L）

一次反向拿黎曼梯度（论文 §6）：adapter 前向写成
``W + A_L·Bᵀ + Z1·B_Rᵀ``，Z1 恒为 0，则 ∇Z1 = G·B_R、∇B = Gᵀ·A_L，
切空间投影 P(G) = (I−A_L A_Lᵀ)·∇Z1·B_Rᵀ + A_L·∇Bᵀ。

每步（全部按形状分桶 batch 计算，fp32）：
    1. 新梯度投影；旧动量 ProjectLR 搬到当前切空间；heavy-ball 累加
    2. OrthoLR：ξ 的非零奇异值全部置 1（QR×2 + 2r×2r SVD），再投影回切空间
    3. RetractionLR：X(1−ηγ) − η·ξ 的秩 r 截断 SVD → 新 A_L / B / B_R

参数里非 Riemannion 的部分（SRA projection 等）走同一 optimizer 内的辅助 AdamW。
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = [
    "RiemannionWithAuxAdam", "project_lr", "ortho_lr", "retract_lr", "riemannian_grad",
]

# 奇异值相对阈值：低于 S.max·tol 的视为 0，不在 OrthoLR 里被抬到 1
_ORTHO_REL_TOL = 1e-6


def _qr(x: Tensor) -> tuple[Tensor, Tensor]:
    return torch.linalg.qr(x, mode="reduced")


def riemannian_grad(a_l: Tensor, grad_z1: Tensor, grad_b: Tensor) -> tuple[Tensor, Tensor]:
    """由 adapter 的两路梯度得切向量 (Ȧ, Ḃ)：Ȧ = (I−A_L A_Lᵀ)·∇Z1，Ḃ = ∇B。"""
    a_dot = grad_z1 - a_l @ (a_l.mT @ grad_z1)
    return a_dot, grad_b


def project_lr(a: Tensor, b: Tensor, a_l: Tensor, b_r: Tensor) -> tuple[Tensor, Tensor]:
    """ProjectLR（Algorithm 2）：把 Z = A·Bᵀ 投影到 (A_L, B_R) 处的切空间。

    P(Z) = A_L A_Lᵀ Z + (I − A_L A_Lᵀ) Z B_R B_Rᵀ
         = (I−A_L A_Lᵀ)·A·(Bᵀ B_R) · B_Rᵀ  +  A_L · (B·(Aᵀ A_L))ᵀ
    """
    a_dot = (a - a_l @ (a_l.mT @ a)) @ (b.mT @ b_r)
    b_dot = b @ (a.mT @ a_l)
    return a_dot, b_dot


def ortho_lr(
    a_l: Tensor, b_r: Tensor, a_dot: Tensor, b_dot: Tensor,
) -> tuple[Tensor, Tensor]:
    """OrthoLR（Algorithm 1）：ξ = [A_L, Ȧ]·[Ḃ, B_R]ᵀ 的非零奇异值置 1。

    返回低秩因子 (Ao, Bo)，Ao·Boᵀ = U·Vᵀ（秩 ≤ 2r）。数值上为 0 的奇异方向
    （如首步 B=0 时）保持为 0，不被抬成 1。
    """
    q1, r1 = _qr(torch.cat([a_l, a_dot], dim=-1))
    q2, r2 = _qr(torch.cat([b_dot, b_r], dim=-1))
    u, s, vh = torch.linalg.svd(r1 @ r2.mT)
    keep = (s > s.amax(dim=-1, keepdim=True) * _ORTHO_REL_TOL).to(u.dtype)
    return q1 @ (u * keep.unsqueeze(-2)), q2 @ vh.mT


def retract_lr(
    a_l: Tensor, b: Tensor, b_r: Tensor, a_dot: Tensor, b_dot: Tensor,
    lr: float, weight_decay: float, rank: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """RetractionLR：X(1−ηγ) − η·ξ 截断到秩 r。

    X(1−ηγ) − η(Ȧ B_Rᵀ + A_L Ḃᵀ) = [A_L, Ȧ] · [(1−ηγ)B − ηḂ, −η B_R]ᵀ，
    两侧 QR 后对 2r×2r 核做 SVD。返回新的 (A_L, B, B_R)。
    """
    q1, r1 = _qr(torch.cat([a_l, a_dot], dim=-1))
    right = torch.cat([(1.0 - lr * weight_decay) * b - lr * b_dot, -lr * b_r], dim=-1)
    q2, r2 = _qr(right)
    u, s, vh = torch.linalg.svd(r1 @ r2.mT)
    new_a_l = q1 @ u[..., :rank]
    new_b_r = q2 @ vh[..., :rank, :].mT
    new_b = new_b_r * s[..., None, :rank]
    return new_a_l, new_b, new_b_r


class RiemannionWithAuxAdam(Optimizer):
    """Riemannion（adapter 标记的 B / Z1 参数）+ 辅助 AdamW（其余参数）。

    adapter 需在参数上挂 ``_riemann_role``（"b" / "z1"）与 ``_riemann_layer``
    （返回所属层的可调用 weakref）；层上有 ``a_l`` / ``b_r`` buffer 与 ``lora_b`` /
    ``lora_z1`` 参数。动量存在 ``self.state[lora_b]``，随 optimizer.state_dict 进 resume。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        adam_betas: tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum,
            adam_betas=tuple(adam_betas), adam_eps=adam_eps, use_riemann=True,
        )
        super().__init__(self._split_groups(params), defaults)

    @staticmethod
    def _split_groups(params) -> list[dict]:
        params = list(params)
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        groups = params if isinstance(params[0], dict) else [{"params": params}]
        split: list[dict] = []
        for group in groups:
            group_params = list(group["params"])
            extra = {k: v for k, v in group.items() if k != "params"}
            riemann = [p for p in group_params if hasattr(p, "_riemann_role")]
            adam = [p for p in group_params if not hasattr(p, "_riemann_role")]
            if riemann:
                split.append({**extra, "params": riemann, "use_riemann": True})
            if adam:
                split.append({**extra, "params": adam, "use_riemann": False})
        return split

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_riemann"]:
                self._riemann_step(group)
            else:
                self._adam_step(group)
        return loss

    def _riemann_step(self, group: dict) -> None:
        lr, wd, beta = group["lr"], group["weight_decay"], group["momentum"]
        buckets: dict[tuple, list] = defaultdict(list)
        for p in group["params"]:
            if p._riemann_role != "b":
                continue
            layer = p._riemann_layer()
            z1 = layer.lora_z1
            if p.grad is None or z1.grad is None:
                continue
            buckets[(tuple(layer.a_l.shape), tuple(p.shape), p.device)].append(layer)

        for layers in buckets.values():
            a_l = torch.stack([ly.a_l for ly in layers]).float()
            b_r = torch.stack([ly.b_r for ly in layers]).float()
            b = torch.stack([ly.lora_b for ly in layers]).float()
            g_z1 = torch.stack([ly.lora_z1.grad for ly in layers]).float()
            g_b = torch.stack([ly.lora_b.grad for ly in layers]).float()
            rank = a_l.shape[-1]

            a_dot, b_dot = riemannian_grad(a_l, g_z1, g_b)
            states = [self.state[ly.lora_b] for ly in layers]
            if "a_hb" in states[0]:
                a_hb = torch.stack([st["a_hb"] for st in states])
                b_hb = torch.stack([st["b_hb"] for st in states])
                a_prev, b_prev = project_lr(a_hb, b_hb, a_l, b_r)
                a_dot = beta * a_prev + a_dot
                b_dot = beta * b_prev + b_dot
            # 缓存动量 ξ = [Ȧ, A_L]·[B_R, Ḃ]ᵀ（下一步再投影到新切空间）
            a_hb_new = torch.cat([a_dot, a_l], dim=-1)
            b_hb_new = torch.cat([b_r, b_dot], dim=-1)

            ao, bo = ortho_lr(a_l, b_r, a_dot, b_dot)
            a_dot, b_dot = project_lr(ao, bo, a_l, b_r)
            new_a_l, new_b, new_b_r = retract_lr(a_l, b, b_r, a_dot, b_dot, lr, wd, rank)

            for i, ly in enumerate(layers):
                ly.a_l.copy_(new_a_l[i])
                ly.b_r.copy_(new_b_r[i])
                ly.lora_b.copy_(new_b[i])
                states[i]["a_hb"] = a_hb_new[i]
                states[i]["b_hb"] = b_hb_new[i]

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
            denom = (exp_avg_sq / (1 - beta2 ** step)).sqrt_().add_(eps)
            p.mul_(1 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-lr / (1 - beta1 ** step))
