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
    2. 基底：Ḃ = B_R·Ḃ_R + Ḃ⊥，对 Ȧ、Ḃ⊥ 各做一次 QR（Q_a ⊥ A_L，Q_b ⊥ B_R），
       则 ξ = [A_L, Q_a]·K·[B_R, Q_b]ᵀ，K = [[Ḃ_Rᵀ, R_bᵀ], [R_a, 0]]（2r×2r）
    3. OrthoLR：K 上做 Newton-Schulz（= ξ 的奇异值推到 ~1）；切空间投影 = 置零右下块
    4. RetractionLR：X(1−ηγ) − η·ξ = 同一基底下的 2r×2r 核，SVD 取前 r → 新 A_L / B / B_R

数学与论文 Algorithm 1/2/4 等价（基底写法见 Vandereycken 2013 的低秩流形回缩），
差别只在数值路径。性能取舍（Anima 280 层实测，朴素写法每步 ~1.1s）：
- 每步只有 2 次 m×r / n×r 的窄 QR，且两侧都不会出现「B 与 B_R 共线」这种病态拼接；
  QR 用 CholeskyQR2（fp64 Gram，Cholesky 失败回退 torch.linalg.qr）
- OrthoLR 用 Newton-Schulz（Muon 同款系数，fp32）求极分解因子，而非 SVD 置 1：
  零奇异值天然保持为 0；论文 §4 亦称近似正交化对收敛几乎无影响
- 回缩的 2r×2r SVD 在 CUDA 上用 gesvda 驱动（对步长部分相对误差 ~1e-4）

参数里非 Riemannion 的部分（SRA projection 等）走同一 optimizer 内的辅助 AdamW。
"""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = [
    "RiemannionWithAuxAdam", "project_lr", "riemannian_grad", "riemannion_update",
]

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)  # 与 Muon 相同（Keller Jordan, 2024）
NS_STEPS = 5


def _cholesky_qr(x: Tensor) -> tuple[Tensor, Tensor] | None:
    gram = (x.mT @ x).double()
    chol, info = torch.linalg.cholesky_ex(gram)
    if bool((info != 0).any()):
        return None
    r = chol.mT.to(x.dtype)
    q = torch.linalg.solve_triangular(r, x, upper=True, left=False)
    return q, r


def _qr(x: Tensor) -> tuple[Tensor, Tensor]:
    """瘦高矩阵（[..., m, k]，k≪m）的 reduced QR：CholeskyQR2，失败回退 Householder。"""
    first = _cholesky_qr(x)
    if first is not None:
        second = _cholesky_qr(first[0])
        if second is not None:
            return second[0], second[1] @ first[1]
    return torch.linalg.qr(x, mode="reduced")


def _qr_perp(x: Tensor, basis: Tensor) -> tuple[Tensor, Tensor]:
    """x 已（解析上）⊥ basis；QR 后再投影一次去掉数值泄漏，保证 [basis, Q] 列正交。"""
    q, r = _qr(x - basis @ (basis.mT @ x))
    q2, r2 = _qr(q - basis @ (basis.mT @ q))
    return q2, r2 @ r


def _polar_newton_schulz(core: Tensor) -> Tensor:
    """方阵 batch 的极分解因子近似（奇异值推到 ~1，零奇异值保持 0），fp32。"""
    a, b, c = NS_COEFFICIENTS
    x = core / core.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    for _ in range(NS_STEPS):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x


def _svd(core: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if core.is_cuda:
        try:
            return torch.linalg.svd(core, driver="gesvda")
        except torch.linalg.LinAlgError:
            # 核有大量近零 / 重复奇异值时 gesvda 不收敛（如从有效秩远低于 r 的普通
            # LoRA 热启动）；回退 CPU 标准 SVD，只在出错的这一步付出代价
            u, s, vh = torch.linalg.svd(core.cpu())
            return u.to(core.device), s.to(core.device), vh.to(core.device)
    return torch.linalg.svd(core)


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


def riemannion_update(
    a_l: Tensor, b: Tensor, b_r: Tensor, a_dot: Tensor, b_dot: Tensor,
    lr: float, weight_decay: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """OrthoLR → 切空间投影 → RetractionLR，返回新的 (A_L, B, B_R)。

    输入切向量 ξ = Ȧ·B_Rᵀ + A_L·Ḃᵀ（Ȧ ⊥ A_L）。全部运算落在
    [A_L, Q_a] × [B_R, Q_b] 基底下的 2r×2r 核上（见模块 docstring）。
    """
    rank = a_l.shape[-1]
    b_dot_r = b_r.mT @ b_dot                       # r×r
    q_a, r_a = _qr_perp(a_dot, a_l)                # Ȧ = Q_a R_a
    q_b, r_b = _qr_perp(b_dot - b_r @ b_dot_r, b_r)  # Ḃ⊥ = Q_b R_b
    zeros = torch.zeros_like(r_a)
    core = torch.cat([
        torch.cat([b_dot_r.mT, r_b.mT], dim=-1),
        torch.cat([r_a, zeros], dim=-1),
    ], dim=-2)
    step = _polar_newton_schulz(core)
    step[..., rank:, rank:] = 0                    # 投影回切空间：去掉 (I−P_A)·O·(I−P_B)
    x_core = torch.zeros_like(core)
    x_core[..., :rank, :rank] = (1.0 - lr * weight_decay) * (b_r.mT @ b).mT  # X = A_L·Cᵀ·B_Rᵀ
    u, s, vh = _svd(x_core - lr * step)
    new_a_l = torch.cat([a_l, q_a], dim=-1) @ u[..., :rank]
    new_b_r = torch.cat([b_r, q_b], dim=-1) @ vh[..., :rank, :].mT
    return new_a_l, new_b_r * s[..., None, :rank], new_b_r


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

            new_a_l, new_b, new_b_r = riemannion_update(a_l, b, b_r, a_dot, b_dot, lr, wd)
            # 全零梯度的层（本步未参与）切空间退化：保持原值。用 batch 掩码而非逐层
            # .any()，避免每层一次 GPU 同步（280 层时 optimizer.step 30ms → 100ms）
            active = ((g_z1.abs().amax(dim=(-2, -1)) > 0) | (g_b.abs().amax(dim=(-2, -1)) > 0))
            active = active[:, None, None]
            new_a_l = torch.where(active, new_a_l, a_l)
            new_b = torch.where(active, new_b, b)
            new_b_r = torch.where(active, new_b_r, b_r)

            # 批量写回：逐层 copy_ 是 3×层数 次 kernel launch
            torch._foreach_copy_(
                [ly.a_l for ly in layers] + [ly.b_r for ly in layers] + [ly.lora_b for ly in layers],
                [*new_a_l.unbind(0), *new_b_r.unbind(0), *new_b.unbind(0)],
            )
            for st, a_hb, b_hb in zip(states, a_hb_new.unbind(0), b_hb_new.unbind(0)):
                st["a_hb"] = a_hb
                st["b_hb"] = b_hb

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
