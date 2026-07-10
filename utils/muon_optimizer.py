"""Muon & Muon-ScheduleFree optimizer.

Muon (MomentUm Orthogonalized by Newton-Schulz, Keller Jordan 2024):
  2D 参数走 SGD-momentum + Newton-Schulz 正交化（近似 SVD 的 UV^T），给
  出类似 Shampoo 的矩阵预条件效果，但 *不需要* GG/Q 矩阵 -- 优化器状态
  仅 1 个 momentum buffer（同参数大小），而非 SOAP 的 6×（exp_avg +
  exp_avg_sq + GG[0] + GG[1] + Q[0] + Q[1]）。

  Newton-Schulz 迭代原生在 bf16 下稳定（与 Shampoo 的 coupled-Newton
  不同，后者必须 fp32），因此不需要 SOAP 那套 _restore_fp32_state /
  _fp32_tree 的 bf16 补丁。

  1D 参数（bias、DoRA scale）走标准 AdamW fallback。

MuonScheduleFree:
  Schedule-Free 轨迹 + Newton-Schulz 正交化。Drop momentum（SF 的
  Polyak-Ruppert 平均替代之），保留 z base-sequence + 2D 走 NS、1D 走
  AdamW（second-moment only）。train()/eval() swap 遵循 SOAPScheduleFree
  的同一套 y/x 模式。

参考:
  - 原始 repo: github.com/KellerJordan/Muon
  - Kimi Moonlight scaling: arXiv:2502.16982
  - Fantastic Pretraining Optimizers: arXiv:2509.02046
  - Schedule-Free: arXiv:2405.15682
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import torch
from torch.optim import Optimizer
from torch import Tensor


# =============================================================================
# Newton-Schulz 核心迭代
# =============================================================================

@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """5-step Newton-Schulz 多项式迭代，近似计算 G 的正交极因子 (UV^T)。

    在 bf16 下数值稳定（显式设计目标）。返回与 G 同形状的正交化矩阵。

    系数 (3.4445, -4.7750, 2.0315) 拟合了 P(G) = 3X - 4X^3 + ... 的
    quintic Newton-Schulz 迭代，5 步内对 ||X|| <= 1 的矩阵收敛到
    正交极因子。
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    # 归一化到谱范数 <= 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.float()


# =============================================================================
# Muon optimizer (momentum + Newton-Schulz)
# =============================================================================

class Muon(Optimizer):
    """Muon: SGD-momentum + Newton-Schulz 正交化 for 2D params, AdamW for 1D.

  Memory profile (per 2D param of shape (A, B)):
    SOAP:  exp_avg + exp_avg_sq + GG[A,A] + GG[B,B] + Q[A,A] + Q[B,B]
           = 2*A*B + 2*(A^2+B^2)  (bytes: fp32)
    Muon:  momentum_buffer = A*B       (bytes: fp32)
    -> Muon uses 6x less memory than SOAP for large dims.

  Hyperparameters:
    lr:           base learning rate (default 0.02, but with Kimi-style
                  update-RMS scaling this behaves like AdamW lr).
    momentum:     SGD momentum (default 0.95). "Usually fine" (original repo).
    nesterov:     Nesterov momentum (default True). Works better in all tests.
    ns_steps:     Newton-Schulz iterations (default 5). 10 = more accurate
                  but no better performance.
    weight_decay: decoupled weight decay (default 0).
    eps:          AdamW eps for 1D params (default 1e-8).
    betas:        (beta1, beta2) AdamW betas for 1D params only.
    correct_bias: AdamW bias correction for 1D params (default True).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        betas: tuple[float, float] = (0.9, 0.999),
        correct_bias: bool = True,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            eps=eps,
            betas=betas,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            correct_bias = group["correct_bias"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.detach()

                if param.ndim >= 2:
                    # ── 2D: Muon path (momentum + Newton-Schulz) ──────────
                    state = self.state[param]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(
                            param, dtype=torch.float32
                        )

                    buf = state["momentum_buffer"]
                    buf.lerp_(grad.float(), 1.0 - momentum)

                    if nesterov:
                        update = grad.float().lerp_(buf, momentum)
                    else:
                        update = buf.clone()

                    # Newton-Schulz orthogonalization (runs in bf16 internally)
                    orig_shape = update.shape
                    if update.ndim > 2:
                        update = update.view(update.shape[0], -1)
                    update_ns = zeropower_via_newtonschulz5(update, steps=ns_steps)

                    # Update RMS scaling: normalize so that the update has
                    # similar magnitude to AdamW regardless of matrix shape.
                    # Original Muon: scale by sqrt(max(1, rows/cols))
                    rows, cols = update_ns.shape
                    update_ns = update_ns * max(1.0, rows / cols) ** 0.5

                    if wd != 0:
                        update_ns = update_ns.add(param.detach().float(), alpha=wd)

                    param.add_(update_ns.view(orig_shape).to(dtype=param.dtype), alpha=-lr)
                    state["step"] += 1

                else:
                    # ── 1D: AdamW fallback (bias, DoRA scale, etc.) ─────────
                    state = self.state[param]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    g = grad.float()

                    exp_avg.lerp_(g, 1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    bc1 = 1.0 - beta1 ** (state["step"] + 1) if correct_bias else 1.0
                    bc2 = 1.0 - beta2 ** (state["step"] + 1) if correct_bias else 1.0
                    step_size = lr * (bc2 ** 0.5) / bc1 if correct_bias else lr

                    update = exp_avg / (exp_avg_sq.sqrt() + eps)
                    if wd != 0:
                        update = update.add(param.detach().float(), alpha=wd)

                    param.add_(update.to(dtype=param.dtype), alpha=-step_size)
                    state["step"] += 1

        return loss


# =============================================================================
# MuonScheduleFree (Schedule-Free + Newton-Schulz)
# =============================================================================

class MuonScheduleFree(Optimizer):
    """Schedule-Free Muon: Polyak-Ruppert averaging + Newton-Schulz.

    Drops momentum (SF averaging replaces it). 2D params get NS-preconditioned
    gradient applied to z; 1D params get AdamW-normalized gradient applied to z.

    The parameter tensor holds the gradient-evaluation point ``y`` while in
    train mode; call :meth:`eval` before sampling/checkpointing to swap to
    the averaged iterate ``x``, and :meth:`train` to swap back.

    Schedule-Free specific args:
        weight_lr_power: power on lr in the Polyak averaging weight (default 2.0).
        r: power on step index (default 0.0 = uniform average).
        warmup_steps: linear lr warmup (default 0).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        betas: tuple[float, float] = (0.9, 0.95),
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        weight_lr_power: float = 2.0,
        r: float = 0.0,
        warmup_steps: int = 0,
        correct_bias: bool = True,
    ):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(
            lr=lr,
            betas=betas,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            eps=eps,
            weight_lr_power=weight_lr_power,
            r=r,
            warmup_steps=warmup_steps,
            correct_bias=correct_bias,
            k=0,
            weight_sum=0.0,
            lr_max=0.0,
            train_mode=True,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def train(self) -> None:
        """Swap parameter from eval point x back to gradient point y."""
        for group in self.param_groups:
            beta1 = group["betas"][0]
            if not group.get("train_mode", True):
                for param in group["params"]:
                    z = self.state.get(param, {}).get("z")
                    if z is not None:
                        y = param.detach().float()
                        y.lerp_(z, weight=1.0 - beta1)
                        param.copy_(y.to(dtype=param.dtype))
                group["train_mode"] = True

    @torch.no_grad()
    def eval(self) -> None:
        """Swap parameter to the Polyak-averaged iterate x (for sampling/saving)."""
        for group in self.param_groups:
            beta1 = group["betas"][0]
            if group.get("train_mode", True):
                for param in group["params"]:
                    z = self.state.get(param, {}).get("z")
                    if z is not None:
                        x = param.detach().float()
                        x.lerp_(z, weight=1.0 - 1.0 / beta1)
                        param.copy_(x.to(dtype=param.dtype))
                group["train_mode"] = False

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group.get("train_mode", True):
                raise RuntimeError(
                    "MuonScheduleFree.step() called in eval mode; call optimizer.train() first."
                )
            beta1, beta2 = group["betas"]
            ns_steps = group["ns_steps"]
            eps = group["eps"]
            decay = group["weight_decay"]
            lr = group["lr"]
            warmup_steps = group["warmup_steps"]

            k = group["k"]
            sched = (k + 1) / warmup_steps if (warmup_steps > 0 and k < warmup_steps) else 1.0
            bias_correction2 = (1.0 - beta2 ** (k + 1)) if group["correct_bias"] else 1.0
            lr_eff = lr * sched * (bias_correction2 ** 0.5)

            lr_max = group["lr_max"] = max(lr_eff, group["lr_max"])
            weight = ((k + 1) ** group["r"]) * (lr_max ** group["weight_lr_power"])
            weight_sum = group["weight_sum"] = group["weight_sum"] + weight
            ckp1 = weight / weight_sum if weight_sum > 0 else 0.0
            adaptive_y_lr = lr_eff * (beta1 * (1.0 - ckp1) - 1.0)

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.detach().float()
                state = self.state[param]

                if len(state) == 0:
                    state["step"] = 0
                    state["z"] = param.detach().clone().float()

                z = state["z"]

                if param.ndim >= 2:
                    # ── 2D: Newton-Schulz path ─────────────────────────────
                    orig_shape = grad.shape
                    if grad.ndim > 2:
                        grad_flat = grad.view(grad.shape[0], -1)
                    else:
                        grad_flat = grad

                    update = zeropower_via_newtonschulz5(grad_flat, steps=ns_steps)

                    # Update RMS scaling
                    rows, cols = update.shape
                    update = update * max(1.0, rows / cols) ** 0.5

                    if decay != 0.0:
                        y = param.detach().float()
                        update = update.add(y, alpha=decay)

                    # Schedule-Free y/z update (fp32), cast y back to param dtype
                    y = param.detach().float()
                    y.lerp_(z, weight=ckp1)
                    y.add_(update, alpha=adaptive_y_lr)
                    param.copy_(y.to(dtype=param.dtype))
                    z.sub_(update, alpha=lr_eff)

                else:
                    # ── 1D: AdamW-SF path (second moment only) ────────────
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)

                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    update = grad / (exp_avg_sq.sqrt() + eps)

                    if decay != 0.0:
                        y = param.detach().float()
                        update = update.add(y, alpha=decay)

                    y = param.detach().float()
                    y.lerp_(z, weight=ckp1)
                    y.add_(update, alpha=adaptive_y_lr)
                    param.copy_(y.to(dtype=param.dtype))
                    z.sub_(update, alpha=lr_eff)

                state["step"] += 1

            group["k"] = k + 1

        return loss
