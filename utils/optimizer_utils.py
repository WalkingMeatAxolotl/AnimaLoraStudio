"""
Optimizer Utils Module - 优化器创建
===================================
支持多种优化器：
1. 标准 AdamW - PyTorch 内置
2. 8-bit AdamW (bitsandbytes) - 内存高效
3. Prodigy (prodigyopt) - 无需调 lr 的自适应优化器
4. ProdigyPlusScheduleFree (prodigy-plus-schedule-free) - Schedule-Free + Prodigy，
   解决 Prodigy 在扩散 LoRA 训练中的 mutation ep / 风格突变问题。
5. Lion - EvoLved Sign Momentum (Chen et al., 2023, arxiv 2302.06675)
6. Automagic - Per-parameter adaptive lr via sign-agreement tracking
   原作者: Ostris (https://github.com/ostris/ai-toolkit, MIT license, Copyright (c)
   2024 Ostris, LLC). bf16 Kahan summation path 借鉴自 tdrussell/diffusion-pipe.
7. SOAP - Adam in the Shampoo eigenbasis (Vyas et al., 2024, arxiv 2409.11321)
8. SOAP-SF - Schedule-Free SOAP，SOAP 预条件 + Schedule-Free trajectory
   (Defazio et al., 2024, "The Road Less Scheduled", arxiv 2405.15682)。SOAP 类
   实现在 utils/soap_optimizer.py（MIT，Copyright (c) 2024 Nikhil Vyas）。
9. CAME - Confidence-guided Adaptive Memory Efficient optimizer
   (Luo et al., 2023, ACL 2023, arxiv 2307.02047)。Adafactor 式分解二阶矩省显存
   + 置信度引导（instability EMA）压制分解近似带来的更新噪声。
   派生自官方实现 yangluo7/CAME（MIT，Copyright (c) 2023 Yang Luo）。
"""

from __future__ import annotations

from contextlib import contextmanager
import inspect
import logging
from typing import List, Dict, Any, Optional, Iterator

import torch
from torch import nn
from torch.optim import Optimizer, AdamW

logger = logging.getLogger(__name__)

# 尝试导入 bitsandbytes
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False

try:
    from optimum.quanto import QBytesTensor
except ImportError:
    QBytesTensor = ()


def _is_param_groups(params: Any) -> bool:
    if isinstance(params, (list, tuple)) and len(params) > 0:
        return isinstance(params[0], dict) and "params" in params[0]
    return False


def _as_float(value: Any) -> Optional[float]:
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().item())
        return float(value)
    except (RuntimeError, TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# 用户通过 schema (ppsf_* 字段) 显式配置的 kwarg。如果上游版本不接受这些，
# silent drop 会让用户的勾选/数值悄悄失效，可能 8 小时后才发现训练效果不对——
# 所以这些必须 fail loud，而不是只 log warning。
_USER_EXPOSED_PPSF_KWARGS = frozenset({
    "d_coef", "prodigy_steps",
    "split_groups", "split_groups_mean",
    "use_speed", "fused_back_pass",
    "use_stableadamw",
})


def _filter_kwargs_by_signature(cls_or_fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(cls_or_fn)
    except (TypeError, ValueError):
        return dict(kwargs)

    accepted, has_var_keyword = set(), False
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            break
        accepted.add(name)

    if has_var_keyword:
        return dict(kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = [k for k in kwargs if k not in accepted]
    if dropped:
        exposed_dropped = [k for k in dropped if k in _USER_EXPOSED_PPSF_KWARGS]
        if exposed_dropped:
            cls_name = getattr(cls_or_fn, "__name__", str(cls_or_fn))
            raise RuntimeError(
                f"[optimizer] {cls_name} 不支持以下用户配置的 kwarg："
                f"{exposed_dropped}。可能是 prodigy-plus-schedule-free 库版本不匹配 "
                f"（pip show prodigy-plus-schedule-free 检查版本）。"
                f"升级/降级依赖，或在 yaml 关掉对应字段。"
            )
        logger.warning(
            f"[optimizer] Dropped unsupported kwargs for "
            f"{getattr(cls_or_fn, '__name__', cls_or_fn)}: {dropped}"
        )
    return filtered


def create_optimizer(
    optimizer_type: str,
    params: Iterator[nn.Parameter],
    learning_rate: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    **kwargs
) -> Optimizer:
    """
    创建优化器

    根据配置创建不同类型的优化器。这是工厂模式的应用，
    将优化器创建逻辑集中管理，便于维护和扩展。

    Args:
        optimizer_type: 优化器类型 ("adamw", "adamw8bit", "prodigy")
        params: 模型参数迭代器
        learning_rate: 学习率
        betas: Adam beta 参数 (beta1, beta2)
        weight_decay: 权重衰减系数
        eps: 数值稳定性 epsilon
        **kwargs: 其他优化器特定参数

    Returns:
        Optimizer: 创建的优化器实例

    Raises:
        ValueError: 如果优化器类型不支持
        ImportError: 如果需要的库未安装
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adamw8bit":
        return create_8bit_adamw(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs
        )

    elif optimizer_type == "adamw":
        return create_standard_adamw(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs
        )

    elif optimizer_type == "prodigy":
        return create_prodigy(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs
        )

    elif optimizer_type == "lion":
        return create_lion(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs,
        )

    elif optimizer_type == "automagic":
        return create_automagic(
            params=params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

    elif optimizer_type == "automagic_v2":
        return create_automagic_v2(
            params=params,
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )

    elif optimizer_type == "prodigy_plus_schedulefree":
        return create_prodigy_plus_schedulefree(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs,
        )

    elif optimizer_type == "soap":
        return create_soap(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs,
        )

    elif optimizer_type == "soap_sf":
        return create_soap_sf(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs,
        )

    elif optimizer_type == "came":
        return create_came(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            **kwargs,
        )

    elif optimizer_type == "muon":
        from utils.muon_optimizer import Muon
        valid_keys = {"momentum", "nesterov", "ns_steps", "correct_bias"}
        muon_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        ignored = [k for k in kwargs if k not in valid_keys]
        if ignored:
            logger.warning(f"[Muon] Ignored unsupported params: {ignored}")
        param_list = params if _is_param_groups(params) else list(params)
        logger.info(f"Creating Muon (lr={learning_rate}, wd={weight_decay}, betas={betas})")
        return Muon(
            param_list, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **muon_kwargs,
        )

    elif optimizer_type in {"muon_sf", "muonsf"}:
        from utils.muon_optimizer import MuonScheduleFree
        if betas == (0.9, 0.999):
            betas = (0.9, 0.95)
        valid_keys = {
            "ns_steps", "weight_lr_power", "r", "warmup_steps", "correct_bias",
        }
        sf_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        ignored = [k for k in kwargs if k not in valid_keys]
        if ignored:
            logger.warning(f"[Muon-SF] Ignored unsupported params: {ignored}")
        param_list = params if _is_param_groups(params) else list(params)
        logger.info(f"Creating Muon-SF (lr={learning_rate}, wd={weight_decay}, betas={betas})")
        return MuonScheduleFree(
            param_list, lr=learning_rate, betas=betas,
            weight_decay=weight_decay, eps=eps, **sf_kwargs,
        )

    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Choose from: adamw, automagic, automagic_v2, came, lion, muon, "
            f"muon_sf, prodigy, prodigy_plus_schedulefree, soap, soap_sf"
        )


def create_8bit_adamw(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    min_8bit_size: int = 4096,
    **kwargs
) -> Optimizer:
    """
    创建 8-bit AdamW 优化器
    
    8-bit AdamW 是 bitsandbytes 库提供的内存高效优化器。
    它将优化器状态（动量、二阶矩）量化为 8-bit，可以
    减少约 50% 的优化器显存占用。
    
    原理：
    - 大多数深度学习参数不需要完整的 32-bit 精度来存储优化器状态
    - 通过分块量化和动态范围调整，8-bit 可以保持良好的优化性能
    
    适用场景：
    - 显存受限的训练（如单卡 RTX 3090 训练大模型）
    - LoRA 训练（虽然 LoRA 参数少，但 8-bit 可以进一步节省内存）
    
    参数说明：
    - min_8bit_size: 小于此大小的张量将保持 32-bit
      这是因为小张量的 8-bit 量化收益不大，反而可能损失精度
    
    Args:
        params: 模型参数
        lr: 学习率
        betas: Adam beta 参数
        weight_decay: 权重衰减
        eps: epsilon
        min_8bit_size: 8-bit 量化的最小张量大小
        
    Returns:
        bnb.optim.AdamW8bit: 8-bit AdamW 优化器
    """
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError(
            "bitsandbytes is required for 8-bit AdamW. "
            "Install with: pip install bitsandbytes"
        )
    
    print(f"Creating 8-bit AdamW optimizer (lr={lr}, weight_decay={weight_decay})")
    print(f"  min_8bit_size: {min_8bit_size}")
    
    # 将参数转换为列表（bitsandbytes 需要可索引的参数）
    param_list = list(params)
    
    # 创建优化器
    optimizer = bnb.optim.AdamW8bit(
        param_list,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        min_8bit_size=min_8bit_size,
        **kwargs
    )
    
    # 计算内存节省
    total_params = sum(p.numel() for p in param_list)
    # 8-bit 优化器状态：约 2 bytes per parameter (vs 8 bytes for 32-bit)
    # 节省约 75% 的优化器状态内存
    estimated_savings_gb = (total_params * 6) / (1024 ** 3)  # 节省 6 bytes per param
    print(f"  [OK] 8-bit AdamW created (estimated memory savings: {estimated_savings_gb:.2f} GB)")
    
    return optimizer


def create_standard_adamw(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    **kwargs
) -> Optimizer:
    """
    创建标准 AdamW 优化器
    
    标准的 PyTorch AdamW 实现。作为后备选项，当其他
    优化器不可用时使用。
    
    AdamW 特点：
    - 将权重衰减与梯度更新解耦（decoupled weight decay）
    - 比 Adam + L2 正则化效果更好
    - 现代深度学习的事实标准优化器
    
    Args:
        params: 模型参数
        lr: 学习率
        betas: Adam beta 参数
        weight_decay: 权重衰减
        eps: epsilon
        
    Returns:
        AdamW: 标准 AdamW 优化器
    """
    print(f"Creating standard AdamW optimizer (lr={lr}, weight_decay={weight_decay})")
    
    # 将参数转换为列表
    param_list = list(params)
    
    optimizer = AdamW(
        param_list,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        **kwargs
    )
    
    print("  [OK] AdamW optimizer created")

    return optimizer


# ============================================================================
# Automagic optimizer + Auto8bitTensor helper
#
# Adapted from ostris/ai-toolkit (https://github.com/ostris/ai-toolkit), which
# also flowed through tdrussell/diffusion-pipe (added Kahan summation for
# bfloat16). The core sign-agreement schedule, 8-bit lr_mask, Adafactor-style
# factored 2nd moment, and stochastic-rounding helpers below are Ostris's
# design. We re-implement on top of the upstream structure and align with
# diffusion-pipe's bf16 Kahan path.
#
# MIT License — Copyright (c) 2024 Ostris, LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ============================================================================


class Auto8bitTensor:
    def __init__(self, data: torch.Tensor | dict[str, Any]) -> None:
        if isinstance(data, dict):
            self.quantized = data["quantized"]
            self.scale = data["scale"]
            self.orig_dtype = data["orig_dtype"]
            return
        abs_max = data.abs().max().item()
        self.scale = abs_max / 127.0 if abs_max > 0 else 1.0
        self.quantized = (data / self.scale).round().clamp(-127, 127).to(torch.int8)
        self.orig_dtype = data.dtype

    def dequantize(self) -> torch.Tensor:
        return self.quantized.to(dtype=torch.float32) * self.scale

    def to(self, *args, **kwargs):
        return self.dequantize().to(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        return {
            "quantized": self.quantized,
            "scale": self.scale,
            "orig_dtype": self.orig_dtype,
        }


def _copy_stochastic_bf16(target: torch.Tensor, source: torch.Tensor) -> None:
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=(1 << 16))
    result.add_(source.view(dtype=torch.int32))
    result.bitwise_and_(-65536)
    target.copy_(result.view(dtype=torch.float32))


def _copy_stochastic(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.dtype == torch.float32:
        target.copy_(source)
        return
    if target.dtype == torch.bfloat16:
        _copy_stochastic_bf16(target, source)
        return
    target.copy_(source.to(target.dtype))


# Note on stochastic-rounding grad accumulation:
# Upstream (ostris/ai-toolkit, tdrussell/diffusion-pipe) registers a
# post-accumulate-grad hook to do fp32 grad accumulation with stochastic
# rounding. Automagic (v1) deliberately does NOT enable that path because it
# silently breaks two common training paths (Automagic2 / v2 below is the
# explicit experimental opt-in for the fused-backward design — gated off in
# the UI by default, see SystemConfig.enable_automagic_v2):
#   1. torch.cuda.amp.GradScaler.unscale_ skips params where p.grad is None
#      — the hook deletes p.grad after each backward, so unscale never runs
#      and the optimizer would step on still-scaled gradients.
#   2. torch.nn.utils.clip_grad_norm_ skips p.grad is None, so grad clipping
#      becomes a no-op.
# bf16 numerical stability is instead handled inside Automagic.step via
# Kahan compensated summation (see `shift` buffer below). See upstream
# diffusion-pipe optimizers/automagic.py:72-79 (commented out by author).


class Automagic(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 1e-6,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if min_lr < 0.0:
            raise ValueError(f"Invalid min_lr: {min_lr}")
        if max_lr <= 0.0 or max_lr < min_lr:
            raise ValueError(f"Invalid max_lr: {max_lr}")
        if lr_bump < 0.0:
            raise ValueError(f"Invalid lr_bump: {lr_bump}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if clip_threshold <= 0.0:
            raise ValueError(f"Invalid clip_threshold: {clip_threshold}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        self.lr = min(lr, max_lr)
        if lr > 1e-3:
            logger.warning("Automagic start lr %s is high; clamping to 1e-6", lr)
            self.lr = 1e-6
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        super().__init__(
            params,
            dict(
                lr=self.lr,
                eps=eps,
                clip_threshold=clip_threshold,
                beta2=beta2,
                weight_decay=weight_decay,
            ),
        )
        self.base_lrs = [self.lr for _ in self.param_groups]

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @staticmethod
    def _get_lr(param_state: dict[str, Any]) -> float:
        avg_lr = param_state.get("avg_lr")
        if avg_lr is None:
            return 0.0
        if isinstance(avg_lr, torch.Tensor):
            return float(avg_lr.detach().float().item())
        return float(avg_lr)

    def _get_group_lr(self, group: dict[str, Any]) -> float:
        lrs = [self._get_lr(self.state[p]) for p in group["params"] if p in self.state]
        if not lrs:
            return self.lr
        return sum(lrs) / len(lrs)

    def get_learning_rates(self) -> list[float]:
        lrs = [self._get_group_lr(group) for group in self.param_groups]
        return lrs or self.base_lrs

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs)

    def initialize_state(self, p: nn.Parameter) -> None:
        state = self.state[p]
        state["step"] = 0
        if "lr_mask" not in state:
            state["lr_mask"] = Auto8bitTensor(
                torch.ones(p.shape, device=p.device, dtype=torch.float32) * self.lr
            )
        state["avg_lr"] = torch.mean(state["lr_mask"].to(torch.float32))
        if "last_polarity" not in state:
            state["last_polarity"] = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
        if len(p.shape) >= 2:
            state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1]).to(p)
            state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:]).to(p)
        else:
            state["exp_avg_sq"] = torch.zeros_like(p)
        state["RMS"] = 0
        # Kahan compensated summation for bf16 — keeps the rounded-off portion
        # of each update so it can be applied on the next step. Aligns with
        # upstream diffusion-pipe (optimizers/automagic.py:354-356).
        if p.dtype == torch.bfloat16 and "shift" not in state:
            state["shift"] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue
                grad = p.grad
                if grad.dtype != torch.float32:
                    grad = grad.to(torch.float32)
                if grad.is_sparse:
                    raise RuntimeError("Automagic does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    self.initialize_state(p)
                factored = len(grad.shape) >= 2
                if factored:
                    state.setdefault("exp_avg_sq_row", torch.zeros(p.shape[:-1]).to(grad))
                    state.setdefault("exp_avg_sq_col", torch.zeros(p.shape[:-2] + p.shape[-1:]).to(grad))
                    state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                    state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                else:
                    state.setdefault("exp_avg_sq", torch.zeros_like(grad))
                    state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.dequantize() if QBytesTensor and isinstance(p, QBytesTensor) else p
                if p.dtype != torch.float32:
                    p_data_fp32 = p_data_fp32.clone().float()

                state["step"] = state.get("step", 0) + 1
                state["RMS"] = self._rms(p_data_fp32)
                beta2 = group["beta2"]
                eps = group["eps"]
                update = (grad ** 2) + eps
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=(1.0 - beta2))
                    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=(1.0 - beta2))
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update, alpha=(1.0 - beta2))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                if "last_polarity" not in state or "lr_mask" not in state:
                    self.initialize_state(p)
                current_polarity = (update > 0).to(torch.bool)
                sign_agreement = torch.where(state["last_polarity"] == current_polarity, 1, -1)
                state["last_polarity"] = current_polarity
                lr_mask = state["lr_mask"].to(torch.float32)
                new_lr = torch.where(
                    sign_agreement > 0,
                    lr_mask + self.lr_bump,
                    lr_mask - self.lr_bump,
                )
                new_lr = torch.clamp(new_lr, min=self.min_lr, max=self.max_lr)
                update.mul_(new_lr)
                state["lr_mask"] = Auto8bitTensor(new_lr)
                state["avg_lr"] = torch.mean(new_lr)

                if group["weight_decay"] != 0:
                    weight_decay_update = p_data_fp32 * (-group["weight_decay"]) * new_lr
                else:
                    weight_decay_update = None

                if p.dtype == torch.bfloat16:
                    # Kahan compensated summation — matches upstream
                    # diffusion-pipe (optimizers/automagic.py:308-318). Trades
                    # one extra bf16 buffer (`state['shift']`) for unbiased,
                    # zero-variance accumulation over long bf16 training; the
                    # alternative stochastic-rounding path injects ~scale/2
                    # noise per step into the lr_mask sign-agreement signal.
                    update.mul_(-1)
                    if weight_decay_update is not None:
                        update.add_(weight_decay_update)
                    shift = state.setdefault("shift", torch.zeros_like(p))
                    shift.add_(update)
                    # Use grad tensor as scratch buffer for the pre-update p,
                    # so shift carries forward only the bf16 rounding error.
                    grad.copy_(p.detach())
                    p.add_(shift)
                    shift.add_(grad.sub_(p))
                else:
                    if weight_decay_update is not None:
                        p_data_fp32.add_(weight_decay_update)
                    p_data_fp32.add_(-update)
                    if p.dtype != torch.float32:
                        _copy_stochastic(p, p_data_fp32)

        return loss

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["state"] = {
            p: {
                **{k: v for k, v in state.items() if k != "lr_mask"},
                **({"lr_mask": state["lr_mask"].state_dict()} if "lr_mask" in state else {}),
            }
            for p, state in state_dict["state"].items()
        }
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        converted = {
            "state": {},
            "param_groups": state_dict.get("param_groups", []),
        }
        for param_id, state in state_dict.get("state", {}).items():
            converted_state = dict(state)
            if isinstance(converted_state.get("lr_mask"), dict):
                converted_state["lr_mask"] = Auto8bitTensor(converted_state["lr_mask"])
            converted["state"][param_id] = converted_state
        result = super().load_state_dict(converted)
        # PyTorch 只把 float state 搬到 param 的 dtype/device；bool tensor 与
        # Auto8bitTensor 内部的 int8 不参与，resume 后留在 CPU。统一搬回。
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st is None:
                    continue
                device = p.device
                if isinstance(st.get("last_polarity"), torch.Tensor):
                    st["last_polarity"] = st["last_polarity"].to(device=device)
                if isinstance(st.get("lr_mask"), Auto8bitTensor):
                    st["lr_mask"].quantized = st["lr_mask"].quantized.to(device=device)
        return result


def create_automagic(
    params: Iterator[nn.Parameter],
    lr: float,
    min_lr: float = 1e-7,
    max_lr: float = 1e-3,
    lr_bump: float = 1e-6,
    eps: float = 1e-30,
    clip_threshold: float = 1.0,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
    **kwargs,
) -> Optimizer:
    # Automagic 上游推荐 init lr=1e-6（每参数自适应起点）；> 1e-5 量级是 AdamW
    # 风格 lr 误用，sign-agreement 调度需要很多 step 才能从过高起点收敛回工作区间。
    # UI 切换 optimizer_type 时会自动改写 lr=1e-6；这里兜底 saved config / CLI 路径。
    if lr > 1e-5:
        logger.warning(
            "Automagic 初始 lr=%.2e 远高于推荐 1e-6；sign-agreement 自适应从过高起点"
            "收敛慢，建议设为 1e-6（per-param lr 由 [min_lr, max_lr] 自动调）",
            lr,
        )
    param_list = params if _is_param_groups(params) else list(params)
    optimizer = Automagic(
        param_list,
        lr=lr,
        min_lr=min_lr,
        max_lr=max_lr,
        lr_bump=lr_bump,
        eps=eps,
        clip_threshold=clip_threshold,
        beta2=beta2,
        weight_decay=weight_decay,
        **kwargs,
    )
    print(
        f"Creating Automagic optimizer (lr={lr}, min_lr={min_lr}, max_lr={max_lr}, "
        f"lr_bump={lr_bump}, beta2={beta2}, weight_decay={weight_decay})"
    )
    print("  [OK] Automagic optimizer created")
    return optimizer


class Automagic2(Optimizer):
    """Automagic v2 — per-parameter scalar LR, fused into backward pass.

    Instead of a per-element lr_mask, keeps one scalar lr per parameter tensor.
    The optimizer step is fused into backward via register_post_accumulate_grad_hook:
    each parameter is updated and its grad freed as soon as autograd finishes
    accumulating into it. .step() is therefore a no-op and peak VRAM stays low.

    Incompatible with GradScaler, clip_grad_norm_ and gradient accumulation
    (params update during backward) — see the design note above class Automagic.
    Experimental: UI 默认隐藏（SystemConfig.enable_automagic_v2）。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        min_lr: float = 1e-7,
        max_lr: float = 1e-3,
        lr_bump: float = 1e-6,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        agreement_threshold: float = 0.5,
    ) -> None:
        if lr > 1e-3:
            logger.warning("Automagic2 start lr %s is high; forcing to 1e-6.", lr)
            lr = 1e-6
        defaults = dict(
            lr=lr, min_lr=min_lr, max_lr=max_lr, lr_bump=lr_bump,
            beta2=beta2, eps=eps, clip_threshold=clip_threshold,
            weight_decay=weight_decay, agreement_threshold=agreement_threshold,
        )
        super().__init__(params, defaults)

        self._hook_handles: list = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    handle = p.register_post_accumulate_grad_hook(
                        self._make_backward_hook(group)
                    )
                    self._hook_handles.append(handle)

        total = sum(p.numel() for g in self.param_groups for p in g["params"])
        logger.info(f"Automagic2 total training params: {total:,}")

    @staticmethod
    def _rms(t: torch.Tensor) -> torch.Tensor:
        return t.norm(2) / (t.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        return torch.mul(r, c)

    def _init_state(self, p: torch.Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = 0
        state["lr"] = torch.full((), float(group["lr"]), dtype=torch.float32, device=p.device)
        state["last_polarity"] = torch.zeros(p.shape, dtype=torch.bool, device=p.device)
        # 二阶矩固定 fp32：bf16 存储下 (1-beta2)=1e-3 的每步相对增量小于 bf16
        # 相对分辨率 (~3.9e-3)，EMA 会在写回时被 round 吞掉而停滞。
        if p.dim() >= 2:
            state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1], dtype=torch.float32, device=p.device)
            state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:], dtype=torch.float32, device=p.device)
        else:
            state["exp_avg_sq"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)

    def _make_backward_hook(self, group):
        def _hook(p: torch.Tensor):
            self._update_param(p, group)
        return _hook

    @torch.no_grad()
    def _update_param(self, p: torch.Tensor, group: dict) -> None:
        if p.grad is None:
            return
        state = self.state[p]
        if len(state) == 0:
            self._init_state(p, group)

        grad = p.grad
        if grad.is_sparse:
            raise RuntimeError("Automagic2 does not support sparse gradients.")
        if grad.dtype != torch.float32:
            grad = grad.to(torch.float32)

        # fused 路径绕过了训练循环的 step 边界 NaN 梯度检查（hook 跑完 p.grad
        # 已是 None），必须在这里自卫 —— 否则一个坏 micro-batch 直接毒化权重。
        if not torch.isfinite(grad).all():
            logger.warning(
                "Automagic2: param %s 梯度含 NaN/Inf，跳过本次 fused update",
                tuple(p.shape),
            )
            p.grad = None
            return

        beta2 = group["beta2"]
        eps = group["eps"]
        sq = (grad * grad).add_(eps)

        if p.dim() >= 2:
            row_state = state["exp_avg_sq_row"]
            col_state = state["exp_avg_sq_col"]
            if row_state.dtype == torch.float32:
                row, col = row_state, col_state
            else:
                row = row_state.to(torch.float32)
                col = col_state.to(torch.float32)
            row.mul_(beta2).add_(sq.mean(dim=-1), alpha=1.0 - beta2)
            col.mul_(beta2).add_(sq.mean(dim=-2), alpha=1.0 - beta2)
            if row_state.dtype != torch.float32:
                row_state.copy_(row.to(row_state.dtype))
                col_state.copy_(col.to(col_state.dtype))
            update = self._approx_sq_grad(row, col).mul_(grad)
        else:
            v_state = state["exp_avg_sq"]
            if v_state.dtype == torch.float32:
                v = v_state
            else:
                v = v_state.to(torch.float32)
            v.mul_(beta2).add_(sq, alpha=1.0 - beta2)
            if v_state.dtype != torch.float32:
                v_state.copy_(v.to(v_state.dtype))
            update = v.rsqrt().mul_(grad)

        update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

        # Per-element sign agreement collapsed to a single scalar lr decision
        cur_polarity = update > 0
        last_polarity = state["last_polarity"]
        agreement = (cur_polarity == last_polarity).to(torch.float32).mean()
        state["last_polarity"] = cur_polarity

        lr_t = state["lr"]
        if state["step"] > 0:
            direction = (agreement >= group["agreement_threshold"]).to(lr_t.dtype) * 2.0 - 1.0
            lr_t.add_(direction, alpha=group["lr_bump"]).clamp_(
                min=group["min_lr"], max=group["max_lr"]
            )
        state["step"] += 1

        update.mul_(lr_t)
        wd = group["weight_decay"]

        if p.dtype == torch.bfloat16:
            new_p_fp32 = p.to(torch.float32)
            if wd != 0.0:
                update.addcmul_(new_p_fp32, lr_t, value=wd)
            new_p_fp32.sub_(update)
            # Stochastic rounding fp32 → bf16
            as_int = new_p_fp32.view(torch.int32)
            as_int.add_(torch.randint_like(as_int, 1 << 16)).bitwise_and_(-65536)
            p.copy_(new_p_fp32)
        else:
            if wd != 0.0:
                p_fp32 = p if p.dtype == torch.float32 else p.to(torch.float32)
                update.addcmul_(p_fp32, lr_t, value=wd)
            p.add_(update.to(p.dtype), alpha=-1.0)

        p.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        return loss

    def get_learning_rates(self) -> "list[float]":
        out = []
        for group in self.param_groups:
            lrs = [
                float(self.state[p]["lr"])
                for p in group["params"]
                if p in self.state and "lr" in self.state[p]
            ]
            out.append(sum(lrs) / len(lrs) if lrs else float(group["lr"]))
        return out

    def get_avg_learning_rate(self) -> float:
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs) if lrs else float(self.defaults["lr"])

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            for k, v in self.defaults.items():
                group[k] = v
            for p in group["params"]:
                st = self.state.get(p)
                if st is None:
                    continue
                # PyTorch load 会把 float state cast 到 param dtype（bf16 训练时
                # fp32 标量/二阶矩会被降级），这里统一恢复 fp32。
                for key in ("lr", "exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_sq"):
                    v = st.get(key)
                    if isinstance(v, torch.Tensor) and v.dtype != torch.float32:
                        st[key] = v.to(torch.float32)


def create_automagic_v2(
    params: Iterator[nn.Parameter],
    lr: float,
    min_lr: float = 1e-7,
    max_lr: float = 1e-3,
    lr_bump: float = 1e-6,
    eps: float = 1e-30,
    clip_threshold: float = 1.0,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
    agreement_threshold: float = 0.5,
    **kwargs,
) -> Optimizer:
    if lr > 1e-5:
        logger.warning(
            "Automagic2 初始 lr=%.2e 远高于推荐 1e-6；sign-agreement 自适应从过高起点"
            "收敛慢，建议设为 1e-6", lr,
        )
    param_list = params if _is_param_groups(params) else list(params)
    optimizer = Automagic2(
        param_list, lr=lr, min_lr=min_lr, max_lr=max_lr, lr_bump=lr_bump,
        eps=eps, clip_threshold=clip_threshold, beta2=beta2, weight_decay=weight_decay,
        agreement_threshold=agreement_threshold,
    )
    logger.info(
        f"Creating Automagic v2 (lr={lr}, min_lr={min_lr}, max_lr={max_lr}, "
        f"lr_bump={lr_bump}, beta2={beta2}, wd={weight_decay}, "
        f"agreement_threshold={agreement_threshold})"
    )
    return optimizer


class Lion(Optimizer):
    """EvoLved Sign Momentum optimizer (Chen et al., 2023).

    Paper: "Symbolic Discovery of Optimization Algorithms"
        https://arxiv.org/abs/2302.06675  (Google Brain)

    Reference implementations:
    - Google official:    https://github.com/google/automl/tree/master/lion
    - Community PyTorch:  https://github.com/lucidrains/lion-pytorch

    This is a minimal re-implementation aligning with the paper's Algorithm 1
    and Google's reference; no dependency on `lion-pytorch`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                if weight_decay != 0.0:
                    p.mul_(1 - lr * weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]

                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def create_lion(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.99),
    weight_decay: float = 0.0,
    **kwargs,
) -> Optimizer:
    # Lion 论文（Chen et al. 2023, arxiv 2302.06675 §4.3）经验：lr ≈ AdamW lr / 3，
    # weight_decay 3-10× AdamW。从 AdamW 默认 lr=1e-4 直切 Lion 容易发散；这里在
    # lr 落在 AdamW 量级（1e-4 及以上）时提示一下。详细见 docs/user-guide/optimizers.md。
    if lr >= 1e-4:
        logger.warning(
            "Lion lr=%.2e 接近/高于 AdamW 量级；论文推荐 lr ≈ AdamW lr / 3 "
            "（如 AdamW 1e-4 → Lion ~3e-5）。继续训练但可能发散，详见 "
            "docs/user-guide/optimizers.md",
            lr,
        )
    param_list = params if _is_param_groups(params) else list(params)
    optimizer = Lion(param_list, lr=lr, betas=betas, weight_decay=weight_decay, **kwargs)
    print(f"Creating Lion optimizer (lr={lr}, betas={betas}, weight_decay={weight_decay})")
    print("  [OK] Lion optimizer created")
    return optimizer


class CAME(Optimizer):
    """Confidence-guided Adaptive Memory Efficient optimizer (Luo et al., 2023).

    Paper: "CAME: Confidence-guided Adaptive Memory Efficient Optimization"
        https://arxiv.org/abs/2307.02047  (ACL 2023 Outstanding Paper)

    Adafactor 式行/列分解二阶矩省显存 + 置信度引导：用 update 与其动量 exp_avg
    的残差平方 EMA（instability）近似逐元素置信度，对动量更新做逆方差加权，
    压制分解近似带来的更新噪声。

    派生自官方实现 yangluo7/CAME（MIT License, Copyright (c) 2023 Yang Luo），
    做了本仓库统一的工程调整（算法公式与官方 step 逐行对齐，未改动）：
    - optimizer state 固定 fp32（bf16 LoRA/LoKr 训练数值稳定，同 SOAP）
    - bf16 参数写回走 stochastic rounding（_copy_stochastic 的 bit-trick 仅
      适用 bf16；fp16 写回为普通 cast，但计算与 state 仍全程 fp32）
    - load_state_dict 后恢复 fp32 state（PyTorch load 会 cast 到 param dtype）

    Args:
        params: 参数或 param group 列表
        lr: 学习率（真实 lr，AdamW 量级；不像 Prodigy 填 1.0）
        eps: (eps1, eps2) — 分别用于二阶矩正则与 instability 正则
        clip_threshold: update RMS 裁剪阈值
        betas: (β1, β2, β3) — 动量 / 二阶矩 / instability EMA 衰减
        weight_decay: 解耦权重衰减（L2）
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        eps: tuple[float, float] = (1e-30, 1e-16),
        clip_threshold: float = 1.0,
        betas: tuple[float, float, float] = (0.9, 0.999, 0.9999),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # β=1.0 会让对应 EMA 永远停在初始零值，_approx_sq_grad 算出 0/0=NaN
        # 首步即毒化参数，故与 Lion 同样取开区间上界
        if len(betas) != 3 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError(f"Invalid betas: {betas} (CAME needs 0 <= β1, β2, β3 < 1)")
        # eps=0 在整张梯度为零（如 LoRA 零初始化首步）时同样产生 0/0=NaN
        if len(eps) != 2 or not all(e > 0.0 for e in eps):
            raise ValueError(f"Invalid eps: {eps} (CAME needs positive (eps1, eps2))")
        if clip_threshold <= 0.0:
            raise ValueError(f"Invalid clip_threshold: {clip_threshold}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(
            lr=lr,
            eps=tuple(eps),
            clip_threshold=clip_threshold,
            betas=tuple(betas),
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row: torch.Tensor, exp_avg_sq_col: torch.Tensor) -> torch.Tensor:
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        # PyTorch load 会把 float state cast 到 param dtype；CAME state 固定
        # fp32（bf16 下 (1-β2)=1e-3 的 EMA 增量会被 bf16 round 吞掉），统一恢复。
        # 不列 key 白名单：CAME 创建的张量 state 全是 fp32（"step" 是 int），
        # 恢复所有浮点张量即可，未来新增 state key 也不会漏。
        for state in self.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point() \
                        and value.dtype != torch.float32:
                    state[key] = value.float()
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group["betas"]
            eps1, eps2 = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients")
                if grad.dtype != torch.float32:
                    grad = grad.float()

                state = self.state[p]
                factored = grad.dim() >= 2

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad.shape[:-1], device=grad.device, dtype=torch.float32)
                        state["exp_avg_sq_col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], device=grad.device, dtype=torch.float32)
                        state["exp_avg_res_row"] = torch.zeros(grad.shape[:-1], device=grad.device, dtype=torch.float32)
                        state["exp_avg_res_col"] = torch.zeros(grad.shape[:-2] + grad.shape[-1:], device=grad.device, dtype=torch.float32)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                state["step"] += 1

                update = (grad ** 2) + eps1
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(update.mean(dim=-1), alpha=1.0 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(update.mean(dim=-2), alpha=1.0 - beta2)
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update, alpha=1.0 - beta2)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(update, alpha=1.0 - beta1)

                # Confidence-guided strategy: instability = (update - exp_avg)^2 的
                # EMA，行/列分解后取 rsqrt 作为逆方差权重乘到动量上
                if factored:
                    res = (update - exp_avg) ** 2 + eps2
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]
                    exp_avg_res_row.mul_(beta3).add_(res.mean(dim=-1), alpha=1.0 - beta3)
                    exp_avg_res_col.mul_(beta3).add_(res.mean(dim=-2), alpha=1.0 - beta3)
                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    update = res_approx.mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                # .float() 对非 fp32 张量本身就返回新拷贝，无需先 clone
                p_data_fp32 = p if p.dtype == torch.float32 else p.detach().float()

                if weight_decay != 0.0:
                    p_data_fp32.add_(p_data_fp32, alpha=-weight_decay * lr)

                update.mul_(lr)
                p_data_fp32.add_(-update)

                if p.dtype != torch.float32:
                    _copy_stochastic(p, p_data_fp32)

        return loss


def create_came(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.999, 0.9999),
    weight_decay: float = 0.0,
    eps: tuple = (1e-30, 1e-16),
    clip_threshold: float = 1.0,
    **kwargs,
) -> Optimizer:
    """创建 CAME 优化器（Luo et al. 2023, arxiv 2307.02047）。

    lr 用 AdamW 量级真实值（LoRA 常用 1e-4 量级），不像 Prodigy 填 1.0。
    可配常规 lr_scheduler。

    Args:
        betas: (β1, β2, β3) — 动量 / 分解二阶矩 / instability EMA 衰减。
            上层 create_optimizer 传其 Adam 默认 (0.9, 0.999) 时补论文默认 β3；
            其余 2 元组视为配置错误，交给 CAME.__init__ 报 Invalid betas。
        eps: (eps1, eps2) — 二阶矩正则 / instability 正则。上层 create_optimizer
            传其 Adam 默认标量 1e-8 时回退论文默认 (1e-30, 1e-16)；显式传其他
            标量则报错（标量 eps 对 CAME 无意义，静默替换会吞掉用户配置）。
        clip_threshold: update RMS 裁剪阈值（同 Adafactor d 参数）。
    """
    betas = tuple(betas)
    if betas == (0.9, 0.999):
        # create_optimizer 的通用 Adam 默认；CAME 需要 (β1, β2, β3)，同 PPSF
        # 的默认哨兵检测，只映射默认值，显式 2 元组由 CAME.__init__ 拒绝
        betas = (0.9, 0.999, 0.9999)
    if isinstance(eps, (int, float)):
        if eps == 1e-8:
            # create_optimizer 的通用 Adam 默认标量；回退论文默认
            eps = (1e-30, 1e-16)
        else:
            raise ValueError(
                f"CAME needs eps as an (eps1, eps2) pair, got scalar {eps}"
            )

    param_list = params if _is_param_groups(params) else list(params)
    optimizer = CAME(
        param_list,
        lr=lr,
        betas=betas,
        eps=tuple(eps),
        clip_threshold=clip_threshold,
        weight_decay=weight_decay,
        **kwargs,
    )
    logger.info(
        f"Creating CAME optimizer (lr={lr}, betas={betas}, eps={tuple(eps)}, "
        f"clip_threshold={clip_threshold}, weight_decay={weight_decay})"
    )
    return optimizer


def create_prodigy(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    d_coef: float = 1.0,
    safeguard_warmup: bool = True,
    use_bias_correction: bool = True,
    **kwargs,
) -> Optimizer:
    """
    创建 Prodigy 优化器 (https://github.com/konstmish/prodigy)

    Prodigy 自适应估计学习率，lr 应设为 1.0（这里的 lr 是 d 的放大系数）。
    如果传入的 lr != 1.0，会强制覆盖为 1.0 并打印警告。

    推荐默认：safeguard_warmup=True, use_bias_correction=True, d_coef=1.0。
    使用 constant 或 cosine scheduler 即可，不建议叠 restart。

    Args:
        params: 模型参数
        lr: 学习率（Prodigy 要求 1.0）
        betas: Adam beta 参数
        weight_decay: 权重衰减
        eps: epsilon
        d_coef: d 的初始缩放系数
        safeguard_warmup: warmup 期间保护 d 不过快增长
        use_bias_correction: 是否使用偏差修正

    Returns:
        Prodigy: Prodigy 优化器
    """
    try:
        from prodigyopt import Prodigy
    except ImportError as e:
        raise ImportError(
            "prodigyopt is required for Prodigy optimizer. "
            "Install with: pip install prodigyopt"
        ) from e

    if abs(lr - 1.0) > 1e-9:
        print(
            f"[WARN] Prodigy requires lr=1.0 (received {lr}); forcing lr=1.0. "
            f"Tune d_coef/weight_decay instead of lr."
        )
        lr = 1.0

    print(
        f"Creating Prodigy optimizer (lr={lr}, weight_decay={weight_decay}, "
        f"d_coef={d_coef}, safeguard_warmup={safeguard_warmup})"
    )

    param_list = list(params)

    optimizer = Prodigy(
        param_list,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        d_coef=d_coef,
        safeguard_warmup=safeguard_warmup,
        use_bias_correction=use_bias_correction,
        **kwargs,
    )

    print("  [OK] Prodigy optimizer created")

    return optimizer


def create_prodigy_plus_schedulefree(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.999),
    weight_decay: float = 0.0,
    eps: Optional[float] = None,
    d_coef: float = 1.0,
    prodigy_steps: int = 0,
    split_groups: bool = True,
    split_groups_mean: bool = False,
    use_speed: bool = False,
    fused_back_pass: bool = False,
    use_stableadamw: bool = True,
    **kwargs,
) -> Optimizer:
    """
    创建 ProdigyPlusScheduleFree 优化器
    (https://github.com/LoganBooker/prodigy-plus-schedule-free)

    Prodigy + Schedule-Free 的合体。相对普通 Prodigy 解决的核心问题：
    - **Schedule-Free 的 averaged weights**: 维护训练权重 y 和 averaged 权重 x，
      sample/save 用 x 出图 → 风格突变 ep 现象基本消失。
      *使用要求*: sample/eval/save 前必须调 optimizer.eval()，事后 optimizer.train()。
      用 optimizer_eval_mode(optimizer) context manager 包装最稳。
    - **prodigy_steps 冻结 d**: 到某 step 后不再更新 d，避免后期跳档。
    - **split_groups 细粒度估计**: 按 param group 分别估 d。

    *学习率*: 必须固定为 1.0（如同普通 Prodigy）。Schedule-Free 不需要 scheduler，
    调用方应强制 lr_scheduler=none 并在启动期校验。

    *betas 默认*: PPSF 上游默认 (0.9, 0.99) 而非 PyTorch AdamW 的 (0.9, 0.999)。
    本工厂检测到传入是 PyTorch 默认时自动覆盖到 PPSF 推荐值；如调用方显式传入则尊重。

    Args:
        params: 模型参数
        lr: 学习率（PPSF 要求 1.0）
        betas: Adam beta 参数（PPSF 推荐 (0.9, 0.99)）
        weight_decay: 权重衰减
        eps: epsilon
        d_coef: d 的初始缩放系数（小数据集建议 0.5）
        prodigy_steps: 在第 N 步后冻结 d（0 = 不冻结，整个训练继续更新）
        split_groups: 按 param group 分别估 d
        split_groups_mean: split_groups=True 时是否取各组 d 的均值
            (PPSF 默认 False；SimpleTuner 改成 True 但理由是给 transformer-only 训练，
             我们走 LoRA + LoKr 多 param group 不适合，保持 False)
        use_speed: 启用加速模式（实验性）
        fused_back_pass: 与 PyTorch fused-backward 路径集成（显存吃紧时开）
        use_stableadamw: 用 stable AdamW 归一化策略

    Returns:
        ProdigyPlusScheduleFree: 优化器实例
    """
    try:
        # pip 包名 `prodigy-plus-schedule-free`，import 名 `prodigyplus`
        from prodigyplus import ProdigyPlusScheduleFree
    except ImportError as e:
        raise ImportError(
            "prodigy-plus-schedule-free is required for ProdigyPlusScheduleFree "
            "optimizer. Install with: pip install 'prodigy-plus-schedule-free>=2.0.0'"
        ) from e

    if abs(lr - 1.0) > 1e-9:
        logger.warning(
            f"[ProdigyPlus] Forcing lr=1.0 (got {lr}); "
            f"Prodigy adapts step size internally via d."
        )
    lr = 1.0

    if isinstance(eps, (int, float)) and eps <= 0:
        logger.warning(f"[ProdigyPlus] eps={eps} non-positive, falling back to None (Adam-atan2).")
        eps = None

    # 上层 create_optimizer 默认 betas=(0.9, 0.999)（适合 AdamW），但 PPSF 推荐
    # (0.9, 0.99)。如果调用方没改默认，覆盖到 PPSF 推荐值。
    if tuple(betas) == (0.9, 0.999):
        betas = (0.9, 0.99)

    candidate = dict(
        lr=lr,
        betas=tuple(betas),
        eps=eps,
        weight_decay=weight_decay,
        d_coef=d_coef,
        prodigy_steps=prodigy_steps,
        split_groups=split_groups,
        split_groups_mean=split_groups_mean,
        use_speed=use_speed,
        fused_back_pass=fused_back_pass,
        use_stableadamw=use_stableadamw,
        **kwargs,
    )
    safe_kwargs = _filter_kwargs_by_signature(ProdigyPlusScheduleFree, candidate)

    param_list = params if _is_param_groups(params) else list(params)

    logger.info(
        f"Creating ProdigyPlusScheduleFree "
        f"(d_coef={d_coef}, betas={tuple(betas)}, wd={weight_decay}, "
        f"eps={eps}, stableadamw={use_stableadamw})"
    )
    logger.info(f"[ProdigyPlus] Effective kwargs: {list(safe_kwargs.keys())}")

    optimizer = ProdigyPlusScheduleFree(param_list, **safe_kwargs)

    total = sum(p.numel() for g in optimizer.param_groups for p in g["params"] if p.requires_grad)
    logger.info(f"[ProdigyPlus] Trainable params: {total:,}")
    return optimizer


def create_soap(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.95, 0.95),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    shampoo_beta: float = -1.0,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precond_in_state: bool = True,
    **kwargs,
) -> Optimizer:
    """创建 SOAP 优化器（Vyas et al. 2024, arxiv 2409.11321）。

    SOAP = Adam 跑在 Shampoo 的 eigenbasis 里：用梯度协方差的特征基旋转梯度，
    在该基里做标准 Adam，再旋转回来。相比纯 Adam，对矩阵型参数（LoRA/LoKr 的
    低秩因子）拟合更快；相比 Shampoo，预条件刷新更省（precondition_frequency）。

    实现见 utils/soap_optimizer.py（MIT, Copyright (c) 2024 Nikhil Vyas）。

    Args:
        betas: (Adam β1, β2)，SOAP 论文默认 (0.95, 0.95)。
        shampoo_beta: Shampoo 协方差 EMA 衰减；< 0 时复用 β2。
        precondition_frequency: 每 N 步刷新一次特征基（越大越省算力、越旧）。
        max_precond_dim: 逐维阈值——某轴维度 ≤ 此值才建满秩预条件，> 此值该轴退化
            为 Adam。设大（如 10000）让大特征维也做二阶 = 提速来源；设小 = SOAP-lite。
        precond_in_state: False 时把可重算的 Shampoo 矩阵（GG/Q）剔出 state_dict，
            ckpt 更小，resume 时冷重建（从零训练不 resume 时零代价）。
    """
    from utils.soap_optimizer import SOAP

    param_list = params if _is_param_groups(params) else list(params)
    optimizer = SOAP(
        param_list,
        lr=lr,
        betas=tuple(betas),
        shampoo_beta=shampoo_beta,
        eps=eps,
        weight_decay=weight_decay,
        precondition_frequency=precondition_frequency,
        max_precond_dim=max_precond_dim,
        precond_in_state=precond_in_state,
        **kwargs,
    )
    logger.info(
        f"Creating SOAP optimizer (lr={lr}, betas={tuple(betas)}, wd={weight_decay}, "
        f"precond_freq={precondition_frequency}, max_precond_dim={max_precond_dim})"
    )
    return optimizer


def create_soap_sf(
    params: Iterator[nn.Parameter],
    lr: float,
    betas: tuple = (0.9, 0.95),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    shampoo_beta: float = -1.0,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precond_in_state: bool = True,
    weight_lr_power: float = 2.0,
    r: float = 0.0,
    warmup_steps: int = 0,
    **kwargs,
) -> Optimizer:
    """创建 Schedule-Free SOAP（SOAP 预条件 + Schedule-Free 轨迹）。

    在 SOAP 的 Adam-in-Shampoo-eigenbasis update 外面套 Schedule-Free 机制
    （Defazio et al. 2024, "The Road Less Scheduled", arxiv 2405.15682）：丢掉
    一阶动量 buffer，用 base 序列 z 与 Polyak-Ruppert 平均 x 的插值取代 LR 调度。
    因此**不需要 lr_scheduler**（调用方应强制 lr_scheduler=none 并启动期校验），
    并像其他 schedule-free 优化器一样暴露 train()/eval()：sample/save 前 eval()
    切到平均权重 x，事后 train() 切回 y（trainer 的 optimizer_eval_mode 已统一处理）。

    实现见 utils/soap_optimizer.py `SOAPScheduleFree`。

    Args:
        betas: (SF 插值权重 β1, 二阶矩衰减 β2)。SF 下 β1 不是动量而是 z↔x 插值系数。
        weight_lr_power / r / warmup_steps: Schedule-Free 专属——Polyak 权重里 lr
            的幂 / step index 的幂（0=均匀平均）/ 线性 lr warmup 步数。
        其余同 create_soap。

    注意：SF 的 Polyak 平均在极短训练（≤ ~100 步）严重滞后（x ≈ 轨迹质心 = 欠拟合），
    那种 regime 用纯 ``soap`` 而非 ``soap_sf``；千步级训练 SF 正常。
    """
    from utils.soap_optimizer import SOAPScheduleFree

    param_list = params if _is_param_groups(params) else list(params)
    optimizer = SOAPScheduleFree(
        param_list,
        lr=lr,
        betas=tuple(betas),
        shampoo_beta=shampoo_beta,
        eps=eps,
        weight_decay=weight_decay,
        precondition_frequency=precondition_frequency,
        max_precond_dim=max_precond_dim,
        precond_in_state=precond_in_state,
        weight_lr_power=weight_lr_power,
        r=r,
        warmup_steps=warmup_steps,
        **kwargs,
    )
    logger.info(
        f"Creating Schedule-Free SOAP optimizer (lr={lr}, betas={tuple(betas)}, "
        f"wd={weight_decay}, precond_freq={precondition_frequency}, "
        f"max_precond_dim={max_precond_dim}, weight_lr_power={weight_lr_power}, r={r})"
    )
    return optimizer


@contextmanager
def optimizer_eval_mode(optimizer: Optimizer):
    """切换 Schedule-Free 系优化器（PPSF 等）到 eval 模式的 context manager。

    Schedule-Free 优化器在内部维护两套权重：训练权重 y 和 averaged 权重 x。
    sample / validation / save 应该用 x（averaged），训练 step 用 y。
    PPSF 的 optimizer.eval() / optimizer.train() 通过 p.lerp_() in-place 切换参数张量
    指向哪一套；忘记切回 train() 会让训练继续用 averaged 权重，结果错乱。

    用法:
        with optimizer_eval_mode(optimizer):
            model.eval()
            img = sample_image(...)
            model.train()

    对非 Schedule-Free 优化器（AdamW / Prodigy 等）无 .train/.eval 方法 — 此 context
    manager 静默 no-op，所以调用方不需要分支判断 optimizer_type。
    """
    has_eval = hasattr(optimizer, "eval") and callable(getattr(optimizer, "eval"))
    has_train = hasattr(optimizer, "train") and callable(getattr(optimizer, "train"))
    if has_eval and has_train:
        optimizer.eval()
        try:
            yield
        finally:
            optimizer.train()
    else:
        yield


def create_optimizer_grouped_parameters(
    model: nn.Module,
    weight_decay: float,
    no_decay_modules: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    创建分组的优化器参数
    
    某些参数（如偏置和 LayerNorm 的权重）通常不应该应用
    权重衰减。这个函数将参数分为两组：
    1. 需要权重衰减的参数（权重矩阵）
    2. 不需要权重衰减的参数（偏置、LayerNorm）
    
    这是 Transformer 训练的最佳实践。
    
    Args:
        model: 模型
        weight_decay: 权重衰减系数
        no_decay_modules: 不应用权重衰减的模块名称列表
        
    Returns:
        List[Dict]: 分组后的参数列表
    """
    if no_decay_modules is None:
        # 默认：偏置和 LayerNorm 参数不应用权重衰减
        no_decay_modules = ["bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"]
    
    # 分组参数
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # 检查是否需要权重衰减
        needs_decay = True
        for no_decay_pattern in no_decay_modules:
            if no_decay_pattern in name:
                needs_decay = False
                break
        
        if needs_decay:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    
    # 构建参数组
    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]
    
    # 打印统计信息
    num_decay_params = sum(p.numel() for p in decay_params)
    num_no_decay_params = sum(p.numel() for p in no_decay_params)
    print(f"Parameter groups:")
    print(f"  With weight decay: {len(decay_params)} params, {num_decay_params:,} elements")
    print(f"  Without weight decay: {len(no_decay_params)} params, {num_no_decay_params:,} elements")
    
    return optimizer_grouped_parameters


def get_optimizer_info(optimizer: Optimizer) -> Dict[str, Any]:
    """
    获取优化器信息
    
    用于日志记录和调试
    
    Args:
        optimizer: 优化器实例
        
    Returns:
        Dict: 优化器信息字典
    """
    info = {
        "type": type(optimizer).__name__,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "num_param_groups": len(optimizer.param_groups),
    }
    
    # 获取总参数数
    total_params = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            total_params += p.numel()
    
    info["total_trainable_params"] = total_params

    # 添加优化器特定信息（duck typing：AdamW / Prodigy / PPSF / bnb AdamW8bit 都有这些字段）
    pg0 = optimizer.param_groups[0] if optimizer.param_groups else {}
    if "betas" in pg0:
        info["betas"] = pg0["betas"]
    if "weight_decay" in pg0:
        info["weight_decay"] = pg0["weight_decay"]
    if "eps" in pg0:
        info["eps"] = pg0["eps"]
    # PPSF 内部 d 估计 — 调试时有用
    if "d" in pg0:
        info["d"] = pg0["d"]
    if hasattr(optimizer, "get_avg_learning_rate") and callable(getattr(optimizer, "get_avg_learning_rate")):
        info["learning_rate"] = optimizer.get_avg_learning_rate()

    return info


def get_optimizer_monitor_metrics(optimizer: Optimizer) -> Dict[str, float]:
    """Return LR metrics suitable for train_monitor / logs.

    AdamW-style optimizers expose a real `lr` in the param group. Prodigy-style
    optimizers keep the UI-facing base lr at 1.0 and adapt the step size through
    `d`; PPSF v2 additionally exposes `effective_lr` for logging. This helper
    normalizes those shapes into one monitor point while preserving the raw
    ingredients for debugging.
    """
    if hasattr(optimizer, "get_avg_learning_rate") and callable(getattr(optimizer, "get_avg_learning_rate")):
        avg_lr = _as_float(optimizer.get_avg_learning_rate())
        if avg_lr is not None:
            return {"lr": avg_lr, "actual_lr": avg_lr}

    groups = list(getattr(optimizer, "param_groups", []) or [])
    if not groups:
        return {"lr": 0.0}

    base_lrs: list[float] = []
    d_values: list[float] = []
    effective_lrs: list[float] = []
    actual_lrs: list[float] = []

    for group in groups:
        base_lr = _as_float(group.get("lr"))
        if base_lr is not None:
            base_lrs.append(base_lr)

        d_source = group.get("d")
        if (
            group.get("split_groups")
            and group.get("split_groups_mean")
            and group.get("shared_d") is not None
        ):
            d_source = group.get("shared_d")
        d_value = _as_float(d_source)
        if d_value is None:
            continue
        d_values.append(d_value)

        # PPSF v2 recommends logging d * effective_lr. Older Prodigy/PPSF
        # versions do not expose it, so fall back to the base group lr.
        effective_lr = _as_float(group.get("effective_lr"))
        if effective_lr is None:
            effective_lr = base_lr
        if effective_lr is None:
            continue
        effective_lrs.append(effective_lr)
        actual_lrs.append(d_value * effective_lr)

    if not actual_lrs:
        return {"lr": base_lrs[0] if base_lrs else 0.0}

    metrics: Dict[str, float] = {
        "lr": _mean(actual_lrs),
        "actual_lr": _mean(actual_lrs),
        "base_lr": _mean(base_lrs),
        "d": _mean(d_values),
    }
    if effective_lrs:
        metrics["effective_lr"] = _mean(effective_lrs)
    if len(d_values) > 1:
        metrics["d_min"] = min(d_values)
        metrics["d_max"] = max(d_values)
    if len(actual_lrs) > 1:
        metrics["actual_lr_min"] = min(actual_lrs)
        metrics["actual_lr_max"] = max(actual_lrs)
    return metrics
