"""
Optimizer Utils Module - 优化器创建
===================================
支持多种优化器：
1. 标准 AdamW - PyTorch 内置
2. 8-bit AdamW (bitsandbytes) - 内存高效
3. Prodigy (prodigyopt) - 无需调 lr 的自适应优化器
4. ProdigyPlusScheduleFree (prodigy-plus-schedule-free) - Schedule-Free + Prodigy，
   解决 Prodigy 在扩散 LoRA 训练中的 mutation ep / 风格突变问题。
5. Automagic - Per-parameter adaptive lr via sign-agreement tracking
   原作者: Ostris (https://github.com/ostris/ai-toolkit, MIT license, Copyright (c)
   2024 Ostris, LLC). bf16 Kahan summation path 借鉴自 tdrussell/diffusion-pipe.
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

    elif optimizer_type == "prodigy_plus_schedulefree":
        return create_prodigy_plus_schedulefree(
            params=params,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
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

    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Choose from: adamw, adamw8bit, automagic, automagic_v2, prodigy, prodigy_plus_schedulefree"
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


# ============================================================================
# Automagic optimizer + Auto8bitTensor helper
#
# Adapted from ostris/ai-toolkit (https://github.com/ostris/ai-toolkit), which
# also flowed through tdrussell/diffusion-pipe (added Kahan summation for
# bfloat16). The core sign-agreement schedule, 8-bit lr_mask, Adafactor-style
# factored 2nd moment, and stochastic-rounding helpers below are Ostris's
# design.
#
# MIT License — Copyright (c) 2024 Ostris, LLC
# ============================================================================


class Auto8bitTensor:
    def __init__(self, data: "torch.Tensor | dict[str, Any]") -> None:
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

    def state_dict(self) -> "dict[str, Any]":
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


# --- PLACEHOLDER_AUTOMAGIC_CLASSES ---


class Automagic(Optimizer):
    """Per-element adaptive LR optimizer via sign-agreement tracking (v1).

    Uses Adafactor-style factored second moment + per-element lr_mask stored
    as int8 (Auto8bitTensor). bf16 training uses Kahan compensated summation
    with fp32 shift buffer for lossless accumulation.
    """

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
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")

        self.lr = min(lr, max_lr)
        if lr > 1e-3:
            logger.warning("Automagic start lr %s is high; clamping to 1e-6", lr)
            self.lr = 1e-6
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump
        super().__init__(
            params,
            dict(lr=self.lr, eps=eps, clip_threshold=clip_threshold,
                 beta2=beta2, weight_decay=weight_decay),
        )
        self.base_lrs = [self.lr for _ in self.param_groups]

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        r = (row / row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c = col.unsqueeze(-2).rsqrt()
        return torch.mul(r, c)

    @staticmethod
    def _get_lr(param_state: "dict[str, Any]") -> float:
        avg_lr = param_state.get("avg_lr")
        if avg_lr is None:
            return 0.0
        if isinstance(avg_lr, torch.Tensor):
            return float(avg_lr.detach().float().item())
        return float(avg_lr)

    def _get_group_lr(self, group: "dict[str, Any]") -> float:
        lrs = [self._get_lr(self.state[p]) for p in group["params"] if p in self.state]
        return sum(lrs) / len(lrs) if lrs else self.lr

    def get_learning_rates(self) -> "list[float]":
        lrs = [self._get_group_lr(g) for g in self.param_groups]
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
        # Kahan compensated summation — fp32 shift tracks bf16 rounding error
        if p.dtype == torch.bfloat16 and "shift" not in state:
            state["shift"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)

    # --- PLACEHOLDER_AUTOMAGIC_STEP ---

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

                p_data_fp32 = p
                if p.dtype != torch.float32:
                    p_data_fp32 = p.clone().float()

                state["step"] = state.get("step", 0) + 1
                state["RMS"] = self._rms(p_data_fp32)
                beta2 = group["beta2"]
                eps = group["eps"]
                update = (grad ** 2) + eps
                if factored:
                    row = state["exp_avg_sq_row"]
                    col = state["exp_avg_sq_col"]
                    row.mul_(beta2).add_(update.mean(dim=-1), alpha=(1.0 - beta2))
                    col.mul_(beta2).add_(update.mean(dim=-2), alpha=(1.0 - beta2))
                    update = self._approx_sq_grad(row, col)
                    update.mul_(grad)
                else:
                    v = state["exp_avg_sq"]
                    v.mul_(beta2).add_(update, alpha=(1.0 - beta2))
                    update = v.rsqrt().mul_(grad)

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
                    wd_update = p_data_fp32 * (-group["weight_decay"]) * new_lr
                else:
                    wd_update = None

                if p.dtype == torch.bfloat16:
                    # Kahan compensated summation with fp32 shift buffer
                    delta = -update
                    if wd_update is not None:
                        delta.add_(wd_update)
                    shift = state.setdefault(
                        "shift", torch.zeros(p.shape, dtype=torch.float32, device=p.device)
                    )
                    shift.add_(delta)
                    old_p_fp32 = p.float()
                    new_p_fp32 = old_p_fp32 + shift
                    p.copy_(new_p_fp32.to(torch.bfloat16))
                    # shift carries forward the rounding error
                    shift.copy_(new_p_fp32 - p.float())
                else:
                    if wd_update is not None:
                        p_data_fp32.add_(wd_update)
                    p_data_fp32.add_(-update)
                    if p.dtype != torch.float32:
                        _copy_stochastic(p, p_data_fp32)

        return loss

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd["state"] = {
            pid: {
                **{k: v for k, v in st.items() if k != "lr_mask"},
                **({"lr_mask": st["lr_mask"].state_dict()} if "lr_mask" in st else {}),
            }
            for pid, st in sd["state"].items()
        }
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):
        converted = {"state": {}, "param_groups": state_dict.get("param_groups", [])}
        for pid, st in state_dict.get("state", {}).items():
            cs = dict(st)
            if isinstance(cs.get("lr_mask"), dict):
                cs["lr_mask"] = Auto8bitTensor(cs["lr_mask"])
            converted["state"][pid] = cs
        return super().load_state_dict(converted)


# --- PLACEHOLDER_AUTOMAGIC2 ---


class Automagic2(Optimizer):
    """Automagic v2 — per-parameter scalar LR, fused into backward pass.

    Instead of a per-element lr_mask, keeps one scalar lr per parameter tensor.
    The optimizer step is fused into backward via register_post_accumulate_grad_hook:
    each parameter is updated and its grad freed as soon as autograd finishes
    accumulating into it. .step() is therefore a no-op and peak VRAM stays low.

    Incompatible with GradScaler and clip_grad_norm_ (params update during backward).
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
        if p.dim() >= 2:
            state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1], dtype=p.dtype, device=p.device)
            state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + p.shape[-1:], dtype=p.dtype, device=p.device)
        else:
            state["exp_avg_sq"] = torch.zeros(p.shape, dtype=p.dtype, device=p.device)

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
                if st is not None and isinstance(st.get("lr"), torch.Tensor):
                    st["lr"] = st["lr"].to(torch.float32)


# --- PLACEHOLDER_FACTORIES ---


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
    if lr > 1e-5:
        logger.warning(
            "Automagic 初始 lr=%.2e 远高于推荐 1e-6；sign-agreement 自适应从过高起点"
            "收敛慢，建议设为 1e-6（per-param lr 由 [min_lr, max_lr] 自动调）", lr,
        )
    param_list = params if _is_param_groups(params) else list(params)
    optimizer = Automagic(
        param_list, lr=lr, min_lr=min_lr, max_lr=max_lr, lr_bump=lr_bump,
        eps=eps, clip_threshold=clip_threshold, beta2=beta2, weight_decay=weight_decay,
    )
    logger.info(
        f"Creating Automagic v1 (lr={lr}, min_lr={min_lr}, max_lr={max_lr}, "
        f"lr_bump={lr_bump}, beta2={beta2}, wd={weight_decay})"
    )
    return optimizer


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
