"""
Optimizer Utils Module - 优化器创建
===================================
支持多种优化器：
1. 标准 AdamW - PyTorch 内置
2. 8-bit AdamW (bitsandbytes) - 内存高效
3. Prodigy (prodigyopt) - 无需调 lr 的自适应优化器
4. ProdigyPlusScheduleFree (prodigy-plus-schedule-free) - Schedule-Free + Prodigy，
   解决 Prodigy 在扩散 LoRA 训练中的 mutation ep / 风格突变问题。
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

    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Choose from: adamw, adamw8bit, prodigy, prodigy_plus_schedulefree"
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
