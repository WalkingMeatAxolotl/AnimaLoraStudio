"""
Manual partial / scheduled OrthoGrad for LoRA / LoKr training.

背景
====
OrthoGrad 来自 Prieto et al. 2025 (arXiv 2501.04697)。原意是在分类模型 grokking
场景下，去掉梯度中"沿当前权重方向、只放大权重 scale 而不改变预测"的 NLM 漂移。

ProdigyPlusScheduleFree 内置的 use_orthograd 实现把整个参数张量按 view(-1) 视为
一个向量，做：

    proj   = <w, g> / (<w, w> + 1e-30)
    g_orth = g - proj * w
    g_orth_scaled = g_orth * (||g|| / (||g_orth|| + 1e-30))

注意第三步：把 g_orth 的范数重新拉回到原始 ||g||，意味着每一步梯度量级不变，
但完全没有沿 w 方向的分量。

在 LoKr / LoRA 从近零初始化的低秩适配器上的副作用
=================================================
LoKr 的 forward 是 ΔW ≈ scaling · w1 ⊗ (w2_a · w2_b)，其中：
  * lokr_w1  initialized normal(0, 0.1)  ——  小但非零
  * lokr_w2_a initialized kaiming        ——  中等 scale
  * lokr_w2_b initialized zeros          ——  从零开始

整个适配器的"幅度"几乎全部来自 w2_b 从零向上的增长。OrthoGrad 把径向分量
精确地置零（且通过重缩放确保张量分量不漂移）：
  * w2_b 被永久锁定在初次增长的微小尺度 → ΔW 幅度被压制，模型只能调整
    "调谁"，无法调整"调多少"——表现为全局结构特征拟合下降。

修复策略
========
1. 参数类排除：lora_B / lokr_w2_b / lokr_w1 永不应用 OrthoGrad
2. 模块级排除：cross_attn / output_proj / mlp.layer2 等可选额外排除
3. 延迟启用：enable_after_step 之前不投影，让前期结构学习走全梯度
4. 强度混合：strength ∈ (0,1] 时混合原梯度，ramp_steps 线性 ramp

使用方法
========
1) yaml 加 orthograd_mode: "manual"；同时把 optimizer_args.use_orthograd: false
2) 训练循环里在 optimizer.step() 前调用 apply_partial_orthograd_()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# 默认排除：从零或小尺度初始化的"幅度承载"参数。
DEFAULT_EXCLUDE_PARAM_KEYWORDS: Tuple[str, ...] = (
    "lokr_w1",
    "lokr_w2_b",
    "lora_B",
)

DEFAULT_EXCLUDE_MODULE_KEYWORDS: Tuple[str, ...] = ()


@dataclass
class OrthoGradConfig:
    enable: bool = False
    enable_after_step: int = 0
    ramp_steps: int = 0
    strength: float = 1.0
    rescale_to_original_norm: bool = True
    exclude_param_keywords: Tuple[str, ...] = DEFAULT_EXCLUDE_PARAM_KEYWORDS
    exclude_module_keywords: Tuple[str, ...] = DEFAULT_EXCLUDE_MODULE_KEYWORDS

    _logged_excluded: bool = field(default=False, repr=False)
    _logged_first_apply: bool = field(default=False, repr=False)


def _normalize_keywords(value) -> Tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (value,) if value else tuple()
    return tuple(str(v) for v in value if str(v))


def build_orthograd_config(args) -> OrthoGradConfig:
    """从 argparse Namespace / dict-like 构建配置。

    yaml 键：
      orthograd_mode: "off" | "manual"
      orthograd_enable_after: int      (0 = 全程)
      orthograd_ramp_steps: int
      orthograd_strength: float        (0..1)
      orthograd_rescale: bool
      orthograd_exclude_param_keywords: list
      orthograd_exclude_module_keywords: list
    """
    mode = str(getattr(args, "orthograd_mode", "off") or "off").lower()
    if mode != "manual":
        return OrthoGradConfig(enable=False)

    excl_param_raw = getattr(args, "orthograd_exclude_param_keywords", None)
    excl_param = (
        _normalize_keywords(excl_param_raw)
        if excl_param_raw is not None
        else DEFAULT_EXCLUDE_PARAM_KEYWORDS
    )
    excl_module = _normalize_keywords(
        getattr(args, "orthograd_exclude_module_keywords", None) or []
    )

    cfg = OrthoGradConfig(
        enable=True,
        enable_after_step=int(getattr(args, "orthograd_enable_after", 0) or 0),
        ramp_steps=max(int(getattr(args, "orthograd_ramp_steps", 0) or 0), 0),
        strength=float(getattr(args, "orthograd_strength", 1.0) or 1.0),
        rescale_to_original_norm=bool(getattr(args, "orthograd_rescale", True)),
        exclude_param_keywords=excl_param,
        exclude_module_keywords=excl_module,
    )
    cfg.strength = max(0.0, min(1.0, cfg.strength))
    logger.info(
        "[orthograd] manual mode ON  enable_after=%d ramp=%d strength=%.3f rescale=%s "
        "exclude_param=%s exclude_module=%s",
        cfg.enable_after_step, cfg.ramp_steps, cfg.strength, cfg.rescale_to_original_norm,
        cfg.exclude_param_keywords, cfg.exclude_module_keywords,
    )
    return cfg


def _should_apply(name: str, cfg: OrthoGradConfig) -> bool:
    for kw in cfg.exclude_param_keywords:
        if kw in name:
            return False
    for kw in cfg.exclude_module_keywords:
        if kw in name:
            return False
    return True


def _current_strength(step: int, cfg: OrthoGradConfig) -> float:
    if step < cfg.enable_after_step:
        return 0.0
    if cfg.ramp_steps <= 0:
        return cfg.strength
    progress = (step - cfg.enable_after_step) / float(cfg.ramp_steps)
    return cfg.strength * max(0.0, min(1.0, progress))


@torch.no_grad()
def apply_partial_orthograd_(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]],
    step: int,
    cfg: OrthoGradConfig,
    eps: float = 1e-30,
) -> None:
    """In-place 修改 p.grad，对未排除且已启用的参数应用 OrthoGrad。

    Args:
        named_params: (full_name, parameter) 可迭代序列
        step:         当前 optimizer step（非 micro-batch）
        cfg:          由 build_orthograd_config 构造
        eps:          数值稳定项
    """
    if not cfg.enable:
        return
    s = _current_strength(step, cfg)
    if s <= 0.0:
        return

    excluded_for_log: List[str] = []
    applied_count = 0

    for name, p in named_params:
        if p.grad is None:
            continue
        if not _should_apply(name, cfg):
            if not cfg._logged_excluded:
                excluded_for_log.append(name)
            continue

        g = p.grad
        w = p.data
        w_flat = w.view(-1)
        g_flat = g.view(-1)

        w_norm_sq = torch.dot(w_flat, w_flat)
        if float(w_norm_sq) <= eps:
            # 权重还几乎在零附近（如 lokr_w2_b 第一步），跳过
            continue

        proj_coef = torch.dot(w_flat, g_flat) / (w_norm_sq + eps)
        g_orth_flat = g_flat - proj_coef * w_flat

        if cfg.rescale_to_original_norm:
            g_norm = g_flat.norm(2)
            g_orth_norm = g_orth_flat.norm(2)
            g_orth_flat = g_orth_flat * (g_norm / (g_orth_norm + eps))

        new_grad_flat = s * g_orth_flat + (1.0 - s) * g_flat if s < 1.0 else g_orth_flat
        p.grad.copy_(new_grad_flat.view_as(g))
        applied_count += 1

    if not cfg._logged_excluded and excluded_for_log:
        sample = ", ".join(excluded_for_log[:6])
        more = "" if len(excluded_for_log) <= 6 else f" 等 {len(excluded_for_log)} 个"
        logger.info("[orthograd] excluded params (sample): %s%s", sample, more)
        cfg._logged_excluded = True

    if not cfg._logged_first_apply and applied_count > 0:
        logger.info(
            "[orthograd] first applied at step=%d strength=%.3f, applied to %d params",
            step, s, applied_count,
        )
        cfg._logged_first_apply = True


def warn_double_orthograd(args) -> None:
    """manual 模式下若 ProdigyPlus 内置 use_orthograd 也开着，给出警告。"""
    mode = str(getattr(args, "orthograd_mode", "off") or "off").lower()
    if mode != "manual":
        return
    opt_args = getattr(args, "optimizer_args", None) or {}
    if isinstance(opt_args, dict) and bool(opt_args.get("use_orthograd", False)):
        logger.warning(
            "[orthograd] orthograd_mode=manual 同时 optimizer_args.use_orthograd=true："
            "OrthoGrad 会被应用两次，训练几乎不动。建议把 optimizer_args.use_orthograd 改为 false。"
        )
