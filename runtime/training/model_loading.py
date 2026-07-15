"""模型加载基础设施：前缀推断、safetensors 读取、路径解析、xformers / 梯度检查点。

抽自原 runtime/anima_train.py L370-612（ADR 0003 PR-A）。这里都是相对底层的 utils；
更上层的 load_anima_model / load_vae / load_text_encoders 在 training.models。

公开（被 sister script 用）：
- find_diffusion_pipe_root / resolve_path_best_effort / enable_xformers
- forward_with_optional_checkpoint（被 train loop 调）

内部：
- _strip_prefixes / _pick_best_prefix_remap — checkpoint key 前缀自动推断
- _load_safetensors_state_dict / _load_weights_best_effort — 容错加载
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


# ============================================================================
# 梯度检查点
# ============================================================================


# ============================================================================
# xformers 支持
# ============================================================================

def enable_xformers(model):
    """为模型启用 xformers memory efficient attention。"""
    try:
        from xformers.ops import memory_efficient_attention  # noqa: F401
    except ImportError:
        logger.warning("xformers 未安装，跳过启用")
        return False

    enabled_count = 0
    module_switches = 0
    module_names = {
        cls.__module__
        for cls in type(model).__mro__
        if getattr(cls, "__module__", None)
    }
    # exec-load 退役后模块身份唯一（多模型 PR-2a）：MRO 即可覆盖真实模块名
    # （modeling.anima.cosmos_predict2_modeling / anima_modeling），无需别名广播。
    for module_name in sorted(module_names):
        module = sys.modules.get(module_name)
        fn = getattr(module, "set_xformers_enabled", None) if module is not None else None
        if fn is None:
            continue
        try:
            if fn(True):
                module_switches += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("xformers 模组开关启用失败 (%s): %s", module_name, exc)

    for name, module in model.named_modules():
        # 查找 attention 模块并替换
        if hasattr(module, "set_use_memory_efficient_attention_xformers"):
            module.set_use_memory_efficient_attention_xformers(True)
            enabled_count += 1
        elif hasattr(module, "enable_xformers_memory_efficient_attention"):
            module.enable_xformers_memory_efficient_attention()
            enabled_count += 1

    if module_switches > 0 or enabled_count > 0:
        logger.info(
            "xformers 已启用: module_switches=%d, module_hooks=%d",
            module_switches,
            enabled_count,
        )
        return True

    logger.warning("xformers 已安装，但当前模型没有可启用的 xformers attention hook")
    return False


# ============================================================================
# 模型代码 / 路径定位
# ============================================================================

def find_diffusion_pipe_root():
    """[deprecated shim] 返回 Anima 模型代码目录（`modeling/anima/`）。

    模型代码随仓库发布并走正常 import（exec-load 与外部 diffusion-pipe checkout
    兼容已退役，多模型 PR-2a）。函数名与返回语义保留 —— 它是 sister 契约 7 名
    之一（docs/AGENTS.md §3.2「可加不可减不可改签名」），下游把返回值作为
    `load_anima_model(..., repo_root=)` 透传，该参数现被忽略。

    `DIFFUSION_PIPE_ROOT` 环境变量已不支持：检测到设置时打一次 warning，
    下个 release 删除此检测。
    """
    if os.environ.get("DIFFUSION_PIPE_ROOT"):
        logger.warning(
            "DIFFUSION_PIPE_ROOT 已不支持（模型代码随仓库发布，走正常 import），忽略该变量"
        )
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "modeling" / "anima"


# ============================================================================
# checkpoint key 前缀推断 + 容错加载
# ============================================================================

def _strip_prefixes(key: str, prefixes: list[str]) -> str:
    """反复剥离前缀（支持 module.model. 这种复合前缀）。"""
    if not prefixes:
        return key
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if key.startswith(p):
                key = key[len(p) :]
                changed = True
    return key


def _pick_best_prefix_remap(sd_keys: list[str], model_keys: set[str]) -> tuple[list[str], int]:
    """
    从常见前缀组合里选择"命中最多 model_keys"的 remap 方案。
    返回 (prefixes, matched_count)。
    """
    candidates: list[tuple[str, list[str]]] = [
        ("none", []),
        ("net.", ["net."]),
        ("model.", ["model."]),
        ("module.", ["module."]),
        ("module.+model.", ["module.", "model."]),
        ("module.model.", ["module.model."]),
        ("diffusion_model.", ["diffusion_model."]),
        ("model.diffusion_model.", ["model.diffusion_model."]),
        ("transformer.", ["transformer."]),
        ("vae.", ["vae."]),
        ("first_stage_model.", ["first_stage_model."]),
        ("net.+model.", ["net.", "model."]),
        ("net.model.", ["net.model."]),
    ]

    best_prefixes: list[str] = []
    best_matched = -1
    for _name, prefixes in candidates:
        matched = 0
        for k in sd_keys:
            kk = _strip_prefixes(k, prefixes)
            if kk in model_keys:
                matched += 1
        if matched > best_matched:
            best_matched = matched
            best_prefixes = prefixes
    return best_prefixes, best_matched


def _load_safetensors_state_dict(path: Path) -> dict:
    from safetensors import safe_open

    sd = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    return sd


def resolve_path_best_effort(path_str: str, bases: list[Path]) -> str:
    """
    将相对路径按多个 base 尝试解析到一个真实存在的路径。
    主要用于：无论从 repo 根目录还是 AnimaLoraToolkit 目录启动，都能找到 models/* 文件。
    """
    if not path_str:
        return path_str

    p = Path(path_str)
    if p.is_absolute():
        return str(p)

    # 先按原样（相对 cwd）试一下
    if p.exists():
        return str(p)

    # 逐 base 拼接尝试
    for b in bases:
        if not b:
            continue
        try:
            cand = (Path(b) / p).resolve()
        except Exception:
            cand = Path(b) / p
        if cand.exists():
            return str(cand)

    # 常见：配置写了 AnimaLoraToolkit/xxx，但启动目录已经在 AnimaLoraToolkit 下
    parts = p.parts
    if parts and parts[0].lower() in ("animaloratoolkit", "anima_trainer", "anima-trainer"):
        p2 = Path(*parts[1:])
        if p2.exists():
            return str(p2)
        for b in bases:
            if not b:
                continue
            cand = Path(b) / p2
            if cand.exists():
                return str(cand)

    return path_str


def _load_weights_best_effort(model: torch.nn.Module, sd: dict, label: str) -> dict:
    """
    更健壮的权重加载：
    - 自动尝试剥离常见前缀（model./module./...）
    - 打印匹配率、missing/unexpected
    - 关键模块未加载时直接报错（避免"采样全噪点"还继续训练）
    """
    model_keys = set(model.state_dict().keys())
    sd_keys = list(sd.keys())
    prefixes, matched = _pick_best_prefix_remap(sd_keys, model_keys)
    # Common path is no prefix remap. Reusing the original dict avoids building
    # another large key->tensor mapping while loading multi-GB checkpoints.
    remapped = sd if not prefixes else {_strip_prefixes(k, prefixes): v for k, v in sd.items()}

    incompatible = model.load_state_dict(remapped, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])

    matched_after = len(set(remapped.keys()) & model_keys)
    coverage = matched_after / max(1, len(model_keys))
    remap_name = "+".join(prefixes) if prefixes else "none"

    logger.info(
        f"{label} 权重加载: remap={remap_name}, 匹配 {matched_after}/{len(model_keys)} ({coverage:.1%}), "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

    # 关键层缺失会直接导致输出接近 0，采样就是纯噪点
    critical_prefixes = ("x_embedder.", "blocks.", "final_layer.")
    critical_missing = [k for k in missing if k.startswith(critical_prefixes)]
    if coverage < 0.60 or len(critical_missing) > 0:
        preview_missing = ", ".join(critical_missing[:8])
        raise RuntimeError(
            f"{label} 权重看起来没有正确加载（remap={remap_name}, coverage={coverage:.1%}）。"
            f"关键参数缺失: {preview_missing or 'N/A'}。\n"
            f"这通常表示你选错了 .safetensors（不是完整 transformer/vae 权重），或 checkpoint key 前缀不匹配。"
        )
    return {
        "remap": remap_name,
        "coverage": coverage,
        "missing": missing,
        "unexpected": unexpected,
    }


# 兼容 re-export（sister/loop 现有 import 面；多模型 PR-2b 移居 families/anima/forward.py）
from training.families.anima.forward import forward_with_optional_checkpoint  # noqa: E402,F401
