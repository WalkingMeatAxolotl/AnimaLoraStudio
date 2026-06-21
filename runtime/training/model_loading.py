"""模型加载基础设施：前缀推断、safetensors 读取、路径解析、xformers / 梯度检查点。

抽自原 runtime/anima_train.py L370-612（ADR 0003 PR-A）。这里都是相对底层的 utils；
更上层的 load_anima_model / load_vae / load_text_encoders 在 training.models。

公开（被 sister script 用）：
- find_diffusion_pipe_root / resolve_path_best_effort / enable_xformers
- forward_with_optional_checkpoint（被 train loop 调）

内部：
- _strip_prefixes / _pick_best_prefix_remap — checkpoint key 前缀自动推断
- _load_safetensors_state_dict / _load_weights_best_effort — 容错加载
- load_module_from_path — 动态加载 anima_modeling.py
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

def forward_with_optional_checkpoint(model, latents, timesteps, cross, padding_mask, use_checkpoint=False):
    """带可选梯度检查点的前向传播。"""
    if not use_checkpoint:
        return model(latents, timesteps, cross, padding_mask=padding_mask)
    from torch.utils.checkpoint import checkpoint

    x_B_T_H_W_D, rope_emb, extra_pos_emb = model.prepare_embedded_sequence(
        latents, fps=None, padding_mask=padding_mask,
    )
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    t_embedding, adaln_lora = model.t_embedder(timesteps)
    t_embedding = model.t_embedding_norm(t_embedding)

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb,
        "adaln_lora_B_T_3D": adaln_lora,
        "extra_per_block_pos_emb": extra_pos_emb,
    }

    for block in model.blocks:
        def custom_forward(x, blk=block):
            return blk(x, t_embedding, cross, **block_kwargs)
        x_B_T_H_W_D = checkpoint(custom_forward, x_B_T_H_W_D, use_reentrant=False)

    x_B_T_H_W_O = model.final_layer(x_B_T_H_W_D, t_embedding, adaln_lora_B_T_3D=adaln_lora)
    return model.unpatchify(x_B_T_H_W_O)


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
    module_names.update({
        "modeling.cosmos_predict2_modeling",
        "models.cosmos_predict2_modeling",  # 兼容外部 diffusion-pipe checkout
        "cosmos_predict2_modeling",
        "modeling.anima_modeling",
        "models.anima_modeling",
        "anima_modeling",
    })
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
    """查找 diffusion-pipe 模型代码路径。

    候选顺序（首个命中即返回）：
      1. 仓库根 `modeling/`（本仓库自带的模型架构代码，现位置）
      2. 脚本同目录 `diffusion_models/` / `models/`（CLI 直接 cd 进 scripts/ 跑）
      3. 仓库根 `models/` / `diffusion_models/`（兼容旧布局 / 外部 diffusion-pipe
         checkout 把代码放 models/ 的情况）
      4. 环境变量 `DIFFUSION_PIPE_ROOT`（覆盖路径用）
    """
    # 注：本模块从 runtime/training/model_loading.py 调用时，__file__ 在 runtime/training/
    # 下，往上两级才是 repo_root。原 anima_train.py 在 runtime/，只往上一级。
    # 用 __file__.parent.parent.parent 保持等价语义。
    module_dir = Path(__file__).resolve().parent  # runtime/training
    runtime_dir = module_dir.parent                 # runtime
    repo_root = runtime_dir.parent                  # repo root
    candidates = [
        repo_root / "modeling",          # 本仓库自带的模型架构代码（现位置）
        runtime_dir / "diffusion_models",
        runtime_dir / "models",
        repo_root / "models",            # 兼容旧布局 / 外部 diffusion-pipe checkout
        repo_root / "diffusion_models",
        Path(os.environ.get("DIFFUSION_PIPE_ROOT", "")) if os.environ.get("DIFFUSION_PIPE_ROOT") else None,
    ]
    for candidate in candidates:
        if candidate and (candidate / "anima_modeling.py").exists():
            return candidate
        if candidate and (candidate / "models" / "anima_modeling.py").exists():
            return candidate / "models"
    raise RuntimeError("找不到 anima_modeling.py，请设置 DIFFUSION_PIPE_ROOT 或放置模型代码")


def load_module_from_path(module_name, file_path):
    """动态加载 Python 模块。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
