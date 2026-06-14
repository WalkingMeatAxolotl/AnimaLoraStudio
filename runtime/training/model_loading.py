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

def tread_route_indices(n_tokens: int, ratio: float, device) -> "torch.Tensor":
    """TREAD（arXiv 2501.04765）随机保留索引（batch 内共享一套）。

    返回升序 1-D 索引 (N_keep,)，N_keep = round(N·(1-ratio))（至少 1）。整个 batch
    共用同一套保留位置，因此 RoPE freqs 子集保持 (N_keep,1,1,Dh) 4-D，与
    apply_rotary_pos_emb 的形状契约一致（freqs 期望 [seq,1,1,d]，逐样本索引会让它
    变 5-D 触发维度不匹配）。升序保持原 token 顺序便于子集对齐；每步重新随机。
    """
    n_keep = max(int(round(n_tokens * (1.0 - float(ratio)))), 1)
    perm = torch.randperm(n_tokens, device=device)[:n_keep]
    return perm.sort().values


def forward_with_optional_checkpoint(
    model,
    latents,
    timesteps,
    cross,
    padding_mask,
    use_checkpoint=False,
    tread_ratio: float = 0.0,
    tread_start_layer: int = 0,
    tread_end_layer: int = 0,
):
    """带可选梯度检查点的前向传播 + 可选 TREAD token 路由。

    TREAD（arXiv 2501.04765，训练期专用 token 路由，推理不变）：当 ``tread_ratio>0``
    且 ``model.training`` 时，blocks[start:end)（负索引按 python 切片语义，end 开区间）
    只处理每步随机保留的 N·(1-ratio) 个 token（batch 内共享一套位置，保证 RoPE 子集
    4-D 契约），段尾把结果 scatter 回原位、被丢 token 恒等旁路 → 省这些 block 的算力。
    采样 / eval 调用方不传 tread 参数 + model.eval() 双保险关闭。

    实现不动模型定义：保留 token 子集 reshape 成 (B,1,1,N_keep,D) 伪网格，复用现有
    ``Block.forward``（其内部本就 ``rearrange("b t h w d -> b (t h w) d")`` 走 token 注意力，
    rope 按位置应用），对 Anima 图像 latent（T=1）与逐 token 前向逐 bit 等价。
    """
    use_tread = float(tread_ratio) > 0.0 and bool(getattr(model, "training", False))
    if not use_checkpoint and not use_tread:
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

    n_blocks = len(model.blocks)
    seg_s = seg_e = -1
    if use_tread:
        if extra_pos_emb is not None:
            raise RuntimeError(
                "TREAD 路由段不支持 extra_per_block_pos_emb（学习型逐块位置嵌入）；"
                "请关闭 tread 或使用 rope-only 位置编码。"
            )
        if x_B_T_H_W_D.shape[1] != 1:
            raise RuntimeError(
                f"TREAD 伪网格路径目前仅支持 T=1 图像 latent，收到 T={x_B_T_H_W_D.shape[1]}。"
            )
        seg_s = tread_start_layer if tread_start_layer >= 0 else n_blocks + tread_start_layer
        seg_e = tread_end_layer if tread_end_layer > 0 else n_blocks + tread_end_layer
        if not (0 <= seg_s < seg_e <= n_blocks):
            raise ValueError(
                f"非法 TREAD 路由段: blocks[{seg_s}:{seg_e}) / n_blocks={n_blocks}"
            )

    def _run_grid(blk, x):
        def fwd(x_in, _b=blk):
            return _b(x_in, t_embedding, cross, **block_kwargs)
        return checkpoint(fwd, x, use_reentrant=False) if use_checkpoint else fwd(x)

    def _run_routed(blk, x_pseudo, rope_keep):
        # 伪网格 (B,1,1,N_keep,D)：复用 Block.forward，rope 传保留子集、关学习型位置嵌入
        def fwd(x_in, _b=blk):
            return _b(
                x_in, t_embedding, cross,
                rope_emb_L_1_1_D=rope_keep,
                adaln_lora_B_T_3D=adaln_lora,
                extra_per_block_pos_emb=None,
            )
        return checkpoint(fwd, x_pseudo, use_reentrant=False) if use_checkpoint else fwd(x_pseudo)

    b = tt = hh = ww = dd = 0
    x_tok_full = keep_idx = grid_shape = x_keep = rope_keep = None
    for i, block in enumerate(model.blocks):
        if use_tread and i == seg_s:
            b, tt, hh, ww, dd = x_B_T_H_W_D.shape
            grid_shape = (b, tt, hh, ww, dd)
            n_tok = tt * hh * ww
            x_tok_full = x_B_T_H_W_D.reshape(b, n_tok, dd)
            keep_idx = tread_route_indices(n_tok, tread_ratio, x_tok_full.device)  # (N_keep,)
            x_keep = x_tok_full.index_select(1, keep_idx)  # (B, N_keep, D)
            if rope_emb is not None:
                if rope_emb.shape[0] != n_tok:
                    raise RuntimeError(
                        f"rope_emb 第 0 维 ({rope_emb.shape[0]}) != token 数 ({n_tok})，"
                        "TREAD 无法对齐 RoPE 子集"
                    )
                rope_keep = rope_emb.index_select(0, keep_idx)  # (N_keep, 1, 1, Dh) — 4-D，契约一致
        if use_tread and seg_s <= i < seg_e:
            n_keep = x_keep.shape[1]
            x_pseudo = x_keep.reshape(b, 1, 1, n_keep, dd)
            x_pseudo = _run_routed(block, x_pseudo, rope_keep)
            x_keep = x_pseudo.reshape(b, n_keep, dd)
            if i == seg_e - 1:
                # 段尾把处理过的保留 token 放回原位；丢弃 token 保持段首值（恒等旁路）
                x_tok_full = x_tok_full.index_copy(1, keep_idx, x_keep.to(x_tok_full.dtype))
                x_B_T_H_W_D = x_tok_full.reshape(grid_shape)
            continue
        x_B_T_H_W_D = _run_grid(block, x_B_T_H_W_D)

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
        "models.cosmos_predict2_modeling",
        "cosmos_predict2_modeling",
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
      1. 脚本同目录 `diffusion_models/` / `models/`（CLI 直接 cd 进 scripts/ 跑）
      2. 仓库根 `models/` / `diffusion_models/`（训练脚本在 runtime/ 下的 repo
         layout：repo_root/runtime/anima_train.py → ../models/anima_modeling.py）
      3. 环境变量 `DIFFUSION_PIPE_ROOT`（覆盖路径用）
    """
    # 注：本模块从 runtime/training/model_loading.py 调用时，__file__ 在 runtime/training/
    # 下，往上两级才是 repo_root。原 anima_train.py 在 runtime/，只往上一级。
    # 用 __file__.parent.parent.parent 保持等价语义。
    module_dir = Path(__file__).resolve().parent  # runtime/training
    runtime_dir = module_dir.parent                 # runtime
    repo_root = runtime_dir.parent                  # repo root
    candidates = [
        runtime_dir / "diffusion_models",
        runtime_dir / "models",
        repo_root / "models",
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
