"""MOSS int8 convrot 底模的 LoRA merge 回写（ComfyUI 逐位 parity，MOSS 设计稿 §5.9 / M9）。

ComfyUI 给 int8 底模挂 LoRA 不在推理时旁路计算，而是加载时 merge 进权重再重新量化
（经典 ModelPatcher 路径 ``patch_weight_to_device``，作者本地 ComfyUI 实际走这条：
highvram / torch<2.8 不启用 dynamic VRAM）。本模块逐位复刻（oracle = 作者本地
ComfyUI 0.41.0 + comfy_kitchen 0.2.38 eager 后端）：

    W16  = dequant(q, s).to(fp16)                       # fp32 里乘 scale、按组乘 Hᵀ
    W16 += ((strength * alpha/dim) * ΔW_fp32).to(fp16)  # 逐 LoRA，comfy calculate_weight
    q', s' = requantize_convrot(W16, seed)              # fp16 里旋转、逐行 amax/127、随机舍入

- fp16 = comfy ``lora_compute_dtype``（现代卡 should_use_fp16）。
- 种子 = ``string_to_seed("diffusion_model.{comfy 键}.weight")``。Comfy-Org 的 MOSS
  文件 qkv 融合存储，Comfy 的键是 ``layers.N.proj.qkv``：q/k/v 三个 LoRA 落在融合
  权重的行切片上（``narrow(0, i*dim, dim)``），随机舍入的随机数按融合形状一次生成。
  所以这里把 q/k/v 拼回整块再 merge、重新量化，最后拆回本项目的 q/k/v 三个 Linear。
- CUDA 与 CPU 的随机数序列不同，逐位 parity 只在 GPU 上成立（与 Krea2 fp8 相同）。

仅推理路径使用（训练走 W8A16 + LoRA 旁路，见 quant_int8）。派生署名见 THIRD_PARTY_NOTICES。
"""

from __future__ import annotations

import logging
import re

import torch
from torch import Tensor

from studio.infrastructure.log_messages import msg
from training.families.comfy_lora_merge import (
    LORA_PREFIX,
    QuantizedMergeAdapter,
    apply_lora_delta,
    group_lora_layers,
    string_to_seed,
)
from training.families.moss.quant_int8 import dequantize, requantize_convrot

logger = logging.getLogger(__name__)

#: ComfyUI lora_compute_dtype（现代 NVIDIA 卡）
_MERGE_DTYPE = torch.float16
_QKV_RE = re.compile(r"^(layers\.\d+\.proj)\.([qkv])$")


def _comfy_units(model: torch.nn.Module, layer_names: set[str]) -> dict[str, list[str]]:
    """被 LoRA 命中的 Linear → ComfyUI 的权重键（merge / 种子单位）及其成员（按行序）。"""
    fused = bool(getattr(model, "moss_fused_qkv", False))
    units: dict[str, list[str]] = {}
    for name in sorted(layer_names):
        m = _QKV_RE.match(name)
        if fused and m:
            units[f"{m.group(1)}.qkv"] = [f"{m.group(1)}.{p}" for p in ("q", "k", "v")]
        else:
            units[name] = [name]
    return units


def merge_loras_into_int8_model(
    model: torch.nn.Module,
    sources: list[tuple[dict[str, Tensor], float, str]],
    *,
    compute_dtype: torch.dtype = torch.float32,
    chunk_rows: int | None = None,
    keep_backup: bool = True,
) -> QuantizedMergeAdapter:
    """把多份 LoRA 按 ComfyUI 语义 merge 进 MOSS int8 底模并重新量化回写。

    ``sources``：[(state_dict, strength, 来源名), ...]，顺序 = 挂载顺序 = comfy patches
    顺序。``compute_dtype``：ΔW 的计算精度，fp32 = ComfyUI 默认（``lora_merge_precision``
    设置项可选 bf16 降低峰值）。返回持有原始 int8 权重与 scale 的 CPU 备份的还原句柄。
    """
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"LoRA merge compute_dtype 仅支持 fp32/bf16，收到：{compute_dtype}")
    module_index = {
        name.replace(".", "_"): (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    per_layer: dict[str, list[tuple[dict[str, Tensor], float, str]]] = {}
    missing: list[str] = []
    for sd, strength, source in sources:
        grouped = group_lora_layers(sd)
        if not grouped:
            raise ValueError(f"LoRA 文件没有任何 {LORA_PREFIX}* 层：{source}")
        for layer_key, tensors in grouped.items():
            entry = module_index.get(layer_key)
            if entry is None:
                missing.append(f"{source}:{layer_key}")
                continue
            name, module = entry
            if module.weight.dtype != torch.int8:
                raise ValueError(
                    f"int8 底模 merge 遇到非量化层 {name}（{module.weight.dtype}）：MOSS 的 LoRA "
                    f"目标层应全部是 int8，请确认底模文件完整"
                )
            per_layer.setdefault(name, []).append((tensors, strength, source))

    if missing and not per_layer:
        raise ValueError(f"LoRA 全部层都无法对应到当前模型：{missing[:5]} ...")
    if missing:
        logger.warning(
            "int8 merge: %d LoRA layer(s) have no matching layer in the model and "
            "were skipped (%s%s) — those LoRA layers do not affect the output",
            len(missing), missing[:5], " ..." if len(missing) > 5 else "",
        )

    modules = dict(model.named_modules())
    # 运算设备：block swap 下换出层的权重是 CPU pinned 主副本，weight_scale 恒在计算
    # 设备。统一在 GPU 上算，写回时 copy_() 跨设备落回主副本（同 Krea2 fp8 merge）
    merge_device = next(
        (p.device for p in model.parameters() if p.device.type == "cuda"), None,
    )
    # 反量化的 fp32 矩阵乘与 ComfyUI 一样不用 TF32（torch 默认即关闭；此处兜底）
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    backup: dict[str, tuple[Tensor, Tensor | None]] = {}
    units = _comfy_units(model, set(per_layer))
    try:
        for unit, members in units.items():
            lins = [modules[n] for n in members]
            groups = {int(m.convrot_groupsize) for m in lins}
            if len(groups) != 1:
                raise ValueError(f"int8 merge：{unit} 的成员 convrot 组大小不一致 {sorted(groups)}")
            group = groups.pop()
            if keep_backup:
                for n, m in zip(members, lins):
                    backup[n] = (
                        m.weight.detach().to("cpu", copy=True),
                        m.weight_scale.detach().to("cpu", copy=True),
                    )
            device = merge_device or lins[0].weight.device
            q = torch.cat([m.weight.detach().to(device) for m in lins], dim=0)
            s = torch.cat([m.weight_scale.detach().to(device).reshape(-1, 1) for m in lins], dim=0)
            w16 = dequantize(q, s, group).to(_MERGE_DTYPE)
            del q, s
            row = 0
            for n, m in zip(members, lins):
                rows = int(m.out_features)
                for tensors, strength, source in per_layer.get(n, ()):
                    apply_lora_delta(
                        w16[row:row + rows], tensors, strength, n, source, compute_dtype, chunk_rows,
                    )
                row += rows
            seed = string_to_seed(f"diffusion_model.{unit}.weight")
            q_new, s_new = requantize_convrot(w16, group, seed=seed)
            del w16
            row = 0
            for m in lins:
                rows = int(m.out_features)
                m.weight.data.copy_(q_new[row:row + rows])
                m.weight_scale.copy_(s_new[row:row + rows].reshape(m.weight_scale.shape))
                row += rows
            del q_new, s_new
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32

    logger.info(msg("train.int8_merge_done", loras=len(sources), layers=len(per_layer)))
    logger.debug(
        "int8_merge: units=%d delta=%s backup=%s", len(units),
        str(compute_dtype).removeprefix("torch."),
        "kept (detach can restore)" if keep_backup else "none (reload to swap LoRA)",
    )
    return QuantizedMergeAdapter(model, backup, can_restore=keep_backup)
