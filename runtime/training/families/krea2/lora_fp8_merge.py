"""Krea2 fp8 底模的 LoRA merge 回写（ComfyUI 逐位 parity）。

ComfyUI 从不在量化存储上算 LoRA：加载时把 LoRA merge 进权重
（model_patcher.patch_weight_to_device），采样期前向与无 LoRA 完全同路。
本模块逐位复刻该链路（oracle=本地 ComfyUI v0.27.1 + comfy_kitchen 0.2.16
eager 路径，docs/design/krea2-fp8-inference.md §1.5）：

    W16    = W_fp8.to(fp16) [* scale.to(fp16)]            # dequant 到 fp16
    W16   += ((strength * alpha/dim) * ΔW_compute).to(fp16) # 逐 LoRA，可选 fp32/bf16
    scale' = amax(|W16|).to(fp32) / 448（fp16 防下溢 clamp）
    W16   *= (1/scale').to(fp16)
    W_fp8' = stochastic_round(W16, seed=CRC32(key))        # ck eager SR

三种层形态（同一模型内混合，用户官方 Turbo fp8 文件 264 Linear 中 256 量化）：
- scaled fp8：dequant 乘 scale；回写 recalculate scale + SR
- 纯 cast fp8：dequant 无 scale；回写直接 SR（无 scale 概念）
- 非量化 Linear：cast fp16 merge；回写直接 cast 回原 dtype（comfy set_weight
  的 layout_type=None 分支，无 SR 无 seed）

数值口径依据（均本地 oracle 逐行核实）：
- dequant 直落 fp16 域：QuantizedTensor.to(fp16) 只改 orig_dtype 标记
  （ck tensor/base.py _handle_to），dequantize 即 qdata.to(fp16)*scale.to(fp16)，
  无 bf16 中转；fp16 = comfy lora_compute_dtype（should_use_fp16 现代卡）
- delta 域：默认按 comfy weight_adapter lokr/lora/loha 把因子 cast fp32 做
  mm/kron/Hadamard；可选 bf16 以降低 merge 峰值与计算量。两者最终都执行
  ``weight += ((strength * alpha) * diff).type(fp16)``，加法仍在 fp16 域
- requantize：quant_ops._TensorCoreFP8LayoutBase.quantize 的
  scale="recalculate" + stochastic_rounding 分支
- SR：comfy/float.py（Generator(device)+manual_seed → randint(0,256,uint8)）
  → ck eager stochastic_rounding_fp8/calc_mantissa。CUDA 与 CPU generator
  序列不同——逐位 parity 在 GPU 上成立（comfy 同在 GPU merge）；CPU 单测
  只验证确定性与公式性质
- seed key = "diffusion_model.{layer}.weight"（comfy ModelPatcher 模型键
  前缀；层名与 checkpoint 键一致——loader 直载 comfy 文件）

派生署名见 THIRD_PARTY_NOTICES（ComfyUI GPL-3.0 + comfy_kitchen）。
仅推理路径使用；merge 后权重 requires_grad=False 不变。
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from studio.infrastructure.log_messages import msg
# 与存储格式无关的 Comfy merge 语义在中立层（MOSS int8 共用）；名字在此重新导出
from training.families.comfy_lora_merge import (  # noqa: F401
    LORA_PREFIX as _LORA_PREFIX,
    QuantizedMergeAdapter,
    apply_lora_delta as _apply_lora_delta,
    group_lora_layers as _group_lora_layers,
    string_to_seed,
)
from training.families.krea2.quant_fp8 import _FP8_TORCH_DTYPES

#: 旧名（daemon / 测试按它识别 merge 句柄）
Fp8LoraMergeAdapter = QuantizedMergeAdapter


logger = logging.getLogger(__name__)

_CUDA_CACHE_TRIM_EVERY_LAYERS = 1
# PEFT/comfy 键后缀 → kohya/lycoris 命名（civitai 生态 krea2 LoRA 常见形态：
# ``diffusion_model.{点分层名}.lora_A/lora_B``，lora_A=down、lora_B=up，
# 通常无 alpha 键——comfy 对缺省 alpha 按缩放 1.0 处理，与 _apply_lora_delta
# 的 "alpha" 缺省分支一致）
_PEFT_SUFFIXES = (
    ("lora_A.weight", "lora_down.weight"),
    ("lora_B.weight", "lora_up.weight"),
    ("lora_down.weight", "lora_down.weight"),
    ("lora_up.weight", "lora_up.weight"),
    ("alpha", "alpha"),
    ("dora_scale", "dora_scale"),
)


def _calc_mantissa(
    abs_x: Tensor,
    exponent: Tensor,
    normal_mask: Tensor,
    mantissa_bits: int,
    exponent_bias: int,
    rng: Tensor,
) -> Tensor:
    # ck eager calc_mantissa（backends/eager/quantization.py:66-74）逐行复刻：
    # 全程 fp16 域，rng/256 提供 [0,1) 的随机进位量，floor 完成随机舍入
    mantissa_scaled = torch.where(
        normal_mask,
        (abs_x / (2.0 ** (exponent - exponent_bias)) - 1.0) * (2 ** mantissa_bits),
        (abs_x / (2.0 ** (-exponent_bias + 1 - mantissa_bits))),
    )
    mantissa_scaled += rng.to(dtype=mantissa_scaled.dtype) * (1.0 / 256.0)
    return mantissa_scaled.floor() / (2 ** mantissa_bits)


def stochastic_round_to_fp8(value: Tensor, fp8_dtype: torch.dtype, seed: int) -> Tensor:
    """comfy.float.stochastic_rounding fp8 分支 + ck eager SR 逐位复刻。"""
    if fp8_dtype == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif fp8_dtype == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15
    else:
        raise ValueError(f"stochastic_round_to_fp8 只支持 fp8 dtype：{fp8_dtype}")

    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    rng = torch.randint(
        0, 256, value.size(),
        dtype=torch.uint8, layout=value.layout, device=value.device,
        generator=generator,
    )

    x = value.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)

    exponent = torch.clamp(
        torch.floor(torch.log2(abs_x)) + exponent_bias,
        0,
        2 ** exponent_bits - 1,
    )
    normal_mask = ~(exponent == 0)

    abs_x[:] = _calc_mantissa(abs_x, exponent, normal_mask, mantissa_bits, exponent_bias, rng)

    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - exponent_bias)) * (1.0 + abs_x),
        (2.0 ** (-exponent_bias + 1)) * abs_x,
    )

    info = torch.finfo(fp8_dtype)
    torch.clamp(sign, min=info.min, max=info.max, out=sign)
    return sign.to(fp8_dtype)


def _requantize_scaled(w16: Tensor, fp8_dtype: torch.dtype, seed: int) -> tuple[Tensor, Tensor]:
    """quant_ops._TensorCoreFP8LayoutBase.quantize 的 recalculate+SR 分支。

    返回 (fp8 qdata, 新 F32 标量 scale)。
    """
    scale = torch.amax(w16.abs()).to(dtype=torch.float32) / torch.finfo(fp8_dtype).max
    # fp16 输入防 scale 过小（comfy 原注释 Prevent scale from being too small）
    if w16.dtype not in (torch.float32, torch.bfloat16):
        tensor_info = torch.finfo(w16.dtype)
        scale = 1.0 / torch.clamp(1.0 / scale, min=tensor_info.min, max=tensor_info.max)
    w16 = w16 * (1.0 / scale).to(w16.dtype)
    return stochastic_round_to_fp8(w16, fp8_dtype, seed), scale


def merge_loras_into_fp8_model(
    model: torch.nn.Module,
    sources: list[tuple[dict[str, Tensor], float, str]],
    *,
    compute_dtype: torch.dtype = torch.float32,
    chunk_rows: int | None = None,
    keep_backup: bool = True,
) -> Fp8LoraMergeAdapter:
    """把多份 LoRA 按 comfy merge 语义烘进（部分）fp8 模型的 Linear 权重。

    ``sources``：[(state_dict, strength, 来源名), ...]，顺序 = 挂载顺序 =
    comfy patches 顺序。``compute_dtype`` 只允许 fp32（ComfyUI 默认）或
    bf16（低峰值模式）。``chunk_rows`` 为正数时，普通 Linear LoRA 按输出
    行分块计算 dense delta；LoHa/LoKr 暂不分块。返回持有原始权重 CPU 备份
    的还原句柄。
    """
    if compute_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"LoRA merge compute_dtype 仅支持 fp32/bf16，收到：{compute_dtype}")
    if chunk_rows is not None and int(chunk_rows) < 0:
        raise ValueError(f"LoRA merge chunk_rows 不能为负数，收到：{chunk_rows}")
    normalized_chunk_rows = None if not chunk_rows else int(chunk_rows)
    module_index = {
        name.replace(".", "_"): (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }

    # 层 → [(tensors, strength, source), ...]，保持挂载顺序
    per_layer: dict[str, list[tuple[dict[str, Tensor], float, str]]] = {}
    missing: list[str] = []
    for sd, strength, source in sources:
        grouped = _group_lora_layers(sd)
        if not grouped:
            raise ValueError(f"LoRA 文件没有任何 {_LORA_PREFIX}* 层：{source}")
        for layer_key, tensors in grouped.items():
            if layer_key not in module_index:
                missing.append(f"{source}:{layer_key}")
                continue
            per_layer.setdefault(layer_key, []).append((tensors, strength, source))

    if missing and not per_layer:
        raise ValueError(f"LoRA 全部层都无法对应到当前模型：{missing[:5]} ...")
    if missing:
        logger.warning(
            "fp8 merge: %d LoRA layer(s) have no matching layer in the model and "
            "were skipped (%s%s) — those LoRA layers do not affect the output",
            len(missing), missing[:5], " ..." if len(missing) > 5 else "",
        )

    # 运算设备：block swap 下换出层的权重是 **CPU pinned 主副本**，而
    # weight_scale 恒在计算设备（swap 的前向需要，见 quant_fp8.patch_fp8_linears）。
    # 若就地按权重所在设备算，`w16 * scale` 会跨设备崩；退一步全在 CPU 上算又
    # 慢得离谱（264 层 6144² 的 fp32 delta）。故统一到 GPU 算，写回时靠
    # `weight.data.copy_()` 跨设备落回主副本 —— 主副本才是持久的那份，之后每次
    # 换入带的都是 merged 权重（docs/design/block-swap.md §9.6）。
    merge_device = next(
        (p.device for p in model.parameters() if p.device.type == "cuda"), None,
    )

    backup: dict[str, tuple[Tensor, Tensor | None]] = {}
    layer_count = len(per_layer)
    for layer_index, (layer_key, layer_sources) in enumerate(per_layer.items(), 1):
        name, module = module_index[layer_key]
        weight = module.weight
        scale = getattr(module, "weight_scale", None)
        is_fp8 = weight.dtype in _FP8_TORCH_DTYPES

        if keep_backup:
            # 注意这份备份与**整个模型等大**：krea2 fp8 实测 261 个 Linear 合计
            # 12.23GB（全模型 12.24GB），且整个会话常驻。开 block swap 时由
            # 调用方关掉（那批用户内存本就紧张），换 LoRA 改走重载兜底。
            backup[name] = (
                weight.detach().to("cpu", copy=True),
                None if scale is None else scale.detach().to("cpu", copy=True),
            )

        # dequant 到 fp16（comfy lora_compute_dtype 域，QuantizedTensor 无
        # bf16 中转）；非量化 Linear 同样 cast fp16 参与 merge
        w16 = weight.detach().to(torch.float16)
        if merge_device is not None and w16.device != merge_device:
            w16 = w16.to(merge_device)
        if scale is not None:
            w16 = w16 * scale.to(device=w16.device, dtype=torch.float16)

        for tensors, strength, source in layer_sources:
            w16 = _apply_lora_delta(
                w16, tensors, strength, name, source, compute_dtype,
                normalized_chunk_rows,
            )

        if is_fp8 and scale is not None:
            seed = string_to_seed(f"diffusion_model.{name}.weight")
            qdata, new_scale = _requantize_scaled(w16, weight.dtype, seed)
            weight.data.copy_(qdata)
            module.weight_scale.copy_(new_scale)
        elif is_fp8:
            # 纯 cast 形态：comfy 无 set_func → 直接 SR 回 fp8，无 scale 重算
            seed = string_to_seed(f"diffusion_model.{name}.weight")
            weight.data.copy_(stochastic_round_to_fp8(w16, weight.dtype, seed))
        else:
            # 非量化 Linear：comfy set_weight 的 layout_type=None 分支——
            # 直接 cast 回存储 dtype，无 SR
            weight.data.copy_(w16.to(weight.dtype))

        # Drop live per-layer temporaries before trimming.  Otherwise CUDA's
        # caching allocator accumulates dead fp16/qdata segments over all 264
        # Linear layers and raises the process high-water mark by several GiB.
        del w16
        if is_fp8 and scale is not None:
            del qdata, new_scale
        if (
            weight.device.type == "cuda"
            and torch.cuda.is_available()
            and (
                layer_index % _CUDA_CACHE_TRIM_EVERY_LAYERS == 0
                or layer_index == layer_count
            )
        ):
            torch.cuda.empty_cache()

    logger.info(msg(
        "train.fp8_merge_done", loras=len(sources), layers=len(per_layer),
    ))
    logger.debug(
        "fp8_merge: delta=%s chunk_rows=%s backup=%s",
        str(compute_dtype).removeprefix("torch."),
        normalized_chunk_rows or "off",
        "kept (detach can restore)" if keep_backup else "none (reload to swap LoRA)",
    )
    return QuantizedMergeAdapter(model, backup, can_restore=keep_backup)
