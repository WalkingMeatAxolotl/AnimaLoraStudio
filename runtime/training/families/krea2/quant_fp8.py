"""Krea2 推理侧 fp8 权重支持（ComfyUI 逐位 parity）。

两种社区/官方 fp8 形态（tmp/quant-inference-design.md）：
- **fp8_scaled**：权重 F8_E4M3 + per-layer F32 标量 ``{layer}.weight_scale``，
  safetensors ``__metadata__._quantization_metadata`` 声明每层格式（Comfy-Org
  官方 Turbo fp8 即此形态）
- **纯 fp8 cast**：权重直接以 fp8 存储，无 scale 无 metadata

数值口径逐位复刻本地 ComfyUI oracle（v0.27.1 + torch2.7/cu128 环境下的
eager 路径，设计文档 §1）：

    W_compute = W_fp8.to(input.dtype) [* scale.to(input.dtype)]
    y = F.linear(x, W_compute, bias)

即 comfy_kitchen ``dequantize_per_tensor_fp8``（backends/eager/quantization.py:59-63，
scale **先转 compute dtype 再乘**）+ comfy ``cast_bias_weight``（目标 dtype =
input.dtype）。不做原生 ``_scaled_mm``（Comfy 默认 ``--fast`` 为空同样不走）。
``full_precision_matrix_mult`` 层标记只影响 Comfy 的原生 matmul 选择，对
dequant 路径无差别——解析后忽略。

前向 patch 思路来自 kohya-ss/musubi-tuner 的 ``apply_fp8_monkey_patch``
（Apache-2.0，见 THIRD_PARTY_NOTICES）；dequant 公式对齐 Comfy（GPL-3.0）。

patch 后权重 requires_grad=False；训练（fp8_base）与推理共用本 patch，
底模恒 frozen，梯度只流经 LoRA 参数。
"""

from __future__ import annotations

import json
import logging
from types import MethodType

import torch


logger = logging.getLogger(__name__)

_FP8_TORCH_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
_SUPPORTED_FORMATS = {"float8_e4m3fn", "float8_e5m2"}


def parse_quantization_metadata(metadata: dict | None) -> dict[str, dict]:
    """解析 safetensors ``__metadata__._quantization_metadata``（Comfy 协议）。

    返回 {layer 名: 层配置 dict}；无声明返回空 dict；声明了不支持的格式报错
    （只支持 fp8 两种——用户裁定范围）。
    """
    if not metadata:
        return {}
    raw = metadata.get("_quantization_metadata")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Krea2 checkpoint 的 _quantization_metadata 不是合法 JSON: {exc}")
    layers = parsed.get("layers")
    if not isinstance(layers, dict):
        raise ValueError("Krea2 _quantization_metadata 缺少 layers 映射")
    unsupported = {
        str(conf.get("format"))
        for conf in layers.values()
        if isinstance(conf, dict) and conf.get("format") not in _SUPPORTED_FORMATS
    }
    if unsupported:
        raise ValueError(
            f"Krea2 checkpoint 声明了不支持的量化格式 {sorted(unsupported)}；"
            f"推理端当前只支持 fp8（{sorted(_SUPPORTED_FORMATS)}）——"
            f"int8/nvfp4/int4 请使用 bf16 或 fp8 版本权重"
        )
    return {str(layer): dict(conf) for layer, conf in layers.items()}


def _fp8_linear_forward(self, input: torch.Tensor) -> torch.Tensor:
    # ComfyUI parity：目标 dtype = input.dtype（cast_bias_weight :286-290）；
    # scale 先转 compute dtype 再乘（ck eager dequantize_per_tensor_fp8）。
    # bias 同 cast_bias_weight cast 到 input.dtype——DiT 场景 bias 本就是
    # input dtype（no-op 等价，parity 不变）；TE fp8（fp32 compute）场景
    # bias 是 fp16 存储，必须 cast。
    weight = self.weight.to(input.dtype)
    scale = getattr(self, "weight_scale", None)
    if scale is not None:
        weight = weight * scale.to(input.dtype)
    bias = self.bias
    if bias is not None and bias.dtype != input.dtype:
        bias = bias.to(input.dtype)
    return torch.nn.functional.linear(input, weight, bias)


def patch_fp8_linears(
    model: torch.nn.Module,
    scales: dict[str, torch.Tensor | None],
    *,
    device: torch.device | None = None,
) -> int:
    """给持 fp8 权重的 Linear 挂 dequant 前向；返回 patch 数。

    ``scales``：layer 名（如 ``blocks.0.attn.gate``）→ F32 标量 scale；
    纯 cast 形态传 None。scale 以非持久 buffer 存进模块（不进 state_dict，
    不破坏既有序列化面）。

    ``device``：scale 放哪。默认跟随该层权重所在设备。**block swap 场景必须
    显式传计算设备**：被换出层的权重此刻在 CPU，跟随它会把 scale 也放 CPU，
    而前向时权重已被搬到 GPU，`weight * scale` 会 device 不匹配（scale.to()
    只改 dtype 不改 device）。scale 是 per-layer 标量，常驻 GPU 开销可忽略。
    """
    patched = 0
    modules = dict(model.named_modules())
    for layer, scale in scales.items():
        module = modules.get(layer)
        if module is None or not isinstance(module, torch.nn.Linear):
            raise ValueError(f"fp8 量化层在模型中不存在或不是 Linear：{layer}")
        if module.weight.dtype not in _FP8_TORCH_DTYPES:
            raise ValueError(
                f"fp8 patch 目标层权重不是 fp8：{layer} ({module.weight.dtype})"
            )
        if scale is not None:
            module.register_buffer(
                "weight_scale",
                scale.to(
                    device=device if device is not None else module.weight.device,
                    dtype=torch.float32,
                ),
                persistent=False,
            )
        module.forward = MethodType(_fp8_linear_forward, module)
        patched += 1
    if patched:
        logger.info("Krea2 fp8 推理：%d 个 Linear 挂 dequant 前向", patched)
    return patched


def model_has_fp8_layers(model: object) -> bool:
    """采样/LoRA 路径判断底模是否为 fp8 量化形态。

    非 nn.Module（测试 fake / 尚未加载）一律 False——fp8 形态只可能来自
    真实 loader 产物。
    """
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return False
    return any(
        isinstance(m, torch.nn.Linear) and m.weight.dtype in _FP8_TORCH_DTYPES
        for m in modules()
    )
