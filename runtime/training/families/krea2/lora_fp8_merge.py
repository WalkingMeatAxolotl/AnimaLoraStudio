"""Krea2 fp8 底模的 LoRA merge 回写（ComfyUI 逐位 parity）。

ComfyUI 从不在量化存储上算 LoRA：加载时把 LoRA merge 进权重
（model_patcher.patch_weight_to_device），采样期前向与无 LoRA 完全同路。
本模块逐位复刻该链路（oracle=本地 ComfyUI v0.27.1 + comfy_kitchen 0.2.16
eager 路径，docs/design/krea2-fp8-inference.md §1.5）：

    W16    = W_fp8.to(fp16) [* scale.to(fp16)]            # dequant 到 fp16
    W16   += ((strength * alpha/dim) * ΔW_fp32).to(fp16)   # 逐 LoRA，fp32 中间
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
- delta 域：comfy weight_adapter lokr/lora——因子 cast fp32 做 mm/kron，
  ``weight += ((strength * alpha) * diff).type(fp16)``，加法在 fp16 域
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
import zlib

import torch
from torch import Tensor

from training.families.krea2.quant_fp8 import _FP8_TORCH_DTYPES


logger = logging.getLogger(__name__)

_LORA_PREFIX = "lora_unet_"
_COMFY_PREFIX = "diffusion_model."
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


def string_to_seed(key: str) -> int:
    """comfy.utils.string_to_seed 等价：标准 CRC-32（0xEDB88320）。

    comfy 手写实现对 str 逐字符 ord() 异或——层名均为 ASCII，与 utf-8 字节
    序列一致，zlib.crc32 逐位相同（单测对拍手写参考实现）。
    """
    if not key.isascii():
        raise ValueError(f"seed key 必须是 ASCII（comfy 层名域）：{key!r}")
    return zlib.crc32(key.encode("utf-8"))


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


def _group_lora_layers(sd: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    """按层聚合，两种键格式归一到 {layer_underscored: {kohya_suffix: tensor}}：

    - kohya/lycoris：``lora_unet_{层名下划线}.{suffix}``（本 app 训练产物）
    - PEFT/comfy：``diffusion_model.{层名点分}.{lora_A|lora_B|alpha}``
      （civitai / musubi / comfy 生态）
    """
    layers: dict[str, dict[str, Tensor]] = {}
    for key, tensor in sd.items():
        if key.startswith(_LORA_PREFIX):
            name, _, suffix = key.partition(".")
            layers.setdefault(name[len(_LORA_PREFIX):], {})[suffix] = tensor
        elif key.startswith(_COMFY_PREFIX):
            rest = key[len(_COMFY_PREFIX):]
            for peft_suffix, kohya_suffix in _PEFT_SUFFIXES:
                if rest.endswith("." + peft_suffix):
                    layer = rest[: -len(peft_suffix) - 1]
                    layers.setdefault(layer.replace(".", "_"), {})[kohya_suffix] = tensor
                    break
            else:
                raise ValueError(
                    f"fp8 merge 无法识别 comfy 形态 LoRA 键的后缀：{key}"
                )
    return layers


def _apply_lora_delta(
    w16: Tensor,
    tensors: dict[str, Tensor],
    strength: float,
    layer: str,
    source: str,
) -> Tensor:
    """单层单 LoRA 的 comfy 顺序 merge：fp32 算 diff，fp16 域相加。"""
    device = w16.device
    if "dora_scale" in tensors:
        raise ValueError(
            f"fp8 底模 merge 不支持 DoRA（weight_decompose）LoRA：{source} 层 {layer}。"
            f"请改用 bf16 版本底模挂载。"
        )

    if "lora_down.weight" in tensors:  # plain LoRA / LoCon Linear
        if "lora_mid.weight" in tensors:
            raise ValueError(f"fp8 merge 不支持 LoCon mid（tucker）形态：{source} 层 {layer}")
        mat1 = tensors["lora_up.weight"].to(device=device, dtype=torch.float32)
        mat2 = tensors["lora_down.weight"].to(device=device, dtype=torch.float32)
        alpha = float(tensors["alpha"]) / mat2.shape[0] if "alpha" in tensors else 1.0
        lora_diff = torch.mm(
            mat1.flatten(start_dim=1), mat2.flatten(start_dim=1),
        ).reshape(w16.shape)
        w16 += ((strength * alpha) * lora_diff).type(w16.dtype)
        return w16

    if "lokr_w1" in tensors or "lokr_w1_a" in tensors:  # LoKr
        if "lokr_t2" in tensors:
            raise ValueError(f"fp8 merge 不支持 LoKr tucker（t2）形态：{source} 层 {layer}")
        # dim 语义照 comfy weight_adapter/lokr.py：w1/w2 各自分解时都会赋值，
        # 两者都分解时 w2_b 的赋值在后、生效（覆盖），两者都是全矩阵时保持
        # None → alpha 系数取 1.0
        dim = None
        if "lokr_w1" in tensors:
            w1 = tensors["lokr_w1"].to(device=device, dtype=torch.float32)
        else:
            dim = tensors["lokr_w1_b"].shape[0]
            w1 = torch.mm(
                tensors["lokr_w1_a"].to(device=device, dtype=torch.float32),
                tensors["lokr_w1_b"].to(device=device, dtype=torch.float32),
            )
        if "lokr_w2" in tensors:
            w2 = tensors["lokr_w2"].to(device=device, dtype=torch.float32)
        else:
            dim = tensors["lokr_w2_b"].shape[0]
            w2 = torch.mm(
                tensors["lokr_w2_a"].to(device=device, dtype=torch.float32),
                tensors["lokr_w2_b"].to(device=device, dtype=torch.float32),
            )
        if "alpha" in tensors and dim is not None:
            alpha = float(tensors["alpha"]) / dim
        else:
            alpha = 1.0
        lora_diff = torch.kron(w1, w2).reshape(w16.shape)
        w16 += ((strength * alpha) * lora_diff).type(w16.dtype)
        return w16

    raise ValueError(
        f"fp8 merge 无法识别 LoRA 层形态：{source} 层 {layer}（{sorted(tensors)}）"
    )


class Fp8LoraMergeAdapter:
    """merge 回写的生命周期句柄——daemon adapters 列表的 duck-type 成员。

    - ``detach()``：从 CPU 备份逐位还原原始权重与 scale（换 LoRA / 卸载）
    - ``network = None``：让 lycoris 侧的 multiplier 设值路径安全 no-op
    - ``supports_hot_reload = False``：merge 无常驻 network，禁用 daemon
      的权重热换路径（必须走 detach → 重 merge）
    """

    network = None
    supports_hot_reload = False

    def __init__(self, model: torch.nn.Module,
                 backup: dict[str, tuple[Tensor, Tensor | None]]) -> None:
        self._model = model
        self._backup = backup

    def detach(self) -> bool:
        if not self._backup:
            return True
        modules = dict(self._model.named_modules())
        for layer, (weight_cpu, scale_cpu) in self._backup.items():
            module = modules[layer]
            module.weight.data.copy_(weight_cpu.to(module.weight.device))
            if scale_cpu is not None:
                module.weight_scale.copy_(scale_cpu.to(module.weight_scale.device))
        self._backup = {}
        return True


def merge_loras_into_fp8_model(
    model: torch.nn.Module,
    sources: list[tuple[dict[str, Tensor], float, str]],
) -> Fp8LoraMergeAdapter:
    """把多份 LoRA 按 comfy merge 语义烘进（部分）fp8 模型的 Linear 权重。

    ``sources``：[(state_dict, strength, 来源名), ...]，顺序 = 挂载顺序 =
    comfy patches 顺序。返回持有原始权重 CPU 备份的还原句柄。
    """
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
            "fp8 merge：%d 个 LoRA 层在模型中无对应 Linear，跳过（comfy 同款行为）：%s%s",
            len(missing), missing[:5], " ..." if len(missing) > 5 else "",
        )

    backup: dict[str, tuple[Tensor, Tensor | None]] = {}
    for layer_key, layer_sources in per_layer.items():
        name, module = module_index[layer_key]
        weight = module.weight
        scale = getattr(module, "weight_scale", None)
        is_fp8 = weight.dtype in _FP8_TORCH_DTYPES

        backup[name] = (
            weight.detach().to("cpu", copy=True),
            None if scale is None else scale.detach().to("cpu", copy=True),
        )

        # dequant 到 fp16（comfy lora_compute_dtype 域，QuantizedTensor 无
        # bf16 中转）；非量化 Linear 同样 cast fp16 参与 merge
        w16 = weight.detach().to(torch.float16)
        if scale is not None:
            w16 = w16 * scale.to(torch.float16)

        for tensors, strength, source in layer_sources:
            w16 = _apply_lora_delta(w16, tensors, strength, name, source)

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

    logger.info(
        "Krea2 fp8 merge：%d 份 LoRA 烘进 %d 个 Linear（含备份，可 detach 还原）",
        len(sources), len(per_layer),
    )
    return Fp8LoraMergeAdapter(model, backup)
