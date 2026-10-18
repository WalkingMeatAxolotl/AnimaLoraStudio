"""量化底模 LoRA merge 的 ComfyUI 通用语义（族无关中立层）。

ComfyUI 从不在量化存储上算 LoRA：加载时把 LoRA merge 进权重
（``model_patcher.patch_weight_to_device`` → ``comfy.lora.calculate_weight``），
再按各量化格式重新量化回写。本模块只放与存储格式无关、多个族共用的部分：

- :func:`string_to_seed`：重新量化随机舍入的种子（``comfy.utils.string_to_seed``）
- :func:`group_lora_layers`：kohya / PEFT 两种 LoRA 键格式按层归并
- :func:`apply_lora_delta`：``weight += ((strength * alpha) * diff).type(dtype)``
  （comfy ``weight_adapter`` 的 lora / loha / lokr 三种形态）
- :class:`QuantizedMergeAdapter`：merge 回写的备份 / 还原句柄

各族的反量化 / 重新量化（Krea2 fp8、MOSS int8 convrot）留在族内。原先这些
代码在 Krea2 的 ``lora_fp8_merge`` 里，MOSS 接入时按「后来的族不 import 先来的
族」的边界纪律抽到这里，Krea2 行为不变。

派生署名见 THIRD_PARTY_NOTICES（ComfyUI GPL-3.0 + comfy_kitchen）。
"""

from __future__ import annotations

import zlib

import torch
from torch import Tensor

LORA_PREFIX = "lora_unet_"
COMFY_PREFIX = "diffusion_model."
# PEFT/comfy 键后缀 → kohya/lycoris 命名（civitai 生态 LoRA 常见形态：
# ``diffusion_model.{点分层名}.lora_A/lora_B``，lora_A=down、lora_B=up，
# 通常无 alpha 键——comfy 对缺省 alpha 按缩放 1.0 处理，与 apply_lora_delta
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


def group_lora_layers(sd: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    """按层聚合，两种键格式归一到 {layer_underscored: {kohya_suffix: tensor}}：

    - kohya/lycoris：``lora_unet_{层名下划线}.{suffix}``（本 app 训练产物）
    - PEFT/comfy：``diffusion_model.{层名点分}.{lora_A|lora_B|alpha}``
      （civitai / musubi / comfy 生态）
    """
    layers: dict[str, dict[str, Tensor]] = {}
    for key, tensor in sd.items():
        if key.startswith(LORA_PREFIX):
            name, _, suffix = key.partition(".")
            layers.setdefault(name[len(LORA_PREFIX):], {})[suffix] = tensor
        elif key.startswith(COMFY_PREFIX):
            rest = key[len(COMFY_PREFIX):]
            for peft_suffix, kohya_suffix in _PEFT_SUFFIXES:
                if rest.endswith("." + peft_suffix):
                    layer = rest[: -len(peft_suffix) - 1]
                    layers.setdefault(layer.replace(".", "_"), {})[kohya_suffix] = tensor
                    break
            else:
                raise ValueError(
                    f"量化底模 merge 无法识别 comfy 形态 LoRA 键的后缀：{key}"
                )
    return layers


def apply_lora_delta(
    w16: Tensor,
    tensors: dict[str, Tensor],
    strength: float,
    layer: str,
    source: str,
    compute_dtype: torch.dtype = torch.float32,
    chunk_rows: int | None = None,
) -> Tensor:
    """单层单 LoRA merge：可选 fp32/bf16 算 diff，在 ``w16`` 的精度域相加（原地）。"""
    device = w16.device
    if "dora_scale" in tensors:
        raise ValueError(
            f"量化底模 merge 不支持 DoRA（weight_decompose）LoRA：{source} 层 {layer}。"
            f"请改用 bf16 版本底模挂载。"
        )

    if "lora_down.weight" in tensors:  # plain LoRA / LoCon Linear
        if "lora_mid.weight" in tensors:
            raise ValueError(f"量化底模 merge 不支持 LoCon mid（tucker）形态：{source} 层 {layer}")
        mat1 = tensors["lora_up.weight"].to(device=device, dtype=compute_dtype)
        mat2 = tensors["lora_down.weight"].to(device=device, dtype=compute_dtype)
        alpha = float(tensors["alpha"]) / mat2.shape[0] if "alpha" in tensors else 1.0
        mat1 = mat1.flatten(start_dim=1)
        mat2 = mat2.flatten(start_dim=1)
        rows = int(mat1.shape[0])
        if chunk_rows is not None and 0 < int(chunk_rows) < rows:
            # A rank-r LoRA is small, but materializing its out×in dense delta
            # can consume almost 1 GiB for Krea2's tproj layer.  Compute row
            # slices and add them immediately so peak workspace is bounded by
            # chunk_rows×in instead of the full weight.  Only plain Linear LoRA
            # uses this path; LoHa/LoKr retain their exact legacy algorithms.
            w_rows = w16.reshape(rows, -1)
            for start in range(0, rows, int(chunk_rows)):
                end = min(start + int(chunk_rows), rows)
                lora_diff = torch.mm(mat1[start:end], mat2)
                w_rows[start:end] += (
                    (strength * alpha) * lora_diff
                ).type(w16.dtype)
                del lora_diff
        else:
            lora_diff = torch.mm(mat1, mat2).reshape(w16.shape)
            w16 += ((strength * alpha) * lora_diff).type(w16.dtype)
        return w16

    if "hada_w1_a" in tensors:  # LoHa
        if "hada_t1" in tensors or "hada_t2" in tensors:
            raise ValueError(f"量化底模 merge 不支持 LoHa tucker（t1/t2）形态：{source} 层 {layer}")
        m1 = torch.mm(
            tensors["hada_w1_a"].to(device=device, dtype=compute_dtype),
            tensors["hada_w1_b"].to(device=device, dtype=compute_dtype),
        )
        m2 = torch.mm(
            tensors["hada_w2_a"].to(device=device, dtype=compute_dtype),
            tensors["hada_w2_b"].to(device=device, dtype=compute_dtype),
        )
        # dim 语义照 comfy weight_adapter/loha.py：divisor = w1_b 的 rank 维
        alpha = float(tensors["alpha"]) / tensors["hada_w1_b"].shape[0] if "alpha" in tensors else 1.0
        lora_diff = (m1 * m2).reshape(w16.shape)
        w16 += ((strength * alpha) * lora_diff).type(w16.dtype)
        return w16

    if "lokr_w1" in tensors or "lokr_w1_a" in tensors:  # LoKr
        if "lokr_t2" in tensors:
            raise ValueError(f"量化底模 merge 不支持 LoKr tucker（t2）形态：{source} 层 {layer}")
        # dim 语义照 comfy weight_adapter/lokr.py：w1/w2 各自分解时都会赋值，
        # 两者都分解时 w2_b 的赋值在后、生效（覆盖），两者都是全矩阵时保持
        # None → alpha 系数取 1.0
        dim = None
        if "lokr_w1" in tensors:
            w1 = tensors["lokr_w1"].to(device=device, dtype=compute_dtype)
        else:
            dim = tensors["lokr_w1_b"].shape[0]
            w1 = torch.mm(
                tensors["lokr_w1_a"].to(device=device, dtype=compute_dtype),
                tensors["lokr_w1_b"].to(device=device, dtype=compute_dtype),
            )
        if "lokr_w2" in tensors:
            w2 = tensors["lokr_w2"].to(device=device, dtype=compute_dtype)
        else:
            dim = tensors["lokr_w2_b"].shape[0]
            w2 = torch.mm(
                tensors["lokr_w2_a"].to(device=device, dtype=compute_dtype),
                tensors["lokr_w2_b"].to(device=device, dtype=compute_dtype),
            )
        if "alpha" in tensors and dim is not None:
            alpha = float(tensors["alpha"]) / dim
        else:
            alpha = 1.0
        lora_diff = torch.kron(w1, w2).reshape(w16.shape)
        w16 += ((strength * alpha) * lora_diff).type(w16.dtype)
        return w16

    raise ValueError(
        f"量化底模 merge 无法识别 LoRA 层形态：{source} 层 {layer}（{sorted(tensors)}）"
    )


class QuantizedMergeAdapter:
    """merge 回写的生命周期句柄——daemon adapters 列表的 duck-type 成员。

    - ``detach()``：从 CPU 备份逐位还原原始权重与 scale（换 LoRA / 卸载）
    - ``network = None``：让 lycoris 侧的 multiplier 设值路径安全 no-op
    - ``supports_hot_reload = False``：merge 无常驻 network，禁用 daemon
      的权重热换路径（必须走 detach → 重 merge）

    ``backup``：``{Linear 模块名: (weight 的 CPU 副本, weight_scale 的 CPU 副本或 None)}``。
    """

    network = None
    supports_hot_reload = False

    def __init__(self, model: torch.nn.Module,
                 backup: dict[str, tuple[Tensor, Tensor | None]],
                 *, can_restore: bool = True) -> None:
        self._model = model
        self._backup = backup
        #: 是否留了备份。False = 换 LoRA 时无法就地还原，detach() 返回 False
        #: 让 daemon 走「重载模型」兜底（省下与整个模型等大的一份 CPU 备份）。
        self._can_restore = can_restore

    def detach(self) -> bool:
        if not self._can_restore:
            # 没留备份：告诉调用方「还不了」，由它重载模型保证干净状态。
            # 模型引用必须一并丢掉——这个句柄还被 _run_generate/_run_xy 的
            # 局部 adapters 变量持着，不置空则重载兜底期间旧模型整份钉在
            # 显存里（XY 逐格换 LoRA 时最多三份模型同驻，直接 OOM）
            self._model = None
            return False
        if not self._backup:
            return True
        modules = dict(self._model.named_modules())
        for layer, (weight_cpu, scale_cpu) in self._backup.items():
            module = modules[layer]
            module.weight.data.copy_(weight_cpu.to(module.weight.device))
            if scale_cpu is not None:
                module.weight_scale.copy_(scale_cpu.to(module.weight_scale.device))
        self._backup = {}
        self._model = None
        return True
