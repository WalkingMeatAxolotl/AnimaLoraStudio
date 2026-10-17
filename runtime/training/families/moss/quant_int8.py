"""ComfyUI ``int8_tensorwise + convrot`` 权重的常驻与前向（MOSS 设计稿 §5.4）。

格式约定（comfy-kitchen 0.2.38 ``tensor/int8_utils.py`` /
``backends/eager/quantization.py``，Apache-2.0，见 THIRD_PARTY_NOTICES）：

- H = 归一化规则 Hadamard（4×4 基元 kron 展开 / √g），对称且自逆
- 存储：``W_rot = W @ Hᵀ``（按输入维每 g 列一组），再逐行 int8 量化；
  ``weight_scale`` 为逐行 F32 ``[out, 1]``
- 反量化：``W = (q · s) @ Hᵀ``（H 自逆）

本项目前向是 W8A16（训练用）：int8 原样常驻，``y = ((x @ H) @ qᵀ) ⊙ sᵀ + b``。
q 是 ±127 内的整数，转 compute dtype 无损；scale 在矩阵乘之后按输出列乘，既不
产生 fp32 临时权重，也不让 scale 先被舍入成 bf16。``x @ H`` 是普通可微 matmul，
底模权重无梯度，autograd 自然给出对输入的梯度——反向不需要自定义 Function。
与 ComfyUI 的 W8A8（激活也量化）差在激活量化这一步；出图 parity 另走 PR-6。
"""

from __future__ import annotations

import json
import math
from types import MethodType

import torch
from torch import Tensor, nn
from torch.nn import functional as F

#: 唯一接受的 Comfy 逐层量化配置
INT8_CONVROT_FORMAT = "int8_tensorwise"

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], Tensor] = {}


def hadamard(size: int, *, device=None, dtype: torch.dtype = torch.float32) -> Tensor:
    """归一化规则 Hadamard 矩阵（comfy-kitchen ``_build_hadamard`` 同构造）。

    元素为 ±1/√size；size 为 4 的幂时 √size 是 2 的幂，bf16 / fp16 下逐元素精确。
    """
    if size < 4 or size & (size - 1) or (size.bit_length() - 1) % 2:
        raise ValueError(f"convrot 组大小须为 4 的幂，收到 {size}")
    device = torch.device(device) if device is not None else torch.device("cpu")
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    out = (h / math.sqrt(size)).to(device=device, dtype=dtype)
    _HADAMARD_CACHE[key] = out
    return out


def parse_comfy_quant(blob: Tensor | bytes) -> dict:
    """解析 ``{layer}.comfy_quant`` U8 blob；不是 int8_tensorwise + convrot 就报错。"""
    raw = bytes(blob.tolist()) if isinstance(blob, Tensor) else bytes(blob)
    try:
        conf = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"comfy_quant 配置不是合法 JSON：{exc}") from exc
    if conf.get("format") != INT8_CONVROT_FORMAT or not conf.get("convrot"):
        raise ValueError(
            f"MOSS 只支持 {INT8_CONVROT_FORMAT} + convrot 的 int8 权重，收到 {conf}"
        )
    group = int(conf.get("convrot_groupsize", 256))
    hadamard(group)  # 组大小合法性
    return {"format": INT8_CONVROT_FORMAT, "convrot": True, "convrot_groupsize": group}


def rotate_activation(x: Tensor, group: int) -> Tensor:
    """``x @ H``（按最后一维每 ``group`` 个一组）。"""
    features = x.shape[-1]
    if features % group:
        raise ValueError(f"输入维 {features} 不能被 convrot 组大小 {group} 整除")
    h = hadamard(group, device=x.device, dtype=x.dtype)
    return (x.reshape(*x.shape[:-1], features // group, group) @ h).reshape(x.shape)


def dequantize(qweight: Tensor, scale: Tensor, group: int) -> Tensor:
    """int8 + 逐行 scale → 原基底 fp32 权重（测试与 LoRA merge 用；前向不走这里）。"""
    w = qweight.float() * scale.float()
    out_f, in_f = w.shape
    h = hadamard(group, device=w.device, dtype=torch.float32)
    return (w.reshape(out_f, in_f // group, group) @ h.T).reshape(out_f, in_f)


def _int8_convrot_forward(self: nn.Linear, input: Tensor) -> Tensor:
    x = rotate_activation(input, self.convrot_groupsize)
    out = F.linear(x, self.weight.to(x.dtype))
    out = out * self.weight_scale.reshape(-1).to(device=out.device, dtype=out.dtype)
    if self.bias is not None:
        out = out + self.bias.to(out.dtype)
    return out


def patch_int8_linear(module: nn.Linear, scale: Tensor, group: int) -> None:
    """给 int8 权重的 Linear 挂 W8A16 前向。``scale`` 以非持久 buffer 存（不进 state_dict）。"""
    if not isinstance(module, nn.Linear):
        raise ValueError(f"int8 patch 目标不是 Linear：{type(module).__name__}")
    if module.weight.dtype != torch.int8:
        raise ValueError(f"int8 patch 目标层权重不是 int8：{module.weight.dtype}")
    if scale.shape not in ((module.out_features, 1), (module.out_features,)):
        raise ValueError(
            f"int8 weight_scale 形状 {tuple(scale.shape)} 与 out_features={module.out_features} 不符"
        )
    if module.in_features % group:
        raise ValueError(f"in_features={module.in_features} 不能被 convrot 组大小 {group} 整除")
    module.register_buffer("weight_scale", scale.to(dtype=torch.float32), persistent=False)
    module.convrot_groupsize = int(group)
    module.forward = MethodType(_int8_convrot_forward, module)


def _int8_convrot_embedding_forward(self: nn.Embedding, input: Tensor) -> Tensor:
    # comfy-kitchen dequantize_embedding 同口径：只取用到的行，逐行反量化后
    # 沿 hidden 维反旋转（W = W_rot @ Hᵀ，H 对称 → 与激活旋转同一个运算）
    rows = self.weight[input].float() * self.weight_scale[input].float()
    return rotate_activation(rows, self.convrot_groupsize).to(self._int8_compute_dtype)


def patch_int8_embedding(module: nn.Embedding, scale: Tensor, group: int,
                         compute_dtype: torch.dtype) -> None:
    """给 int8 权重的 Embedding 挂按行反量化的 lookup（Comfy int8 TE 的 embed_tokens）。"""
    if not isinstance(module, nn.Embedding):
        raise ValueError(f"int8 embedding patch 目标不是 Embedding：{type(module).__name__}")
    if module.weight.dtype != torch.int8:
        raise ValueError(f"int8 embedding patch 目标权重不是 int8：{module.weight.dtype}")
    if scale.shape not in ((module.num_embeddings, 1), (module.num_embeddings,)):
        raise ValueError(
            f"int8 embedding weight_scale 形状 {tuple(scale.shape)} 与词表 {module.num_embeddings} 不符"
        )
    if module.embedding_dim % group:
        raise ValueError(f"embedding_dim={module.embedding_dim} 不能被 convrot 组大小 {group} 整除")
    module.register_buffer("weight_scale", scale.reshape(-1, 1).to(dtype=torch.float32), persistent=False)
    module.convrot_groupsize = int(group)
    module._int8_compute_dtype = compute_dtype
    module.forward = MethodType(_int8_convrot_embedding_forward, module)


def model_has_int8_layers(model: object) -> bool:
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return False
    return any(isinstance(m, nn.Linear) and m.weight.dtype == torch.int8 for m in modules())
