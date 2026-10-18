"""ComfyUI ``int8_tensorwise + convrot`` 权重的常驻与前向（MOSS 设计稿 §5.4）。

格式约定（comfy-kitchen 0.2.38 ``tensor/int8_utils.py`` /
``backends/eager/quantization.py``，Apache-2.0，见 THIRD_PARTY_NOTICES）：

- H = 归一化规则 Hadamard（4×4 基元 kron 展开 / √g），对称且自逆
- 存储：``W_rot = W @ Hᵀ``（按输入维每 g 列一组），再逐行 int8 量化；
  ``weight_scale`` 为逐行 F32 ``[out, 1]``
- 反量化：``W = (q · s) @ Hᵀ``（H 自逆）

三种前向（``patch_int8_linear(..., forward=...)``，int8 都原样常驻）：

- ``w8a16``（训练）：``y = ((x @ H) @ qᵀ) ⊙ sᵀ + b``。q 是 ±127 内的整数，转 compute
  dtype 无损；scale 在矩阵乘之后按输出列乘，既不产生 fp32 临时权重，也不让 scale
  先被舍入成 bf16。``x @ H`` 是普通可微 matmul，autograd 自然给出对输入的梯度。
- ``w8a8``（出图 DiT，ComfyUI parity）：逐位复刻 comfy-kitchen eager ``int8_linear``
  ——激活旋转后逐行动态 int8 量化，``torch._int_mm`` 整数矩阵乘，再按
  ``x_scale × w_scale``（fp32）缩放回输出 dtype。作者本地 ComfyUI（torch cu128）
  会禁用 comfy-kitchen 的 cuda / triton 后端，实际跑的就是 eager。
- ``dequant``（TE，ComfyUI parity）：ComfyUI 编码时 TE 不走 W8A8，每次前向把权重
  反量化成 compute dtype 再做普通 ``F.linear``（``full_precision_mm``）。

LoRA merge 后的重新量化（随机舍入）见 :func:`requantize_convrot`。
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
    # 缓存是进程级的：TE 编码在 inference_mode 里首次构造的 H 若是 inference tensor，
    # 之后被 DiT 训练前向（同 device / dtype / 组大小）复用时，autograd 存不下它，
    # 反向直接报错。构造时显式退出 inference_mode，产物恒为普通张量。
    with torch.inference_mode(False):
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


def _rotate_activation_comfy(x: Tensor, group: int) -> Tensor:
    """comfy-kitchen ``_rotate_activation``：``(-1, 组数, 组大小)`` 的 3 维批量
    matmul，H 在 ``x.dtype``。与 :func:`rotate_activation` 数学相同，形状照搬是为了
    让 cuBLAS 选同一种算法（W8A8 parity 路径用）。"""
    features = x.shape[-1]
    if features % group:
        raise ValueError(f"输入维 {features} 不能被 convrot 组大小 {group} 整除")
    h = hadamard(group, device=x.device, dtype=x.dtype)
    return torch.matmul(x.reshape(-1, features // group, group), h).reshape(x.shape)


def _rotate_weight_comfy(w: Tensor, group: int) -> Tensor:
    """comfy-kitchen ``_rotate_weight``：``W @ Hᵀ``（按输入维分组），H 在 ``w.dtype``。"""
    out_f, in_f = w.shape
    if in_f % group:
        raise ValueError(f"in_features {in_f} 不能被 convrot 组大小 {group} 整除")
    h = hadamard(group, device=w.device, dtype=w.dtype)
    return torch.matmul(w.reshape(out_f, in_f // group, group), h.T).reshape(out_f, in_f)


def _round_int8(scaled: Tensor, seed: int = 0) -> Tensor:
    """comfy-kitchen eager ``_round_int8``：seed > 0 时随机舍入（同 dtype 的
    ``torch.rand``，设备上的 Generator），否则四舍六入五成双。原地改写 ``scaled``。"""
    if seed > 0:
        generator = torch.Generator(device=scaled.device)
        generator.manual_seed(int(seed))
        scaled.add_(torch.rand(
            scaled.shape, dtype=scaled.dtype, layout=scaled.layout,
            device=scaled.device, generator=generator,
        ))
        return scaled.floor_().clamp_(-128.0, 127.0).to(torch.int8)
    return scaled.round_().clamp_(-128.0, 127.0).to(torch.int8)


def quantize_rowwise(x: Tensor, *, seed: int = 0) -> tuple[Tensor, Tensor]:
    """comfy-kitchen eager ``quantize_int8_rowwise``：逐行 amax/127（fp32 scale，
    下限 1e-30）；除法用 scale 转成 ``x.dtype`` 的值（0 换成该 dtype 的 tiny）。"""
    abs_max = x.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max.float() / 127.0).clamp(min=1e-30)
    divisor = scale.to(device=x.device, dtype=x.dtype)
    divisor = torch.where(divisor == 0, torch.full_like(divisor, torch.finfo(x.dtype).tiny), divisor)
    return _round_int8(x / divisor, seed), scale


def int8_mm(a: Tensor, b: Tensor) -> Tensor:
    """comfy-kitchen eager ``_int8_matmul_accumulate``：``torch._int_mm`` 前按 Comfy
    的规则补零（CUDA 上 M 补到 32 的倍数且 ≥32，K 补到 8 的倍数，N 补到 8 / Turing
    32 的倍数）。整数累加精确，补零只为满足 cuBLASLt 的形状约束。"""
    def _round_up(value: int, alignment: int) -> int:
        return ((value + alignment - 1) // alignment) * alignment

    orig_m, k, orig_n = a.size(0), a.size(1), b.size(1)
    if orig_m == 0 or k == 0:
        return torch._int_mm(a, b)
    padded_m = _round_up(max(orig_m, 32), 32) if a.is_cuda else orig_m
    if padded_m != orig_m:
        a = torch.cat((a, a.new_zeros((padded_m - orig_m, k))), dim=0)
    padded_k = _round_up(k, 8)
    if padded_k != k:
        a = torch.cat((a, a.new_zeros((a.size(0), padded_k - k))), dim=1)
        b = torch.cat((b, b.new_zeros((padded_k - k, b.size(1)))), dim=0)
    turing = a.is_cuda and torch.cuda.get_device_capability(a.device) == (7, 5)
    padded_n = _round_up(orig_n, 32 if turing else 8)
    if padded_n != orig_n:
        b = torch.cat((b, b.new_zeros((b.size(0), padded_n - orig_n))), dim=1)
    out = torch._int_mm(a, b)
    if out.size(0) != orig_m or out.size(1) != orig_n:
        out = out[:orig_m, :orig_n]
    return out


def w8a8_linear(x: Tensor, qweight: Tensor, weight_scale: Tensor, group: int,
                bias: Tensor | None = None) -> Tensor:
    """comfy-kitchen eager ``int8_linear``（convrot）逐位复刻；输出 dtype = ``x.dtype``。"""
    out_dtype = x.dtype
    x = _rotate_activation_comfy(x, group)
    orig_shape = x.shape
    x_8, x_scale = quantize_rowwise(x.reshape(-1, x.shape[-1]))
    acc = int8_mm(x_8, qweight.to(device=x.device).T.contiguous())
    w_scale = weight_scale.to(device=x.device, dtype=torch.float32).reshape(1, -1)
    # Comfy 按行分块只为省显存，逐元素运算与整块相同：(x_scale·w_scale) 先乘
    out = (acc.float() * (x_scale.float() * w_scale)).to(out_dtype)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype).reshape(1, -1)
    return out.reshape(*orig_shape[:-1], qweight.shape[0])


def requantize_convrot(weight: Tensor, group: int, *, seed: int) -> tuple[Tensor, Tensor]:
    """merge 后的权重重新量化回 int8（comfy-kitchen eager ``quantize_int8_convrot_weight``
    + ``stochastic_rounding=seed``）：在 ``weight.dtype``（Comfy 为 fp16）里旋转、逐行
    amax、随机舍入。返回 ``(int8 [N, K], fp32 scale [N, 1])``。"""
    return quantize_rowwise(_rotate_weight_comfy(weight, group), seed=seed)


def dequantize(qweight: Tensor, scale: Tensor, group: int) -> Tensor:
    """int8 + 逐行 scale → 原基底 fp32 权重（comfy-kitchen ``dequantize_int8_convrot_weight``：
    fp32 里乘 scale 再按组乘 Hᵀ；调用方按需 ``.to(dtype)``）。"""
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


def _int8_convrot_w8a8_forward(self: nn.Linear, input: Tensor) -> Tensor:
    return w8a8_linear(input, self.weight, self.weight_scale, self.convrot_groupsize, self.bias)


def _int8_convrot_dequant_forward(self: nn.Linear, input: Tensor) -> Tensor:
    weight = dequantize(self.weight, self.weight_scale, self.convrot_groupsize).to(input.dtype)
    bias = None if self.bias is None else self.bias.to(input.dtype)
    return F.linear(input, weight, bias)


_INT8_FORWARDS = {
    "w8a16": _int8_convrot_forward,
    "w8a8": _int8_convrot_w8a8_forward,
    "dequant": _int8_convrot_dequant_forward,
}


def patch_int8_linear(module: nn.Linear, scale: Tensor, group: int, *,
                      forward: str = "w8a16") -> None:
    """给 int8 权重的 Linear 挂前向（三种见模块 docstring）。``scale`` 以非持久
    buffer 存（不进 state_dict）。"""
    if forward not in _INT8_FORWARDS:
        raise ValueError(f"未知 int8 前向 {forward!r}，可选 {sorted(_INT8_FORWARDS)}")
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
    module.forward = MethodType(_INT8_FORWARDS[forward], module)


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
