"""MOSS-Image DiT 单文件加载（MOSS 设计稿 §5.3）。

接受 ComfyUI / civitai 分发的两种单文件：

- **bf16**：``layers.N.proj.qkv`` 融合（Comfy-Org 打包），或 q/k/v 分开（HF 导出）
- **int8**：``int8_tensorwise + convrot``，``layers.N.proj.*`` 四类 Linear 为 I8 +
  逐行 F32 ``weight_scale`` + U8 ``comfy_quant`` 配置；其余张量 bf16

内部统一用 HF 命名（q/k/v 分开），所以 LoRA 键名在融合 / 非融合、bf16 / int8
四种底模上都一样（ComfyUI 把分开的 q/k/v LoRA 映射到融合 qkv 的行切片）。
融合 qkv 按行三等分即可精确拆开：convrot 旋转只作用在输入维（列），int8 的
scale 是逐行的，行与行互相独立。

层数、宽度等结构参数一律从 checkpoint header 推断，不查表。

block swap（``blocks_to_swap`` > 0）：末尾 N 层的权重直接载到 CPU pinned、不经过
显存，由 ``training.block_swap.PinnedBlockSwap`` 就地接管（与 Krea2 loader 同口径）。
int8 层的 ``weight_scale`` 恒放计算设备——它是 buffer 不归 swap 管，跟着权重落 CPU
会让前向每步做一次 H2D。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from modeling.moss.moss_modeling import MossConfig, MossImageTransformer
from training.families.moss.quant_int8 import parse_comfy_quant, patch_int8_linear

logger = logging.getLogger(__name__)

#: 外层前缀候选（按序尝试；"" = 无前缀，Comfy-Org / civitai 文件即此形态）
_PREFIX_CANDIDATES = ("", "model.diffusion_model.", "diffusion_model.", "model.", "transformer.")
_PROBE_KEY = "layers.0.norm1.weight"
#: checkpoint 里存在但不是权重的键：派生 RoPE 频率、ComfyUI 注意力内核配置
_IGNORED_KEY_RE = re.compile(r"^(rope\.bands|layers\.\d+\.comfy_attention\.config)$")
_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_QKV_RE = re.compile(r"^(layers\.\d+\.proj)\.qkv\.(weight|weight_scale|comfy_quant)$")
_FLOAT_DTYPES = {"BF16", "F16", "F32"}


@dataclass
class MossCheckpointInfo:
    config: MossConfig
    prefix: str
    #: qkv 是否融合存储
    fused_qkv: bool
    #: int8 量化层（HF 命名的 Linear 路径）→ convrot 组大小
    int8_layers: dict[str, int] = field(default_factory=dict)

    @property
    def is_int8(self) -> bool:
        return bool(self.int8_layers)


def _checkpoint_path(path) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_dir() or checkpoint.suffix != ".safetensors":
        raise ValueError(f"MOSS 底模须为单个 .safetensors 文件：{checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MOSS 底模文件不存在：{checkpoint}")
    return checkpoint


def _choose_prefix(keys: list[str]) -> str:
    key_set = set(keys)
    for prefix in _PREFIX_CANDIDATES:
        if prefix + _PROBE_KEY in key_set:
            return prefix
    raise ValueError(f"不是 MOSS-Image DiT 权重：找不到 {_PROBE_KEY}")


def _config_from_shapes(shapes: dict[str, tuple[int, ...]]) -> MossConfig:
    def shape(key: str) -> tuple[int, ...]:
        if key not in shapes:
            raise ValueError(f"MOSS checkpoint 缺少结构探测键 {key}")
        return shapes[key]

    dim = shape("video_embedder.bias")[0]
    head_dim = shape("layers.0.q_norm.weight")[0]
    layers = {int(m.group(1)) for k in shapes if (m := _LAYER_RE.match(k))}
    num_layers = max(layers) + 1
    if layers != set(range(num_layers)):
        raise ValueError(f"MOSS checkpoint 的 block 序号不连续：{sorted(set(range(num_layers)) - layers)}")
    return MossConfig(
        dim=dim,
        ffn_dim=shape("layers.0.proj.ffn_up_gate.weight")[0] // 2,
        num_heads=dim // head_dim,
        num_layers=num_layers,
        text_dim=shape("text_embedder.proj.weight")[1],
        freq_dim=shape("time_embedding.0.weight")[1],
        latent_channels=shape("final_linear_video.weight")[0],
    )


def _split_rows(tensor: torch.Tensor, parts: int = 3) -> list[torch.Tensor]:
    if tensor.shape[0] % parts:
        raise ValueError(f"融合 qkv 的行数 {tensor.shape[0]} 不能三等分")
    return list(torch.chunk(tensor, parts, dim=0))


def inspect_moss_checkpoint(path) -> MossCheckpointInfo:
    """只读 header：推断结构、前缀、qkv 融合与否、int8 层清单（不读权重数据）。"""
    checkpoint = _checkpoint_path(path)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        prefix = _choose_prefix(keys)
        shapes: dict[str, tuple[int, ...]] = {}
        dtypes: dict[str, str] = {}
        for key in keys:
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            sl = handle.get_slice(key)
            shapes[name] = tuple(sl.get_shape())
            dtypes[name] = sl.get_dtype()
        quant_blobs = {
            name: handle.get_tensor(prefix + name)
            for name in shapes if name.endswith(".comfy_quant")
        }

    config = _config_from_shapes(shapes)
    fused = "layers.0.proj.qkv.weight" in shapes
    int8_layers: dict[str, int] = {}
    for name, dtype in dtypes.items():
        if not name.endswith(".weight") or _IGNORED_KEY_RE.match(name):
            continue
        if dtype in _FLOAT_DTYPES:
            continue
        layer = name[: -len(".weight")]
        if dtype != "I8":
            raise ValueError(
                f"MOSS 底模 {layer} 的权重类型 {dtype} 不受支持（只支持 bf16 / fp16 / fp32 "
                "与 ComfyUI int8_tensorwise + convrot）"
            )
        if f"{layer}.weight_scale" not in shapes or f"{layer}.comfy_quant" not in quant_blobs:
            raise ValueError(f"MOSS int8 层 {layer} 缺少 weight_scale 或 comfy_quant 配置")
        group = parse_comfy_quant(quant_blobs[f"{layer}.comfy_quant"])["convrot_groupsize"]
        if m := re.match(r"^(layers\.\d+\.proj)\.qkv$", layer):
            for part in ("q", "k", "v"):
                int8_layers[f"{m.group(1)}.{part}"] = group
        else:
            int8_layers[layer] = group
    return MossCheckpointInfo(config=config, prefix=prefix, fused_qkv=fused, int8_layers=int8_layers)


_DTYPE_BYTES = {"I8": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4}


def _swapped_prefixes(depth: int, blocks_to_swap: int) -> tuple[str, ...]:
    """被换出层的键前缀（末尾 N 层，超界 clamp，与 PinnedBlockSwap 同口径）。"""
    num_swap = min(max(int(blocks_to_swap), 0), depth)
    return tuple(f"layers.{i}." for i in range(depth - num_swap, depth))


def _header_layer_numel(path) -> tuple[dict[int, int], int]:
    """header 里每层参数量与全模型参数量（不读权重；weight_scale / comfy_quant 不计）。"""
    checkpoint = _checkpoint_path(path)
    per_layer: dict[int, int] = {}
    total = 0
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        prefix = _choose_prefix(keys)
        for key in keys:
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if _IGNORED_KEY_RE.match(name) or name.endswith((".comfy_quant", ".weight_scale")):
                continue
            numel = 1
            for dim in handle.get_slice(key).get_shape():
                numel *= int(dim)
            total += numel
            if m := _LAYER_RE.match(name):
                per_layer[int(m.group(1))] = per_layer.get(int(m.group(1)), 0) + numel
    return per_layer, total


def swapped_param_ratio(path, blocks_to_swap: int) -> float:
    """换出层占全模型参数的比例（显存预算折扣用，按参数量不按字节——int8 与 bf16
    文件大小差一倍，按字节折扣会在 int8 场景把护栏折扣穿；同 Krea2 口径）。"""
    if blocks_to_swap <= 0:
        return 0.0
    per_layer, total = _header_layer_numel(path)
    depth = len(per_layer)
    first = depth - min(int(blocks_to_swap), depth)
    swapped = sum(n for i, n in per_layer.items() if i >= first)
    return swapped / total if total else 0.0


def _swapped_pinned_bytes(path, prefixes: tuple[str, ...], dtype: torch.dtype) -> int:
    """换出层落 pinned 后的精确字节数：int8 原样 1 字节，浮点按计算 dtype。"""
    checkpoint = _checkpoint_path(path)
    float_bytes = torch.empty(0, dtype=dtype).element_size()
    need = 0
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        prefix = _choose_prefix(keys)
        for key in keys:
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if not name.startswith(prefixes) or _IGNORED_KEY_RE.match(name):
                continue
            if name.endswith((".comfy_quant", ".weight_scale")):
                continue
            sl = handle.get_slice(key)
            numel = 1
            for dim in sl.get_shape():
                numel *= int(dim)
            need += numel * (1 if sl.get_dtype() == "I8" else float_bytes)
    return need


def load_moss_model(path, device, dtype: torch.dtype, *,
                    blocks_to_swap: int = 0, int8_forward: str = "w8a16") -> MossImageTransformer:
    """strict 加载 MOSS DiT：浮点张量 cast 到 ``dtype``，int8 原样常驻并挂 W8A16 前向。

    结果 ``requires_grad_(False)``（底模冻结，LoRA 另挂）。``blocks_to_swap`` > 0 时
    末尾 N 层直接落 CPU pinned（见模块 docstring）。``int8_forward``：训练用
    ``w8a16``，出图用 ``w8a8``（ComfyUI parity，见 quant_int8）。

    模型上记 ``moss_fused_qkv``：checkpoint 的 qkv 是否融合存储。ComfyUI 对融合
    qkv 按整块 ``layers.N.proj.qkv`` merge 与重新量化（随机舍入的种子和随机数形状
    都按整块取），parity merge 要据此把 q/k/v 拼回去。
    """
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"MOSS loader 不支持 dtype={dtype}")
    target = torch.device(device)
    if target.type == "meta":
        raise ValueError("MOSS loader 的目标 device 不能是 meta")
    checkpoint = _checkpoint_path(path)
    info = inspect_moss_checkpoint(checkpoint)

    with torch.device("meta"):
        model = MossImageTransformer(info.config)
    model.requires_grad_(False)
    expected = set(model.state_dict().keys())

    swapped = _swapped_prefixes(info.config.num_layers, blocks_to_swap)
    packer = None
    if swapped:
        from training.block_swap import PinnedPacker
        from training.sysmem import check_pinned_budget

        # 预算护栏先行（任何权重读取之前，失败即 fail-fast）；再按精确总量一次性
        # 预分配 pinned 大块（逐张量 pin 会被 host allocator 按 2 的幂取整白锁）
        need = _swapped_pinned_bytes(checkpoint, swapped, dtype)
        check_pinned_budget(need, blocks=len(swapped))
        packer = PinnedPacker(need)

    state: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith(info.prefix):
                continue
            name = key[len(info.prefix):]
            if _IGNORED_KEY_RE.match(name) or name.endswith(".comfy_quant"):
                continue
            tensor = handle.get_tensor(key)
            qkv = _QKV_RE.match(name)
            if qkv:
                base, kind = qkv.group(1), qkv.group(2)
                pieces = dict(zip(("q", "k", "v"), _split_rows(tensor)))
            else:
                base, kind, pieces = None, None, None
            targets = (
                {f"{base}.{part}.{kind}": piece.contiguous() for part, piece in pieces.items()}
                if pieces is not None else {name: tensor}
            )
            for target_name, value in targets.items():
                if target_name.endswith(".weight_scale"):
                    scales[target_name[: -len(".weight_scale")]] = value.to(device=target)
                elif packer is not None and target_name.startswith(swapped):
                    state[target_name] = packer.pin(
                        value, dtype=None if value.dtype == torch.int8 else dtype,
                    )
                elif value.dtype == torch.int8:
                    state[target_name] = value.to(device=target)
                else:
                    state[target_name] = value.to(device=target, dtype=dtype)

    missing = sorted(expected - state.keys())
    unexpected = sorted(state.keys() - expected)
    if missing or unexpected:
        raise ValueError(
            f"MOSS checkpoint 键不匹配：缺少 {missing[:5]}（共 {len(missing)}），"
            f"多出 {unexpected[:5]}（共 {len(unexpected)}）"
        )
    if set(scales) != set(info.int8_layers):
        raise ValueError("MOSS int8 层与 weight_scale 不一一对应")

    model.load_state_dict(state, strict=True, assign=True)
    del state
    modules = dict(model.named_modules())
    for layer, group in info.int8_layers.items():
        patch_int8_linear(modules[layer], scales[layer], group, forward=int8_forward)
    model.moss_fused_qkv = bool(info.fused_qkv)
    if packer is not None:
        logger.debug(
            "block_swap: placement swapped=%d/%d pinned=%.2f GB locked=%.2f GB",
            len(swapped), info.config.num_layers, packer.packed_bytes / 1024 ** 3,
            packer.allocated_bytes / 1024 ** 3,
        )
    return model.eval()
