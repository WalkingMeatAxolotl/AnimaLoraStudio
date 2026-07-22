"""Strict single-file Krea2 checkpoint inspection and loading.

接受 Comfy/musubi 单文件布局（非 diffusers 分片布局）。**Raw 与 Turbo
结构全等（同键同形状）——本 loader 对两者一视同仁，无法也不尝试区分**；
「Turbo 不建议做训练底模」的防呆靠 studio 侧 catalog variant 的 purpose
元数据（P4-4），不在加载层。

The meta-device + ``assign=True`` loading strategy was adapted from
kohya-ss/musubi-tuner (Apache-2.0):
Copyright 2026 Kohya S. and musubi-tuner contributors.
https://github.com/kohya-ss/musubi-tuner/blob/8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6/src/musubi_tuner/krea2/krea2_utils.py

Fingerprint validation, prefix handling, and diagnostics are original to this
repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from modeling.krea2 import KREA2_CONFIG, Krea2Config, SingleStreamDiT
from training.families.krea2.quant_fp8 import (
    parse_quantization_metadata,
    patch_fp8_linears,
)


_PREFIX_CANDIDATES = (
    "",
    "diffusion_model.",
    "model.diffusion_model.",
    "module.",
    "model.",
    "transformer.",
)
_FLOAT_DTYPES = {
    "BF16",
    "F16",
    "F32",
    "F64",
}
# fp8 权重原样常驻 + Linear forward 逐层 dequant（quant_fp8）。推理与训练
# （fp8_base：底模 frozen，LoRA 参数全精度，kohya/musubi 生态标准做法）都
# 支持——绝不静默 upcast 回 bf16（显存零收益丢精度，A7'/C13）。
_FP8_DTYPES = {"F8_E4M3", "F8_E5M2"}


@dataclass(frozen=True)
class Krea2CheckpointInfo:
    path: Path
    prefix: str
    key_count: int
    parameter_count: int


def _checkpoint_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Krea2 checkpoint 不存在：{resolved}")
    if resolved.is_dir():
        raise ValueError(
            "Krea2 loader 需要单文件 raw.safetensors，不能传 diffusers transformer "
            f"分片目录：{resolved}"
        )
    if resolved.suffix.lower() != ".safetensors":
        raise ValueError(f"Krea2 checkpoint 必须是 .safetensors 文件：{resolved}")
    return resolved


def _expected_state(config: Krea2Config):
    with torch.device("meta"):
        model = SingleStreamDiT(config)
    return model, {
        key: tuple(value.shape)
        for key, value in model.state_dict().items()
    }


def _choose_prefix(source_keys: list[str], expected_keys: set[str]) -> str:
    best_prefix = ""
    best_score = -1
    for prefix in _PREFIX_CANDIDATES:
        normalized = {
            key[len(prefix):] if prefix and key.startswith(prefix) else key
            for key in source_keys
        }
        score = len(normalized & expected_keys)
        if score > best_score:
            best_prefix = prefix
            best_score = score
    if best_score <= 0:
        diffusers_hint = any(
            key.startswith(("transformer_blocks.", "img_in.", "text_fusion."))
            for key in source_keys
        )
        hint = (
            "；检测到 diffusers 风格键，请改用仓库根目录的 raw.safetensors"
            if diffusers_hint
            else ""
        )
        raise ValueError(f"不是可识别的 Krea2 checkpoint：结构指纹零命中{hint}")
    return best_prefix


def _normalize_key(key: str, prefix: str) -> str:
    return key[len(prefix):] if prefix and key.startswith(prefix) else key


def _format_key_delta(label: str, keys: set[str]) -> str:
    sample = ", ".join(sorted(keys)[:5])
    suffix = " ..." if len(keys) > 5 else ""
    return f"{label} {len(keys)} 个：{sample}{suffix}"


def _inspect(
    path: str | Path,
    config: Krea2Config,
    *,
    expected_shapes: dict[str, tuple[int, ...]] | None = None,
    allow_fp8: bool = False,
) -> tuple[Krea2CheckpointInfo, dict[str, str], dict[str, str]]:
    """校验 checkpoint 并返回 (info, 权重键映射, fp8 scale 键映射)。

    ``allow_fp8=True``（推理路径）时：接受 fp8 权重；``{layer}.weight_scale``
    F32 标量键被单独收集（fp8_scaled 形态），不参与键集比对。
    """
    checkpoint = _checkpoint_path(path)
    if expected_shapes is None:
        _, expected_shapes = _expected_state(config)
    expected_keys = set(expected_shapes)

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        quant_meta = (
            parse_quantization_metadata(handle.metadata()) if allow_fp8 else {}
        )
        all_source_keys = list(handle.keys())
        prefix = _choose_prefix(all_source_keys, expected_keys)
        # fp8_scaled 的 per-layer scale 键：归一后形如 blocks.N.xxx.weight_scale。
        # 只在 allow_fp8 时剥离；显式 allow_fp8=False 的调用面它们照旧落进
        # 「多出」报错。
        scale_to_source: dict[str, str] = {}
        source_keys = []
        for source_key in all_source_keys:
            normalized = _normalize_key(source_key, prefix)
            if allow_fp8 and normalized.endswith(".weight_scale"):
                layer = normalized[: -len(".weight_scale")]
                if f"{layer}.weight" in expected_keys:
                    scale_to_source[layer] = source_key
                    continue
            source_keys.append(source_key)
        normalized_to_source: dict[str, str] = {}
        for source_key in source_keys:
            normalized = _normalize_key(source_key, prefix)
            if normalized in normalized_to_source:
                raise ValueError(f"Krea2 checkpoint 前缀归一后键冲突：{normalized}")
            normalized_to_source[normalized] = source_key

        actual_keys = set(normalized_to_source)
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys
        errors = []
        if missing:
            errors.append(_format_key_delta("缺少", missing))
        if unexpected:
            errors.append(_format_key_delta("多出", unexpected))

        shape_mismatches = []
        dtype_mismatches = []
        fp8_keys = []
        for normalized in sorted(expected_keys & actual_keys):
            source_key = normalized_to_source[normalized]
            tensor_slice = handle.get_slice(source_key)
            actual_shape = tuple(tensor_slice.get_shape())
            expected_shape = expected_shapes[normalized]
            if actual_shape != expected_shape:
                shape_mismatches.append(
                    f"{normalized}: {actual_shape} != {expected_shape}"
                )
            actual_dtype = tensor_slice.get_dtype()
            if actual_dtype in _FP8_DTYPES:
                fp8_keys.append(normalized)
            elif actual_dtype not in _FLOAT_DTYPES:
                dtype_mismatches.append(f"{normalized}: {actual_dtype}")
        if fp8_keys and not allow_fp8:
            raise ValueError(
                f"Krea2 checkpoint 含 fp8 参数（{len(fp8_keys)} 个，如 "
                f"{fp8_keys[0]}），但调用方要求全精度权重。"
            )
        # fp8_scaled 一致性：有 scale 的层权重必须是 fp8；metadata 声明的层
        # 若既无 fp8 权重也无 scale 属异常文件
        if allow_fp8:
            fp8_set = set(fp8_keys)
            for layer in scale_to_source:
                if f"{layer}.weight" not in fp8_set:
                    errors.append(f"{layer} 带 weight_scale 但权重不是 fp8")
            for layer in quant_meta:
                if f"{layer}.weight" not in fp8_set:
                    errors.append(f"metadata 声明量化层 {layer} 但权重不是 fp8")
        if shape_mismatches:
            sample = "; ".join(shape_mismatches[:5])
            suffix = " ..." if len(shape_mismatches) > 5 else ""
            errors.append(f"shape 不匹配 {len(shape_mismatches)} 个：{sample}{suffix}")
        if dtype_mismatches:
            sample = "; ".join(dtype_mismatches[:5])
            suffix = " ..." if len(dtype_mismatches) > 5 else ""
            errors.append(f"非浮点 tensor {len(dtype_mismatches)} 个：{sample}{suffix}")
        if errors:
            raise ValueError("Krea2 checkpoint 结构指纹不匹配；" + "；".join(errors))

    parameter_count = sum(
        math_product(shape) for shape in expected_shapes.values()
    )
    return (
        Krea2CheckpointInfo(
            path=checkpoint,
            prefix=prefix,
            key_count=len(source_keys),
            parameter_count=parameter_count,
        ),
        normalized_to_source,
        scale_to_source,
    )


def math_product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def inspect_krea2_checkpoint(
    path: str | Path,
    *,
    config: Krea2Config = KREA2_CONFIG,
) -> Krea2CheckpointInfo:
    """Validate keys and shapes from the safetensors header without reading payloads.

    bf16 与 fp8（scaled / 纯 cast）两种形态都是合法 checkpoint。
    """
    info, _, _ = _inspect(path, config, allow_fp8=True)
    return info


def checkpoint_contains_fp8(path: str | Path) -> bool:
    """轻量探测：safetensors header 里是否有 fp8 权重（不读 payload）。

    供训练启动期防呆用（fp8 底模 + grad_checkpoint 关闭等组合要 fail-fast，
    不能等 13GB 加载完才崩）。非 safetensors / 读失败返回 False——真正的
    结构校验由 loader 兜底。
    """
    try:
        checkpoint = _checkpoint_path(path)
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            return any(
                handle.get_slice(key).get_dtype() in _FP8_DTYPES
                for key in handle.keys()
            )
    except Exception:
        return False


def _swapped_block_prefixes(config: Krea2Config, blocks_to_swap: int) -> tuple[str, ...]:
    """被换出的层的 state_dict 键前缀（末尾 N 层，与 PinnedBlockSwap 同口径）。"""
    if blocks_to_swap <= 0:
        return ()
    first = max(config.layers - blocks_to_swap, 0)
    return tuple(f"blocks.{i}." for i in range(first, config.layers))


def _swapped_param_counts(
    blocks_to_swap: int, config: Krea2Config,
) -> tuple[int, int]:
    """(换出层参数量, 全模型参数量)。meta 模型数参数，不读盘、不占显存。"""
    prefixes = _swapped_block_prefixes(config, blocks_to_swap)
    with torch.device("meta"):
        probe = SingleStreamDiT(config)
    swapped = total = 0
    for name, param in probe.named_parameters():
        total += param.numel()
        if prefixes and name.startswith(prefixes):
            swapped += param.numel()
    return swapped, total


def swapped_param_ratio(
    blocks_to_swap: int, *, config: Krea2Config = KREA2_CONFIG,
) -> float:
    """换出层占全模型参数的比例 —— **显存预算折扣用这个，不要用字节数**。

    折扣必须 dtype 无关：`check_load_budget` 的 need 来自权重文件实际大小，
    fp8 checkpoint 只有 bf16 的一半。若折扣按 bf16 字节算，会把 fp8 场景的
    需求折扣过头（need 13GB 减掉 11.3GB → 以为只要 1.7GB，实际常驻 7.2GB），
    护栏就形同虚设。按比例乘文件实际大小则两种精度都正确。
    """
    if blocks_to_swap <= 0:
        return 0.0
    swapped, total = _swapped_param_counts(blocks_to_swap, config)
    return (swapped / total) if total else 0.0


def estimate_swapped_bytes(
    blocks_to_swap: int,
    dtype: torch.dtype,
    *,
    config: Krea2Config = KREA2_CONFIG,
) -> int:
    """换出层的权重字节数（**pinned 内存预算专用**）。

    这里按计算 dtype 估，对 fp8 底模是高估 —— 对 pinned 预算而言高估是安全方向
    （提前拒绝，不会让用户在锁定内存上翻车）。显存折扣不能用它，见
    ``swapped_param_ratio`` 的说明。
    """
    if blocks_to_swap <= 0:
        return 0
    swapped, _total = _swapped_param_counts(blocks_to_swap, config)
    return swapped * torch.empty(0, dtype=dtype).element_size()


def _swapped_bytes_from_checkpoint(
    checkpoint: Path,
    prefixes: tuple[str, ...],
    normalized_to_source: dict,
) -> int:
    """换出层在 checkpoint 里的**实际**字节数（只读 header，不加载数据）。

    必须用实际 dtype 而非计算 dtype：fp8 checkpoint 只有 bf16 的一半，按 bf16
    估会把 28 层算成 22.6GB（实际 11.3GB），在 37.5GB 内存的机器上撞 60% 安全
    线被**误拒** —— 恰好挡死 B12 的目标配置。高估在这里不是保守，是假阴性。

    header 读不出来时返回 0，由调用方回退到按计算 dtype 的估算。
    """
    total = 0
    try:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            for normalized, source_key in normalized_to_source.items():
                if not normalized.startswith(prefixes):
                    continue
                slice_ = handle.get_slice(source_key)
                numel = 1
                for dim in slice_.get_shape():
                    numel *= dim
                total += numel * _dtype_size(slice_.get_dtype())
    except Exception:  # noqa: BLE001
        return 0
    return total


#: safetensors dtype 字符串 → 字节宽度（header 里是字符串不是 torch.dtype）
_DTYPE_BYTES = {
    "F64": 8, "I64": 8,
    "F32": 4, "I32": 4,
    "F16": 2, "BF16": 2, "I16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}


def _dtype_size(name: str) -> int:
    return _DTYPE_BYTES.get(str(name).upper(), 2)


def _check_swap_budget(
    config: Krea2Config,
    blocks_to_swap: int,
    dtype: torch.dtype,
    normalized_to_source: dict,
    checkpoint: Path,
) -> None:
    """换出层落 pinned 之前先过内存预算护栏（B6：失败即报错，不静默降级）。

    在**任何权重读取之前**调用 —— 失败要 fail-fast，不能等搬了一半才炸。
    """
    from training.sysmem import check_pinned_budget

    prefixes = _swapped_block_prefixes(config, blocks_to_swap)
    need = _swapped_bytes_from_checkpoint(checkpoint, prefixes, normalized_to_source)
    if need <= 0:  # header 读不出：回退到按计算 dtype 估
        need = estimate_swapped_bytes(blocks_to_swap, dtype, config=config)
    check_pinned_budget(need, blocks=blocks_to_swap)


def load_krea2_model(
    path: str | Path,
    device: str | torch.device,
    dtype: torch.dtype,
    *,
    config: Krea2Config = KREA2_CONFIG,
    purpose: str = "train",
    blocks_to_swap: int = 0,
) -> SingleStreamDiT:
    """Strict-load a single-file Krea2 checkpoint into a frozen meta-created model.

    fp8 权重（纯 cast 与 fp8_scaled 两种形态）推理与训练都接受：fp8 张量
    原样常驻显存，Linear 前向逐层 dequant 到 compute dtype（ComfyUI parity，
    见 quant_fp8）。训练即 kohya/musubi 生态的 fp8_base 语义——底模 frozen
    无梯度，LoRA 参数全精度；显存收益依赖 grad checkpointing（dequant 临时
    权重随重算段释放），该约束由 trainer 启动期校验强制（phases/models）。
    ``blocks_to_swap`` > 0 时（block swap，见 docs/design/block-swap.md）：末尾
    N 层的权重**直接载到 CPU pinned**，不经过显存 —— 这是「12/16GB 消费级卡
    跑 K2」的前提，若先全量上卡再搬下来，峰值仍等于完整模型。落地后由
    ``training.block_swap.PinnedBlockSwap`` 就地接管这批 CPU 张量。

    ``purpose`` 当前不影响加载行为，保留作调用面语义标注。
    """
    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError(f"Krea2 loader 不支持 dtype={dtype}")
    target_device = torch.device(device)
    if target_device.type == "meta":
        raise ValueError("Krea2 loader 的目标 device 不能是 meta")
    allow_fp8 = True

    model, expected_shapes = _expected_state(config)
    _, normalized_to_source, scale_to_source = _inspect(
        path,
        config,
        expected_shapes=expected_shapes,
        allow_fp8=allow_fp8,
    )

    state_dict = {}
    fp8_scales: dict[str, torch.Tensor | None] = {}
    checkpoint = _checkpoint_path(path)

    swapped_prefixes = _swapped_block_prefixes(config, blocks_to_swap)
    if swapped_prefixes:
        _check_swap_budget(
            config, blocks_to_swap, dtype, normalized_to_source, checkpoint,
        )

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for normalized, source_key in normalized_to_source.items():
            tensor = handle.get_tensor(source_key)
            if not tensor.is_floating_point():
                raise ValueError(
                    f"Krea2 checkpoint 含非浮点参数 {source_key}: {tensor.dtype}"
                )
            # 被换出的层不上卡：直接落 CPU pinned（见 docstring）
            swapped = normalized.startswith(swapped_prefixes) if swapped_prefixes else False
            if allow_fp8 and tensor.dtype in (
                torch.float8_e4m3fn, torch.float8_e5m2,
            ):
                # fp8 原样常驻（显存收益所在），不 cast dtype
                state_dict[normalized] = (
                    tensor.pin_memory() if swapped
                    else tensor.to(device=target_device)
                )
                if normalized.endswith(".weight"):
                    layer = normalized[: -len(".weight")]
                    fp8_scales.setdefault(layer, None)
            elif swapped:
                state_dict[normalized] = tensor.to(dtype=dtype).pin_memory()
            else:
                state_dict[normalized] = tensor.to(
                    device=target_device,
                    dtype=dtype,
                )
        for layer, source_key in scale_to_source.items():
            fp8_scales[layer] = handle.get_tensor(source_key).to(
                device=target_device,
            )

    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    model.requires_grad_(False)
    if fp8_scales:
        # scale 恒放计算设备：换出层的权重此刻在 CPU，跟随它会导致前向 device
        # 不匹配（见 patch_fp8_linears docstring）
        patch_fp8_linears(model, fp8_scales, device=target_device)
    return model
