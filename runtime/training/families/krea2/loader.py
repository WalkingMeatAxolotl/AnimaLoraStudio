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
# fp8 是运行时量化技术（load-time 量化 + Linear forward patch + scale buffer），
# 不是 checkpoint 交换格式——社区的「纯 fp8 cast」权重若静默 upcast 回 bf16，
# 显存零收益还丢精度（A7'/C13）。显式报错并指路 bf16 版本。
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
) -> tuple[Krea2CheckpointInfo, dict[str, str]]:
    checkpoint = _checkpoint_path(path)
    if expected_shapes is None:
        _, expected_shapes = _expected_state(config)
    expected_keys = set(expected_shapes)

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        prefix = _choose_prefix(source_keys, expected_keys)
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
        if fp8_keys:
            raise ValueError(
                f"Krea2 checkpoint 含 fp8 参数（{len(fp8_keys)} 个，如 "
                f"{fp8_keys[0]}）。fp8 是推理端运行时量化格式，本训练器不支持"
                f"（静默转回 bf16 显存零收益还丢精度）——请下载 bf16 版本权重"
                f"（官方 Raw/Turbo 均为 bf16）。"
            )
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
    """Validate keys and shapes from the safetensors header without reading payloads."""
    info, _ = _inspect(path, config)
    return info


def load_krea2_model(
    path: str | Path,
    device: str | torch.device,
    dtype: torch.dtype,
    *,
    config: Krea2Config = KREA2_CONFIG,
) -> SingleStreamDiT:
    """Strict-load a single-file Krea2 checkpoint into a frozen meta-created model."""
    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError(f"Krea2 loader 不支持 dtype={dtype}")
    target_device = torch.device(device)
    if target_device.type == "meta":
        raise ValueError("Krea2 loader 的目标 device 不能是 meta")

    model, expected_shapes = _expected_state(config)
    _, normalized_to_source = _inspect(
        path,
        config,
        expected_shapes=expected_shapes,
    )

    state_dict = {}
    checkpoint = _checkpoint_path(path)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for normalized, source_key in normalized_to_source.items():
            tensor = handle.get_tensor(source_key)
            if not tensor.is_floating_point():
                raise ValueError(
                    f"Krea2 checkpoint 含非浮点参数 {source_key}: {tensor.dtype}"
                )
            state_dict[normalized] = tensor.to(
                device=target_device,
                dtype=dtype,
            )

    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    model.requires_grad_(False)
    return model
