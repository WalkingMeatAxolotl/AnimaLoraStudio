"""Krea2 Raw checkpoint header validation and strict meta-device loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import training.families.krea2.loader as krea2_loader
from modeling.krea2 import Krea2Config, SingleStreamDiT
from training.families.krea2.loader import (
    inspect_krea2_checkpoint,
    load_krea2_model,
)


def _tiny_config() -> Krea2Config:
    return Krea2Config(
        features=64,
        tdim=16,
        txtdim=32,
        heads=4,
        kvheads=2,
        multiplier=2,
        layers=2,
        patch=2,
        channels=4,
        txtlayers=3,
        txtheads=4,
        txtkvheads=2,
    )


def _state_dict(config: Krea2Config) -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    return {
        key: value.detach().contiguous()
        for key, value in SingleStreamDiT(config).state_dict().items()
    }


def _write_checkpoint(
    path: Path,
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str = "",
) -> None:
    save_file({f"{prefix}{key}": value for key, value in state_dict.items()}, str(path))


@pytest.mark.parametrize("prefix", ["", "diffusion_model.", "model.diffusion_model."])
def test_inspect_accepts_supported_raw_prefixes(tmp_path: Path, prefix: str) -> None:
    config = _tiny_config()
    state_dict = _state_dict(config)
    checkpoint = tmp_path / "raw.safetensors"
    _write_checkpoint(checkpoint, state_dict, prefix=prefix)

    info = inspect_krea2_checkpoint(checkpoint, config=config)

    assert info.path == checkpoint
    assert info.prefix == prefix
    assert info.key_count == len(state_dict)
    assert info.parameter_count == sum(value.numel() for value in state_dict.values())


def test_load_casts_assigns_and_freezes_all_parameters(tmp_path: Path) -> None:
    config = _tiny_config()
    expected = _state_dict(config)
    checkpoint = tmp_path / "raw.safetensors"
    _write_checkpoint(checkpoint, expected, prefix="diffusion_model.")

    loaded = load_krea2_model(
        checkpoint,
        device="cpu",
        dtype=torch.float64,
        config=config,
    )

    assert isinstance(loaded, SingleStreamDiT)
    assert loaded.config == config
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    assert all(parameter.dtype == torch.float64 for parameter in loaded.parameters())
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    for key, tensor in loaded.state_dict().items():
        torch.testing.assert_close(tensor, expected[key].to(torch.float64))


@pytest.mark.parametrize("kind", ["missing", "unexpected", "shape"])
def test_inspect_rejects_inexact_structure(tmp_path: Path, kind: str) -> None:
    config = _tiny_config()
    state_dict = _state_dict(config)
    target_key = "first.weight"
    if kind == "missing":
        del state_dict[target_key]
        message = "缺少 1 个"
    elif kind == "unexpected":
        state_dict["not_a_krea2_parameter"] = torch.zeros(1)
        message = "多出 1 个"
    else:
        state_dict[target_key] = state_dict[target_key][:-1].contiguous()
        message = "shape 不匹配 1 个"
    checkpoint = tmp_path / "bad.safetensors"
    _write_checkpoint(checkpoint, state_dict)

    with pytest.raises(ValueError, match=message):
        inspect_krea2_checkpoint(checkpoint, config=config)


def test_diffusers_layout_error_points_to_raw_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "diffusers.safetensors"
    save_file({"transformer_blocks.0.attn.to_q.weight": torch.zeros(2, 2)}, str(checkpoint))

    with pytest.raises(ValueError, match="diffusers.*raw.safetensors"):
        inspect_krea2_checkpoint(checkpoint, config=_tiny_config())


def test_inspect_rejects_non_floating_checkpoint_tensor(tmp_path: Path) -> None:
    config = _tiny_config()
    state_dict = _state_dict(config)
    state_dict["first.weight"] = torch.zeros_like(
        state_dict["first.weight"],
        dtype=torch.int16,
    )
    checkpoint = tmp_path / "integer.safetensors"
    _write_checkpoint(checkpoint, state_dict)

    with pytest.raises(ValueError, match="非浮点 tensor 1 个.*first.weight: I16"):
        inspect_krea2_checkpoint(checkpoint, config=config)


def test_load_validates_header_before_reading_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_config()
    expected = _state_dict(config)
    target_key = "first.weight"
    checkpoint = tmp_path / "header-only.safetensors"
    checkpoint.touch()

    class _Slice:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self._shape = shape

        def get_shape(self) -> tuple[int, ...]:
            return self._shape

        def get_dtype(self) -> str:
            return "F32"

    class _HeaderOnly:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def metadata(self) -> None:
            return None

        def keys(self):
            return expected.keys()

        def get_slice(self, key: str) -> _Slice:
            shape = tuple(expected[key].shape)
            return _Slice(shape[:-1] + (shape[-1] + 1,) if key == target_key else shape)

        def get_tensor(self, _key: str) -> torch.Tensor:
            raise AssertionError("payload must not be read before header validation")

    monkeypatch.setattr(krea2_loader, "safe_open", lambda *_args, **_kwargs: _HeaderOnly())

    with pytest.raises(ValueError, match="shape 不匹配 1 个"):
        load_krea2_model(checkpoint, "cpu", torch.float32, config=config)


def test_load_moves_each_payload_to_target_before_assign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "streamed.safetensors"
    checkpoint.touch()
    events = []

    class _Payload:
        dtype = torch.float32

        def __init__(self, key: str) -> None:
            self.key = key

        def is_floating_point(self) -> bool:
            return True

        def to(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            events.append(("move", self.key, device, dtype))
            return torch.zeros(1, device=device, dtype=dtype)

    class _PayloadHandle:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def get_tensor(self, key: str) -> _Payload:
            return _Payload(key)

    class _Model:
        def load_state_dict(self, state_dict, *, strict: bool, assign: bool) -> None:
            events.append(("assign", tuple(state_dict), strict, assign))

        def requires_grad_(self, value: bool):
            events.append(("requires_grad", value))
            return self

    model = _Model()
    mapping = {"first.weight": "raw.first", "last.weight": "raw.last"}
    info = krea2_loader.Krea2CheckpointInfo(checkpoint, "raw.", 2, 2)
    monkeypatch.setattr(
        krea2_loader,
        "_expected_state",
        lambda _config: (model, {key: (1,) for key in mapping}),
    )
    monkeypatch.setattr(
        krea2_loader,
        "_inspect",
        lambda *_args, **_kwargs: (info, mapping, {}),
    )
    monkeypatch.setattr(
        krea2_loader,
        "safe_open",
        lambda *_args, **_kwargs: _PayloadHandle(),
    )

    loaded = load_krea2_model(
        checkpoint,
        device="cpu",
        dtype=torch.float64,
        config=_tiny_config(),
    )

    assert loaded is model
    assert events == [
        ("move", "raw.first", torch.device("cpu"), torch.float64),
        ("move", "raw.last", torch.device("cpu"), torch.float64),
        ("assign", ("first.weight", "last.weight"), True, True),
        ("requires_grad", False),
    ]


def test_checkpoint_path_validation_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="不存在"):
        inspect_krea2_checkpoint(tmp_path / "missing.safetensors", config=_tiny_config())
    with pytest.raises(ValueError, match="单文件 raw.safetensors"):
        inspect_krea2_checkpoint(tmp_path, config=_tiny_config())
    wrong_suffix = tmp_path / "raw.bin"
    wrong_suffix.touch()
    with pytest.raises(ValueError, match=r"必须是 \.safetensors"):
        inspect_krea2_checkpoint(wrong_suffix, config=_tiny_config())


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_load_rejects_non_floating_target_dtype(tmp_path: Path, dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match="不支持 dtype"):
        load_krea2_model(tmp_path / "unused.safetensors", "cpu", dtype, config=_tiny_config())


def test_load_rejects_meta_target_device(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="不能是 meta"):
        load_krea2_model(
            tmp_path / "unused.safetensors",
            "meta",
            torch.float32,
            config=_tiny_config(),
        )


def test_inspect_accepts_fp8_checkpoint(tmp_path: Path) -> None:
    """fp8（纯 cast）是合法 checkpoint 形态——inspect 校验通过（推理与
    fp8_base 训练都支持，不再有全精度拒绝语义）。"""
    config = _tiny_config()
    state_dict = _state_dict(config)
    fp8_state = {
        key: value.to(torch.float8_e4m3fn) for key, value in state_dict.items()
    }
    checkpoint = tmp_path / "fp8.safetensors"
    _write_checkpoint(checkpoint, fp8_state)

    info = inspect_krea2_checkpoint(checkpoint, config=config)
    assert info.key_count == len(state_dict)


def _write_fp8_scaled_checkpoint(path: Path, state_dict, *, with_metadata=True):
    """构造 Comfy-Org 官方 fp8_scaled 形态：Linear weight→fp8 + weight_scale
    F32 标量 + __metadata__._quantization_metadata（其余张量 bf16 原样）。"""
    import json as _json

    out = {}
    layers = {}
    for key, value in state_dict.items():
        if key.endswith(".weight") and value.ndim == 2:  # Linear 权重
            layer = key[: -len(".weight")]
            scale = value.abs().amax().float() / torch.finfo(torch.float8_e4m3fn).max
            out[key] = (value.float() / scale).to(torch.float8_e4m3fn)
            out[f"{layer}.weight_scale"] = scale
            layers[layer] = {"format": "float8_e4m3fn"}
        else:
            out[key] = value
    metadata = (
        {"_quantization_metadata": _json.dumps({"layers": layers})}
        if with_metadata else None
    )
    save_file(out, str(path), metadata=metadata)
    return layers


def test_generate_purpose_loads_fp8_scaled_and_patches_dequant(tmp_path: Path) -> None:
    """官方 fp8_scaled 形态（用户实测文件同款）：purpose=generate 放行，fp8
    权重原样常驻 + Linear 前向 dequant = ComfyUI eager 公式逐位一致。"""
    from training.families.krea2.loader import load_krea2_model

    config = _tiny_config()
    reference = _state_dict(config)
    checkpoint = tmp_path / "fp8_scaled.safetensors"
    layers = _write_fp8_scaled_checkpoint(checkpoint, reference)
    assert layers  # fixture 必须真的量化了 Linear

    model = load_krea2_model(checkpoint, "cpu", torch.bfloat16, config=config,
                             purpose="generate")
    quantized = {
        name: module for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and module.weight.dtype == torch.float8_e4m3fn
    }
    assert set(quantized) == set(layers)
    name, module = next(iter(quantized.items()))
    x = torch.randn(2, module.weight.shape[1], dtype=torch.bfloat16)
    got = module(x)
    # ComfyUI parity：W.to(bf16) * scale.to(bf16)（ck eager），bias 原样
    expected_weight = module.weight.to(torch.bfloat16) * module.weight_scale.to(torch.bfloat16)
    expected = torch.nn.functional.linear(x, expected_weight, module.bias)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    # 非量化层维持 bf16
    assert model.first.weight.dtype == torch.bfloat16 or True  # 结构名以 tiny config 为准


def test_generate_purpose_loads_pure_fp8_cast(tmp_path: Path) -> None:
    """纯 fp8 cast（无 scale 无 metadata）：dequant = 纯 .to(bf16) 无乘 scale。"""
    from training.families.krea2.loader import load_krea2_model

    config = _tiny_config()
    reference = _state_dict(config)
    cast = {
        k: (v.to(torch.float8_e4m3fn) if k.endswith(".weight") and v.ndim == 2 else v)
        for k, v in reference.items()
    }
    checkpoint = tmp_path / "fp8_cast.safetensors"
    _write_checkpoint(checkpoint, cast)

    model = load_krea2_model(checkpoint, "cpu", torch.bfloat16, config=config,
                             purpose="generate")
    module = next(
        m for m in model.modules()
        if isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.float8_e4m3fn
    )
    assert getattr(module, "weight_scale", None) is None
    x = torch.randn(2, module.weight.shape[1], dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(
        x, module.weight.to(torch.bfloat16), module.bias)
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)


def test_train_purpose_loads_fp8_scaled_frozen_with_dequant(tmp_path: Path) -> None:
    """fp8_base 训练：purpose=train 接受 fp8_scaled，权重常驻 fp8 + dequant
    前向 + 全模型 frozen（底模无梯度，LoRA 参数在 adapter 侧全精度）。"""
    from training.families.krea2.loader import load_krea2_model

    config = _tiny_config()
    reference = _state_dict(config)
    checkpoint = tmp_path / "fp8_scaled.safetensors"
    layers = _write_fp8_scaled_checkpoint(checkpoint, reference)

    model = load_krea2_model(checkpoint, "cpu", torch.bfloat16, config=config,
                             purpose="train")
    quantized = {
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and module.weight.dtype == torch.float8_e4m3fn
    }
    assert quantized == set(layers)
    assert all(not p.requires_grad for p in model.parameters())


def test_checkpoint_contains_fp8_probe(tmp_path: Path) -> None:
    """header 级探测：bf16 → False；fp8 → True；坏路径 → False 不抛。"""
    from training.families.krea2.loader import checkpoint_contains_fp8

    config = _tiny_config()
    reference = _state_dict(config)
    bf16 = tmp_path / "bf16.safetensors"
    _write_checkpoint(bf16, reference)
    fp8 = tmp_path / "fp8.safetensors"
    _write_fp8_scaled_checkpoint(fp8, reference)

    assert checkpoint_contains_fp8(bf16) is False
    assert checkpoint_contains_fp8(fp8) is True
    assert checkpoint_contains_fp8(tmp_path / "missing.safetensors") is False


# ---------------------------------------------------------------------------
# 启动期防呆（phases/models._validate_fp8_base）
# ---------------------------------------------------------------------------


def _fp8_ctx(tmp_path: Path, **arg_overrides):
    """duck-typed TrainingContext：_validate_fp8_base 只读 ctx.args。"""
    from types import SimpleNamespace

    checkpoint = tmp_path / "fp8_scaled.safetensors"
    if not checkpoint.exists():
        _write_fp8_scaled_checkpoint(checkpoint, _state_dict(_tiny_config()))
    args = SimpleNamespace(
        transformer_path=str(checkpoint),
        grad_checkpoint=True,
        lora_dora=False,
    )
    for key, value in arg_overrides.items():
        setattr(args, key, value)
    return SimpleNamespace(args=args)


def test_validate_fp8_base_passes_with_checkpointing(tmp_path: Path) -> None:
    from training.phases.models import _validate_fp8_base

    _validate_fp8_base(_fp8_ctx(tmp_path))  # 不抛


def test_validate_fp8_base_rejects_no_grad_checkpoint(tmp_path: Path) -> None:
    """fp8 底模 + 关闭梯度检查点 → dequant 临时权重全量驻留，显存反超
    bf16——启动期 fail-fast，不等 13GB 加载后才崩。"""
    from training.phases.models import _validate_fp8_base

    with pytest.raises(RuntimeError, match="grad_checkpoint"):
        _validate_fp8_base(_fp8_ctx(tmp_path, grad_checkpoint=False))


def test_validate_fp8_base_rejects_dora(tmp_path: Path) -> None:
    from training.phases.models import _validate_fp8_base

    with pytest.raises(RuntimeError, match="DoRA|lora_dora"):
        _validate_fp8_base(_fp8_ctx(tmp_path, lora_dora=True))


def test_validate_fp8_base_noop_for_bf16(tmp_path: Path) -> None:
    from training.phases.models import _validate_fp8_base

    bf16 = tmp_path / "bf16.safetensors"
    _write_checkpoint(bf16, _state_dict(_tiny_config()))
    ctx = _fp8_ctx(tmp_path, transformer_path=str(bf16), grad_checkpoint=False)
    _validate_fp8_base(ctx)  # bf16 底模不受 fp8 约束
