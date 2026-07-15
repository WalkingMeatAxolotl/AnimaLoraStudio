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
        lambda *_args, **_kwargs: (info, mapping),
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
