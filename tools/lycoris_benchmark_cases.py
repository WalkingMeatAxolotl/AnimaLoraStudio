"""Architecture-derived synthetic Linear fixtures; no model assets or algorithms copied."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from tools.benchmark_lycoris import (
    activate_backend, eager_backend, metadata, read_json, runtime_backend, write_json,
)

TOLERANCES = {"float32": {"atol": 1e-5, "rtol": 1e-5},
              "bfloat16": {"atol": 0.02, "rtol": 0.02}}


@dataclass(frozen=True)
class LayerCase:
    profile: str
    module_path: str
    in_features: int
    out_features: int
    bias: bool
    input_shape: tuple[int, ...]
    layout: str
    provenance: str
    source: str


def discover_cases(profile: str, resolution: int, text_tokens: int) -> list[LayerCase]:
    if profile == "cpu-smoke":
        return [LayerCase(profile, "blocks.0.self_attn.q_proj", 32, 32, False,
                          (1, 4, 32), "BNC", "synthetic_smoke", "explicit tiny fixture")]
    if profile not in {"anima-2048", "anima-5120"}:
        raise ValueError("unknown layer profile")
    from modeling.anima.cosmos_predict2_modeling import Block
    from training.families.latent_spaces import WAN21_F8C16

    channels = int(profile.split("-")[1])
    # Same supported channel/head/context configuration as the Anima loader.
    # Only meta parameters: never materialize a full block/model or load assets.
    with torch.device("meta"):
        block = Block(channels, 1024, channels // 128, use_adaln_lora=True,
                      self_attention_backend="torch", cross_attention_backend="torch")
    side = resolution // WAN21_F8C16.spatial_stride // 2
    selected = ["self_attn.q_proj", "cross_attn.k_proj", "mlp.layer1", "mlp.layer2"]
    cases = []
    for path in selected:
        layer = block.get_submodule(path)
        if path.startswith("mlp"):
            shape, layout = (1, 1, side, side, layer.in_features), "BTHWC"
        else:
            tokens = text_tokens if path.startswith("cross") else side * side
            shape, layout = (1, tokens, layer.in_features), "BNC"
        cases.append(LayerCase(
            profile, f"blocks.0.{path}", layer.in_features, layer.out_features,
            layer.bias is not None, shape, layout, "architecture_derived_synthetic",
            "modeling/anima/cosmos_predict2_modeling.py:Block; "
            "runtime/training/families/anima/loader.py; WAN21_F8C16/patch2",
        ))
    return cases


def build_fixture(case: LayerCase, config: dict, device: str, *, backend: str | None = None):
    activate_backend(config, config["algorithm"], device, override=backend)
    from training.families.anima.preset import ANIMA_PRESET
    from utils.lycoris_adapter import LycorisAdapter

    dtype = getattr(torch, config["dtype"])
    torch.manual_seed(config["seed"])
    host = nn.Module()
    parent = host
    parts = case.module_path.split(".")
    for part in parts[:-1]:
        child = nn.Module()
        parent.add_module(part, child)
        parent = child
    layer = nn.Linear(case.in_features, case.out_features, bias=case.bias)
    parent.add_module(parts[-1], layer)
    host.to(device=device, dtype=dtype).requires_grad_(False)
    adapter = LycorisAdapter(preset=ANIMA_PRESET, algo=config["algorithm"],
                             rank=config["rank"], alpha=config["alpha"], factor=config["factor"])
    modules = adapter.inject(host)
    if len(modules) != 1 or not list(adapter.network.parameters()):
        raise ValueError("fixture must inject exactly one adapter")
    # Synthetic fixtures only: make every multiplicative factor active rather
    # than comparing the base model with a default zero-up adapter.
    with torch.no_grad():
        for parameter in adapter.network.parameters():
            if parameter.requires_grad:
                parameter.uniform_(-0.1, 0.1)
    x = torch.randn(case.input_shape, device=device, dtype=dtype).requires_grad_(True)
    cotangent = torch.randn((*case.input_shape[:-1], case.out_features), device=device, dtype=dtype)
    return host, layer, adapter, x, cotangent


def adapter_metadata(adapter) -> list[dict]:
    return [{
        "name": module.lora_name,
        "bypass": bool(getattr(module, "bypass_mode", False)),
        "rank": getattr(module, "lora_dim", None),
        "use_w2": getattr(module, "use_w2", None),
        "scale": float(getattr(module, "scale", 1.0)),
        "parameters": {name: {"shape": list(p.shape), "dtype": str(p.dtype),
                               "requires_grad": p.requires_grad}
                       for name, p in module.named_parameters()},
    } for module in adapter.network.loras]


def vector_jacobian(layer, adapter, x, cotangent, *, validate=True) -> dict[str, torch.Tensor]:
    adapter.network.zero_grad(set_to_none=True)
    x.grad = None
    y = layer(x)
    (y.float() * cotangent.float()).sum().backward()
    values = {"output": y.detach(), "input_gradient": x.grad}
    for name, p in adapter.network.named_parameters():
        if p.requires_grad:
            values[f"gradient.{name}"] = p.grad
    if validate:
        validate_values(values)
    return values


def validate_values(values: dict) -> None:
    for value in values.values():
        if value is None or not torch.isfinite(value).all() or not torch.count_nonzero(value):
            raise ValueError("nonfinite, absent or degenerate synthetic tensor")


def tensor_inventory(values: dict) -> dict:
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype), "stride": list(v.stride())}
            for k, v in values.items()}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_reference(case, config, fixture, directory: Path, *, backend: dict | None = None) -> Path:
    _, layer, adapter, x, cotangent = fixture
    source_backend = backend or eager_backend()
    if source_backend.get("resolved") != "torch":
        raise ValueError("reference source must use the torch backend")
    source_backend = {key: source_backend[key] for key in ("version", "requested", "resolved")}
    values = vector_jacobian(layer, adapter, x, cotangent)
    with torch.no_grad():
        base = torch.nn.functional.linear(x, layer.weight, layer.bias)
        if not torch.count_nonzero(values["output"] - base):
            raise ValueError("adapter has no observable synthetic effect")
    values.update({"input": x, "cotangent": cotangent, "base.weight": layer.weight})
    if layer.bias is not None:
        values["base.bias"] = layer.bias
    values.update({f"adapter.{k}": v for k, v in adapter.network.state_dict().items()})
    values = {k: v.detach().cpu().contiguous().clone() for k, v in values.items()}
    directory.mkdir(parents=True, exist_ok=False)
    payload = directory / "tensors.safetensors"
    save_file(values, str(payload))
    manifest = {
        "schema_version": 1, "kind": "synthetic_layer_reference", "backend": source_backend,
        "source_environment": metadata(str(x.device), source_backend),
        "case": asdict(case), "experiment": config,
        "adapter": adapter_metadata(adapter), "tolerance": TOLERANCES[config["dtype"]],
        "loss": "sum(output.float * cotangent.float)", "file": payload.name,
        "sha256": file_hash(payload), "tensors": tensor_inventory(values),
    }
    path = directory / "reference-manifest.json"
    write_json(path, manifest)
    return path


def replay_reference(path: Path, device: str, *, backend: str = "torch") -> dict:
    from tools.benchmark_lycoris import validate_config

    manifest = read_json(path)
    keys = {"schema_version", "kind", "backend", "source_environment", "case", "experiment", "adapter",
            "tolerance", "loss", "file", "sha256", "tensors"}
    if set(manifest) != keys or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported reference manifest")
    if (manifest["kind"] != "synthetic_layer_reference" or manifest["file"] != "tensors.safetensors"
            or manifest["loss"] != "sum(output.float * cotangent.float)"):
        raise ValueError("invalid reference kind, loss or payload path")
    config = manifest["experiment"]
    if not isinstance(config, dict) or "algorithm" not in config:
        raise ValueError("invalid reference experiment")
    validate_config({k: v for k, v in config.items() if k != "algorithm"})
    if config["algorithm"] not in config["algorithms"]:
        raise ValueError("invalid reference algorithm")
    candidates = discover_cases(config["profile"], config["resolution"], config["text_tokens"])
    case = next((c for c in candidates if json.loads(json.dumps(asdict(c))) == manifest["case"]), None)
    if case is None or manifest["tolerance"] != TOLERANCES[config["dtype"]]:
        raise ValueError("unknown shape/layout or tolerance")
    if manifest["backend"] != {"version": "4.0.0", "requested": "torch", "resolved": "torch"}:
        raise ValueError("reference source must be LyCORIS 4.0.0 torch")
    if backend not in {"torch", "triton"}:
        raise ValueError("replay backend must be torch or triton")
    payload = path.parent / manifest["file"]
    if payload.is_symlink() or file_hash(payload) != manifest["sha256"]:
        raise ValueError("reference hash mismatch")
    tensors = load_file(str(payload), device="cpu")
    if tensor_inventory(tensors) != manifest["tensors"]:
        raise ValueError("reference inventory mismatch")
    validate_values({k: v for k, v in tensors.items() if k in {"input", "cotangent", "output", "input_gradient"}
                     or k.startswith("gradient.")})
    fixture = build_fixture(case, config, device, backend=backend)
    target_backend = runtime_backend(backend)
    _, layer, adapter, x, cotangent = fixture
    expected = {"input": x, "cotangent": cotangent, "base.weight": layer.weight,
                "output": cotangent, "input_gradient": x}
    if layer.bias is not None:
        expected["base.bias"] = layer.bias
    expected.update({f"adapter.{k}": v for k, v in adapter.network.state_dict().items()})
    expected.update({f"gradient.{k}": p for k, p in adapter.network.named_parameters() if p.requires_grad})
    if tensor_inventory({k: v.detach().cpu().contiguous() for k, v in expected.items()}) != manifest["tensors"]:
        raise ValueError("reference full key/shape/dtype mismatch")
    actual_adapter = adapter_metadata(adapter)
    source_adapter = manifest["adapter"]
    without_bypass = lambda rows: [{k: v for k, v in row.items() if k != "bypass"} for row in rows]
    if without_bypass(actual_adapter) != without_bypass(source_adapter):
        raise ValueError("reference adapter path mismatch")
    if backend == "torch" and actual_adapter != source_adapter:
        raise ValueError("torch replay adapter metadata mismatch")
    if backend == "triton" and (config["algorithm"] not in {"lora", "loha"}
                                or not all(row["bypass"] for row in actual_adapter)):
        raise ValueError("Triton replay must use LoRA/LoHa bypass")
    with torch.no_grad():
        layer.weight.copy_(tensors["base.weight"])
        if layer.bias is not None:
            layer.bias.copy_(tensors["base.bias"])
        x.copy_(tensors["input"])
        cotangent.copy_(tensors["cotangent"])
    adapter.network.load_state_dict({k[8:]: v for k, v in tensors.items() if k.startswith("adapter.")}, strict=True)
    actual = vector_jacobian(layer, adapter, x, cotangent)
    errors = {}
    for key, value in actual.items():
        reference = tensors[key].to(device)
        if not torch.allclose(value, reference, **manifest["tolerance"]):
            raise ValueError("reference numerical mismatch")
        difference = (value.float() - reference.float()).abs()
        errors[key] = {"max_abs": difference.max().item(),
                       "rms": difference.square().mean().sqrt().item(),
                       "max_rel": (difference / reference.float().abs().clamp_min(1e-8)).max().item()}
    with torch.no_grad():
        base = torch.nn.functional.linear(x, layer.weight, layer.bias)
        if not torch.count_nonzero(actual["output"] - base):
            raise ValueError("replayed adapter is degenerate")
    return {"schema_version": 1, "status": "passed", "kind": manifest["kind"],
            "source_backend": manifest["backend"], "target_backend": target_backend, "errors": errors}
