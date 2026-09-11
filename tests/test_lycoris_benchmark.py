"""R1 manifests, eager synthetic references and honest metrics (CPU only)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools import benchmark_lycoris as bench
from tools import lycoris_benchmark_cases as cases

CONFIG = Path(__file__).resolve().parents[1] / "tools/benchmark_configs/lycoris_cpu_smoke_v1.json"


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "torch")
    return bench.validate_config(bench.read_json(CONFIG))


@pytest.mark.parametrize("algorithm", ["lora", "lokr", "loha"])
def test_nonzero_reference_roundtrip(tmp_path, config, algorithm, monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda *a, **k: pytest.fail("eager must not compile"))
    case = cases.discover_cases("cpu-smoke", 512, 512)[0]
    experiment = {**config, "algorithm": algorithm}
    fixture = cases.build_fixture(case, experiment, "cpu")
    manifest = cases.save_reference(case, experiment, fixture, tmp_path / algorithm)
    result = cases.replay_reference(manifest, "cpu")
    assert result["status"] == "passed"
    assert all(value["max_abs"] == 0 for value in result["errors"].values())
    info = bench.read_json(manifest)
    assert info["adapter"][0]["bypass"] == (algorithm == "lora")
    assert info["case"]["provenance"] == "synthetic_smoke"
    assert any(key.startswith("gradient.") for key in info["tensors"])


@pytest.mark.parametrize("profile,channels", [("anima-2048", 2048), ("anima-5120", 5120)])
def test_architecture_derived_shapes_and_layouts(profile, channels):
    discovered = cases.discover_cases(profile, 512, 512)
    assert [(c.in_features, c.out_features) for c in discovered] == [
        (channels, channels), (1024, channels), (channels, 4 * channels), (4 * channels, channels)]
    assert discovered[0].input_shape == (1, 1024, channels)
    assert discovered[1].input_shape == (1, 512, 1024)
    assert discovered[2].input_shape == (1, 1, 32, 32, channels)
    assert all(c.provenance == "architecture_derived_synthetic" for c in discovered)


@pytest.mark.parametrize("value", [{}, [], {"schema_version": 999}, {"unknown": 1}])
def test_reject_empty_malformed_unknown_config(value):
    with pytest.raises((ValueError, TypeError)):
        bench.validate_config(value)


@pytest.mark.parametrize("field,value", [("repeats", 0), ("warmup", 0), ("rank", True), ("dtype", "fp8"),
                                         ("alpha", float("nan")), ("profile", "real_training"),
                                         ("algorithms", []), ("algorithms", ["lora", "lora"])])
def test_bounded_config(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        bench.validate_config(config)


@pytest.mark.parametrize("text", ["", "[]", '{"a":1,"a":2}', '{"a":NaN}', "not json"])
def test_json_fail_closed(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        bench.read_json(path)


@pytest.fixture
def reference(tmp_path, config):
    experiment = {**config, "algorithm": "lora"}
    case = cases.discover_cases("cpu-smoke", 512, 512)[0]
    return cases.save_reference(case, experiment, cases.build_fixture(case, experiment, "cpu"), tmp_path / "ref")


@pytest.mark.parametrize("damage", ["hash", "missing_file", "missing_key", "extra_key", "shape", "nan", "zero",
                                   "output", "unknown", "path", "tolerance", "backend", "layout"])
def test_corruption_rejected(reference, damage):
    info = bench.read_json(reference)
    payload = reference.parent / "tensors.safetensors"
    if damage == "hash":
        info["sha256"] = "bad"
    elif damage == "missing_file":
        payload.unlink()
    elif damage == "unknown":
        info["unknown"] = True
    elif damage == "path":
        info["file"] = "../tensors.safetensors"
    elif damage == "tolerance":
        info["tolerance"]["atol"] = 1000
    elif damage == "backend":
        info["backend"]["resolved"] = "compile"
    elif damage == "layout":
        info["case"]["layout"] = "fake"
    else:
        values = {key: value.clone() for key, value in load_file(str(payload)).items()}
        if damage == "missing_key":
            values.pop(next(k for k in values if k.startswith("gradient.")))
        elif damage == "extra_key":
            values["evil"] = torch.ones(1)
        elif damage == "shape":
            values["input"] = values["input"].reshape(-1)
        elif damage == "nan":
            values["output"].fill_(float("nan"))
        elif damage == "zero":
            values["input_gradient"].zero_()
        elif damage == "output":
            values["output"].add_(10)
        save_file(values, str(payload))
        # A forged self-consistent inventory/hash must still fail against the
        # derived full expected keys/shapes and numerical replay, not just SHA.
        info["sha256"] = cases.file_hash(payload)
        info["tensors"] = cases.tensor_inventory(values)
    bench.write_json(reference, info)
    with pytest.raises((ValueError, FileNotFoundError)):
        cases.replay_reference(reference, "cpu")


@pytest.mark.parametrize("backend", ["compile", "auto", "triton", "tilelang", "unknown"])
def test_backend_fail_closed(monkeypatch, backend):
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", backend)
    with pytest.raises(ValueError, match="torch backend"):
        bench.eager_backend()


def test_resolved_backend_and_version_fail_closed(monkeypatch):
    from utils import lycoris_backend
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "torch")
    for version, resolved in [("3.2.0", "torch"), ("4.0.0", "compile")]:
        monkeypatch.setattr(lycoris_backend, "get_lycoris_runtime_info", lambda: {"version": version, "resolved": resolved})
        with pytest.raises(ValueError):
            bench.eager_backend()


def test_metrics_stats_and_cpu_unavailable():
    assert bench.summarize([1, 2, 8])["median"] == 2
    assert bench.summarize([4])["sample_stdev"]["value"] is None
    assert bench.summarize([4])["sample_stdev"]["unavailable_reason"] == "insufficient_repeats"
    assert all(m["value"] is None for m in bench.memory_metrics("cpu", "test").values())
    assert bench.kernel_measurement(lambda: pytest.fail("CPU must not profile CUDA"), "cpu")["unavailable_reason"] == "cpu_no_cuda"
    with pytest.raises(ValueError):
        bench.metric(None, "s", "clock", "window")
    with pytest.raises(ValueError):
        bench.summarize([float("nan")])


def test_profiler_duration_is_leaf_sum_not_parent_or_memory():
    def event(name, device, duration, children=()):
        return SimpleNamespace(name=name, device_type=device, cpu_children=children,
                               time_range=SimpleNamespace(elapsed_us=lambda: duration))
    cuda, cpu = torch.autograd.DeviceType.CUDA, torch.autograd.DeviceType.CPU
    result = bench.kernel_events_metric([event("kernel_a", cuda, 100), event("kernel_b", cuda, 200),
                                        event("Memcpy HtoD", cuda, 900), event("Memset", cuda, 500),
                                        event("aten::mm", cpu, 300), event("parent", cuda, 300, [1])])
    assert result["value"] == 0.3
    assert "not critical path" in result["scope"]


def test_layer_worker_cpu_result_privacy(tmp_path, config, monkeypatch):
    monkeypatch.setenv("SECRET_CANARY", "PRIVATE_ASSET_CAPTION_CANARY")
    result = bench.layer_worker(config, 0, "lora", "cpu", tmp_path)
    text = json.dumps(result)
    assert "PRIVATE_ASSET_CAPTION_CANARY" not in text
    assert str(tmp_path) not in text
    assert result["training_it_s"]["value"] is None
    assert result["compile_tuning"]["unavailable_reason"] == "not_applicable_eager"
    assert result["fb_seconds"]["n"] == config["measured"]
    assert copy.deepcopy(result) == result


def test_eager_layer_without_optional_kernel_imports(tmp_path):
    import os
    import subprocess
    import sys

    code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'triton', 'tilelang'}:
        raise ImportError('optional kernels deliberately unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('torch.compile must not execute')
torch.compile = forbidden
from tools import benchmark_lycoris as bench
from tools import lycoris_benchmark_cases as cases
config = bench.read_json(bench.REPO / 'tools/benchmark_configs/lycoris_cpu_smoke_v1.json')
config['algorithm'] = 'lora'
case = cases.discover_cases('cpu-smoke', 512, 512)[0]
_, layer, adapter, x, cotangent = cases.build_fixture(case, config, 'cpu')
cases.vector_jacobian(layer, adapter, x, cotangent)
'''
    process = subprocess.run([sys.executable, "-c", code], cwd=bench.REPO, capture_output=True, text=True,
                             env={**os.environ, "LYCORIS_KERNEL_BACKEND": "torch", "PYTHONDONTWRITEBYTECODE": "1"})
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("manifest", [{}, {"schema_version": 0}, {"schema_version": True}, {"unknown": 1}])
def test_empty_unknown_reference_manifest(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    bench.write_json(path, manifest)
    with pytest.raises(ValueError):
        cases.replay_reference(path, "cpu")
