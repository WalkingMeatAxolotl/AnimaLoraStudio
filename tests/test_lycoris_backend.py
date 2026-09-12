"""LyCORIS v4 backend policy and isolated preflight regression tests."""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from utils import lycoris_backend


def test_adapter_registry_does_not_import_lycoris_before_preflight() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(root), str(root / "runtime")))
    code = (
        "import sys; import training.adapters; "
        "assert 'lycoris' not in sys.modules; "
        "assert 'lycoris.kernels' not in sys.modules; "
        "assert 'utils.lycoris_backend' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_backend_defaults_to_eager_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LYCORIS_KERNEL_BACKEND", raising=False)

    assert lycoris_backend.configure_lycoris_backend() == "torch"
    assert lycoris_backend.os.environ["LYCORIS_KERNEL_BACKEND"] == "torch"


def test_explicit_backend_override_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "triton")

    assert lycoris_backend.configure_lycoris_backend() == "triton"
    assert lycoris_backend.os.environ["LYCORIS_KERNEL_BACKEND"] == "triton"


def test_explicit_config_backend_overrides_ambient_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "auto")

    assert lycoris_backend.configure_lycoris_backend("torch") == "torch"
    assert lycoris_backend.os.environ["LYCORIS_KERNEL_BACKEND"] == "torch"


def test_backend_normalizes_case_and_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "  ToRcH  ")

    assert lycoris_backend.configure_lycoris_backend() == "torch"
    assert lycoris_backend.os.environ["LYCORIS_KERNEL_BACKEND"] == "torch"


def test_invalid_backend_fails_before_lycoris_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "fastest")

    with pytest.raises(ValueError, match="LYCORIS_KERNEL_BACKEND"):
        lycoris_backend.configure_lycoris_backend()


def test_runtime_info_reports_installed_version_and_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "torch")

    info = lycoris_backend.get_lycoris_runtime_info()

    assert info["version"] is not None
    assert info["requested"] == "torch"
    assert info["resolved"] in {"legacy", "torch"}
    assert isinstance(info["available"], tuple)
    assert isinstance(info["fused"], tuple)


def test_runtime_log_contains_version_and_resolved_backend(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "torch")
    monkeypatch.setattr(lycoris_backend, "_OPTIONAL_DECISION_REPORTED", False)
    lycoris_backend.log_lycoris_runtime_once.cache_clear()

    with caplog.at_level(logging.INFO, logger="utils.lycoris_backend"):
        lycoris_backend.log_lycoris_runtime_once()

    assert any(
        "LyCORIS runtime:" in record.message
        and "resolved_backend=" in record.message
        for record in caplog.records
    )


def _response(
    *,
    ok: bool,
    category: str | None = None,
    exception_type: str | None = None,
    resolved: str | None = "triton",
) -> str:
    payload = {
        "schema_version": 1,
        "ok": ok,
        "category": category,
        "exception_type": exception_type,
        "resolved": resolved,
        "available": ["triton", "compile", "torch"],
        "fused": ["triton"],
    }
    return lycoris_backend._PROBE_RESULT_PREFIX + json.dumps(payload) + "\n"


def _prepare(**overrides):
    kwargs = {
        "algorithm": "lora",
        "device": "cuda",
        "dtype": "bfloat16",
        "rank": 32,
        "alpha": 16.0,
        "factor": 8,
    }
    kwargs.update(overrides)
    return lycoris_backend.prepare_lycoris_backend(**kwargs)


def test_safe_default_does_not_start_probe_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LYCORIS_KERNEL_BACKEND", raising=False)
    monkeypatch.setattr(
        lycoris_backend,
        "_run_probe_process",
        lambda *args, **kwargs: pytest.fail("safe torch must not spawn a probe"),
    )

    decision = _prepare()

    assert decision.status == "safe_default"
    assert decision.configured == "torch"


def test_triton_policy_falls_back_before_probe_for_unsupported_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lycoris_backend,
        "_run_probe_process",
        lambda *args, **kwargs: pytest.fail("policy rejection must not spawn a probe"),
    )

    unsupported = _prepare(requested_backend="triton", algorithm="lokr")
    assert unsupported.reason == "unsupported_algorithm"
    assert unsupported.configured == "torch"

    dora = _prepare(requested_backend="triton", weight_decompose=True)
    assert dora.reason == "dora_not_supported"
    assert dora.configured == "torch"

    dropout = _prepare(requested_backend="triton", rank_dropout=0.1)
    assert dropout.reason == "dropout_not_supported"
    assert dropout.configured == "torch"


def test_successful_probe_preserves_requested_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "triton")
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=_response(ok=True), stderr="",
    )
    monkeypatch.setattr(
        lycoris_backend, "_run_probe_process", lambda *args, **kwargs: completed,
    )

    decision = _prepare()

    assert decision.status == "passed"
    assert decision.requested == "triton"
    assert decision.configured == "triton"
    assert decision.resolved == "triton"
    assert decision.available == ("triton", "compile", "torch")
    assert lycoris_backend.os.environ["LYCORIS_KERNEL_BACKEND"] == "triton"


def test_optional_preflight_suppresses_duplicate_runtime_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(lycoris_backend, "_OPTIONAL_DECISION_REPORTED", False)
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "triton")
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=_response(ok=True), stderr="",
    )
    monkeypatch.setattr(
        lycoris_backend, "_run_probe_process", lambda *args, **kwargs: completed,
    )
    with caplog.at_level(logging.INFO, logger="utils.lycoris_backend"):
        assert _prepare().status == "passed"
        lycoris_backend.log_lycoris_runtime_once.cache_clear()
        lycoris_backend.log_lycoris_runtime_once()

    messages = [record.message for record in caplog.records]
    assert sum("LyCORIS backend probe:" in message for message in messages) == 1
    assert not any("LyCORIS runtime:" in message for message in messages)


@pytest.mark.parametrize(
    ("category", "exception_type"),
    [
        ("backend_unavailable", "RuntimeError"),
        ("import_error", "ImportError"),
        ("compile_error", "TritonMissing"),
        ("execution_error", "RuntimeError"),
        ("nonfinite", None),
    ],
)
def test_probe_failure_falls_back_to_torch_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    category: str,
    exception_type: str | None,
) -> None:
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "triton")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout=_response(
            ok=False,
            category=category,
            exception_type=exception_type,
        ),
        stderr="private child diagnostics",
    )
    monkeypatch.setattr(
        lycoris_backend, "_run_probe_process", lambda *args, **kwargs: completed,
    )

    with caplog.at_level(logging.WARNING, logger="utils.lycoris_backend"):
        decision = _prepare()

    assert decision.status == "fallback"
    assert decision.reason == category
    assert decision.detail == exception_type
    assert decision.configured == "torch"
    assert decision.resolved == "torch"
    assert decision.probe_resolved == "triton"
    assert lycoris_backend.os.environ["LYCORIS_KERNEL_BACKEND"] == "torch"
    records = [r for r in caplog.records if "LyCORIS backend probe:" in r.message]
    assert len(records) == 1
    assert "duration_ms=" in records[0].message
    assert "private child diagnostics" not in records[0].message


def test_probe_timeout_falls_back_to_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "compile")

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="probe", timeout=1)

    monkeypatch.setattr(lycoris_backend, "_run_probe_process", _timeout)

    decision = _prepare(timeout_seconds=1)

    assert decision.reason == "timeout"
    assert decision.configured == "torch"
    assert lycoris_backend.os.environ["LYCORIS_KERNEL_BACKEND"] == "torch"


def test_optional_backend_on_cpu_falls_back_without_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "triton")
    monkeypatch.setattr(
        lycoris_backend,
        "_run_probe_process",
        lambda *args, **kwargs: pytest.fail("CPU path must not spawn a CUDA probe"),
    )

    decision = _prepare(device="cpu")

    assert decision.reason == "cuda_unavailable"
    assert decision.configured == "torch"


def test_probe_results_are_not_cached_between_training_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=_response(ok=True), stderr="",
    )

    def _run(*args, **kwargs):
        calls.append((args, kwargs))
        return completed

    monkeypatch.setattr(lycoris_backend, "_run_probe_process", _run)
    for _ in range(2):
        monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "triton")
        assert _prepare().status == "passed"

    assert len(calls) == 2


def test_invalid_worker_response_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LYCORIS_KERNEL_BACKEND", "auto")
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="ordinary output only", stderr="",
    )
    monkeypatch.setattr(
        lycoris_backend, "_run_probe_process", lambda *args, **kwargs: completed,
    )

    decision = _prepare()

    assert decision.reason == "invalid_worker_response"
    assert decision.configured == "torch"


def test_probe_worker_detects_nonfinite_values() -> None:
    torch = pytest.importorskip("torch")
    from utils import _lycoris_probe_worker as worker

    assert worker._check_finite([torch.ones(1)]) is True
    assert worker._check_finite([torch.tensor([float("nan")])]) is False
    assert worker._classify_exception(ImportError()) == "import_error"

    TritonMissing = type("TritonMissing", (RuntimeError,), {"__module__": "torch._inductor"})
    assert worker._classify_exception(TritonMissing()) == "compile_error"


def test_adapter_registry_prepares_only_kernel_backed_algorithms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from training import adapters

    calls = []

    def _prepare_adapter(args, **kwargs):
        calls.append((args.lora_type, kwargs))
        return "decision"

    monkeypatch.setitem(adapters.PREPARERS, "lora", _prepare_adapter)
    result = adapters.prepare_adapter(
        argparse.Namespace(lora_type="lora"),
        device="cuda",
        dtype="bfloat16",
        fp8_base=False,
    )
    skipped = adapters.prepare_adapter(
        argparse.Namespace(lora_type="ortho"),
        device="cuda",
        dtype="bfloat16",
        fp8_base=False,
    )
    missing = adapters.prepare_adapter(
        argparse.Namespace(),
        device="cpu",
        dtype="float32",
        fp8_base=False,
    )

    assert result == "decision"
    assert skipped is None
    assert missing is None
    assert calls == [("lora", {"device": "cuda", "dtype": "bfloat16", "fp8_base": False})]


def test_lycoris_preparer_forwards_effective_training_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from training.adapters import lycoris as plugin

    captured = {}

    def _prepare_backend(**kwargs):
        captured.update(kwargs)
        return "decision"

    monkeypatch.setattr(lycoris_backend, "prepare_lycoris_backend", _prepare_backend)
    args = argparse.Namespace(
        lora_type="lokr",
        lora_rank=48,
        lora_alpha=24.0,
        lokr_factor=12,
        lora_dora=True,
        lora_rs=False,
    )

    assert plugin.prepare(
        args,
        device="cuda",
        dtype=torch.bfloat16,
        fp8_base=True,
    ) == "decision"
    assert captured == {
        "algorithm": "lokr",
        "device": "cuda",
        "dtype": "bfloat16",
        "rank": 48,
        "alpha": 24.0,
        "factor": 12,
        "weight_decompose": True,
        "rs_lora": False,
        "fp8_base": True,
        "requested_backend": None,
        "dropout": 0.0,
        "rank_dropout": 0.0,
        "module_dropout": 0.0,
    }


def test_tlora_preparer_skips_ortho_and_probes_compatibility_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from training.adapters import lycoris as lycoris_plugin
    from training.adapters import tlora as tlora_plugin

    calls = []
    monkeypatch.setattr(
        lycoris_plugin,
        "prepare",
        lambda args, **kwargs: calls.append((args, kwargs)) or "decision",
    )
    ortho_args = argparse.Namespace(lora_type="tlora", tlora_use_ortho=True)
    compat_args = argparse.Namespace(lora_type="tlora", tlora_use_ortho=False)

    assert tlora_plugin.prepare(
        ortho_args, device="cuda", dtype="bfloat16", fp8_base=False,
    ) is None
    assert tlora_plugin.prepare(
        compat_args, device="cuda", dtype="bfloat16", fp8_base=True,
    ) == "decision"
    assert calls == [(
        compat_args,
        {"device": "cuda", "dtype": "bfloat16", "fp8_base": True},
    )]


def test_models_phase_prepares_backend_before_loading_dit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from training import adapters
    from training.phases import models
    from training import sysmem

    events = []
    ctx = types.SimpleNamespace(
        family=types.SimpleNamespace(
            spec=types.SimpleNamespace(text=types.SimpleNamespace(strategy="online")),
        ),
        args=argparse.Namespace(blocks_to_swap=0),
        device="cuda",
        dtype="bfloat16",
    )
    monkeypatch.setattr(models, "_resolve_paths", lambda _ctx: events.append("paths"))
    monkeypatch.setattr(models, "_validate_fp8_base", lambda _ctx: False)
    monkeypatch.setattr(models, "_defer_dit_for_text_cache", lambda _ctx: False)
    monkeypatch.setattr(models, "_swap_vram_discount", lambda _ctx: 0.0)
    monkeypatch.setattr(models, "_load_dit", lambda _ctx: events.append("dit"))
    monkeypatch.setattr(models, "_load_vae", lambda _ctx: events.append("vae"))
    monkeypatch.setattr(models, "_load_text", lambda _ctx: events.append("text"))
    monkeypatch.setattr(models, "_inject_adapter", lambda _ctx: events.append("inject"))
    monkeypatch.setattr(models, "_log_train_start_vram", lambda _ctx: None)
    monkeypatch.setattr(sysmem, "check_load_budget", lambda *args, **kwargs: None)
    monkeypatch.setattr(sysmem, "guard_enabled_from_env", lambda: False)
    monkeypatch.setattr(
        adapters,
        "prepare_adapter",
        lambda *args, **kwargs: events.append("probe"),
    )

    models.run(ctx)

    assert events == ["paths", "probe", "dit", "vae", "text", "inject"]
