"""Isolated CUDA forward/backward probe for an opted-in LyCORIS backend.

This module is an internal subprocess entry point.  The parent sends one JSON
payload on stdin and reads the single prefixed JSON result from stdout.  Running
in a disposable process keeps failed imports/compilation and CUDA state out of
the training process.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

_RESULT_PREFIX = "__ANIMALORA_LYCORIS_PROBE__="
_SCHEMA_VERSION = 1
_VALID_ALGORITHMS = frozenset({"lora", "lokr", "loha", "tlora"})
_VALID_DTYPES = frozenset({"float16", "bfloat16", "float32"})
_EXPECTED_KEYS = {
    "schema_version",
    "requested",
    "algorithm",
    "dtype",
    "rank",
    "alpha",
    "factor",
    "weight_decompose",
    "rs_lora",
    "fp8_base",
}


def _emit(result: dict[str, Any]) -> None:
    print(_RESULT_PREFIX + json.dumps(result, ensure_ascii=True, sort_keys=True), flush=True)


def _failure(category: str, exc: BaseException | None = None, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "ok": False,
        "category": category,
        "exception_type": type(exc).__name__ if exc is not None else None,
        "resolved": extra.pop("resolved", None),
        "available": extra.pop("available", []),
        "fused": extra.pop("fused", []),
    }
    result.update(extra)
    return result


def _classify_exception(exc: BaseException) -> str:
    if isinstance(exc, (ImportError, ModuleNotFoundError, OSError)):
        return "import_error"
    qualified = f"{type(exc).__module__}.{type(exc).__name__}".lower()
    if any(token in qualified for token in ("triton", "inductor", "compile", "dynamo")):
        return "compile_error"
    return "execution_error"


def _read_payload() -> dict[str, Any]:
    payload = json.loads(sys.stdin.read())
    if not isinstance(payload, dict) or set(payload) != _EXPECTED_KEYS:
        raise ValueError("invalid probe payload keys")
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("unsupported probe payload schema")
    if payload["algorithm"] not in _VALID_ALGORITHMS:
        raise ValueError("unsupported probe algorithm")
    if payload["dtype"] not in _VALID_DTYPES:
        raise ValueError("unsupported probe dtype")
    if payload["requested"] not in {"auto", "triton", "tilelang", "compile"}:
        raise ValueError("probe requires an optional backend request")
    for name in ("rank", "factor"):
        if type(payload[name]) is not int or payload[name] <= 0:
            raise ValueError(f"invalid {name}")
    if not isinstance(payload["alpha"], (int, float)):
        raise ValueError("invalid alpha")
    for name in ("weight_decompose", "rs_lora", "fp8_base"):
        if type(payload[name]) is not bool:
            raise ValueError(f"invalid {name}")
    return payload


def _check_finite(values: list[Any]) -> bool:
    import torch

    return bool(values) and all(
        value is not None and bool(torch.isfinite(value).all()) for value in values
    )


def run_probe(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one representative production-wrapper path on CUDA."""
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except Exception as exc:  # pragma: no cover - exercised in parent response tests
        return _failure("import_error", exc)

    if not torch.cuda.is_available():
        return _failure("cuda_unavailable")

    requested = str(payload["requested"])
    try:
        from utils.lycoris_backend import get_lycoris_runtime_info

        runtime = get_lycoris_runtime_info()
        resolved = runtime.get("resolved")
        available = list(runtime.get("available") or ())
        fused = list(runtime.get("fused") or ())
    except Exception as exc:
        return _failure("backend_unavailable", exc)

    if requested != "auto" and resolved != requested:
        return _failure(
            "resolved_mismatch",
            resolved=resolved,
            available=available,
            fused=fused,
        )

    dtype = getattr(torch, str(payload["dtype"]))
    device = torch.device("cuda")
    try:
        from utils.lycoris_adapter import LycorisAdapter

        class ProbeHost(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Keep a normal floating parameter first so the production
                # adapter selects the requested compute dtype for FP8 bases.
                self.anchor = nn.Parameter(
                    torch.zeros(1, device=device, dtype=dtype),
                    requires_grad=False,
                )
                self.probe = nn.Linear(
                    64, 64, bias=False, device=device, dtype=dtype,
                ).requires_grad_(False)

        host = ProbeHost()
        if payload["fp8_base"]:
            fp8_dtype = torch.float8_e4m3fn
            host.probe.weight = nn.Parameter(
                host.probe.weight.detach().to(fp8_dtype), requires_grad=False,
            )

            def _fp8_forward(module, value):
                return F.linear(value, module.weight.to(dtype=value.dtype), None)

            host.probe.forward = types.MethodType(_fp8_forward, host.probe)

        preset = {
            "enable_conv": False,
            "target_module": [],
            "target_name": ["probe"],
            "exclude_name": [],
            "use_fnmatch": True,
            "lora_prefix": "lora_unet",
            "module_algo_map": {},
            "name_algo_map": {},
        }
        adapter = LycorisAdapter(
            preset=preset,
            algo=str(payload["algorithm"]),
            rank=min(int(payload["rank"]), 64),
            alpha=float(payload["alpha"]),
            factor=min(int(payload["factor"]), 32),
            weight_decompose=bool(payload["weight_decompose"]),
            rs_lora=bool(payload["rs_lora"]),
        )
        modules = adapter.inject(host)
        if len(modules) != 1 or adapter.network is None:
            return _failure(
                "adapter_injection_failed",
                resolved=resolved,
                available=available,
                fused=fused,
            )

        with torch.no_grad():
            for parameter in adapter.network.parameters():
                if parameter.requires_grad and parameter.is_floating_point():
                    parameter.uniform_(-0.05, 0.05)

        value = torch.randn(
            2, 8, 64, device=device, dtype=dtype, requires_grad=True,
        )
        output = host.probe(value)
        output.float().square().mean().backward()
        torch.cuda.synchronize(device)
        gradients = [
            parameter.grad
            for parameter in adapter.network.parameters()
            if parameter.requires_grad
        ]
        if not gradients or not _check_finite([output, value.grad, *gradients]):
            return _failure(
                "nonfinite",
                resolved=resolved,
                available=available,
                fused=fused,
            )
        return {
            "schema_version": _SCHEMA_VERSION,
            "ok": True,
            "category": None,
            "exception_type": None,
            "resolved": resolved,
            "available": available,
            "fused": fused,
            "probe_rank": min(int(payload["rank"]), 64),
            "probe_factor": min(int(payload["factor"]), 32),
        }
    except Exception as exc:  # isolated failure becomes a parent-side fallback
        return _failure(
            _classify_exception(exc),
            exc,
            resolved=resolved,
            available=available,
            fused=fused,
        )


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        payload = _read_payload()
        result = run_probe(payload)
    except Exception as exc:
        result = _failure("invalid_payload", exc)
    _emit(result)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
