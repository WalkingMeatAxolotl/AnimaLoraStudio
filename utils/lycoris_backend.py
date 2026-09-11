"""LyCORIS v4 kernel backend selection, capability probe and diagnostics.

LyCORIS 4 defaults to ``auto`` and may select its per-op ``torch.compile``
backend on CUDA even when Triton is not installed.  On supported Windows
setups that can fail on the first training forward with ``TritonMissing``.
Studio therefore defaults to the eager ``torch`` backend.

An explicit optional backend is validated in a disposable subprocess before
training imports LyCORIS.  A failed import, compile, CUDA forward/backward,
finite-gradient check or timeout changes only this training process to the safe
``torch`` backend.  The probe is representative rather than exhaustive: real
shape/path failures must still propagate from training.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_NAME = "LYCORIS_KERNEL_BACKEND"
_DEFAULT_BACKEND = "torch"
_VALID_BACKENDS = frozenset({"auto", "triton", "tilelang", "compile", "torch"})
_OPTIONAL_BACKENDS = _VALID_BACKENDS - {_DEFAULT_BACKEND}
_PROBE_RESULT_PREFIX = "__ANIMALORA_LYCORIS_PROBE__="
_PROBE_SCHEMA_VERSION = 1
_DEFAULT_PROBE_TIMEOUT_SECONDS = 120.0
_OPTIONAL_DECISION_REPORTED = False


@dataclass(frozen=True)
class LycorisBackendDecision:
    """One training process's backend preflight decision."""

    requested: str
    configured: str
    resolved: str | None
    probe_resolved: str | None
    status: str
    reason: str | None
    detail: str | None
    available: tuple[str, ...]
    fused: tuple[str, ...]
    duration_ms: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _installed_version() -> str | None:
    try:
        return version("lycoris-lora")
    except PackageNotFoundError:
        return None


def configure_lycoris_backend() -> str:
    """Set the safe Studio default before any LyCORIS module is imported.

    The environment variable is the upstream public configuration surface, so
    an explicit valid value always wins.  Empty values are treated as unset.
    """
    requested = os.environ.get(_ENV_NAME, "").strip().lower()
    if not requested:
        requested = _DEFAULT_BACKEND
    elif requested not in _VALID_BACKENDS:
        valid = ", ".join(sorted(_VALID_BACKENDS))
        raise ValueError(f"{_ENV_NAME} must be one of: {valid}; got {requested!r}")
    os.environ[_ENV_NAME] = requested
    return requested


def _probe_payload(
    *,
    requested: str,
    algorithm: str,
    dtype: str,
    rank: int,
    alpha: float,
    factor: int,
    weight_decompose: bool,
    rs_lora: bool,
    fp8_base: bool,
) -> dict[str, Any]:
    return {
        "schema_version": _PROBE_SCHEMA_VERSION,
        "requested": requested,
        "algorithm": algorithm,
        "dtype": dtype,
        "rank": int(rank),
        "alpha": float(alpha),
        "factor": int(factor),
        "weight_decompose": bool(weight_decompose),
        "rs_lora": bool(rs_lora),
        "fp8_base": bool(fp8_base),
    }


def _run_probe_process(
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    worker = Path(__file__).with_name("_lycoris_probe_worker.py")
    env = os.environ.copy()
    env[_ENV_NAME] = str(payload["requested"])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, str(worker)],
        input=json.dumps(payload, ensure_ascii=True),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout_seconds,
        check=False,
    )


def _parse_probe_result(stdout: str) -> dict[str, Any]:
    matches = [
        line[len(_PROBE_RESULT_PREFIX):]
        for line in stdout.splitlines()
        if line.startswith(_PROBE_RESULT_PREFIX)
    ]
    if len(matches) != 1:
        raise ValueError("probe worker did not emit exactly one result")
    result = json.loads(matches[0])
    required = {
        "schema_version", "ok", "category", "exception_type",
        "resolved", "available", "fused",
    }
    if (
        not isinstance(result, dict)
        or not required.issubset(result)
        or result["schema_version"] != _PROBE_SCHEMA_VERSION
        or type(result["ok"]) is not bool
        or not isinstance(result["available"], list)
        or not isinstance(result["fused"], list)
    ):
        raise ValueError("invalid probe worker result")
    return result


def _decision(
    *,
    requested: str,
    configured: str,
    resolved: str | None,
    status: str,
    probe_resolved: str | None = None,
    reason: str | None = None,
    detail: str | None = None,
    available: tuple[str, ...] = (),
    fused: tuple[str, ...] = (),
    duration_ms: float | None = None,
) -> LycorisBackendDecision:
    return LycorisBackendDecision(
        requested=requested,
        configured=configured,
        resolved=resolved,
        probe_resolved=probe_resolved,
        status=status,
        reason=reason,
        detail=detail,
        available=available,
        fused=fused,
        duration_ms=duration_ms,
    )


def _fallback_decision(
    requested: str,
    *,
    reason: str,
    detail: str | None = None,
    probe_resolved: str | None = None,
    available: tuple[str, ...] = (),
    fused: tuple[str, ...] = (),
    duration_ms: float | None = None,
) -> LycorisBackendDecision:
    global _OPTIONAL_DECISION_REPORTED

    os.environ[_ENV_NAME] = _DEFAULT_BACKEND
    result = _decision(
        requested=requested,
        configured=_DEFAULT_BACKEND,
        resolved=_DEFAULT_BACKEND,
        probe_resolved=probe_resolved,
        status="fallback",
        reason=reason,
        detail=detail,
        available=available,
        fused=fused,
        duration_ms=duration_ms,
    )
    logger.warning(
        "LyCORIS backend probe: version=%s requested_backend=%s "
        "configured_backend=torch resolved_backend=torch "
        "probe_status=fallback reason=%s detail=%s probe_resolved_backend=%s "
        "duration_ms=%s available_backends=%s",
        _installed_version(),
        requested,
        reason,
        detail,
        probe_resolved,
        f"{duration_ms:.1f}" if duration_ms is not None else None,
        available,
    )
    _OPTIONAL_DECISION_REPORTED = True
    return result


def prepare_lycoris_backend(
    *,
    algorithm: str,
    device: str,
    dtype: str,
    rank: int,
    alpha: float,
    factor: int,
    weight_decompose: bool = False,
    rs_lora: bool = False,
    fp8_base: bool = False,
    timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> LycorisBackendDecision:
    """Probe an explicit optional backend before the parent imports LyCORIS.

    No cross-process or on-disk result cache is used: every opted-in training
    task probes its own interpreter, dependency set, CUDA device and adapter
    path.  ``torch`` (including the unset safe default) does not spawn a child.
    """
    global _OPTIONAL_DECISION_REPORTED

    requested = configure_lycoris_backend()
    if requested == _DEFAULT_BACKEND:
        return _decision(
            requested=requested,
            configured=requested,
            resolved=_DEFAULT_BACKEND,
            status="safe_default",
        )
    if requested not in _OPTIONAL_BACKENDS:  # defensive; configure validated it
        return _fallback_decision(requested, reason="invalid_backend")
    if str(device).split(":", 1)[0].lower() != "cuda":
        return _fallback_decision(requested, reason="cuda_unavailable")
    if algorithm not in {"lora", "lokr", "loha", "tlora"}:
        return _fallback_decision(requested, reason="unsupported_algorithm")
    if dtype not in {"float16", "bfloat16", "float32"}:
        return _fallback_decision(requested, reason="unsupported_dtype")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    payload = _probe_payload(
        requested=requested,
        algorithm=algorithm,
        dtype=dtype,
        rank=rank,
        alpha=alpha,
        factor=factor,
        weight_decompose=weight_decompose,
        rs_lora=rs_lora,
        fp8_base=fp8_base,
    )
    started = time.perf_counter()
    try:
        completed = _run_probe_process(payload, timeout_seconds=timeout_seconds)
    except subprocess.TimeoutExpired:
        duration_ms = (time.perf_counter() - started) * 1000.0
        return _fallback_decision(
            requested,
            reason="timeout",
            duration_ms=duration_ms,
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000.0
        return _fallback_decision(
            requested,
            reason="worker_start_failed",
            detail=type(exc).__name__,
            duration_ms=duration_ms,
        )

    duration_ms = (time.perf_counter() - started) * 1000.0
    try:
        response = _parse_probe_result(completed.stdout)
    except Exception as exc:
        return _fallback_decision(
            requested,
            reason="invalid_worker_response",
            detail=type(exc).__name__,
            duration_ms=duration_ms,
        )

    available = tuple(str(item) for item in response["available"])
    fused = tuple(str(item) for item in response["fused"])
    resolved = response["resolved"] if isinstance(response["resolved"], str) else None
    if completed.returncode != 0 or not response["ok"]:
        reason = response["category"] if isinstance(response["category"], str) else "worker_failed"
        detail = (
            response["exception_type"]
            if isinstance(response["exception_type"], str)
            else None
        )
        return _fallback_decision(
            requested,
            reason=reason,
            detail=detail,
            probe_resolved=resolved,
            available=available,
            fused=fused,
            duration_ms=duration_ms,
        )

    result = _decision(
        requested=requested,
        configured=requested,
        resolved=resolved,
        probe_resolved=resolved,
        status="passed",
        available=available,
        fused=fused,
        duration_ms=duration_ms,
    )
    logger.info(
        "LyCORIS backend probe: version=%s requested_backend=%s configured_backend=%s "
        "resolved_backend=%s probe_status=passed duration_ms=%.1f "
        "available_backends=%s fused_backends=%s; this is a representative "
        "preflight, not proof that every training operator uses fused kernels",
        _installed_version(),
        requested,
        requested,
        resolved,
        duration_ms,
        available,
        fused,
    )
    _OPTIONAL_DECISION_REPORTED = True
    return result


def get_lycoris_runtime_info() -> dict[str, Any]:
    """Return version/backend diagnostics without requiring v4 on v3 installs."""
    installed = _installed_version()
    if installed is None:
        return {
            "version": None,
            "requested": os.environ.get(_ENV_NAME),
            "resolved": None,
            "available": (),
            "fused": (),
        }

    requested = os.environ.get(_ENV_NAME, _DEFAULT_BACKEND)
    try:
        from lycoris.kernels import (  # noqa: PLC0415 - optional v4 API
            available_backends,
            fused_backends,
            resolve_backend,
        )
    except ImportError:
        # LyCORIS 3.x has no kernel dispatcher.  The environment setting is
        # intentionally harmless there and keeps the same adapter import path.
        return {
            "version": installed,
            "requested": requested,
            "resolved": "legacy",
            "available": (),
            "fused": (),
        }

    return {
        "version": installed,
        "requested": requested,
        "resolved": resolve_backend(),
        "available": tuple(available_backends()),
        "fused": tuple(fused_backends()),
    }


@lru_cache(maxsize=1)
def log_lycoris_runtime_once() -> None:
    """Log version/backend once unless the optional preflight already did."""
    if _OPTIONAL_DECISION_REPORTED:
        return
    info = get_lycoris_runtime_info()
    logger.info(
        "LyCORIS runtime: version=%s requested_backend=%s resolved_backend=%s "
        "available_backends=%s fused_backends=%s",
        info["version"],
        info["requested"],
        info["resolved"],
        info["available"],
        info["fused"],
    )
