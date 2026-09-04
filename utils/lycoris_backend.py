"""LyCORIS v4 kernel backend selection and runtime diagnostics.

LyCORIS 4 defaults to ``auto`` and may select its per-op ``torch.compile``
backend on CUDA even when Triton is not installed.  On supported Windows
setups that can fail on the first training forward with ``TritonMissing``.
Studio therefore defaults to the eager ``torch`` backend until the optional
kernel stack has passed its own capability probe.  An explicit environment
value remains an opt-in override for advanced users and benchmarks.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

logger = logging.getLogger(__name__)

_ENV_NAME = "LYCORIS_KERNEL_BACKEND"
_DEFAULT_BACKEND = "torch"
_VALID_BACKENDS = frozenset({"auto", "triton", "tilelang", "compile", "torch"})


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


def get_lycoris_runtime_info() -> dict[str, Any]:
    """Return version/backend diagnostics without requiring v4 on v3 installs."""
    try:
        installed = version("lycoris-lora")
    except PackageNotFoundError:
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
    """Log the installed version and effective backend once per process."""
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
