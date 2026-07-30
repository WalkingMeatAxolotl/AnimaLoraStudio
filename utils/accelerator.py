"""Small, dependency-light PyTorch accelerator probes.

PyTorch intentionally exposes AMD ROCm devices through the ``torch.cuda`` API.
Checking only ``torch.version.cuda`` therefore misclassifies a working ROCm
wheel as CPU-only.  Runtime and Studio use this module as the shared source of
truth while keeping the actual device string (``cuda``) unchanged.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AcceleratorInfo:
    backend: str
    available: bool
    build: str | None
    device_name: str | None
    device_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_accelerator(torch_module=None) -> AcceleratorInfo:
    """Return ``cuda``, ``rocm`` or ``cpu`` without importing torch eagerly."""
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]  # noqa: PLC0415
        except ImportError:
            return AcceleratorInfo("cpu", False, None, None, 0)

    version = getattr(torch_module, "version", None)
    hip_version = getattr(version, "hip", None)
    cuda_version = getattr(version, "cuda", None)
    cuda_api = getattr(torch_module, "cuda", None)
    available = bool(cuda_api is not None and cuda_api.is_available())

    if hip_version:
        backend = "rocm"
        build = f"rocm{hip_version}"
    elif cuda_version:
        backend = "cuda"
        build = f"cu{str(cuda_version).replace('.', '')}"
    else:
        backend = "cpu"
        build = "cpu"

    device_count = 0
    device_name = None
    if available:
        try:
            device_count = int(cuda_api.device_count())
        except Exception:  # noqa: BLE001
            device_count = 1
        try:
            device_name = str(cuda_api.get_device_name(0))
        except Exception:  # noqa: BLE001
            device_name = "?"

    return AcceleratorInfo(backend, available, build, device_name, device_count)


def is_rocm(torch_module=None) -> bool:
    return detect_accelerator(torch_module).backend == "rocm"


def torch_device_type(torch_module=None) -> str:
    """Return the device string expected by torch (ROCm still uses ``cuda``)."""
    info = detect_accelerator(torch_module)
    return "cuda" if info.available and info.backend in {"cuda", "rocm"} else "cpu"


def configure_miopen_cache(base_dir: str | Path, environ=None) -> dict[str, str]:
    """Point Windows MIOpen's writable caches at an application-owned directory.

    Some embedded ROCm distributions resolve MIOpen's default database to a
    protected user-profile directory. Setting both documented runtime paths
    before importing torch makes command-line and Studio-launched training
    behave consistently. Existing user overrides always win.
    """
    env = os.environ if environ is None else environ
    root = Path(env.get("ANIMA_ROCM_CACHE_DIR") or Path(base_dir) / ".cache" / "miopen")
    db_dir = root / "db"
    kernel_dir = root / "kernels"
    db_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("MIOPEN_USER_DB_PATH", str(db_dir))
    env.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(kernel_dir))
    return {
        "MIOPEN_USER_DB_PATH": env["MIOPEN_USER_DB_PATH"],
        "MIOPEN_CUSTOM_CACHE_DIR": env["MIOPEN_CUSTOM_CACHE_DIR"],
    }
