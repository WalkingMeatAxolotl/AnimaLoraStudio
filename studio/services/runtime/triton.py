"""Managed Triton installation for the experimental LyCORIS kernel path.

The supported surface is deliberately narrow: one audited package/version per
platform, a CUDA-enabled Torch 2.11 runtime, and no dependency resolution that
could replace the project's Torch build.  Training still performs the isolated
LyCORIS capability probe before using Triton.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TritonPackageSpec:
    package: str
    version: str


# Torch 2.11 uses Triton 3.8.  Windows needs the community wheel that exposes
# the same ``triton`` import package; Linux uses the official PyPI distribution.
_PINS: dict[tuple[str, str], TritonPackageSpec] = {
    ("windows", "x86_64"): TritonPackageSpec("triton-windows", "3.8.0.post28"),
    ("linux", "x86_64"): TritonPackageSpec("triton", "3.8.0"),
}
_KNOWN_PACKAGES = ("triton", "triton-windows")
_SUPPORTED_TORCH = (2, 11)
_SUPPORTED_PYTHON_MIN = (3, 10)
_SUPPORTED_PYTHON_MAX = (3, 14)
_RESTART_REQUIRED = False


def _normalized_platform() -> tuple[str, str]:
    system = platform.system().strip().lower()
    machine = platform.machine().strip().lower()
    if machine in {"amd64", "x64"}:
        machine = "x86_64"
    return system, machine


def _package_spec() -> TritonPackageSpec | None:
    return _PINS.get(_normalized_platform())


def _installed_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in _KNOWN_PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


_START_INSTALLED_VERSIONS = _installed_versions()


def _torch_environment() -> dict[str, Any]:
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return {
            "version": None,
            "cuda_version": None,
            "cuda_available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        cuda_available = False
    return {
        "version": str(getattr(torch, "__version__", "")) or None,
        "cuda_version": str(cuda_version) if cuda_version else None,
        "cuda_available": cuda_available,
        "error": None,
    }


def _environment_status() -> dict[str, Any]:
    spec = _package_spec()
    system, machine = _normalized_platform()
    py = (sys.version_info.major, sys.version_info.minor)
    torch_env = _torch_environment()
    torch_version = torch_env["version"]
    match = re.match(r"^(\d+)\.(\d+)", torch_version or "")
    torch_minor = (int(match.group(1)), int(match.group(2))) if match else None

    reason = "supported"
    supported = True
    if spec is None:
        supported, reason = False, "platform_unsupported"
    elif not (_SUPPORTED_PYTHON_MIN <= py <= _SUPPORTED_PYTHON_MAX):
        supported, reason = False, "python_unsupported"
    elif torch_version is None:
        supported, reason = False, "torch_missing"
    elif torch_minor != _SUPPORTED_TORCH:
        supported, reason = False, "torch_unsupported"
    elif torch_env["cuda_version"] is None:
        supported, reason = False, "cuda_torch_required"

    return {
        "platform": f"{system}_{machine}",
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "torch_cuda_version": torch_env["cuda_version"],
        "torch_cuda_available": torch_env["cuda_available"],
        "supported": supported,
        "reason": reason,
        "expected_package": spec.package if spec else None,
        "expected_version": spec.version if spec else None,
    }


def current_status() -> dict[str, Any]:
    """Return package facts and the audited compatibility decision."""
    installed = _installed_versions()
    env = _environment_status()
    expected_package = env["expected_package"]
    expected_version = env["expected_version"]
    conflicts = [name for name in installed if name != expected_package]

    available = bool(
        env["supported"]
        and expected_package
        and installed.get(expected_package) == expected_version
        and not conflicts
    )
    if conflicts:
        reason = "package_conflict"
    elif expected_package in installed and installed.get(expected_package) != expected_version:
        reason = "version_mismatch"
    elif not env["supported"]:
        reason = env["reason"]
    elif not installed:
        reason = "not_installed"
    else:
        reason = "available"

    restart_required = _RESTART_REQUIRED or installed != _START_INSTALLED_VERSIONS
    if restart_required:
        state = "restart_required"
    elif available:
        state = "available"
    elif installed:
        state = "incompatible"
    else:
        state = "not_installed"

    return {
        "state": state,
        "installed": bool(installed),
        "available": available,
        "installed_packages": installed,
        "package": expected_package,
        "version": installed.get(expected_package) if expected_package else None,
        "expected_package": expected_package,
        "expected_version": expected_version,
        "compatible": available,
        "reason": reason,
        "restart_required": restart_required,
        "environment": env,
    }


def _run_pip(args: list[str], *, action: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Triton {action}超时（10 分钟）") from exc
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-2000:]
        raise RuntimeError(
            f"Triton {action}失败 (exit {result.returncode}):\n{tail}"
        )
    return result


def install() -> dict[str, Any]:
    """Install/repair the exact supported wheel without touching Torch."""
    global _RESTART_REQUIRED

    env = _environment_status()
    if not env["supported"]:
        raise RuntimeError(
            "当前环境不在 Triton 实验支持矩阵内："
            f"reason={env['reason']}, platform={env['platform']}, "
            f"Python={env['python_version']}, Torch={env['torch_version']}, "
            f"Torch CUDA={env['torch_cuda_version']}"
        )
    spec = _package_spec()
    assert spec is not None

    outputs: list[str] = []
    installed = _installed_versions()
    conflicts = [name for name in installed if name != spec.package]
    if conflicts:
        result = _run_pip(["uninstall", "-y", *conflicts], action="冲突包卸载")
        outputs.append(result.stdout or result.stderr or "")

    result = _run_pip(
        [
            "install",
            "--no-deps",
            "--only-binary=:all:",
            "--force-reinstall",
            f"{spec.package}=={spec.version}",
        ],
        action="安装",
    )
    outputs.append(result.stdout or result.stderr or "")
    importlib.invalidate_caches()
    _RESTART_REQUIRED = True

    status = current_status()
    if status["version"] != spec.version:
        raise RuntimeError(
            "Triton 安装命令成功，但未检测到预期制品 "
            f"{spec.package}=={spec.version}；请重启 Studio 后刷新状态"
        )
    return {
        **status,
        "stdout_tail": "\n".join(outputs)[-2000:],
        "restart_required": True,
    }


def uninstall() -> dict[str, Any]:
    """Remove managed/conflicting Triton distributions; eager Torch is untouched."""
    global _RESTART_REQUIRED

    installed = _installed_versions()
    if not installed:
        return {**current_status(), "stdout_tail": ""}

    result = _run_pip(
        ["uninstall", "-y", *sorted(installed)],
        action="卸载",
    )
    importlib.invalidate_caches()
    _RESTART_REQUIRED = True
    return {
        **current_status(),
        "stdout_tail": (result.stdout or result.stderr or "")[-2000:],
        "restart_required": True,
    }
