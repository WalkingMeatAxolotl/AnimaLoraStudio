"""Managed Triton runtime service tests."""
from __future__ import annotations

import importlib.metadata
import sys
import types
from unittest.mock import MagicMock

import pytest

from studio.services.runtime import triton as triton_setup


def _versions(values: dict[str, str]):
    def get(name: str) -> str:
        if name not in values:
            raise importlib.metadata.PackageNotFoundError(name)
        return values[name]
    return get


def _windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(triton_setup.platform, "system", lambda: "Windows")
    monkeypatch.setattr(triton_setup.platform, "machine", lambda: "AMD64")


def _torch(monkeypatch: pytest.MonkeyPatch, version: str, cuda: str | None) -> None:
    fake = types.ModuleType("torch")
    fake.__version__ = version  # type: ignore[attr-defined]
    fake.version = types.SimpleNamespace(cuda=cuda)  # type: ignore[attr-defined]
    fake.cuda = types.SimpleNamespace(is_available=lambda: cuda is not None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)


def _pip_result(returncode: int = 0, stdout: str = "ok", stderr: str = ""):
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_status_reports_exact_windows_pin_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _windows(monkeypatch)
    _torch(monkeypatch, "2.11.0+cu128", "12.8")
    monkeypatch.setattr(
        triton_setup.importlib.metadata,
        "version",
        _versions({"triton-windows": "3.8.0.post28"}),
    )
    monkeypatch.setattr(
        triton_setup,
        "_START_INSTALLED_VERSIONS",
        {"triton-windows": "3.8.0.post28"},
    )
    monkeypatch.setattr(triton_setup, "_RESTART_REQUIRED", False)

    status = triton_setup.current_status()

    assert status["state"] == "available"
    assert status["available"] is True
    assert status["package"] == "triton-windows"
    assert status["version"] == "3.8.0.post28"
    assert status["environment"]["supported"] is True


@pytest.mark.parametrize(
    ("system", "machine", "package", "version", "expected"),
    [
        ("Windows", "AMD64", "triton-windows", "3.8.0.post28", True),
        ("Windows", "AMD64", "triton-windows", "3.7.0", False),
        ("Linux", "x86_64", "triton", "3.8.0", True),
        ("Linux", "x86_64", "triton", "3.7.0", False),
    ],
)
def test_compatible_requires_exact_platform_pin(
    monkeypatch: pytest.MonkeyPatch,
    system: str, machine: str, package: str, version: str, expected: bool,
) -> None:
    monkeypatch.setattr(triton_setup.platform, "system", lambda: system)
    monkeypatch.setattr(triton_setup.platform, "machine", lambda: machine)
    _torch(monkeypatch, "2.11.0+cu128", "12.8")
    installed = {package: version}
    monkeypatch.setattr(triton_setup.importlib.metadata, "version", _versions(installed))
    monkeypatch.setattr(triton_setup, "_START_INSTALLED_VERSIONS", dict(installed))
    monkeypatch.setattr(triton_setup, "_RESTART_REQUIRED", False)

    status = triton_setup.current_status()

    assert status["compatible"] is expected
    assert status["available"] is expected
    assert status["state"] == ("available" if expected else "incompatible")
    assert status["reason"] == ("available" if expected else "version_mismatch")


def test_status_rejects_wrong_version_and_conflicting_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _windows(monkeypatch)
    _torch(monkeypatch, "2.11.0+cu128", "12.8")
    monkeypatch.setattr(
        triton_setup.importlib.metadata,
        "version",
        _versions({"triton-windows": "3.7.0", "triton": "3.8.0"}),
    )
    monkeypatch.setattr(triton_setup, "_START_INSTALLED_VERSIONS", {})
    monkeypatch.setattr(triton_setup, "_RESTART_REQUIRED", False)

    status = triton_setup.current_status()

    assert status["state"] == "restart_required"
    assert status["available"] is False
    assert status["reason"] == "package_conflict"


@pytest.mark.parametrize(
    ("torch_version", "cuda", "reason"),
    [
        ("2.10.0+cu128", "12.8", "torch_unsupported"),
        ("2.11.0+cpu", None, "cuda_torch_required"),
    ],
)
def test_status_rejects_unsupported_torch_environment(
    monkeypatch: pytest.MonkeyPatch,
    torch_version: str,
    cuda: str | None,
    reason: str,
) -> None:
    _windows(monkeypatch)
    _torch(monkeypatch, torch_version, cuda)
    monkeypatch.setattr(
        triton_setup.importlib.metadata, "version", _versions({}),
    )
    monkeypatch.setattr(triton_setup, "_START_INSTALLED_VERSIONS", {})
    monkeypatch.setattr(triton_setup, "_RESTART_REQUIRED", False)

    status = triton_setup.current_status()

    assert status["compatible"] is False
    assert status["reason"] == reason


def test_install_removes_conflict_and_uses_exact_no_deps_wheel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _windows(monkeypatch)
    monkeypatch.setattr(triton_setup, "_environment_status", lambda: {
        "platform": "windows_x86_64",
        "python_version": "3.13.0",
        "torch_version": "2.11.0+cu128",
        "torch_cuda_version": "12.8",
        "torch_cuda_available": True,
        "supported": True,
        "reason": "supported",
        "expected_package": "triton-windows",
        "expected_version": "3.8.0.post28",
    })
    installed = {"triton": "3.8.0"}
    monkeypatch.setattr(
        triton_setup.importlib.metadata,
        "version",
        lambda name: installed[name] if name in installed else (
            "3.8.0.post28" if name == "triton-windows" else (_ for _ in ()).throw(
                importlib.metadata.PackageNotFoundError(name)
            )
        ),
    )
    commands: list[list[str]] = []

    def run(cmd, **_kwargs):
        commands.append(cmd)
        if "uninstall" in cmd:
            installed.pop("triton", None)
        return _pip_result()

    monkeypatch.setattr(triton_setup.subprocess, "run", run)
    monkeypatch.setattr(triton_setup, "_RESTART_REQUIRED", False)

    result = triton_setup.install()

    assert commands[0][-3:] == ["uninstall", "-y", "triton"]
    assert commands[1][-4:] == [
        "--no-deps", "--only-binary=:all:", "--force-reinstall",
        "triton-windows==3.8.0.post28",
    ]
    assert "--no-deps" in commands[1]
    assert result["restart_required"] is True
    assert result["version"] == "3.8.0.post28"


def test_install_refuses_unsupported_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(triton_setup, "_environment_status", lambda: {
        "platform": "darwin_arm64", "python_version": "3.13.0",
        "torch_version": "2.11.0", "torch_cuda_version": None,
        "torch_cuda_available": False, "supported": False,
        "reason": "platform_unsupported", "expected_package": None,
        "expected_version": None,
    })
    monkeypatch.setattr(
        triton_setup.subprocess, "run",
        lambda *_a, **_k: pytest.fail("unsupported environment must not run pip"),
    )

    with pytest.raises(RuntimeError, match="platform_unsupported"):
        triton_setup.install()


def test_uninstall_removes_all_known_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = {"triton": "3.8.0", "triton-windows": "3.8.0.post28"}
    monkeypatch.setattr(
        triton_setup.importlib.metadata,
        "version",
        lambda name: installed[name] if name in installed else (_ for _ in ()).throw(
            importlib.metadata.PackageNotFoundError(name)
        ),
    )
    commands: list[list[str]] = []

    def run(cmd, **_kwargs):
        commands.append(cmd)
        installed.clear()
        return _pip_result(stdout="removed")

    monkeypatch.setattr(triton_setup.subprocess, "run", run)
    monkeypatch.setattr(triton_setup, "_START_INSTALLED_VERSIONS", dict(installed))
    monkeypatch.setattr(triton_setup, "_RESTART_REQUIRED", False)

    result = triton_setup.uninstall()

    assert commands[0][-4:] == ["uninstall", "-y", "triton", "triton-windows"]
    assert result["installed"] is False
    assert result["restart_required"] is True
