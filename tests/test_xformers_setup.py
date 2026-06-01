"""xformers_setup 单测 —— current_status / install / _torch_cuda_index 路径覆盖。

参考 test_flash_attention_setup.py 的风格但更简洁（xformers service 比
flash_attention 简单很多）。
"""
from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from studio.services.runtime import xformers as xs


# ---------------------------------------------------------------------------
# current_status
# ---------------------------------------------------------------------------


def test_current_status_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """importlib.metadata 找不到 xformers → installed=False。"""
    import importlib.metadata as md
    def boom(name: str) -> str:
        raise md.PackageNotFoundError(name)
    monkeypatch.setattr(xs.importlib.metadata, "version", boom)
    s = xs.current_status()
    assert s == {"installed": False, "version": None}


def test_current_status_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xs.importlib.metadata, "version", lambda _: "0.0.28")
    s = xs.current_status()
    assert s == {"installed": True, "version": "0.0.28"}


# ---------------------------------------------------------------------------
# _torch_cuda_index — 与 flash_attention_setup.detect_env 同样从 torch 拿 ABI tag
# ---------------------------------------------------------------------------


def _patch_torch(monkeypatch: pytest.MonkeyPatch, version: str | None) -> None:
    """注入 / 移除 fake torch 模块（version=None 模拟未装 torch）。"""
    if version is None:
        monkeypatch.setitem(sys.modules, "torch", None)  # type: ignore[arg-type]
        return
    fake = types.ModuleType("torch")
    fake.__version__ = version  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_torch_cuda_index_from_torch_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch.__version__='2.11.0+cu128' → cu128 index URL。"""
    _patch_torch(monkeypatch, "2.11.0+cu128")
    assert xs._torch_cuda_index() == "https://download.pytorch.org/whl/cu128"


def test_torch_cuda_index_no_cu_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU-only torch（无 +cu）→ None；caller 走 PyPI default。"""
    _patch_torch(monkeypatch, "2.11.0")
    assert xs._torch_cuda_index() is None


def test_torch_cuda_index_no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_torch(monkeypatch, None)
    assert xs._torch_cuda_index() is None


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def _make_pip_result(returncode: int, stdout: str = "", stderr: str = "") -> Any:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_install_success_with_torch_cu(monkeypatch: pytest.MonkeyPatch) -> None:
    """成功路径：cmd 含 --index-url cu_tag；返回 status + restart_required。"""
    _patch_torch(monkeypatch, "2.11.0+cu128")
    captured: list[list[str]] = []
    def fake_run(cmd, **_kw):
        captured.append(cmd)
        return _make_pip_result(0, stdout="Successfully installed xformers-0.0.28")
    monkeypatch.setattr(xs.subprocess, "run", fake_run)
    monkeypatch.setattr(xs.importlib.metadata, "version", lambda _: "0.0.28")

    result = xs.install()

    assert result["installed"] is True
    assert result["version"] == "0.0.28"
    assert result["restart_required"] is True
    assert "Successfully installed" in result["stdout_tail"]
    # cmd 含 --index-url cu128
    assert "--index-url" in captured[0]
    idx = captured[0].index("--index-url")
    assert captured[0][idx + 1] == "https://download.pytorch.org/whl/cu128"


def test_install_no_torch_falls_back_to_pypi(monkeypatch: pytest.MonkeyPatch) -> None:
    """无 torch / 无 cu 后缀 → 不传 --index-url，走 PyPI default。"""
    _patch_torch(monkeypatch, None)
    captured: list[list[str]] = []
    def fake_run(cmd, **_kw):
        captured.append(cmd)
        return _make_pip_result(0, stdout="Successfully installed xformers-0.0.27")
    monkeypatch.setattr(xs.subprocess, "run", fake_run)
    monkeypatch.setattr(xs.importlib.metadata, "version", lambda _: "0.0.27")

    result = xs.install()

    assert result["installed"] is True
    assert "--index-url" not in captured[0]


def test_install_pip_failure_raises_with_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    """pip exit != 0 → RuntimeError 含 stderr 末尾。"""
    _patch_torch(monkeypatch, "2.11.0+cu128")
    monkeypatch.setattr(
        xs.subprocess, "run",
        lambda *a, **k: _make_pip_result(1, stderr="ERROR: No matching distribution found"),
    )
    with pytest.raises(RuntimeError) as exc_info:
        xs.install()
    assert "No matching distribution found" in str(exc_info.value)
    assert "exit 1" in str(exc_info.value)


def test_install_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_torch(monkeypatch, "2.11.0+cu128")
    def boom(*_a, **_k):
        raise xs.subprocess.TimeoutExpired(cmd="pip", timeout=600)
    monkeypatch.setattr(xs.subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="超时"):
        xs.install()
