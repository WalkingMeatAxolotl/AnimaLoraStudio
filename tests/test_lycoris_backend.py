"""LyCORIS v4 backend policy regression tests."""
from __future__ import annotations

import logging

import pytest

from utils import lycoris_backend


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
    lycoris_backend.log_lycoris_runtime_once.cache_clear()

    with caplog.at_level(logging.INFO, logger="utils.lycoris_backend"):
        lycoris_backend.log_lycoris_runtime_once()

    assert any(
        "LyCORIS runtime:" in record.message
        and "resolved_backend=" in record.message
        for record in caplog.records
    )
