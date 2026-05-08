"""PR-S2.1 — pending_install marker 读写 + apply_pending 调度。

不真跑 pip：torch_setup.reinstall 用 monkeypatch 替成假实现验证流程。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio.services import pending_install


@pytest.fixture
def isolated_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """每个测试独立 marker 路径，避免互相污染。"""
    marker = tmp_path / ".pending-pip-install.json"
    monkeypatch.setattr(pending_install, "STUDIO_DATA", tmp_path)
    monkeypatch.setattr(pending_install, "PENDING_MARKER", marker)
    return marker


def test_register_writes_marker(isolated_marker: Path) -> None:
    pending_install.register_torch_reinstall("cu128")
    assert isolated_marker.exists()
    data = pending_install.read_pending()
    assert data == {"kind": "torch", "target": "cu128"}


def test_register_overwrites_previous(isolated_marker: Path) -> None:
    pending_install.register_torch_reinstall("cu118")
    pending_install.register_torch_reinstall("cu128")  # 覆盖
    assert pending_install.read_pending()["target"] == "cu128"


def test_read_pending_returns_none_when_missing(isolated_marker: Path) -> None:
    assert pending_install.read_pending() is None


def test_read_pending_returns_none_on_corrupt_marker(isolated_marker: Path) -> None:
    isolated_marker.write_text("not-json{", encoding="utf-8")
    assert pending_install.read_pending() is None


def test_clear_pending_removes_marker(isolated_marker: Path) -> None:
    pending_install.register_torch_reinstall("cu128")
    pending_install.clear_pending()
    assert not isolated_marker.exists()
    assert pending_install.read_pending() is None


def test_clear_pending_no_marker_no_error(isolated_marker: Path) -> None:
    """没 marker 时 clear 也应静默成功。"""
    pending_install.clear_pending()
    assert not isolated_marker.exists()


def test_apply_pending_no_marker_is_noop(
    isolated_marker: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没 pending → 不应触发 torch_setup.reinstall。"""
    from studio.services import torch_setup
    called: list[str] = []
    monkeypatch.setattr(torch_setup, "reinstall", lambda t: called.append(t))
    pending_install.apply_pending()
    assert called == []


def test_apply_pending_runs_torch_reinstall_and_clears(
    isolated_marker: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """有 pending=torch → 调 reinstall + 清 marker。"""
    pending_install.register_torch_reinstall("cu128")
    from studio.services import torch_setup
    captured: list[str] = []

    def fake_reinstall(target):
        captured.append(target)
        return {
            "target": target, "tag": "cu128",
            "index_url": "https://x/cu128",
            "version": "2.5.0+cu128",
            "stdout_tail": "ok",
            "restart_required": True,
        }

    monkeypatch.setattr(torch_setup, "reinstall", fake_reinstall)
    pending_install.apply_pending()
    assert captured == ["cu128"]
    assert not isolated_marker.exists()  # 成功后清掉
    out = capsys.readouterr().out
    assert "torch 重装完成" in out


def test_apply_pending_keeps_marker_on_failure(
    isolated_marker: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """reinstall 抛 RuntimeError → marker 保留，下次启动重试。"""
    pending_install.register_torch_reinstall("cu128")
    from studio.services import torch_setup

    def fake_reinstall(_t):
        raise RuntimeError("network failed")

    monkeypatch.setattr(torch_setup, "reinstall", fake_reinstall)
    pending_install.apply_pending()
    # marker 保留
    assert isolated_marker.exists()
    assert pending_install.read_pending()["target"] == "cu128"
    err = capsys.readouterr().err
    assert "torch 重装失败" in err
    assert "network failed" in err


def test_apply_pending_unknown_kind_clears_marker(
    isolated_marker: Path, capsys
) -> None:
    """未知 kind → warn + 清 marker（防止永久卡住）。"""
    isolated_marker.write_text(
        '{"kind": "modelscope", "target": "auto"}', encoding="utf-8"
    )
    pending_install.apply_pending()
    assert not isolated_marker.exists()
    err = capsys.readouterr().err
    assert "未知" in err
