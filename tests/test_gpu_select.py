"""services/runtime/gpu_select.py — 计算显卡选择的 env 注入（#491）。

注入形态固定为 CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=<n>：
PCI 序与 NVML/nvidia-smi 同构，设置存的序号、监控显示的序号、CUDA 选中
的卡才是同一个意思。marker 区分「我们注入」与「用户手设」——launcher
常驻进程里上一轮注入必须能被新设置覆盖/撤销，手设的永远不动。
"""
from __future__ import annotations

import pytest

from studio.services.runtime import gpu_select


@pytest.fixture()
def two_gpus(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gpu_select, "_nvml_device_count", lambda: 2)


def _set_selection(monkeypatch: pytest.MonkeyPatch, idx):
    monkeypatch.setattr(gpu_select, "_selected_index", lambda: idx)


def test_unset_selection_is_noop(monkeypatch, two_gpus):
    _set_selection(monkeypatch, None)
    env: dict[str, str] = {}
    gpu_select.apply_gpu_selection_env(env)
    assert env == {}


def test_selection_injects_pci_order_pair(monkeypatch, two_gpus):
    _set_selection(monkeypatch, 1)
    env: dict[str, str] = {}
    gpu_select.apply_gpu_selection_env(env)
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env[gpu_select._MARKER] == "1"


def test_manual_env_is_respected(monkeypatch, two_gpus):
    """用户在 studio.bat 手设过 CUDA env（#491 里的 workaround）→ 永不覆盖。"""
    _set_selection(monkeypatch, 1)
    env = {"CUDA_VISIBLE_DEVICES": "0"}
    gpu_select.apply_gpu_selection_env(env)
    assert env == {"CUDA_VISIBLE_DEVICES": "0"}


def test_relaunch_updates_previous_injection(monkeypatch, two_gpus):
    """launcher 常驻：用户改设置 → 重启循环重跑 → 上轮注入被新值覆盖。"""
    _set_selection(monkeypatch, 0)
    env: dict[str, str] = {}
    gpu_select.apply_gpu_selection_env(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    _set_selection(monkeypatch, 1)
    gpu_select.apply_gpu_selection_env(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_clearing_selection_revokes_injection(monkeypatch, two_gpus):
    """设置清回默认 → 撤销注入，回 CUDA 自选。"""
    _set_selection(monkeypatch, 1)
    env: dict[str, str] = {}
    gpu_select.apply_gpu_selection_env(env)
    _set_selection(monkeypatch, None)
    gpu_select.apply_gpu_selection_env(env)
    assert env == {}


def test_out_of_range_index_skipped_and_revoked(monkeypatch, two_gpus):
    """卡不在了（eGPU 拔线）：忽略选择，不能让 torch 面对空设备列表。"""
    _set_selection(monkeypatch, 5)
    env: dict[str, str] = {}
    gpu_select.apply_gpu_selection_env(env)
    assert env == {}
    # 上轮注入过、这轮卡没了 → 撤销
    _set_selection(monkeypatch, 1)
    gpu_select.apply_gpu_selection_env(env)
    _set_selection(monkeypatch, 5)
    gpu_select.apply_gpu_selection_env(env)
    assert env == {}


def test_nvml_unavailable_still_injects(monkeypatch):
    """NVML 查不到卡数（驱动/库缺失）→ 保守放行注入，保持用户意图。"""
    monkeypatch.setattr(gpu_select, "_nvml_device_count", lambda: None)
    _set_selection(monkeypatch, 1)
    env: dict[str, str] = {}
    gpu_select.apply_gpu_selection_env(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_selected_index_reads_secrets(monkeypatch):
    """secrets.system.gpu_index → int；None/负数 → None。"""
    from studio.infrastructure import secrets as secrets_infra

    class _Sys:
        gpu_index = 1

    class _S:
        system = _Sys()

    monkeypatch.setattr(secrets_infra, "load", lambda: _S())
    assert gpu_select._selected_index() == 1
    _Sys.gpu_index = None
    assert gpu_select._selected_index() is None
    _Sys.gpu_index = -1
    assert gpu_select._selected_index() is None
