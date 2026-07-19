"""commit 20：AnimaLycorisAdapter.detach() 撤销 inject 的钩子。

让 daemon 切换 LoRA path 时不必重 load 整个 transformer。
mock-based —— 不依赖真 lycoris 包（dev env 通常没装）。
"""
from __future__ import annotations

from unittest.mock import MagicMock

from utils.lycoris_adapter import AnimaLycorisAdapter
from training.families.anima.preset import ANIMA_PRESET


def test_detach_noop_when_not_injected() -> None:
    """没 inject 过的 adapter，detach 是 noop 返 True。"""
    a = AnimaLycorisAdapter(preset=ANIMA_PRESET, )
    assert a.detach() is True
    assert a.network is None


def test_detach_calls_restore_and_clears_state() -> None:
    """有 LycorisNetwork.restore() 时被调用 + model.train 还原 + 引用清空。"""
    a = AnimaLycorisAdapter(preset=ANIMA_PRESET, )
    fake_network = MagicMock()
    # restore 是第一个被尝试的方法名
    fake_network.restore = MagicMock()
    a.network = fake_network

    fake_orig_train = MagicMock()
    fake_model = MagicMock()
    a._orig_train = fake_orig_train
    a._injected_model = fake_model

    assert a.detach() is True
    fake_network.restore.assert_called_once()
    # model.train 应该被还原（assigned back to _orig_train）
    assert fake_model.train is fake_orig_train
    # 引用清空让 GC
    assert a.network is None
    assert a._injected_model is None
    assert a._orig_train is None


def test_detach_falls_back_to_restore_apply() -> None:
    """没 restore 但有 restore_apply 时，调 restore_apply。"""
    a = AnimaLycorisAdapter(preset=ANIMA_PRESET, )
    fake_network = MagicMock(spec=["restore_apply", "remove_apply"])
    fake_network.restore_apply = MagicMock()
    a.network = fake_network
    a._orig_train = MagicMock()
    a._injected_model = MagicMock()

    assert a.detach() is True
    fake_network.restore_apply.assert_called_once()


def test_detach_returns_false_when_no_restore_interface() -> None:
    """lycoris 三个接口都没（旧版本）→ 返 False，调用方应 fallback 到 reload。"""
    a = AnimaLycorisAdapter(preset=ANIMA_PRESET, )
    # spec=[] 让 MagicMock 不自动生成 restore* 属性 → getattr 返 None
    fake_network = MagicMock(spec=[])
    a.network = fake_network
    a._orig_train = MagicMock()
    a._injected_model = MagicMock()

    assert a.detach() is False
    # 即使 restore 失败，model.train 仍应被还原
    assert a.network is None


def test_detach_idempotent() -> None:
    """重复 detach 不抛错，第二次 noop。"""
    a = AnimaLycorisAdapter(preset=ANIMA_PRESET, )
    fake_network = MagicMock()
    fake_network.restore = MagicMock()
    a.network = fake_network
    a._orig_train = MagicMock()
    a._injected_model = MagicMock()

    a.detach()
    # 第二次：network 已 None
    assert a.detach() is True
