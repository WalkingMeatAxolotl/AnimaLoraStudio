"""utils.lycoris_patch — LyCORIS v3/v4 LoKr rank-dropout device patch。

覆盖：
- 每个已确认受影响版本都会 patch 真实 LokrModule.get_weight
- patch 委托原始 weight builder，并在 weight.device 上重放 rank dropout
- 未装 lycoris → skipped_not_installed
- 未知版本 → skipped_version_unknown + debug
- 同进程内幂等 → skipped_already_patched
"""
from __future__ import annotations

import importlib
import logging
import sys

import pytest


@pytest.fixture
def fresh_patch_module(monkeypatch: pytest.MonkeyPatch):
    """每个测试拿一份新 imported 的 lycoris_patch + 重置 LokrModule.get_weight。

    LokrModule 是单例；同进程内 patch 后属性会持久。fixture 在测试前后
    把 get_weight 还原到上游版本，并清掉 _PATCHED_FLAG，让每个测试都从
    「未 patch」状态起跑。
    """
    # 重新加载模块（清掉 module 级缓存）
    if "utils.lycoris_patch" in sys.modules:
        del sys.modules["utils.lycoris_patch"]
    mod = importlib.import_module("utils.lycoris_patch")

    # 备份并清理 LokrModule.get_weight，让幂等检查重置
    try:
        from lycoris.modules.lokr import LokrModule
        orig_get_weight = LokrModule.get_weight
        had_flag = getattr(LokrModule, mod._PATCHED_FLAG, False)
        had_original = hasattr(LokrModule, mod._ORIGINAL_GET_WEIGHT_ATTR)
        original_attr = getattr(LokrModule, mod._ORIGINAL_GET_WEIGHT_ATTR, None)
        if had_flag:
            delattr(LokrModule, mod._PATCHED_FLAG)
    except Exception:
        LokrModule = None  # type: ignore[assignment]
        orig_get_weight = None
        had_flag = False
        had_original = False
        original_attr = None

    yield mod

    # 还原
    if LokrModule is not None and orig_get_weight is not None:
        LokrModule.get_weight = orig_get_weight
        if had_flag:
            setattr(LokrModule, mod._PATCHED_FLAG, True)
        elif hasattr(LokrModule, mod._PATCHED_FLAG):
            delattr(LokrModule, mod._PATCHED_FLAG)
        if had_original:
            setattr(LokrModule, mod._ORIGINAL_GET_WEIGHT_ATTR, original_attr)
        elif hasattr(LokrModule, mod._ORIGINAL_GET_WEIGHT_ATTR):
            delattr(LokrModule, mod._ORIGINAL_GET_WEIGHT_ATTR)


@pytest.mark.parametrize(
    "affected_version",
    ["3.4.0", "4.0.0", "4.0.1.dev20260902072855"],
)
def test_apply_on_known_affected_version_patches_get_weight(
    fresh_patch_module,
    monkeypatch: pytest.MonkeyPatch,
    affected_version: str,
) -> None:
    """每个确认受影响版本都应 patch，并在 weight.device 上创建 mask。"""
    pytest.importorskip("lycoris.modules.lokr")
    import torch
    from lycoris.modules.lokr import LokrModule

    monkeypatch.setattr(fresh_patch_module, "version", lambda _: affected_version)

    status = fresh_patch_module.apply_lokr_device_patch()
    assert status == "applied"
    assert getattr(LokrModule, fresh_patch_module._PATCHED_FLAG) is True

    # 行为验证：让 get_weight 走到 rank_dropout 分支并触发 torch.rand —— mask
    # 必须生成在 weight.device 上。用 patch 把 torch.rand 替换成探针。
    captured: dict[str, object] = {}
    real_rand = torch.rand

    def _spy_rand(*args, **kwargs):
        captured["device"] = kwargs.get("device", None)
        return real_rand(*args, **kwargs)

    monkeypatch.setattr(torch, "rand", _spy_rand)

    # 用 mock self（不构造完整 LokrModule，单测只关心 torch.rand 这一行）
    class _FakeSelf:
        training = True
        rank_dropout = 0.5
        rank_dropout_scale = False
        use_w1 = True
        use_w2 = True
        tucker = False
        scale = 1.0
        lokr_w1 = torch.eye(4)
        lokr_w2 = torch.eye(4)

    fake = _FakeSelf()
    # 直接调被替换后的 get_weight；shape=None 让 weight 保持原状
    LokrModule.get_weight(fake, None)
    assert captured["device"] == fake.lokr_w1.device
    assert fake.rank_dropout == 0.5


def test_apply_when_not_installed_returns_skipped(
    fresh_patch_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没装 lycoris-lora（PackageNotFoundError）→ 静默跳过。"""
    def _raise(_pkg):
        raise fresh_patch_module.PackageNotFoundError
    monkeypatch.setattr(fresh_patch_module, "version", _raise)

    assert fresh_patch_module.apply_lokr_device_patch() == "skipped_not_installed"


def test_apply_unknown_version_skips_quietly(
    fresh_patch_module, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """未知版本（如上游已修的 3.5.0）→ 跳过，避免覆盖上游修复。

    这是「什么都没做」的正常路径：新版用户每次训练吃一条 WARNING 属误报，
    按 leveling-rules R4 记 DEBUG（版本号仍在，排障可查）。
    """
    pytest.importorskip("lycoris.modules.lokr")
    monkeypatch.setattr(fresh_patch_module, "version", lambda _: "999.0.0")
    with caplog.at_level(logging.DEBUG, logger="utils.lycoris_patch"):
        status = fresh_patch_module.apply_lokr_device_patch()
    assert status == "skipped_version_unknown"
    assert any("999.0.0" in rec.message for rec in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_apply_idempotent(
    fresh_patch_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同进程内重复调用：第一次 applied，之后 skipped_already_patched。"""
    pytest.importorskip("lycoris.modules.lokr")
    monkeypatch.setattr(fresh_patch_module, "version", lambda _: "3.4.0")
    assert fresh_patch_module.apply_lokr_device_patch() == "applied"
    assert fresh_patch_module.apply_lokr_device_patch() == "skipped_already_patched"


def test_patch_restores_rank_dropout_when_weight_build_fails(
    fresh_patch_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """上游 weight builder 抛错时也不能把模块永久留在 rank_dropout=0。"""
    pytest.importorskip("lycoris.modules.lokr")
    from lycoris.modules.lokr import LokrModule

    monkeypatch.setattr(fresh_patch_module, "version", lambda _: "3.4.0")
    assert fresh_patch_module.apply_lokr_device_patch() == "applied"

    class _BrokenSelf:
        training = True
        rank_dropout = 0.5

    broken = _BrokenSelf()
    with pytest.raises(AttributeError):
        LokrModule.get_weight(broken, None)
    assert broken.rank_dropout == 0.5
