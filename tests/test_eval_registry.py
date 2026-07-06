"""评估指标 registry：默认集合 + runner 门控逻辑。"""
from __future__ import annotations

from studio.services import eval_registry


def test_default_enabled_is_existing_three() -> None:
    assert eval_registry.DEFAULT_ENABLED == ["clip_t", "clip_i", "dino_i"]


def test_enabled_runners_default() -> None:
    # 默认三指标 → clip + dino 两个 runner（clip_t/clip_i 共用 clip）
    assert eval_registry.enabled_runners(["clip_t", "clip_i", "dino_i"]) == ["clip", "dino"]


def test_enabled_runners_none_falls_back_to_default() -> None:
    assert eval_registry.enabled_runners(None) == ["clip", "dino"]
    assert eval_registry.enabled_runners([]) == ["clip", "dino"]


def test_runner_dedup_shared_metrics() -> None:
    # 只开 clip_t：clip runner 在列（与 clip_i 共用），不重复
    assert eval_registry.enabled_runners(["clip_t"]) == ["clip"]


def test_enabled_runners_new_metrics() -> None:
    assert eval_registry.enabled_runners(["ccip_i"]) == ["ccip"]
    assert eval_registry.enabled_runners(["tag_recall"]) == ["tag"]
    # 顺序按 METRICS 定义（clip→dino→ccip→tag）
    assert eval_registry.enabled_runners(["tag_recall", "clip_i"]) == ["clip", "tag"]


def test_normalize_drops_unknown_keeps_default_when_all_invalid() -> None:
    assert eval_registry.normalize_enabled(["clip_t", "bogus"]) == {"clip_t"}
    assert eval_registry.normalize_enabled(["bogus"]) == set(eval_registry.DEFAULT_ENABLED)


def test_public_catalog_shape() -> None:
    cat = eval_registry.public_catalog()
    keys = [m["key"] for m in cat]
    assert keys == ["clip_t", "clip_i", "dino_i", "ccip_i", "tag_recall"]
    for m in cat:
        assert {"key", "label", "default", "desc", "note", "models"} <= set(m)
