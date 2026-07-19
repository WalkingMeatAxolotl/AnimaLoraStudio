"""schema.noise_enhancement_type + migrate_noise_enhancement_type 回归。

对齐 kohya-ss/sd-scripts PR #477（raise error when both noise_offset and
multires）：noise_offset 与金字塔噪声在 Anima 这边走单一 type 字段管控，
schema / migration 层强制清零反组字段（kohya_ss issue #2599 教训：UI 隐藏
不等于清值，序列化层要互斥）。
"""
from __future__ import annotations

import pytest

from studio.schema import TrainingConfig, migrate_noise_enhancement_type


# ---------------------------------------------------------------------------
# migrate_noise_enhancement_type（dict 层 helper）
# ---------------------------------------------------------------------------


def test_migrate_legacy_only_offset() -> None:
    """只设 noise_offset > 0 → type=offset。"""
    out = migrate_noise_enhancement_type({"noise_offset": 0.05})
    assert out["noise_enhancement_type"] == "offset"
    assert out["pyramid_noise_iters"] == 0


def test_migrate_legacy_only_pyramid() -> None:
    """只设 pyramid_noise_iters > 0 → type=pyramid，noise_offset 清零。"""
    out = migrate_noise_enhancement_type({"pyramid_noise_iters": 3})
    assert out["noise_enhancement_type"] == "pyramid"
    assert out["noise_offset"] == 0.0


def test_migrate_legacy_neither() -> None:
    """两者都 0 / 未设 → type=none。"""
    out = migrate_noise_enhancement_type({})
    assert out["noise_enhancement_type"] == "none"


def test_migrate_legacy_both_set_pyramid_wins() -> None:
    """历史 bug 配置：两者都 > 0 → pyramid 优先，offset 清零。

    理由：Anima 旧 make_noise 末尾归一化稀释了 noise_offset 的常数偏移，
    实际生效的主要是 pyramid。
    """
    out = migrate_noise_enhancement_type({
        "noise_offset": 0.05,
        "pyramid_noise_iters": 3,
    })
    assert out["noise_enhancement_type"] == "pyramid"
    assert out["noise_offset"] == 0.0


# ---------------------------------------------------------------------------
# runtime 纵深防御：noise_params_from_args 按 type 分派（审计 #3）
# ---------------------------------------------------------------------------


def test_noise_params_dispatch_by_type() -> None:
    """runtime 消费端以 noise_enhancement_type 为准 —— 绕过配置构造的 args
    （pause snapshot 旧格式等）残留反组参数不参与，杜绝 offset+pyramid 叠加。"""
    from argparse import Namespace

    from training.noise import noise_params_from_args

    both = Namespace(
        noise_enhancement_type="offset",
        noise_offset=0.05, pyramid_noise_iters=3, pyramid_noise_discount=0.5,
    )
    assert noise_params_from_args(both) == (0.05, 0, 0.5)

    both.noise_enhancement_type = "pyramid"
    assert noise_params_from_args(both) == (0.0, 3, 0.5)

    both.noise_enhancement_type = "none"
    assert noise_params_from_args(both) == (0.0, 0, 0.5)


def test_migrate_explicit_type_offset_clears_pyramid() -> None:
    """显式 type=offset → pyramid_noise_iters 强制清零（issue #2599 教训）。"""
    out = migrate_noise_enhancement_type({
        "noise_enhancement_type": "offset",
        "noise_offset": 0.05,
        "pyramid_noise_iters": 3,  # 残值，必须清掉
    })
    assert out["noise_enhancement_type"] == "offset"
    assert out["noise_offset"] == 0.05
    assert out["pyramid_noise_iters"] == 0


def test_migrate_explicit_type_pyramid_clears_offset() -> None:
    out = migrate_noise_enhancement_type({
        "noise_enhancement_type": "pyramid",
        "noise_offset": 0.05,  # 残值
        "pyramid_noise_iters": 3,
    })
    assert out["noise_enhancement_type"] == "pyramid"
    assert out["noise_offset"] == 0.0
    assert out["pyramid_noise_iters"] == 3


def test_migrate_explicit_type_none_clears_both() -> None:
    out = migrate_noise_enhancement_type({
        "noise_enhancement_type": "none",
        "noise_offset": 0.05,
        "pyramid_noise_iters": 3,
    })
    assert out["noise_enhancement_type"] == "none"
    assert out["noise_offset"] == 0.0
    assert out["pyramid_noise_iters"] == 0


def test_migrate_idempotent() -> None:
    once = migrate_noise_enhancement_type({"pyramid_noise_iters": 3})
    twice = migrate_noise_enhancement_type(dict(once))
    assert once == twice


def test_migrate_non_dict_passthrough() -> None:
    """非 dict 直接 return 不动（pydantic model_validator(mode='before') 兼容）。"""
    assert migrate_noise_enhancement_type(None) is None
    assert migrate_noise_enhancement_type("notadict") == "notadict"
    assert migrate_noise_enhancement_type(42) == 42


def test_migrate_str_number_coerced() -> None:
    """yaml 偶尔把数字读成 str；不能崩，按 0 处理。"""
    out = migrate_noise_enhancement_type({"noise_offset": "0.05"})
    assert out["noise_enhancement_type"] == "offset"


def test_migrate_invalid_number_treated_as_zero() -> None:
    """坏值（非数字 str / None）→ 当作 0 处理，不抛。"""
    out = migrate_noise_enhancement_type({
        "noise_offset": "abc",
        "pyramid_noise_iters": None,
    })
    assert out["noise_enhancement_type"] == "none"


# ---------------------------------------------------------------------------
# pydantic schema —— TrainingConfig 走 migrate 后能正确构造 + 互斥
# ---------------------------------------------------------------------------


def test_schema_default_is_none() -> None:
    t = TrainingConfig()
    assert t.noise_enhancement_type == "none"
    assert t.noise_offset == 0.0
    assert t.pyramid_noise_iters == 0


def test_schema_legacy_offset_yaml() -> None:
    """老 yaml { noise_offset: 0.05 } → type=offset，pyramid 清零。"""
    t = TrainingConfig(noise_offset=0.05)  # type: ignore[call-arg]
    assert t.noise_enhancement_type == "offset"
    assert t.noise_offset == 0.05
    assert t.pyramid_noise_iters == 0


def test_schema_legacy_pyramid_yaml() -> None:
    t = TrainingConfig(pyramid_noise_iters=3)  # type: ignore[call-arg]
    assert t.noise_enhancement_type == "pyramid"
    assert t.noise_offset == 0.0
    assert t.pyramid_noise_iters == 3


def test_schema_legacy_both_set_pyramid_wins() -> None:
    """老 yaml 两者都设 → pyramid 优先，offset 清零（issue #2599 同款防御）。"""
    t = TrainingConfig(noise_offset=0.05, pyramid_noise_iters=3)  # type: ignore[call-arg]
    assert t.noise_enhancement_type == "pyramid"
    assert t.noise_offset == 0.0
    assert t.pyramid_noise_iters == 3


def test_schema_explicit_offset_clears_pyramid() -> None:
    """显式 type=offset 但 yaml 残留 pyramid 字段 → 清掉。"""
    t = TrainingConfig(
        noise_enhancement_type="offset",
        noise_offset=0.05,
        pyramid_noise_iters=3,  # 残值
    )
    assert t.noise_enhancement_type == "offset"
    assert t.noise_offset == 0.05
    assert t.pyramid_noise_iters == 0


def test_schema_explicit_pyramid_clears_offset() -> None:
    t = TrainingConfig(
        noise_enhancement_type="pyramid",
        noise_offset=0.05,  # 残值
        pyramid_noise_iters=3,
    )
    assert t.noise_enhancement_type == "pyramid"
    assert t.noise_offset == 0.0
    assert t.pyramid_noise_iters == 3


def test_schema_explicit_none_clears_both() -> None:
    t = TrainingConfig(
        noise_enhancement_type="none",
        noise_offset=0.05,
        pyramid_noise_iters=3,
    )
    assert t.noise_enhancement_type == "none"
    assert t.noise_offset == 0.0
    assert t.pyramid_noise_iters == 0


def test_schema_invalid_type_rejected() -> None:
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        TrainingConfig(noise_enhancement_type="multires")  # type: ignore[arg-type]


@pytest.mark.parametrize("t_value", ["none", "offset", "pyramid"])
def test_schema_all_types_validate(t_value: str) -> None:
    t = TrainingConfig(noise_enhancement_type=t_value)  # type: ignore[arg-type]
    assert t.noise_enhancement_type == t_value
