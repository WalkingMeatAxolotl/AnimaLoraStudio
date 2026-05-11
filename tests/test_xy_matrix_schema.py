"""XY 矩阵 schema 测试 —— 字段验证 + 跨字段互斥规则。

覆盖：
  - XYAxisSpec：axis 枚举值 + values 类型按 axis 派生 + lora_index 必要性
  - XYMatrixSpec：y 可选
  - GenerateConfig：xy_matrix 与 prompts 多条 / count>1 互斥
  - GenerateConfig：lora_index 越界检测
  - 默认 xy_matrix=None 不破坏老 cfg
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from studio.schema import (
    GenerateConfig,
    LoraEntry,
    XYAxisSpec,
    XYMatrixSpec,
)


# ---------------------------------------------------------------------------
# XYAxisSpec
# ---------------------------------------------------------------------------


def test_axis_steps_int_values_ok() -> None:
    a = XYAxisSpec(axis="steps", values=[1500, 2000, 2476])
    assert a.values == [1500, 2000, 2476]


def test_axis_lora_scale_float_values_ok() -> None:
    a = XYAxisSpec(axis="lora_scale", values=[0.6, 0.8, 1.0], lora_index=0)
    assert a.lora_index == 0


def test_axis_lora_ckpt_string_values_ok() -> None:
    """lora_ckpt 接受 ckpt 路径字符串列表 + 必须 lora_index。"""
    a = XYAxisSpec(
        axis="lora_ckpt",
        values=["/p/v3/output/v3_step1500.safetensors", "/p/v3/output/v3_step2000.safetensors"],
        lora_index=0,
    )
    assert a.values == [
        "/p/v3/output/v3_step1500.safetensors",
        "/p/v3/output/v3_step2000.safetensors",
    ]
    assert a.lora_index == 0


def test_axis_lora_ckpt_requires_lora_index() -> None:
    """_check_axis_values 在 GenerateConfig validator 里走，所以缺 lora_index 要在
    完整 GenerateConfig 校验中暴露错误。"""
    with pytest.raises(ValidationError, match="必须指定 lora_index"):
        _gen(
            xy_matrix=XYMatrixSpec(
                x=XYAxisSpec(axis="lora_ckpt", values=["/p.safetensors"]),
            ),
        )


def test_axis_sampler_name_now_rejected() -> None:
    """commit: sampler_name 轴已删（我们硬编码 er_sde 不支持其他采样器）。"""
    with pytest.raises(ValidationError):
        XYAxisSpec(axis="sampler_name", values=["er_sde"])  # type: ignore[arg-type]


def test_axis_seed_now_rejected() -> None:
    """commit: seed 轴已删（"测 ep" 场景下应锁种子；用户决策）。"""
    with pytest.raises(ValidationError):
        XYAxisSpec(axis="seed", values=[42])  # type: ignore[arg-type]


def test_axis_invalid_axis_rejected() -> None:
    with pytest.raises(ValidationError):
        XYAxisSpec(axis="not_an_axis", values=[1])  # type: ignore[arg-type]


def test_axis_empty_values_rejected() -> None:
    with pytest.raises(ValidationError):
        XYAxisSpec(axis="steps", values=[])


def test_axis_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        XYAxisSpec(axis="steps", values=[20], extra_field="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# XYMatrixSpec —— y 可选
# ---------------------------------------------------------------------------


def test_matrix_x_only_y_none() -> None:
    m = XYMatrixSpec(x=XYAxisSpec(axis="steps", values=[20, 25, 30]))
    assert m.y is None


def test_matrix_xy_both_set() -> None:
    m = XYMatrixSpec(
        x=XYAxisSpec(axis="steps", values=[20, 25]),
        y=XYAxisSpec(axis="cfg_scale", values=[3.0, 4.0]),
    )
    assert m.y is not None
    assert m.y.axis == "cfg_scale"


# ---------------------------------------------------------------------------
# GenerateConfig 跨字段校验
# ---------------------------------------------------------------------------


def _gen(**overrides):
    """构造一份合法的 GenerateConfig（默认 xy_matrix=None）。"""
    base = dict(
        transformer_path="t",
        vae_path="v",
        text_encoder_path="te",
    )
    base.update(overrides)
    return GenerateConfig(**base)


def test_generate_default_xy_none() -> None:
    g = _gen()
    assert g.xy_matrix is None


def test_generate_xy_with_single_prompt_ok() -> None:
    g = _gen(
        xy_matrix=XYMatrixSpec(x=XYAxisSpec(axis="steps", values=[20, 25])),
    )
    assert g.xy_matrix is not None


def test_generate_xy_with_multi_prompt_rejected() -> None:
    """xy_matrix + 多 prompt 互斥（排列爆炸）。"""
    with pytest.raises(ValidationError, match="xy_matrix 与多 prompt 互斥"):
        _gen(
            prompts=["p1", "p2"],
            xy_matrix=XYMatrixSpec(x=XYAxisSpec(axis="steps", values=[20])),
        )


def test_generate_xy_with_count_gt_1_rejected() -> None:
    with pytest.raises(ValidationError, match="xy_matrix 与 count>1 互斥"):
        _gen(
            count=3,
            xy_matrix=XYMatrixSpec(x=XYAxisSpec(axis="steps", values=[20])),
        )


def test_generate_xy_lora_scale_requires_lora_index() -> None:
    """axis=lora_scale 没填 lora_index → 报错。"""
    with pytest.raises(ValidationError, match="必须指定 lora_index"):
        _gen(
            lora_configs=[LoraEntry(path="/a.safetensors", scale=1.0)],
            xy_matrix=XYMatrixSpec(
                x=XYAxisSpec(axis="lora_scale", values=[0.5, 1.0]),
            ),
        )


def test_generate_xy_lora_index_out_of_range() -> None:
    """lora_index=2 但只有 1 个 lora_configs → 报错。"""
    with pytest.raises(ValidationError, match="lora_index=2 越界"):
        _gen(
            lora_configs=[LoraEntry(path="/a.safetensors", scale=1.0)],
            xy_matrix=XYMatrixSpec(
                x=XYAxisSpec(axis="lora_scale", values=[0.5, 1.0], lora_index=2),
            ),
        )


def test_generate_xy_non_lora_axis_with_lora_index_rejected() -> None:
    """axis=steps 不允许设 lora_index。"""
    with pytest.raises(ValidationError, match="不允许设 lora_index"):
        _gen(
            xy_matrix=XYMatrixSpec(
                x=XYAxisSpec(axis="steps", values=[20], lora_index=0),
            ),
        )


def test_axis_lora_path_now_rejected() -> None:
    """lora_path 轴已删；不同 LoRA 切换通过 lora_ckpt 处理（视为同一 lora_index 不同 path）。"""
    with pytest.raises(ValidationError):
        XYAxisSpec(axis="lora_path", values=["/a/v1.safetensors"], lora_index=0)  # type: ignore[arg-type]


def test_generate_xy_axis_value_type_mismatch() -> None:
    """axis=steps + 浮点 values → 报错（按 axis 类型校验）。"""
    with pytest.raises(ValidationError, match="values 必须为 int"):
        _gen(
            xy_matrix=XYMatrixSpec(
                x=XYAxisSpec(axis="steps", values=[1.5, 2.5]),
            ),
        )


def test_generate_xy_y_axis_validated_too() -> None:
    """y 轴的 lora_index 越界也要被检测到。"""
    with pytest.raises(ValidationError, match="越界"):
        _gen(
            xy_matrix=XYMatrixSpec(
                x=XYAxisSpec(axis="steps", values=[20]),
                y=XYAxisSpec(axis="lora_scale", values=[0.5], lora_index=5),
            ),
        )


def test_generate_legacy_attention_with_xy_compatible() -> None:
    """老 cfg（xformers/flash_attn 双 bool）+ xy_matrix 共存可以正常 migrate。"""
    g = GenerateConfig.model_validate({
        "transformer_path": "t",
        "vae_path": "v",
        "text_encoder_path": "te",
        "xformers": True,  # 老字段
        "xy_matrix": {
            "x": {"axis": "steps", "values": [20, 25]},
        },
    })
    assert g.attention_backend == "xformers"  # migrate 生效
    assert g.xy_matrix is not None


def test_generate_xy_serialize_round_trip() -> None:
    """model_dump → model_validate 等幂（确保 server 端透传不丢字段）。"""
    g = _gen(
        lora_configs=[LoraEntry(path="/a.safetensors", scale=1.0)],
        xy_matrix=XYMatrixSpec(
            x=XYAxisSpec(axis="lora_scale", values=[0.5, 0.8, 1.0], lora_index=0),
            y=XYAxisSpec(axis="steps", values=[20, 25]),
        ),
    )
    dumped = g.model_dump()
    g2 = GenerateConfig.model_validate(dumped)
    assert g2.xy_matrix is not None
    assert g2.xy_matrix.x.axis == "lora_scale"
    assert g2.xy_matrix.x.lora_index == 0
    assert g2.xy_matrix.y is not None
    assert g2.xy_matrix.y.values == [20, 25]
