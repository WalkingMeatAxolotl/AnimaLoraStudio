"""daemon XY 的 fp8 lora_scale 轴：_cell_lora_configs 组装逻辑。

fp8 底模的 LoRA 是 merge 进权重的，lora_scale 轴不能 multiplier 热换——
逐格组装 lora_configs 走 CACHE.apply_loras（detach 还原 + 重 merge，
lora_ckpt 轴同款路径）。bf16 底模的 scale 轴维持 _apply_axis 的
multiplier 热换（返回 None 不重挂载）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 同 test_anima_generate_xy.py：让 import anima_daemon 找到 runtime/ 邻居
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "runtime"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import anima_daemon  # noqa: E402

_cell = anima_daemon._cell_lora_configs

_PATHS = ["a.safetensors", "b.safetensors"]
_SCALES = [0.8, 0.6]


def test_iter_xy_cells_x_lora_axis_uses_x_outer_order() -> None:
    """仅 X 改 LoRA 时，同一 checkpoint 连续跑完所有 Y 值。"""
    cells = list(anima_daemon._iter_xy_cells(
        {"axis": "lora_ckpt", "values": ["a", "b", "c"]},
        {"axis": "cfg_scale", "values": [3.0, 5.0]},
    ))

    assert cells == [
        (0, 0, "a", 3.0), (0, 1, "a", 5.0),
        (1, 0, "b", 3.0), (1, 1, "b", 5.0),
        (2, 0, "c", 3.0), (2, 1, "c", 5.0),
    ]
    assert {(xi, yi) for xi, yi, _xv, _yv in cells} == {
        (xi, yi) for xi in range(3) for yi in range(2)
    }


def test_iter_xy_cells_y_lora_axis_keeps_y_outer_order() -> None:
    cells = list(anima_daemon._iter_xy_cells(
        {"axis": "cfg_scale", "values": [3.0, 5.0]},
        {"axis": "lora_ckpt", "values": ["a", "b"]},
    ))

    assert [(xi, yi, xv, yv) for xi, yi, xv, yv in cells] == [
        (0, 0, 3.0, "a"), (1, 0, 5.0, "a"),
        (0, 1, 3.0, "b"), (1, 1, 5.0, "b"),
    ]


def test_iter_xy_cells_fp8_scale_axis_uses_x_outer_order() -> None:
    cells = list(anima_daemon._iter_xy_cells(
        {"axis": "lora_scale", "values": [0.5, 1.0]},
        {"axis": "steps", "values": [20, 30]},
        fp8_scale_axes=True,
    ))

    assert [(xi, yi) for xi, yi, _xv, _yv in cells] == [
        (0, 0), (0, 1), (1, 0), (1, 1),
    ]


def test_iter_xy_cells_bf16_scale_does_not_use_expensive_x_order() -> None:
    """动态 BF16 multiplier 可热更新，非 FP8 scale 不触发列优先遍历。"""
    cells = list(anima_daemon._iter_xy_cells(
        {"axis": "lora_scale", "values": [0.5, 1.0]},
        {"axis": "steps", "values": [20, 30]},
        fp8_scale_axes=False,
    ))

    assert [(xi, yi) for xi, yi, _xv, _yv in cells] == [
        (0, 0), (1, 0), (0, 1), (1, 1),
    ]


def test_iter_xy_cells_both_lora_axes_keeps_stable_row_major_order() -> None:
    """两轴都改 LoRA 时保持原有稳定的 Y 外层顺序。"""
    cells = list(anima_daemon._iter_xy_cells(
        {"axis": "lora_ckpt", "values": ["a", "b"]},
        {"axis": "lora_scale", "values": [0.5, 1.0]},
        fp8_scale_axes=True,
    ))

    assert [(xi, yi, xv, yv) for xi, yi, xv, yv in cells] == [
        (0, 0, "a", 0.5), (1, 0, "b", 0.5),
        (0, 1, "a", 1.0), (1, 1, "b", 1.0),
    ]


def test_bf16_scale_axis_returns_none() -> None:
    """bf16（fp8_scale_axes=False）scale 轴走 multiplier 热换，不重挂载。"""
    out = _cell(
        {"axis": "lora_scale", "values": []}, None, 0.3, None,
        _PATHS, _SCALES, fp8_scale_axes=False,
    )
    assert out is None


def test_non_lora_axes_return_none() -> None:
    out = _cell(
        {"axis": "steps"}, {"axis": "cfg_scale"}, 20, 3.5,
        _PATHS, _SCALES, fp8_scale_axes=True,
    )
    assert out is None


def test_fp8_x_scale_sets_all_entries() -> None:
    out = _cell(
        {"axis": "lora_scale"}, {"axis": "steps"}, 0.3, 20,
        _PATHS, _SCALES, fp8_scale_axes=True,
    )
    assert out == [
        {"path": "a.safetensors", "scale": 0.3},
        {"path": "b.safetensors", "scale": 0.3},
    ]


def test_fp8_y_scale_sets_all_entries() -> None:
    out = _cell(
        {"axis": "steps"}, {"axis": "lora_scale"}, 20, 0.9,
        _PATHS, _SCALES, fp8_scale_axes=True,
    )
    assert all(lc["scale"] == 0.9 for lc in out)
    assert [lc["path"] for lc in out] == _PATHS


def test_fp8_both_scale_axes_y_wins() -> None:
    """x/y 都是 scale 轴时 y 后写赢——与 _apply_axis 的 x→y 顺序一致。"""
    out = _cell(
        {"axis": "lora_scale"}, {"axis": "lora_scale"}, 0.3, 0.9,
        _PATHS, _SCALES, fp8_scale_axes=True,
    )
    assert all(lc["scale"] == 0.9 for lc in out)


def test_fp8_scale_with_ckpt_axis_combined() -> None:
    """x=scale + y=ckpt：path 换（按 lora_index）+ scale=cell 值同时生效。"""
    out = _cell(
        {"axis": "lora_scale"},
        {"axis": "lora_ckpt", "lora_index": 1},
        0.5, "c.safetensors",
        _PATHS, _SCALES, fp8_scale_axes=True,
    )
    assert out == [
        {"path": "a.safetensors", "scale": 0.5},
        {"path": "c.safetensors", "scale": 0.5},
    ]


def test_ckpt_axis_keeps_base_scales() -> None:
    """ckpt 轴（bf16/fp8 通用）只换 path，scale 保持循环外快照。"""
    out = _cell(
        {"axis": "lora_ckpt", "lora_index": 0}, None, "c.safetensors", None,
        _PATHS, _SCALES, fp8_scale_axes=False,
    )
    assert out == [
        {"path": "c.safetensors", "scale": 0.8},
        {"path": "b.safetensors", "scale": 0.6},
    ]


def test_single_axis_y_none() -> None:
    """单轴（y_spec=None）时 yv=None 不参与。"""
    out = _cell(
        {"axis": "lora_scale"}, None, 0.4, None,
        _PATHS, _SCALES, fp8_scale_axes=True,
    )
    assert all(lc["scale"] == 0.4 for lc in out)
