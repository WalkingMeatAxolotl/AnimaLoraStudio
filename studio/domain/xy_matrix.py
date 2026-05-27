"""XY 矩阵 schema —— 轴枚举 + axis spec + matrix spec + 值校验。

设计：单 task 内循环全图（一次 model load 摊销 ~30s 启动成本）。前端拿到
samples[].xy={xi,yi,xv,yv} 元数据按 (yi, xi) 排成 grid 渲染。

轴值类型按 axis 枚举派生：
  lora_scale / cfg_scale → float
  steps                  → int
  lora_ckpt              → str (ckpt 文件路径)

历史注：lora_ckpt 轴 v1 因 AnimaLycorisAdapter 缺 unhook 接口未实现；
detach()（utils/lycoris_adapter.py）+ CACHE.apply_loras 重 inject 路径之
后补上（runtime/anima_daemon.py:_run_xy）。

轴行为：
  lora_scale：全局轴，遍历所有 adapters 把 multiplier 都覆盖为 cell 值
              （旧版只改 lora_configs[lora_index]，已废弃）。
  lora_ckpt：cell 内 mutate lora_configs[lora_index].path 然后调
              CACHE.apply_loras 重 inject（detach + reload state_dict）。

注意：不使用 `from __future__ import annotations`——Pydantic v2 + Python 3.12+
在延迟求值模式下会将 typing._SpecialForm 当成 schema key，触发 AttributeError。
"""
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

XYAxisType = Literal[
    "lora_scale",   # 把所有 LoRA 的 multiplier 都设成轴值（全局）
    "steps",        # 不同采样步数
    "cfg_scale",    # 不同 CFG
    "lora_ckpt",    # 同一 LoRA 训练过程的不同 step/epoch ckpt（找过拟合拐点）
]


class XYAxisSpec(BaseModel):
    """单轴定义：axis 枚举 + values 列表 + (lora_ckpt 时) lora_index。"""

    model_config = ConfigDict(extra="forbid")
    axis: XYAxisType = Field(..., description="轴绑定的字段")
    values: list[Any] = Field(..., min_length=1, description="此轴扫描的值列表")
    lora_index: Optional[int] = Field(
        None, ge=0,
        description="axis=lora_ckpt 时指定改 lora_configs 哪一项的 path",
    )


class XYMatrixSpec(BaseModel):
    """XY 矩阵：x 轴必填，y 可选（None = 单轴 N×1 退化成一行）。"""

    model_config = ConfigDict(extra="forbid")
    x: XYAxisSpec
    y: Optional[XYAxisSpec] = None


def _check_axis_values(axis: XYAxisSpec) -> None:
    """按 axis 枚举校验 values 类型（浮点 / 整数 / 字符串）。"""
    int_axes = {"steps"}
    float_axes = {"lora_scale", "cfg_scale"}
    str_axes = {"lora_ckpt"}  # ckpt 路径列表
    needs_lora_index = {"lora_ckpt"}  # lora_scale 改为全局轴，不再需要

    if axis.axis in int_axes:
        for v in axis.values:
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"axis={axis.axis} values 必须为 int，收到 {type(v).__name__}")
    elif axis.axis in float_axes:
        for v in axis.values:
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise ValueError(f"axis={axis.axis} values 必须为 number，收到 {type(v).__name__}")
    elif axis.axis in str_axes:
        for v in axis.values:
            if not isinstance(v, str):
                raise ValueError(f"axis={axis.axis} values 必须为 str，收到 {type(v).__name__}")

    if axis.axis in needs_lora_index and axis.lora_index is None:
        raise ValueError(f"axis={axis.axis} 必须指定 lora_index（绑定到 lora_configs 哪一项）")
    if axis.axis not in needs_lora_index and axis.lora_index is not None:
        raise ValueError(f"axis={axis.axis} 不允许设 lora_index（仅 lora_ckpt 可设）")
