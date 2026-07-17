"""遗留 yaml schema 迁移函数 —— 老字段名 → 新字段名映射。

两处调用：
  1. 各 model 的 `@model_validator(mode='before')` —— pydantic 校验前先洗
  2. runtime/anima_train.py 等子进程的 `apply_yaml_config` 之前显式调一次
     —— 因为 argparse_bridge.merge_yaml_into_namespace 不走 pydantic validator,
     需要这层兜底

注意：不使用 `from __future__ import annotations`——Pydantic v2 + Python 3.12+
在延迟求值模式下会将 typing._SpecialForm 当成 schema key，触发 AttributeError。
"""
from typing import Any

# PP6.1 退役的内置 HTTP monitor server 字段 —— 值早已不生效，schema 字段已删。
# 历史上每份 dump 都把这组默认值写进 yaml，所以读老配置时必须静默丢弃，
# 不能进 _tolerant_validate 的 dropped_fields 提示（否则所有旧 config.yaml /
# 预设一打开就弹兼容横幅）。TrainingConfig extra="ignore" 与 argparse_bridge
# 跳过未知键已保证不报错，这个集合只服务 dropped_fields 的降噪。
RETIRED_MONITOR_KEYS = frozenset({"no_monitor", "monitor_host", "monitor_port", "no_browser"})


def migrate_legacy_save_keys(data: Any) -> Any:
    """把老 cfg 的 save_every / save_state_every 改名带单位后缀。

    save_every       → save_every_epochs   (epoch-based)
    save_state_every → save_state_every_steps (step-based)

    Idempotent；新名已存在则丢弃同义旧名。
    """
    if not isinstance(data, dict):
        return data
    for legacy, new in (("save_every", "save_every_epochs"),
                         ("save_state_every", "save_state_every_steps")):
        if legacy in data:
            if new in data:
                data.pop(legacy)
            else:
                data[new] = data.pop(legacy)
    return data


def migrate_noise_enhancement_type(data: Any) -> Any:
    """对齐 kohya: `noise_offset` 与金字塔噪声互斥；用单一 type 字段管控。

    两步：
      1. 老 yaml 没有 `noise_enhancement_type` 时按现状字段推导：
            pyramid_noise_iters > 0   → "pyramid"
            noise_offset > 0          → "offset"
            都为 0                    → "none"
         历史 bug 配置（两者都 > 0）：选 "pyramid"。理由：Anima 旧 `make_noise`
         实现末尾 `noise = cur / cur.std().clamp(...)` 归一化会稀释 noise_offset
         的常数偏移，实际生效的主要是 pyramid，所以推导到 pyramid 跟用户主观
         观察最接近。
      2. 反组字段强制清零 —— kohya_ss issue #2599 教训：序列化层就要互斥，
         UI 隐藏字段不等于清值，否则 yaml 残值会进训练。
         （历史注:argparse 路径曾绕开 pydantic validator;刀 1 / R1 起 trainer
         与 Studio 同走 TrainingConfig 构造,本迁移对两路统一生效。runtime
         make_noise 侧另有 noise_params_from_args 按 type 分派作纵深防御。）

    Idempotent：已显式给了 `noise_enhancement_type` 就直接尊重。
    """
    if not isinstance(data, dict):
        return data
    if "noise_enhancement_type" not in data:
        pyramid = _coerce_num(data.get("pyramid_noise_iters", 0))
        offset = _coerce_num(data.get("noise_offset", 0.0))
        if pyramid > 0:
            data["noise_enhancement_type"] = "pyramid"
        elif offset > 0:
            data["noise_enhancement_type"] = "offset"
        else:
            data["noise_enhancement_type"] = "none"
    t = data["noise_enhancement_type"]
    if t == "offset":
        data["pyramid_noise_iters"] = 0
    elif t == "pyramid":
        data["noise_offset"] = 0.0
    elif t == "none":
        data["noise_offset"] = 0.0
        data["pyramid_noise_iters"] = 0
    return data


def _coerce_num(v: Any) -> float:
    """yaml 可能把数字读成 str / None；为 type 推导宽容地转 float。"""
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
