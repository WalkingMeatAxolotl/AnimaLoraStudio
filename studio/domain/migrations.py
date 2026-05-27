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


def migrate_legacy_attention(data: Any) -> Any:
    """把老 cfg 的 `xformers` / `flash_attn` 双 bool 映射成 `attention_backend`。

    Idempotent：已有 attention_backend 就剥掉老字段；只有老字段时按下面规则映射；
    都没有则保持空（让 schema default 生效）。

    映射规则（与原代码 `use_flash = flash_attn and not xformers` 一致 — xformers 优先）：
        xformers=true  → "xformers"
        xformers=false, flash_attn=true → "flash_attn"
        xformers=false, flash_attn=false → "none"
    """
    if not isinstance(data, dict):
        return data
    for key in (
        "wandb_enabled",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
        "wandb_mode",
        "wandb_log_samples",
    ):
        data.pop(key, None)
    if "attention_backend" in data:
        data.pop("xformers", None)
        data.pop("flash_attn", None)
        return data
    has_legacy = "xformers" in data or "flash_attn" in data
    if not has_legacy:
        return data
    xf = bool(data.pop("xformers", False))
    fa = bool(data.pop("flash_attn", True))
    if xf:
        data["attention_backend"] = "xformers"
    elif fa:
        data["attention_backend"] = "flash_attn"
    else:
        data["attention_backend"] = "none"
    return data


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
