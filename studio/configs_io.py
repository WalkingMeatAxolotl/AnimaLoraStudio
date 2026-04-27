"""配置文件 I/O —— 用 pydantic 验证、用 PyYAML 落盘。

存储位置：`studio_data/configs/{name}.yaml`
名字白名单：`[A-Za-z0-9_-]+`，防止路径穿越和 Windows 非法字符。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .paths import USER_CONFIGS_DIR
from .schema import TrainingConfig

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class ConfigError(Exception):
    """配置 I/O 错误。"""


def _validate_name(name: str) -> None:
    if not NAME_PATTERN.fullmatch(name):
        raise ConfigError(f"非法配置名: {name!r}（只允许字母/数字/下划线/连字符）")


def _config_path(name: str, base: Path | None = None) -> Path:
    _validate_name(name)
    return (base or USER_CONFIGS_DIR) / f"{name}.yaml"


def list_configs(base: Path | None = None) -> list[dict[str, Any]]:
    """返回 `[{name, path, updated_at}]`，按修改时间倒序。"""
    base = base or USER_CONFIGS_DIR
    if not base.exists():
        return []
    items: list[dict[str, Any]] = []
    for p in base.glob("*.yaml"):
        items.append({
            "name": p.stem,
            "path": str(p),
            "updated_at": p.stat().st_mtime,
        })
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items


def read_config(name: str, base: Path | None = None) -> dict[str, Any]:
    """读取并校验配置；返回校验后的 dict（未知字段会被 forbid）。"""
    path = _config_path(name, base)
    if not path.exists():
        raise ConfigError(f"配置不存在: {name}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"配置格式错误（顶层不是 mapping）: {name}")
    try:
        cfg = TrainingConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"配置校验失败: {exc}") from exc
    return cfg.model_dump(mode="python")


def write_config(name: str, data: dict[str, Any], base: Path | None = None) -> Path:
    """先校验后写盘；任何未知字段或类型不匹配都会拒绝。"""
    path = _config_path(name, base)
    try:
        cfg = TrainingConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"配置校验失败: {exc}") from exc
    dumped = cfg.model_dump(mode="python")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dumped, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path


def delete_config(name: str, base: Path | None = None) -> None:
    path = _config_path(name, base)
    if not path.exists():
        raise ConfigError(f"配置不存在: {name}")
    path.unlink()


def duplicate_config(src: str, dst: str, base: Path | None = None) -> Path:
    src_path = _config_path(src, base)
    dst_path = _config_path(dst, base)
    if not src_path.exists():
        raise ConfigError(f"源配置不存在: {src}")
    if dst_path.exists():
        raise ConfigError(f"目标已存在: {dst}")
    dst_path.write_bytes(src_path.read_bytes())
    return dst_path
