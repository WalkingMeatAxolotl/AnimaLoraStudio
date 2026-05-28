"""Preset 双向流（PP6.2）。

`fork_preset_for_version` —— 全局 preset 复制进 version 私有 config，立即
应用项目特定字段（data_dir / output_dir / output_name 等）。

`save_version_config_as_preset` —— version 私有 config 反向导出回全局
preset 池；项目特定字段清回 schema 默认值（不带项目数据走出去）。
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from . import io as presets_io
from ... import secrets
from ...schema import TrainingConfig
from ..models import downloader as model_downloader
from .. import version_config


def _auto_sync_paths() -> bool:
    """读 settings.models.auto_sync_paths（默认 ON）。

    ON  → fork 用 Settings 全局值覆盖 4 个模型字段；「保存为预设」时这 4 字段
          清回 Settings 默认（不带本机自定义路径出去）。
    OFF → fork 尊重预设值；「保存为预设」原样保留预设里的绝对路径。
    """
    try:
        return bool(secrets.load().models.auto_sync_paths)
    except Exception:
        return True


def fork_preset_for_version(
    src_preset_name: str,
    project: dict[str, Any],
    version: dict[str, Any],
) -> dict[str, Any]:
    """从全局 preset 复制一份进 version 私有 config。

    1. 读全局 preset（presets_io 校验；老相对路径已在读取层转绝对）
    2. 应用项目特定字段（data_dir / output_dir / output_name…）
    3. **可选** 应用当前全局模型路径（受 `models.auto_sync_paths` toggle 控制）：
       - toggle ON（默认 / 多数用户）：用 `default_paths_for_new_version()` 覆盖 4
         字段。Settings 切了 selected_anima → 后续新 version 自动用新值。
       - toggle OFF（独立模型用户）：尊重预设里的绝对路径，不覆盖。
    4. 写到 `versions/{label}/config.yaml`
    返回最终落盘的 config dict。
    """
    src = presets_io.read_preset(src_preset_name)
    new_data = deepcopy(src)
    new_data.update(version_config.project_specific_overrides(project, version))
    if _auto_sync_paths():
        new_data.update(model_downloader.default_paths_for_new_version())
    version_config.write_version_config(
        project, version, new_data, force_project_overrides=True
    )
    return version_config.read_version_config(project, version)


def save_version_config_as_preset(
    project: dict[str, Any],
    version: dict[str, Any],
    target_preset_name: str,
    *, overwrite: bool = False,
) -> dict[str, Any]:
    """version 私有 config → 全局 preset。

    1. 读 version 私有 config
    2. 项目特定字段清回 TrainingConfig 默认值（不带项目数据走出去）
    3. **可选** 4 个模型字段清回当前 Settings 默认（受 toggle 控制）：
       - toggle ON：清成 `default_paths_for_new_version()` 算的当前 Settings 绝对值，
         避免把"我本机的自定义路径"带进预设池。
       - toggle OFF：原样保留 version yaml 里的绝对路径（独立模型用户主动设置的值）。
    4. 写 `presets/{target_preset_name}.yaml`
    返回最终落盘的 preset dict。
    """
    src = version_config.read_version_config(project, version)
    cleaned = deepcopy(src)
    defaults = TrainingConfig().model_dump()
    for f in version_config.PROJECT_SPECIFIC_FIELDS:
        cleaned[f] = defaults.get(f)
    if _auto_sync_paths():
        cleaned.update(model_downloader.default_paths_for_new_version())

    target_path = presets_io._preset_path(target_preset_name)  # 校验名字合法
    if target_path.exists() and not overwrite:
        raise presets_io.PresetError(f"预设已存在: {target_preset_name}")
    presets_io.write_preset(target_preset_name, cleaned)
    return presets_io.read_preset(target_preset_name)
