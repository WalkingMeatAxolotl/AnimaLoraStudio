"""Preset 双向流（PP6.2）。

`fork_preset_for_version` —— 全局 preset 复制进 version 私有 config，立即
应用项目特定字段（data_dir / output_dir / output_name 等）。

`save_version_config_as_preset` —— version 私有 config 反向导出回全局
preset 池；项目特定字段清回 schema 默认值（不带项目数据走出去）。
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .. import presets_io
from ..schema import TrainingConfig
from . import model_downloader, version_config


def fork_preset_for_version(
    src_preset_name: str,
    project: dict[str, Any],
    version: dict[str, Any],
) -> dict[str, Any]:
    """从全局 preset 复制一份进 version 私有 config。

    1. 读全局 preset（presets_io 校验）
    2. 应用项目特定字段（data_dir / output_dir / output_name…）
    3. **应用当前全局模型选择**（settings.models.selected_anima 决定 transformer_path；
       VAE / qwen3 / t5 也从 models_root 算成绝对路径）—— 一次性注入，之后用户
       在 Train 页改了不会被「保存」覆盖（write_version_config 不 force 模型路径）
    4. 写到 `versions/{label}/config.yaml`
    返回最终落盘的 config dict。
    """
    src = presets_io.read_preset(src_preset_name)
    new_data = deepcopy(src)
    new_data.update(version_config.project_specific_overrides(project, version))
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
    3. 写 `presets/{target_preset_name}.yaml`
    返回最终落盘的 preset dict。
    """
    src = version_config.read_version_config(project, version)
    cleaned = deepcopy(src)
    defaults = TrainingConfig().model_dump()
    for f in version_config.PROJECT_SPECIFIC_FIELDS:
        cleaned[f] = defaults.get(f)

    target_path = presets_io._preset_path(target_preset_name)  # 校验名字合法
    if target_path.exists() and not overwrite:
        raise presets_io.PresetError(f"预设已存在: {target_preset_name}")
    presets_io.write_preset(target_preset_name, cleaned)
    return presets_io.read_preset(target_preset_name)
