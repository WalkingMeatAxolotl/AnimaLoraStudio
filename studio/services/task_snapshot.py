"""Task config snapshot — ADR-0007 §11.7。

task 启动时把当时的 version config.yaml 冻结一份到
``studio_data/tasks/{task_id}/snapshot/config.yaml``。

设计要点：
- **仅冻 config**，不冻 caption / 图 / 正则集（跨 OS export OK，磁盘代价 KB 级）
- 心智分离 UI：task 详情独立 [关联配置] tab，**不点 task 跳 version config 编辑页**
  → 让 user 理解 config 是历史快照，caption / 图是 version 当前状态
- 冻结时机：supervisor `_spawn_task` 把 cfg_path 给 worker 之前
- 失败不阻塞 task 启动（snapshot 是 forensics 不是必需）

用 user 视角："点 task 详情 [关联配置] 看当时跑的什么参数，按'套用此配置'按钮
跳到 ⑦ 训练 phase 页面 + prefill → 编辑 → 训练 = 新 task" （§11.7 流程）。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Optional

import yaml

from ..paths import task_dir

SNAPSHOT_CONFIG_FILENAME = "config.yaml"


def snapshot_dir(task_id: int) -> Path:
    """``studio_data/tasks/{task_id}/snapshot/``。

    跟 monitor/ samples/ run.log sibling，整组是 task 完整档案。
    路径由 `paths.task_dir` 派生，跟其他 task-scoped helper 同源；
    测试只需 monkeypatch `paths.TASKS_DIR` 一次即可隔离全部。
    """
    return task_dir(task_id) / "snapshot"


def snapshot_config_path(task_id: int) -> Path:
    return snapshot_dir(task_id) / SNAPSHOT_CONFIG_FILENAME


def has_snapshot(task_id: int) -> bool:
    return snapshot_config_path(task_id).exists()


def freeze_config(task_id: int, source: Path) -> Path:
    """复制 source yaml 到 ``snapshot_config_path(task_id)``，返回目标路径。

    重复调用会覆盖（同 task_id 重启场景）。source 不存在时 raise FileNotFoundError。
    """
    if not source.exists():
        raise FileNotFoundError(f"snapshot source not found: {source}")
    dst = snapshot_config_path(task_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dst)
    return dst


def read_snapshot_config(task_id: int) -> Optional[dict[str, Any]]:
    """读 task config snapshot；不存在返回 None。

    返回 ``{"yaml": raw_text, "config": parsed_dict}`` —— UI 既能展示原始 yaml
    （只读 monaco），也能 prefill 训练 config 表单。
    """
    p = snapshot_config_path(task_id)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8")
    parsed = yaml.safe_load(text) or {}
    if not isinstance(parsed, dict):
        parsed = {}
    return {"yaml": text, "config": parsed}
