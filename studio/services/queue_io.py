"""任务队列的 JSON 导入 / 导出（用于在不同机器之间分享一组训练任务）。

导出格式：
    {
      "version": 1,
      "exported_at": <unix-ts>,
      "tasks": [
        {
          "name": ...,
          "config_name": ...,
          "priority": ...,
          "config": { ... 完整 TrainingConfig ... }
        },
        ...
      ]
    }

导入：对每个 task，把 config 写到 USER_PRESETS_DIR/{config_name}.yaml（若已存在
则在名字后加 _imported_{n} 后缀），然后创建 pending 任务。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .. import db
from .presets import io as presets_io


EXPORT_VERSION = 1


def export_tasks(
    task_ids: list[int],
    *,
    db_path: Path | None = None,
    configs_base: Path | None = None,
) -> dict[str, Any]:
    """读出指定 task 与对应 config，组成导出 dict。"""
    out_tasks: list[dict[str, Any]] = []
    with db.connection_for(db_path) as conn:
        for tid in task_ids:
            t = db.get_task(conn, tid)
            if not t:
                continue
            try:
                cfg = presets_io.read_preset(t["config_name"], base=configs_base)
            except presets_io.PresetError:
                cfg = None
            out_tasks.append({
                "name": t["name"],
                "config_name": t["config_name"],
                "priority": t["priority"],
                "config": cfg,
            })
    return {
        "version": EXPORT_VERSION,
        "exported_at": time.time(),
        "tasks": out_tasks,
    }


def _unique_config_name(base_name: str, configs_base: Path | None) -> str:
    """如果 base_name 已存在，加 _imported_N 直到不冲突。"""
    if not presets_io.list_presets(base=configs_base):
        return base_name
    existing = {c["name"] for c in presets_io.list_presets(base=configs_base)}
    if base_name not in existing:
        return base_name
    n = 1
    while f"{base_name}_imported_{n}" in existing:
        n += 1
    return f"{base_name}_imported_{n}"


def import_tasks(
    payload: dict[str, Any],
    *,
    db_path: Path | None = None,
    configs_base: Path | None = None,
) -> dict[str, Any]:
    """从导出 dict 还原任务和配置。"""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != EXPORT_VERSION:
        raise ValueError(f"unsupported export version: {payload.get('version')!r}")
    tasks = payload.get("tasks") or []
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a list")

    created_ids: list[int] = []
    rename_map: dict[str, str] = {}
    with db.connection_for(db_path) as conn:
        for entry in tasks:
            cfg = entry.get("config")
            if cfg is None:
                # 没有附带 config —— 必须能在本地找到同名 config
                try:
                    presets_io.read_preset(entry["config_name"], base=configs_base)
                    final_name = entry["config_name"]
                except presets_io.PresetError:
                    continue  # 忽略此任务
            else:
                final_name = _unique_config_name(entry["config_name"], configs_base)
                presets_io.write_preset(final_name, cfg, base=configs_base)
                if final_name != entry["config_name"]:
                    rename_map[entry["config_name"]] = final_name
            tid = db.create_task(
                conn,
                name=entry.get("name", final_name),
                config_name=final_name,
                priority=int(entry.get("priority", 0) or 0),
            )
            created_ids.append(tid)

    return {
        "imported_count": len(created_ids),
        "task_ids": created_ids,
        "renamed": rename_map,
    }
