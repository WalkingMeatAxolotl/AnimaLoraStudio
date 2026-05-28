"""v3 → v4: tasks 表加 `config_path` 列（PP6.3）。

PP6.3 之后训练 task 的 yaml 不再来自全局 `presets/{config_name}.yaml`，而是
来自 version 私有 config（`projects/{id}-{slug}/versions/{label}/config.yaml`）。
supervisor 优先用 `tasks.config_path`，没有则走 `_configs_dir / config_name.yaml`
兜底（兼容老任务）。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "config_path", "config_path TEXT")
