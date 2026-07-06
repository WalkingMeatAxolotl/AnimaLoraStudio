"""v16 → v17: tasks 加 params（0.17 R-2 台账合并第一步）。

资源档位模型（docs/design/queue-resource-model-0.17.md §5 R-2）：project_jobs
的九类数据作业将统一写入 tasks 表（写路径切换在 R-3），tasks 需要承接 job 的
params JSON（kind 专属参数：打标器配置 / 下载标签词 / 正则构建选项……）。

- params：kind 专属参数 JSON（同 project_jobs.params 语义）。train/reg_ai 恒
  NULL（它们的配置在 config_path yaml）；generate 的参数快照另有 _v14 的
  generate_params（时间线 forward-write，不混用）。

NULLABLE，不 backfill。旧 project_jobs 表保留只读（Q-R2：不迁移旧行、不展示）。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "params", "params TEXT")
