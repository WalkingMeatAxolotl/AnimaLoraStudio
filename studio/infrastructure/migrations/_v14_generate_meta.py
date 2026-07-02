"""v13 → v14: tasks 加 generate_params / generate_cover（0.17 P-I forward-write）。

为未来「纯 DB 出图时间线」铺路：从现在起每次 generate 把参数快照 + 封面图地址落 DB，
前端**暂不读**（右栏仍走 live 队列 ∪ cache/disk 扫盘）。等切到 DB 驱动时数据现成、
无需数据迁移。

- generate_params：enqueue 时写的 params_snapshot（含 mode/prompt/lora/... JSON），
  未来时间线回填 + 展示用。
- generate_cover：出图完成时写的**封面图地址**——落盘走磁盘 PNG url（持久），temp 走
  cache 引用（会话级）。未来时间线据此定位图 + 判存在（在=显示、不在=已释放）。

全部 NULLABLE，不需要 backfill；老 generate task（及所有非 generate task）保持 NULL。
老 generate 图没 task_id 无法回填，本就只能向前生效（见 0.17 讨论）。
"""
from __future__ import annotations

import sqlite3

from ._v2_projects import _add_column_if_missing


def migrate(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "tasks", "generate_params", "generate_params TEXT")
    _add_column_if_missing(conn, "tasks", "generate_cover", "generate_cover TEXT")
