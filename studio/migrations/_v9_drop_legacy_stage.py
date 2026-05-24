"""v8 → v9: 物理删除 projects.stage 与 versions.stage 列 (ADR-0007 PR-5 destructive)。

PR-2 v8 加 status / phase / last_failure_reason；PR-3 / PR-5 commit 1/2/3 把所有
读写 stage 的代码移除。此 migration 是 destructive 收尾，把两个老列从 DB 拔掉。

**显式打破 `studio/migrations/__init__.py` 既有约定**（"不允许向后改写已有列"）。
ADR-0007 §后果 已记录此例外。

SQLite < 3.35 不支持 ``ALTER TABLE DROP COLUMN``，统一走 recreate-table 模式：
- CREATE TABLE {t}_new （不含 stage 列）
- INSERT _new (cols) SELECT cols FROM 原表
- DROP 原表 / RENAME _new → 原名
- 重建 index

幂等性：第二次跑时 `stage` 列已经不存在，``SELECT stage FROM`` 会 sqlite3.OperationalError，
migration 框架不会重复跑（PRAGMA user_version 推进后跳过此函数）。
"""
from __future__ import annotations

import sqlite3


_PROJECTS_NEW_SCHEMA = """
CREATE TABLE projects_new (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    slug              TEXT UNIQUE NOT NULL,
    title             TEXT NOT NULL,
    active_version_id INTEGER,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    note              TEXT
);
"""

_PROJECTS_COLS = "id, slug, title, active_version_id, created_at, updated_at, note"


_VERSIONS_NEW_SCHEMA = """
CREATE TABLE versions_new (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id           INTEGER NOT NULL,
    label                TEXT NOT NULL,
    config_name          TEXT,
    created_at           REAL NOT NULL,
    output_lora_path     TEXT,
    note                 TEXT,
    trigger_word         TEXT NOT NULL DEFAULT '',
    status               TEXT NOT NULL DEFAULT 'preparing',
    phase                TEXT NOT NULL DEFAULT 'curating',
    last_failure_reason  TEXT,
    UNIQUE(project_id, label),
    FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
);
"""

_VERSIONS_COLS = (
    "id, project_id, label, config_name, created_at, output_lora_path, note, "
    "trigger_word, status, phase, last_failure_reason"
)


def migrate(conn: sqlite3.Connection) -> None:
    _drop_column_via_recreate(
        conn, "projects", _PROJECTS_NEW_SCHEMA, _PROJECTS_COLS,
        ("CREATE INDEX IF NOT EXISTS idx_projects_slug ON projects(slug)",),
    )
    _drop_column_via_recreate(
        conn, "versions", _VERSIONS_NEW_SCHEMA, _VERSIONS_COLS,
        ("CREATE INDEX IF NOT EXISTS idx_versions_project ON versions(project_id)",),
    )


def _drop_column_via_recreate(
    conn: sqlite3.Connection,
    table: str,
    new_schema: str,
    cols: str,
    indexes: tuple[str, ...] = (),
) -> None:
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.executescript(new_schema)
        conn.execute(f"INSERT INTO {table}_new ({cols}) SELECT {cols} FROM {table}")
        conn.execute(f"DROP TABLE {table}")
        conn.execute(f"ALTER TABLE {table}_new RENAME TO {table}")
        for idx_sql in indexes:
            conn.execute(idx_sql)
        conn.commit()
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
