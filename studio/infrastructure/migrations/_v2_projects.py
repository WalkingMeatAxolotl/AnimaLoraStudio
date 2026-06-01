"""v1 → v2: 加 projects / versions / project_jobs + tasks 扩 project_id/version_id。"""
from __future__ import annotations

import sqlite3


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, ddl: str
) -> None:
    """SQLite 没有 ADD COLUMN IF NOT EXISTS；用 PRAGMA 自检后跳过重复添加。"""
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            slug              TEXT UNIQUE NOT NULL,
            title             TEXT NOT NULL,
            stage             TEXT NOT NULL DEFAULT 'created',
            active_version_id INTEGER,
            created_at        REAL NOT NULL,
            updated_at        REAL NOT NULL,
            note              TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_projects_slug ON projects(slug);

        CREATE TABLE IF NOT EXISTS versions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id       INTEGER NOT NULL,
            label            TEXT NOT NULL,
            config_name      TEXT,
            stage            TEXT NOT NULL DEFAULT 'curating',
            created_at       REAL NOT NULL,
            output_lora_path TEXT,
            note             TEXT,
            UNIQUE(project_id, label),
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_versions_project ON versions(project_id);

        CREATE TABLE IF NOT EXISTS project_jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  INTEGER NOT NULL,
            version_id  INTEGER,
            kind        TEXT NOT NULL,
            params      TEXT NOT NULL,
            status      TEXT NOT NULL,
            started_at  REAL,
            finished_at REAL,
            pid         INTEGER,
            log_path    TEXT,
            error_msg   TEXT,
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY(version_id) REFERENCES versions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_jobs_project ON project_jobs(project_id);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON project_jobs(status);
        """
    )
    _add_column_if_missing(conn, "tasks", "project_id", "project_id INTEGER")
    _add_column_if_missing(conn, "tasks", "version_id", "version_id INTEGER")
