"""Project 数据模型 + 物理目录 + stage 推进。

Project 是 Pipeline 的最外层容器：每次 LoRA 训练对应一个 project，
包含 download/ 和若干 versions/。slug 一旦生成就不可改（路径锚点）；
title 和 note 可改。

软删：目录搬到 `studio_data/_trash/projects/{slug}/`，db 行真实删除
（CASCADE 删 versions / project_jobs）。配 `empty_trash()` 物理清理。
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from .paths import STUDIO_DATA

PROJECTS_DIR = STUDIO_DATA / "projects"
TRASH_DIR = STUDIO_DATA / "_trash" / "projects"

VALID_STAGES: frozenset[str] = frozenset({
    "created", "downloading", "curating", "tagging",
    "regularizing", "configured", "training", "done",
})


class ProjectError(Exception):
    """Project 业务错误（不存在 / 名字非法 / 冲突）。"""


# ---------------------------------------------------------------------------
# slug
# ---------------------------------------------------------------------------

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(title: str) -> str:
    """转 ASCII 小写 + 连字符。空串 / 全非 ASCII → 'project'。"""
    s = _NON_SLUG.sub("-", title.lower()).strip("-")
    return s or "project"


def _unique_slug(conn: sqlite3.Connection, base: str) -> str:
    """如果 base 已被占用，加 -2 -3 后缀直到不冲突。"""
    n = 1
    candidate = base
    while conn.execute(
        "SELECT 1 FROM projects WHERE slug = ?", (candidate,)
    ).fetchone():
        n += 1
        candidate = f"{base}-{n}"
    return candidate


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def project_dir(project_id: int, slug: str) -> Path:
    return PROJECTS_DIR / f"{project_id}-{slug}"


def _write_project_json(p: dict[str, Any]) -> None:
    """同步 project.json 到磁盘。stage / active_version_id 等字段冗余存。"""
    pdir = project_dir(p["id"], p["slug"])
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "project.json").write_text(
        json.dumps(p, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def _row_to_project(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return dict(row) if row else None


def create_project(
    conn: sqlite3.Connection,
    *,
    title: str,
    slug: Optional[str] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    title = (title or "").strip()
    if not title:
        raise ProjectError("title 不能为空")
    base_slug = slug or slugify(title)
    final_slug = _unique_slug(conn, base_slug)
    now = time.time()
    cur = conn.execute(
        "INSERT INTO projects(slug, title, stage, created_at, updated_at, note) "
        "VALUES (?, ?, 'created', ?, ?, ?)",
        (final_slug, title, now, now, note),
    )
    conn.commit()
    pid = int(cur.lastrowid)
    pdir = project_dir(pid, final_slug)
    (pdir / "download").mkdir(parents=True, exist_ok=True)
    (pdir / "versions").mkdir(parents=True, exist_ok=True)
    p = _must_get(conn, pid)
    _write_project_json(p)
    return p


def get_project(
    conn: sqlite3.Connection, project_id: int
) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM projects WHERE id = ?", (project_id,)
    ).fetchone()
    return _row_to_project(row)


def _must_get(conn: sqlite3.Connection, project_id: int) -> dict[str, Any]:
    p = get_project(conn, project_id)
    if not p:
        raise ProjectError(f"项目不存在: id={project_id}")
    return p


def list_projects(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM projects ORDER BY updated_at DESC"
        )
    ]


_UPDATABLE = {"title", "note", "stage", "active_version_id"}


def update_project(
    conn: sqlite3.Connection, project_id: int, **fields: Any
) -> dict[str, Any]:
    p = _must_get(conn, project_id)
    keep = {k: v for k, v in fields.items() if k in _UPDATABLE}
    if "stage" in keep and keep["stage"] not in VALID_STAGES:
        raise ProjectError(f"非法 stage: {keep['stage']!r}")
    if not keep:
        return p
    cols = ", ".join(f"{k} = ?" for k in keep)
    params: list[Any] = list(keep.values())
    cols += ", updated_at = ?"
    params.append(time.time())
    params.append(project_id)
    conn.execute(f"UPDATE projects SET {cols} WHERE id = ?", params)
    conn.commit()
    p = _must_get(conn, project_id)
    _write_project_json(p)
    return p


def soft_delete_project(conn: sqlite3.Connection, project_id: int) -> None:
    """目录搬到 `_trash/`，db 行删除（CASCADE 清掉 versions/project_jobs）。"""
    p = _must_get(conn, project_id)
    src = project_dir(p["id"], p["slug"])
    if src.exists():
        TRASH_DIR.mkdir(parents=True, exist_ok=True)
        # 冲突时加时间戳后缀
        dst = TRASH_DIR / f"{p['id']}-{p['slug']}"
        if dst.exists():
            dst = TRASH_DIR / f"{p['id']}-{p['slug']}-{int(time.time())}"
        shutil.move(str(src), str(dst))
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()


def empty_trash() -> int:
    """物理删除 `_trash/projects/` 下所有内容；返回删掉的项目目录数。"""
    if not TRASH_DIR.exists():
        return 0
    n = 0
    for child in list(TRASH_DIR.iterdir()):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            n += 1
        else:
            child.unlink(missing_ok=True)
    return n


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------


def advance_stage(
    conn: sqlite3.Connection,
    project_id: int,
    target: str,
) -> dict[str, Any]:
    """显式设 stage（PP1 不强制顺序，由调用方决定何时推进）。"""
    if target not in VALID_STAGES:
        raise ProjectError(f"非法 stage: {target!r}")
    return update_project(conn, project_id, stage=target)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def stats_for_project(p: dict[str, Any]) -> dict[str, Any]:
    """轻量统计：download/ 下的图片数量。version 级统计在 versions.py。"""
    from .datasets import IMAGE_EXTS  # 复用既有扩展名集

    pdir = project_dir(p["id"], p["slug"])
    download = pdir / "download"
    if not download.exists():
        return {"download_image_count": 0}
    cnt = sum(
        1 for f in download.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    return {"download_image_count": cnt}


def projects_with_stats(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        merged = dict(r)
        merged.update(stats_for_project(r))
        out.append(merged)
    return out
