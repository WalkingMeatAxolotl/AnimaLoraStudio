"""Project 数据模型 + 物理目录 (ADR-0007: project 极简，无过程状态)。

Project 是 Pipeline 的最外层容器：每次 LoRA 训练对应一个 project，
包含 download/ 和若干 versions/。slug 一旦生成就不可改（路径锚点）；
title 和 note 可改。

删除：直接 rmtree 项目目录 + DELETE db 行（CASCADE 清 versions /
project_jobs）。无回收站、不可恢复 —— UI 层 confirm 提示用户。

归档（v12）：archived_at 非 NULL = 归档，仅做列表软隐藏（目录 / versions /
任务全部原样）。UI 上"×"先归档，归档视图里再删才走 delete_project。
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from ...paths import STUDIO_DATA

PROJECTS_DIR = STUDIO_DATA / "projects"

from studio.domain.errors import DomainError


class ProjectError(DomainError):
    """Project 业务错误（不存在 / 名字非法 / 冲突）。

    PR-2 C3 加 DomainError base — handler 自动翻 dual-write envelope。
    """
    default_code = "project.error"


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
    """同步 project.json 到磁盘。active_version_id 等字段冗余存。"""
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
        "INSERT INTO projects(slug, title, created_at, updated_at, note) "
        "VALUES (?, ?, ?, ?, ?)",
        (final_slug, title, now, now, note),
    )
    conn.commit()
    pid = int(cur.lastrowid)
    pdir = project_dir(pid, final_slug)
    (pdir / "download").mkdir(parents=True, exist_ok=True)
    (pdir / "preprocess").mkdir(parents=True, exist_ok=True)
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


_UPDATABLE = {"title", "note", "active_version_id"}


def update_project(
    conn: sqlite3.Connection, project_id: int, **fields: Any
) -> dict[str, Any]:
    p = _must_get(conn, project_id)
    keep = {k: v for k, v in fields.items() if k in _UPDATABLE}
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


def set_archived(
    conn: sqlite3.Connection, project_id: int, archived: bool
) -> dict[str, Any]:
    """归档 / 取消归档。只写 archived_at，不动 updated_at —— 恢复后项目
    在"按活跃时间"排序里回到原来的位置，而不是顶到最前。"""
    _must_get(conn, project_id)
    conn.execute(
        "UPDATE projects SET archived_at = ? WHERE id = ?",
        (time.time() if archived else None, project_id),
    )
    conn.commit()
    p = _must_get(conn, project_id)
    _write_project_json(p)
    return p


def delete_project(conn: sqlite3.Connection, project_id: int) -> None:
    """rmtree 项目目录 + DELETE db 行（CASCADE 清掉 versions/project_jobs）。不可恢复。"""
    p = _must_get(conn, project_id)
    src = project_dir(p["id"], p["slug"])
    if src.exists():
        shutil.rmtree(src, ignore_errors=True)
    conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def stats_for_project(p: dict[str, Any]) -> dict[str, Any]:
    """轻量统计：download/ 与 preprocess/ 下的图片数量。version 级统计在 versions.py。"""
    from ...services.dataset.scan import IMAGE_EXTS  # 复用既有扩展名集

    pdir = project_dir(p["id"], p["slug"])

    def _count(d: Path) -> int:
        if not d.exists():
            return 0
        return sum(
            1 for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )

    return {
        "download_image_count": _count(pdir / "download"),
        "preprocess_image_count": _count(pdir / "preprocess"),
    }


def projects_with_stats(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        merged = dict(r)
        merged.update(stats_for_project(r))
        out.append(merged)
    return out
