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
import logging
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from ...paths import STUDIO_DATA

PROJECTS_DIR = STUDIO_DATA / "projects"
logger = logging.getLogger(__name__)

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
        raise ProjectError(
            "Project title is required", code="project.title_required",
        )
    # 用户可在创建时显式指定 slug（前端已校验为 ASCII）。这里仍统一过一遍
    # slugify：既归一化大小写 / 非法字符，也防 API 直连绕过前端校验。留空 / 清理
    # 后为空 → 回退到从 title 派生（全非 ASCII title 仍走 "project" 兜底）。
    slug = (slug or "").strip()
    base_slug = slugify(slug) if slug else slugify(title)
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
        raise ProjectError(
            "Project not found", code="project.not_found",
            details={"id": project_id}, http_status=404,
        )
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


def _unique_project_ids(project_ids: Iterable[int]) -> list[int]:
    """De-duplicate IDs while preserving the caller's visible selection order."""
    return list(dict.fromkeys(int(project_id) for project_id in project_ids))


def _must_get_projects(
    conn: sqlite3.Connection, project_ids: Iterable[int],
) -> list[dict[str, Any]]:
    return [_must_get(conn, project_id) for project_id in _unique_project_ids(project_ids)]


def set_archived_many(
    conn: sqlite3.Connection, project_ids: Iterable[int], archived: bool,
) -> list[dict[str, Any]]:
    """Archive or restore a validated project selection in one DB transaction."""
    selected = _must_get_projects(conn, project_ids)
    expected_archived = not archived
    for project in selected:
        if bool(project.get("archived_at")) != expected_archived:
            action = "archive" if archived else "restore"
            raise ProjectError(
                f"Project cannot be included in batch {action}",
                code="project.batch_state_invalid",
                details={"id": project["id"], "archived": bool(project.get("archived_at"))},
            )

    archived_at = time.time() if archived else None
    conn.executemany(
        "UPDATE projects SET archived_at = ? WHERE id = ?",
        [(archived_at, project["id"]) for project in selected],
    )
    conn.commit()

    updated = [_must_get(conn, project["id"]) for project in selected]
    for project in updated:
        _write_project_json(project)
    return updated


def delete_projects(
    conn: sqlite3.Connection,
    project_ids: Iterable[int],
    *,
    require_archived: bool = False,
) -> dict[str, Any]:
    """Delete validated projects and report any physical-directory failures.

    Task rows and ``studio_data/tasks/<id>`` live outside the project tree and are
    deliberately untouched. Lifecycle validation is all-or-nothing, while physical
    deletion is reported per project because filesystem work cannot be rolled back.
    """
    selected = _must_get_projects(conn, project_ids)
    if require_archived:
        for project in selected:
            if project.get("archived_at") is None:
                raise ProjectError(
                    "Only archived projects can be batch deleted",
                    code="project.batch_delete_requires_archived",
                    details={"id": project["id"]},
                )

    deleted_projects: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for project in selected:
        src = project_dir(project["id"], project["slug"])
        try:
            if src.exists():
                shutil.rmtree(src)
        except OSError:
            logger.exception("Failed to remove project directory for project %s", project["id"])
            failed.append({
                "id": int(project["id"]),
                "code": "project.delete_failed",
                "message": "Could not remove project directory",
            })
        else:
            deleted_projects.append(project)

    if deleted_projects:
        conn.executemany(
            "DELETE FROM projects WHERE id = ?",
            [(project["id"],) for project in deleted_projects],
        )
        conn.commit()
    return {
        "deleted": [int(project["id"]) for project in deleted_projects],
        "failed": failed,
    }


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
