"""统一 LoRA catalog：项目 checkpoint + Studio models/loras + 外部目录。

扫描结果使用短 TTL 缓存；搜索、排序和分页只在内存快照上执行，避免每次输入
搜索词都重扫磁盘。项目 checkpoint 必须通过 projects.versions 的既有 helper
获取，保持 final/step/epoch 的识别与排序规则只有一个权威实现。
"""
from __future__ import annotations

import os
import stat as stat_module
import threading
import time
from hashlib import blake2s
from pathlib import Path
from typing import Any, Literal

from .. import db, secrets
from .models import models_root
from .projects import projects, versions

CACHE_TTL_SECONDS = 20.0
SourceType = Literal["project", "studio_models", "external"]

_cache_lock = threading.Lock()
_cache_key: tuple[str, ...] | None = None
_cache_created_monotonic = 0.0
_cache_snapshot: dict[str, Any] | None = None


def _absolute_path(path: Path) -> Path:
    """不要求路径存在的绝对路径；resolve 失败时仍给稳定的绝对表示。"""
    try:
        return path.expanduser().resolve(strict=False)
    except OSError:
        return Path(os.path.abspath(os.path.expanduser(str(path))))


def _path_identity(path: Path) -> str:
    """按当前平台路径语义归一化，用于完整路径去重。"""
    return os.path.normcase(os.path.normpath(str(_absolute_path(path))))


def _external_source_id(path: Path) -> str:
    digest = blake2s(_path_identity(path).encode("utf-8"), digest_size=6).hexdigest()
    return f"external:{digest}"


def _source(
    source_type: SourceType,
    source_id: str,
    label: str,
    path: Path,
    *,
    project_archived: bool = False,
) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "source_id": source_id,
        "source_label": label,
        "path": str(_absolute_path(path)),
        "item_count": 0,
        "error": None,
        "project_archived": project_archived,
    }


def _error_text(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _project_item(
    checkpoint: dict[str, Any],
    *,
    project: dict[str, Any],
    version: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any] | None:
    path = _absolute_path(Path(str(checkpoint.get("path") or "")))
    if not path.name:
        return None
    try:
        stat = path.stat()
        size = stat.st_size
        mtime = stat.st_mtime
    except OSError:
        size = 0
        mtime = float(checkpoint.get("mtime") or 0.0)
    return {
        "path": str(path),
        "name": path.name,
        "relative_path": path.name,
        "size": size,
        "mtime": mtime,
        "source_type": source["source_type"],
        "source_id": source["source_id"],
        "source_label": source["source_label"],
        "project_id": int(project["id"]),
        "version_id": int(version["version_id"]),
        "project_title": str(project["title"]),
        "version_label": str(version["label"]),
        "project_archived": bool(project.get("archived_at")),
        "kind": str(checkpoint.get("kind") or "other"),
    }


def _is_link_directory(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction and is_junction())
    except OSError:
        return True


def _scan_directory(
    root: Path,
    *,
    source: dict[str, Any],
    missing_is_empty: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    """递归扫描目录，不跟随 symlink/junction；错误限制在当前来源。"""
    root = _absolute_path(root)
    try:
        root_stat = root.stat()
    except FileNotFoundError:
        if missing_is_empty:
            return [], None
        return [], f"Directory is not accessible: {root}"
    except OSError as exc:
        return [], _error_text(exc)
    if not stat_module.S_ISDIR(root_stat.st_mode):
        return [], f"Directory is not accessible: {root}"

    items: list[dict[str, Any]] = []
    errors: list[str] = []

    def _onerror(exc: OSError) -> None:
        errors.append(_error_text(exc))

    try:
        for current, dirnames, filenames in os.walk(
            root, topdown=True, onerror=_onerror, followlinks=False
        ):
            current_path = Path(current)
            dirnames[:] = sorted(
                name
                for name in dirnames
                if not _is_link_directory(current_path / name)
            )
            for filename in sorted(filenames):
                if Path(filename).suffix.lower() != ".safetensors":
                    continue
                path = current_path / filename
                try:
                    if not path.is_file():
                        continue
                    stat = path.stat()
                    relative = path.relative_to(root).as_posix()
                except OSError as exc:
                    errors.append(_error_text(exc))
                    continue
                items.append({
                    "path": str(_absolute_path(path)),
                    "name": path.name,
                    "relative_path": relative,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "source_type": source["source_type"],
                    "source_id": source["source_id"],
                    "source_label": source["source_label"],
                    "project_id": None,
                    "version_id": None,
                    "project_title": None,
                    "version_label": None,
                    "project_archived": False,
                    "kind": "other",
                })
    except OSError as exc:
        errors.append(_error_text(exc))
    error = "; ".join(dict.fromkeys(errors)) or None
    return items, error


def _build_snapshot() -> dict[str, Any]:
    cfg = secrets.load()
    default_root = _absolute_path(models_root() / "loras")
    all_items: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    # 项目是第一优先级：若同一完整路径也被某个目录来源覆盖，保留带项目语义的项。
    try:
        with db.connection_for() as conn:
            project_rows = projects.list_projects(conn)
            for project in project_rows:
                pid = int(project["id"])
                source = _source(
                    "project",
                    f"project:{pid}",
                    str(project["title"]),
                    projects.project_dir(pid, str(project["slug"])),
                    project_archived=bool(project.get("archived_at")),
                )
                sources.append(source)
                try:
                    groups = versions.list_project_lora_ckpts(conn, project)
                    for group in groups:
                        for checkpoint in group["items"]:
                            item = _project_item(
                                checkpoint,
                                project=project,
                                version=group,
                                source=source,
                            )
                            if item is not None:
                                all_items.append(item)
                except Exception as exc:  # 单项目失败不影响其它项目/目录来源
                    source["error"] = _error_text(exc)
    except Exception as exc:  # DB 整体不可用时目录来源仍可浏览
        failed = _source(
            "project", "project:all", "Studio projects", projects.PROJECTS_DIR
        )
        failed["error"] = _error_text(exc)
        sources.append(failed)

    default_source = _source(
        "studio_models", "studio_models", "Studio models/loras", default_root
    )
    sources.append(default_source)
    default_items, default_error = _scan_directory(
        default_root, source=default_source, missing_is_empty=True
    )
    default_source["error"] = default_error
    all_items.extend(default_items)

    seen_external_roots: set[str] = set()
    for raw in cfg.generate.lora_catalog_dirs:
        raw = str(raw).strip()
        if not raw:
            continue
        root = _absolute_path(Path(raw))
        root_key = _path_identity(root)
        if root_key in seen_external_roots:
            continue
        seen_external_roots.add(root_key)
        source = _source(
            "external", _external_source_id(root), root.name or str(root), root
        )
        sources.append(source)
        try:
            items, error = _scan_directory(
                root, source=source, missing_is_empty=False
            )
            source["error"] = error
            all_items.extend(items)
        except Exception as exc:  # 防御性隔离第三方盘/网络盘异常
            source["error"] = _error_text(exc)

    # 规范化完整路径去重，保留上述优先顺序；同 basename 不同路径不会合并。
    unique_items: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for item in all_items:
        identity = _path_identity(Path(item["path"]))
        if identity in seen_paths:
            continue
        seen_paths.add(identity)
        unique_items.append(item)

    counts: dict[str, int] = {}
    for item in unique_items:
        sid = str(item["source_id"])
        counts[sid] = counts.get(sid, 0) + 1
    for source in sources:
        source["item_count"] = counts.get(str(source["source_id"]), 0)

    return {
        "items": unique_items,
        "sources": sources,
        "generated_at": time.time(),
    }


def _current_cache_key() -> tuple[str, ...]:
    cfg = secrets.load()
    roots = tuple(
        _path_identity(Path(str(path).strip()))
        for path in cfg.generate.lora_catalog_dirs
        if str(path).strip()
    )
    return (
        _path_identity(models_root() / "loras"),
        str(getattr(db, "STUDIO_DB", "")),
        _path_identity(projects.PROJECTS_DIR),
        *roots,
    )


def clear_cache() -> None:
    """清空进程内快照；公开函数供测试与未来配置写入钩子使用。"""
    global _cache_key, _cache_created_monotonic, _cache_snapshot
    with _cache_lock:
        _cache_key = None
        _cache_created_monotonic = 0.0
        _cache_snapshot = None


def _snapshot(*, refresh: bool) -> tuple[dict[str, Any], bool]:
    global _cache_key, _cache_created_monotonic, _cache_snapshot
    key = _current_cache_key()
    now = time.monotonic()
    with _cache_lock:
        if (
            not refresh
            and _cache_snapshot is not None
            and _cache_key == key
            and now - _cache_created_monotonic < CACHE_TTL_SECONDS
        ):
            return _cache_snapshot, True
        built = _build_snapshot()
        _cache_key = key
        _cache_created_monotonic = time.monotonic()
        _cache_snapshot = built
        return built, False


def _source_matches(value: str | None, row: dict[str, Any]) -> bool:
    if not value or value == "all":
        return True
    aliases = {
        "projects": "project",
        "studio": "studio_models",
    }
    wanted = aliases.get(value, value)
    return wanted in {row["source_type"], row["source_id"]}


def query_catalog(
    *,
    q: str = "",
    source: str | None = None,
    sort: Literal["recommended", "name", "mtime", "size", "source"] = "recommended",
    order: Literal["asc", "desc"] = "asc",
    include_archived: bool = False,
    limit: int = 100,
    cursor: int = 0,
    refresh: bool = False,
) -> dict[str, Any]:
    snapshot, cached = _snapshot(refresh=refresh)
    needle = q.strip().casefold()

    def _visible(item: dict[str, Any]) -> bool:
        if item["project_archived"] and not include_archived:
            return False
        if not _source_matches(source, item):
            return False
        if not needle:
            return True
        haystack = "\n".join(
            str(item.get(key) or "")
            for key in (
                "name", "relative_path", "path", "source_label",
                "project_title", "version_label", "kind",
            )
        ).casefold()
        return needle in haystack

    items = [item for item in snapshot["items"] if _visible(item)]
    if sort != "recommended":
        if sort == "name":
            key = lambda item: (item["name"].casefold(), item["path"].casefold())
        elif sort == "mtime":
            key = lambda item: (item["mtime"], item["path"].casefold())
        elif sort == "size":
            key = lambda item: (item["size"], item["path"].casefold())
        else:
            key = lambda item: (
                item["source_label"].casefold(),
                item["name"].casefold(),
                item["path"].casefold(),
            )
        items.sort(key=key, reverse=order == "desc")
    elif order == "desc":
        items.reverse()

    visible_sources = [
        row for row in snapshot["sources"]
        if (include_archived or not row["project_archived"])
        and _source_matches(source, row)
    ]
    total = len(items)
    page = items[cursor:cursor + limit]
    next_cursor = cursor + limit if cursor + limit < total else None
    return {
        "items": page,
        "sources": visible_sources,
        "total": total,
        "cursor": cursor,
        "next_cursor": next_cursor,
        "generated_at": snapshot["generated_at"],
        "cached": cached,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
    }
