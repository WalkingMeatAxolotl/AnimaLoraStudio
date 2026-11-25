"""统一 LoRA catalog：项目 checkpoint + Studio models/loras + 外部目录。

两层、按需付费：

- **来源计划**（:func:`_plan`）：只列来源 + 每个来源的文件数。项目来源用 DB 的
  versions 定位 ``output/``，目录来源用 ``os.scandir`` 递归计数；不 stat 单文件、
  不读 safetensors header、不逐文件 ``resolve()``。打开抽屉只需要这一层。
- **来源条目**（:func:`_source_items`）：进入某个来源才扫这一个来源，拿 size /
  mtime 用于排序展示。项目 checkpoint 仍走 ``versions.list_lora_ckpts``，保持
  final/step/epoch 识别与排序只有一个权威实现。

训练底模层数（``base_num_blocks``）只用于列表徽标展示，真正的兼容校验在推理
加载时另行执行，因此不在列表关键路径上：前端拿到条目后再调
:func:`read_arch` 按需补全。

两层都用短 TTL 缓存；``refresh`` 同时丢弃两层。磁盘仍是唯一事实源。
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import blake2s
from pathlib import Path
from typing import Any, Literal

from .. import db, secrets
from .inference.lora_compat import read_lora_base_arch_cached
from .models import models_root
from .projects import projects, versions

CACHE_TTL_SECONDS = 20.0
SourceType = Literal["project", "studio_models", "external"]
_LORA_SUFFIX = ".safetensors"


# ── 路径工具 ────────────────────────────────────────────────────────────────

def _absolute_path(path: Path) -> Path:
    """不要求路径存在的绝对路径；resolve 失败时仍给稳定的绝对表示。

    只用于**根目录**（每个来源一次）。根下的条目路径由根拼接而来，不再逐个
    resolve —— Windows 上每次 resolve 都是真实 IO。
    """
    try:
        return path.expanduser().resolve(strict=False)
    except OSError:
        return Path(os.path.abspath(os.path.expanduser(str(path))))


def _path_identity(path: Path) -> str:
    """按当前平台路径语义归一化（含 resolve），用于根目录去重。"""
    return os.path.normcase(os.path.normpath(str(_absolute_path(path))))


def _lexical_identity(path: str) -> str:
    """不碰磁盘的归一化：用于已基于 resolved 根拼出来的子路径。"""
    return os.path.normcase(os.path.normpath(path))


def _is_within(child: str, parent: str) -> bool:
    """两者都是 identity；child 等于 parent 或位于其下。"""
    if child == parent:
        return True
    prefix = parent if parent.endswith(os.sep) else parent + os.sep
    return child.startswith(prefix)


def _external_source_id(path: Path) -> str:
    digest = blake2s(_path_identity(path).encode("utf-8"), digest_size=6).hexdigest()
    return f"external:{digest}"


def _error_text(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _is_lora_name(name: str) -> bool:
    return name.lower().endswith(_LORA_SUFFIX)


def _entry_is_link_dir(entry: os.DirEntry[str]) -> bool:
    try:
        if entry.is_symlink():
            return True
        is_junction = getattr(entry, "is_junction", None)
        return bool(is_junction and is_junction())
    except OSError:
        return True


# ── 来源计划 ────────────────────────────────────────────────────────────────

@dataclass
class _ProjectVersion:
    version_id: int
    label: str
    vdir: Path


@dataclass
class _SourcePlan:
    info: dict[str, Any]
    project: dict[str, Any] | None = None
    versions: list[_ProjectVersion] = field(default_factory=list)
    root: Path | None = None
    missing_is_empty: bool = False
    #: 位于更高优先级来源之下的子目录（identity），扫描时整棵剪掉
    pruned_roots: frozenset[str] = frozenset()
    #: 整个根都被更高优先级来源覆盖 → 恒为空
    shadowed: bool = False
    #: 根目录存在且可枚举（目录来源）
    scannable: bool = False


@dataclass
class _Plan:
    sources: list[_SourcePlan]
    #: 项目 version 的 output/ 目录（identity）；目录来源扫描时跳过其中的文件，
    #: 让同一文件保留带项目语义的那一条
    project_output_dirs: frozenset[str]
    generated_at: float


def _source(
    source_type: SourceType,
    source_id: str,
    label: str,
    path: Path,
    *,
    project_archived: bool = False,
    created_at: float | None = None,
    updated_at: float | None = None,
) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "source_id": source_id,
        "source_label": label,
        "path": str(path),
        "item_count": 0,
        "error": None,
        "project_archived": project_archived,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def _count_flat_loras(directory: Path) -> int:
    """与 ``versions.list_lora_ckpts`` 同口径的计数（output/*.safetensors 文件）。"""
    try:
        with os.scandir(directory) as entries:
            return sum(
                1 for entry in entries
                if _is_lora_name(entry.name) and entry.is_file()
            )
    except FileNotFoundError:
        return 0


def _iter_directory_loras(
    plan: _SourcePlan,
    *,
    project_output_dirs: frozenset[str],
    errors: list[str],
) -> Iterator[tuple[os.DirEntry[str], str]]:
    """递归遍历目录来源，产出 (entry, 相对路径)。

    不跟随 symlink/junction；错误限制在当前来源（记入 ``errors``）。顺序与旧版
    ``os.walk(topdown)`` + 排序一致：先当前目录文件（按名），再按名深度优先子目录。
    Windows 上 ``DirEntry.is_file()/stat()`` 直接取目录枚举数据，不额外 IO。
    """
    if plan.shadowed or plan.root is None:
        return
    root = str(plan.root)
    stack: list[tuple[str, str]] = [(root, "")]
    while stack:
        current, relative_dir = stack.pop()
        current_identity = _lexical_identity(current)
        skip_files = current_identity in project_output_dirs
        try:
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            errors.append(_error_text(exc))
            continue
        subdirs: list[tuple[str, str]] = []
        for entry in entries:
            relative = f"{relative_dir}/{entry.name}" if relative_dir else entry.name
            try:
                if entry.is_dir(follow_symlinks=False):
                    if _entry_is_link_dir(entry):
                        continue
                    if _lexical_identity(entry.path) in plan.pruned_roots:
                        continue
                    subdirs.append((entry.path, relative))
                    continue
                if skip_files or not _is_lora_name(entry.name):
                    continue
                if not entry.is_file():
                    continue
            except OSError as exc:
                errors.append(_error_text(exc))
                continue
            yield entry, relative
        stack.extend(reversed(subdirs))


def _directory_root_error(plan: _SourcePlan) -> tuple[bool, str | None]:
    """(可扫描, 错误)。缺失的默认目录视为空来源而非错误。"""
    assert plan.root is not None
    try:
        is_dir = plan.root.is_dir()
        exists = is_dir or plan.root.exists()
    except OSError as exc:
        return False, _error_text(exc)
    if is_dir:
        return True, None
    if not exists and plan.missing_is_empty:
        return False, None
    return False, f"Directory is not accessible: {plan.root}"


def _build_plan() -> _Plan:
    cfg = secrets.load()
    plans: list[_SourcePlan] = []
    project_output_dirs: set[str] = set()

    # 项目是第一优先级：若同一完整路径也被某个目录来源覆盖，保留带项目语义的项。
    try:
        projects_root = _absolute_path(projects.PROJECTS_DIR)
        with db.connection_for() as conn:
            for project in projects.list_projects(conn):
                pid = int(project["id"])
                slug = str(project["slug"])
                pdir = projects_root / projects.project_dir(pid, slug).name
                info = _source(
                    "project",
                    f"project:{pid}",
                    str(project["title"]),
                    pdir,
                    project_archived=bool(project.get("archived_at")),
                    created_at=float(project["created_at"]),
                    updated_at=float(project["updated_at"]),
                )
                plan = _SourcePlan(info=info, project=project)
                plans.append(plan)
                try:
                    count = 0
                    for version in versions.list_versions(conn, pid):
                        label = str(version["label"])
                        vdir = pdir / "versions" / label
                        plan.versions.append(
                            _ProjectVersion(int(version["id"]), label, vdir)
                        )
                        output = vdir / "output"
                        project_output_dirs.add(_lexical_identity(str(output)))
                        count += _count_flat_loras(output)
                    info["item_count"] = count
                except Exception as exc:  # 单项目失败不影响其它项目/目录来源
                    info["error"] = _error_text(exc)
    except Exception as exc:  # DB 整体不可用时目录来源仍可浏览
        failed = _source(
            "project", "project:all", "Studio projects",
            _absolute_path(projects.PROJECTS_DIR),
        )
        failed["error"] = _error_text(exc)
        plans.append(_SourcePlan(info=failed))

    directory_plans: list[_SourcePlan] = []
    default_root = _absolute_path(models_root() / "loras")
    directory_plans.append(_SourcePlan(
        info=_source("studio_models", "studio_models", "Studio models/loras", default_root),
        root=default_root,
        missing_is_empty=True,
    ))
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
        directory_plans.append(_SourcePlan(
            info=_source("external", _external_source_id(root), root.name or str(root), root),
            root=root,
        ))

    # 目录来源之间按配置顺序优先：被更早来源覆盖的部分不再计入后来的来源。
    earlier_roots: list[str] = []
    frozen_outputs = frozenset(project_output_dirs)
    for plan in directory_plans:
        assert plan.root is not None
        identity = _lexical_identity(str(plan.root))
        plan.shadowed = any(_is_within(identity, earlier) for earlier in earlier_roots)
        plan.pruned_roots = frozenset(
            earlier for earlier in earlier_roots if _is_within(earlier, identity)
        )
        earlier_roots.append(identity)
        if not plan.shadowed:
            plan.scannable, error = _directory_root_error(plan)
            plan.info["error"] = error
            if plan.scannable:
                errors: list[str] = []
                try:
                    plan.info["item_count"] = sum(
                        1 for _ in _iter_directory_loras(
                            plan, project_output_dirs=frozen_outputs, errors=errors,
                        )
                    )
                except Exception as exc:  # 防御性隔离第三方盘/网络盘异常
                    errors.append(_error_text(exc))
                plan.info["error"] = "; ".join(dict.fromkeys(errors)) or None
        plans.append(plan)

    return _Plan(
        sources=plans,
        project_output_dirs=frozen_outputs,
        generated_at=time.time(),
    )


# ── 来源条目 ────────────────────────────────────────────────────────────────

def _project_items(plan: _SourcePlan) -> list[dict[str, Any]]:
    project = plan.project
    assert project is not None
    source = plan.info
    items: list[dict[str, Any]] = []
    for version in plan.versions:
        for checkpoint in versions.list_lora_ckpts(version.vdir, with_arch=False):
            path = str(checkpoint.get("path") or "")
            name = os.path.basename(path)
            if not name:
                continue
            items.append({
                "path": path,
                "name": name,
                "relative_path": name,
                "size": max(0, int(checkpoint.get("size") or 0)),
                "mtime": float(checkpoint.get("mtime") or 0.0),
                "source_type": source["source_type"],
                "source_id": source["source_id"],
                "source_label": source["source_label"],
                "project_id": int(project["id"]),
                "version_id": version.version_id,
                "project_title": str(project["title"]),
                "version_label": version.label,
                "project_archived": bool(project.get("archived_at")),
                "kind": str(checkpoint.get("kind") or "other"),
            })
    return items


def _directory_items(plan: _SourcePlan, project_output_dirs: frozenset[str]) -> list[dict[str, Any]]:
    source = plan.info
    if plan.shadowed or not plan.scannable:
        return []
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    for entry, relative in _iter_directory_loras(
        plan, project_output_dirs=project_output_dirs, errors=errors,
    ):
        try:
            stat = entry.stat()
        except OSError:
            continue
        items.append({
            "path": entry.path,
            "name": entry.name,
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
    return items


def _scan_source(plan: _SourcePlan, project_output_dirs: frozenset[str]) -> list[dict[str, Any]]:
    try:
        if plan.info["source_type"] == "project":
            return _project_items(plan) if plan.project is not None else []
        return _directory_items(plan, project_output_dirs)
    except Exception as exc:  # 来源间隔离：单个来源失败只影响自己
        plan.info["error"] = _error_text(exc)
        return []


# ── 缓存 ────────────────────────────────────────────────────────────────────

_cache_lock = threading.Lock()
_cache_key: tuple[str, ...] | None = None
_plan_created_monotonic = 0.0
_plan_cache: _Plan | None = None
#: source_id → (创建时刻 monotonic, 条目)
_items_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


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
    """清空进程内缓存；公开函数供测试与未来配置写入钩子使用。"""
    global _cache_key, _plan_created_monotonic, _plan_cache
    with _cache_lock:
        _cache_key = None
        _plan_created_monotonic = 0.0
        _plan_cache = None
        _items_cache.clear()


def _get_plan(*, refresh: bool) -> tuple[_Plan, bool]:
    global _cache_key, _plan_created_monotonic, _plan_cache
    key = _current_cache_key()
    now = time.monotonic()
    with _cache_lock:
        if (
            not refresh
            and _plan_cache is not None
            and _cache_key == key
            and now - _plan_created_monotonic < CACHE_TTL_SECONDS
        ):
            return _plan_cache, True
        built = _build_plan()
        _cache_key = key
        _plan_created_monotonic = time.monotonic()
        _plan_cache = built
        _items_cache.clear()
        return built, False


def _source_items(plan: _Plan, source: _SourcePlan) -> tuple[list[dict[str, Any]], bool]:
    source_id = str(source.info["source_id"])
    now = time.monotonic()
    with _cache_lock:
        hit = _items_cache.get(source_id)
        if hit is not None and now - hit[0] < CACHE_TTL_SECONDS:
            return hit[1], True
    items = _scan_source(source, plan.project_output_dirs)
    with _cache_lock:
        if _plan_cache is plan:  # 期间被 refresh 换掉的计划不回写
            _items_cache[source_id] = (time.monotonic(), items)
    return items, False


# ── 查询 ────────────────────────────────────────────────────────────────────

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
    plan, cached = _get_plan(refresh=refresh)
    needle = q.strip().casefold()

    def _visible_source(row: dict[str, Any]) -> bool:
        return (include_archived or not row["project_archived"]) and _source_matches(source, row)

    def _visible(item: dict[str, Any]) -> bool:
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

    matching = [entry for entry in plan.sources if _visible_source(entry.info)]
    items: list[dict[str, Any]] = []
    # 只扫匹配的来源；不指定来源时会扫全部（慢路径）。只要来源列表的调用方
    # 应走 list_sources。
    for entry in matching:
        source_items, source_cached = _source_items(plan, entry)
        cached = cached and source_cached
        items.extend(item for item in source_items if _visible(item))

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

    total = len(items)
    page = items[cursor:cursor + limit]
    next_cursor = cursor + limit if cursor + limit < total else None
    return {
        "items": page,
        "sources": [entry.info for entry in matching],
        "total": total,
        "cursor": cursor,
        "next_cursor": next_cursor,
        "generated_at": plan.generated_at,
        "cached": cached,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
    }


def list_sources(*, include_archived: bool = False, refresh: bool = False) -> dict[str, Any]:
    """只要来源列表（含计数）：不扫任何来源的条目。"""
    plan, cached = _get_plan(refresh=refresh)
    sources = [
        entry.info for entry in plan.sources
        if include_archived or not entry.info["project_archived"]
    ]
    return {
        "items": [],
        "sources": sources,
        "total": sum(int(source["item_count"]) for source in sources),
        "cursor": 0,
        "next_cursor": None,
        "generated_at": plan.generated_at,
        "cached": cached,
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
    }


# ── 底模层数（按需） ────────────────────────────────────────────────────────

def _catalog_roots() -> list[str]:
    cfg = secrets.load()
    roots = [
        _path_identity(projects.PROJECTS_DIR),
        _path_identity(models_root() / "loras"),
    ]
    roots.extend(
        _path_identity(Path(str(raw).strip()))
        for raw in cfg.generate.lora_catalog_dirs
        if str(raw).strip()
    )
    return roots


def read_arch(paths: list[str]) -> list[dict[str, Any]]:
    """按需读取 LoRA 训练底模层数（lora_compat 契约：元数据 / 键扫描下界 / 未知）。

    只接受 catalog 根目录内的 ``.safetensors``；其它路径、缺失或损坏文件一律
    ``unknown``，不抛错。结果按 (path, size, mtime) 进程内缓存。
    """
    roots = _catalog_roots()
    results: list[dict[str, Any]] = []
    for raw in dict.fromkeys(paths):
        num_blocks: int | None = None
        arch_source = "unknown"
        identity = _lexical_identity(os.path.abspath(raw))
        if _is_lora_name(raw) and any(_is_within(identity, root) for root in roots):
            try:
                stat = os.stat(raw)
            except OSError:
                stat = None
            if stat is not None:
                arch = read_lora_base_arch_cached(raw, stat.st_size, stat.st_mtime)
                num_blocks = arch.num_blocks
                arch_source = arch.source
        results.append({
            "path": raw,
            "base_num_blocks": num_blocks,
            "base_arch_source": arch_source,
        })
    return results
