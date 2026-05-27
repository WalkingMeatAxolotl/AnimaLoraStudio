"""PR-1 安全网 — 冻结 studio.server.app 的全部 route 三元组。

后续重构 PR（拆 router、搬 endpoint）必须保持 (methods, path, name) 集合不变；
任何意外丢路由 / 改路径 / 改函数名都会让本测试 fail 并给出可读 diff。

snapshot 文件：tests/_snapshots/studio_routes.json
- 首次运行（snapshot 不存在）→ 创建并 emit warning，测试通过
- 之后每次运行 → 读出比对，不一致用 pytest.fail 列出 added/removed/changed
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from studio.server import app

SNAPSHOT_PATH = Path(__file__).parent / "_snapshots" / "studio_routes.json"
SNAPSHOT_VERSION = 1


def _route_entry(route: Any) -> dict[str, Any]:
    methods = sorted(getattr(route, "methods", None) or [])
    return {
        "path": getattr(route, "path", ""),
        "methods": methods,
        "name": getattr(route, "name", ""),
        "type": type(route).__name__,
    }


def _collect_routes() -> list[dict[str, Any]]:
    entries = [_route_entry(r) for r in app.routes]
    entries.sort(key=lambda e: (e["path"], ",".join(e["methods"]), e["name"]))
    return entries


def _entry_key(e: dict[str, Any]) -> tuple[str, str, str]:
    return (e["path"], ",".join(e["methods"]), e["type"])


def _load_snapshot() -> dict[str, Any] | None:
    if not SNAPSHOT_PATH.exists():
        return None
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _write_snapshot(routes: list[dict[str, Any]]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": SNAPSHOT_VERSION,
        "generated_from": "studio.server.app",
        "count": len(routes),
        "routes": routes,
    }
    SNAPSHOT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _format_diff(current: list[dict[str, Any]], saved: list[dict[str, Any]]) -> str:
    cur_by_key = {_entry_key(e): e for e in current}
    saved_by_key = {_entry_key(e): e for e in saved}
    added = [k for k in cur_by_key if k not in saved_by_key]
    removed = [k for k in saved_by_key if k not in cur_by_key]
    changed = []
    for k in cur_by_key.keys() & saved_by_key.keys():
        if cur_by_key[k]["name"] != saved_by_key[k]["name"]:
            changed.append(
                (k, saved_by_key[k]["name"], cur_by_key[k]["name"])
            )

    lines: list[str] = []
    if added:
        lines.append(f"新增 {len(added)} 个 route（snapshot 里没有）：")
        for path, methods, type_ in sorted(added):
            lines.append(f"  + [{type_}] {methods or '-'} {path}")
    if removed:
        lines.append(f"丢失 {len(removed)} 个 route（snapshot 里有但当前没有）：")
        for path, methods, type_ in sorted(removed):
            lines.append(f"  - [{type_}] {methods or '-'} {path}")
    if changed:
        lines.append(f"name 改了 {len(changed)} 个 route：")
        for (path, methods, type_), old_name, new_name in sorted(changed):
            lines.append(f"  ~ [{type_}] {methods or '-'} {path}: {old_name} → {new_name}")
    if not lines:
        lines.append("（key 集合相同但 JSON 仍不一致 —— 可能是 snapshot 版本字段或排序变了）")
    lines.append("")
    lines.append("如果你确实增删改了 route 且这是预期：删除 snapshot 文件后重跑生成新的，")
    lines.append(f"然后 commit：{SNAPSHOT_PATH.relative_to(Path(__file__).parent.parent)}")
    return "\n".join(lines)


def test_route_snapshot() -> None:
    current = _collect_routes()
    assert len(current) > 100, (
        f"app.routes 只有 {len(current)} 个，明显过少 —— 可能 server.py 装配出了问题"
    )

    saved = _load_snapshot()
    if saved is None:
        _write_snapshot(current)
        warnings.warn(
            f"snapshot 已创建：{SNAPSHOT_PATH} （{len(current)} 个 route）。"
            "请将该文件 commit 进 git。",
            stacklevel=1,
        )
        return

    if saved.get("routes") == current:
        return

    diff = _format_diff(current, saved.get("routes", []))
    pytest.fail(
        f"studio.server.app.routes snapshot 不匹配。\n"
        f"snapshot 路径：{SNAPSHOT_PATH}\n"
        f"snapshot count = {saved.get('count')} / 当前 count = {len(current)}\n\n"
        f"{diff}"
    )
