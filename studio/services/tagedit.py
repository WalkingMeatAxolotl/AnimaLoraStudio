"""批量标签操作（PP4）。

scope = {kind: 'all' | 'folder' | 'files', folder?, names?}
所有读写都用 `read_caption` / `write_caption`，自动适配 `.txt`（逗号分隔）
与 `.json`（参考 docs/user-guide/caption-format.md，已简化为 {"tags": [...]}）。

写时：
- 如果该图已有 `.json` → 写 `.json`，更新 tags 字段
- 否则写 `.txt`
- add/remove 不会改变文件格式
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

from ..datasets import IMAGE_EXTS

ScopeKind = Literal["all", "folder", "files"]


# ---------------------------------------------------------------------------
# read / write
# ---------------------------------------------------------------------------


def caption_path(image: Path) -> Path | None:
    """返回图片对应的 caption 文件路径；若没有 caption 文件，返回 None。"""
    txt = image.with_suffix(".txt")
    js = image.with_suffix(".json")
    if js.exists():
        return js
    if txt.exists():
        return txt
    return None


def read_tags(image: Path) -> list[str]:
    """统一读 caption（txt / json）；不存在 → []。"""
    p = caption_path(image)
    if p is None:
        return []
    if p.suffix == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(data, dict):
            tags = data.get("tags")
            if isinstance(tags, list):
                return [str(t) for t in tags]
        return []
    # txt: 逗号分隔
    text = p.read_text(encoding="utf-8")
    return [t.strip() for t in text.split(",") if t.strip()]


def write_tags(image: Path, tags: list[str]) -> Path:
    """写 caption。已有 .json 就更新；否则写 .txt。"""
    js = image.with_suffix(".json")
    if js.exists():
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        data["tags"] = list(tags)
        js.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return js
    txt = image.with_suffix(".txt")
    txt.write_text(", ".join(tags), encoding="utf-8")
    return txt


# ---------------------------------------------------------------------------
# scope resolution
# ---------------------------------------------------------------------------


def _scope_image_paths(scope: dict[str, Any], train_dir: Path) -> list[Path]:
    kind: ScopeKind = scope.get("kind", "all")  # type: ignore[assignment]
    if kind == "all":
        out: list[Path] = []
        if train_dir.exists():
            for sub in train_dir.iterdir():
                if sub.is_dir():
                    out.extend(_imgs_in(sub))
        return out
    if kind == "folder":
        folder = str(scope.get("name") or scope.get("folder") or "")
        if not folder:
            return []
        return _imgs_in(train_dir / folder)
    if kind == "files":
        # 新形式（PP4 拆分后用）：items=[{folder, name}, ...]，跨 folder
        items = scope.get("items")
        if isinstance(items, list):
            out: list[Path] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                f = str(it.get("folder") or "")
                n = str(it.get("name") or "")
                if not f or not n:
                    continue
                p = train_dir / f / n
                if p.exists():
                    out.append(p)
            return out
        # 旧形式：folder + names（保留兼容）
        folder = str(scope.get("folder") or "")
        names = scope.get("names") or []
        if not folder or not isinstance(names, list):
            return []
        d = train_dir / folder
        return [d / n for n in names if (d / n).exists()]
    return []


def _imgs_in(d: Path) -> list[Path]:
    if not d.exists() or not d.is_dir():
        return []
    return sorted(
        f for f in d.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )


# ---------------------------------------------------------------------------
# ops
# ---------------------------------------------------------------------------


def stats(
    scope: dict[str, Any], train_dir: Path, top: int = 50
) -> list[tuple[str, int]]:
    """统计 tag 频率，返回 top N。"""
    counter: Counter[str] = Counter()
    for img in _scope_image_paths(scope, train_dir):
        for tag in read_tags(img):
            counter[tag] += 1
    return counter.most_common(top)


def add_tags(
    scope: dict[str, Any],
    train_dir: Path,
    tags: list[str],
    *,
    position: Literal["front", "back"] = "back",
) -> int:
    """对 scope 内所有 caption 加 tags（已有的不重复）。返回受影响文件数。"""
    new = [t.strip() for t in tags if t.strip()]
    if not new:
        return 0
    affected = 0
    for img in _scope_image_paths(scope, train_dir):
        cur = read_tags(img)
        cur_set = set(cur)
        to_add = [t for t in new if t not in cur_set]
        if not to_add:
            continue
        if position == "front":
            merged = to_add + cur
        else:
            merged = cur + to_add
        write_tags(img, merged)
        affected += 1
    return affected


def remove_tags(
    scope: dict[str, Any], train_dir: Path, tags: list[str]
) -> int:
    drop = {t.strip() for t in tags if t.strip()}
    if not drop:
        return 0
    affected = 0
    for img in _scope_image_paths(scope, train_dir):
        cur = read_tags(img)
        kept = [t for t in cur if t not in drop]
        if len(kept) != len(cur):
            write_tags(img, kept)
            affected += 1
    return affected


def replace_tag(
    scope: dict[str, Any], train_dir: Path, old: str, new: str
) -> int:
    old_s = old.strip()
    new_s = new.strip()
    if not old_s or not new_s or old_s == new_s:
        return 0
    affected = 0
    for img in _scope_image_paths(scope, train_dir):
        cur = read_tags(img)
        if old_s not in cur:
            continue
        # 替换并保持顺序；如果 new 已在列表里就只删 old，避免重复
        out: list[str] = []
        seen: set[str] = set()
        for t in cur:
            t_out = new_s if t == old_s else t
            if t_out in seen:
                continue
            seen.add(t_out)
            out.append(t_out)
        write_tags(img, out)
        affected += 1
    return affected


def dedupe(scope: dict[str, Any], train_dir: Path) -> int:
    """每个文件去重（保持顺序，首次出现保留）。返回受影响文件数。"""
    affected = 0
    for img in _scope_image_paths(scope, train_dir):
        cur = read_tags(img)
        seen: set[str] = set()
        out: list[str] = []
        for t in cur:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        if len(out) != len(cur):
            write_tags(img, out)
            affected += 1
    return affected


# ---------------------------------------------------------------------------
# single-image helpers (用在 GET/PUT /captions/{folder}/{filename})
# ---------------------------------------------------------------------------


def list_captions_in_folder(
    train_dir: Path, folder: str, *, preview: int = 5, full: bool = False
) -> list[dict[str, Any]]:
    """列文件夹内所有图片 + tag 信息。

    full=True 时附带完整 tags 和 format（给前端缓存模型用）；
    否则只返回 tag 数 + 前 N 个 preview（给缩略图列表用）。
    """
    d = train_dir / folder
    out: list[dict[str, Any]] = []
    for img in _imgs_in(d):
        tags = read_tags(img)
        cap_path = caption_path(img)
        item: dict[str, Any] = {
            "name": img.name,
            "folder": folder,
            "tag_count": len(tags),
            "tags_preview": tags[:preview],
            "has_caption": cap_path is not None,
        }
        if full:
            item["tags"] = tags
            item["format"] = (
                "json" if cap_path and cap_path.suffix == ".json"
                else "txt" if cap_path
                else "none"
            )
        out.append(item)
    return out


def list_all_captions(
    train_dir: Path, *, preview: int = 5, full: bool = False
) -> list[dict[str, Any]]:
    """列 train/ 下所有 folder 的图片 + tag 信息，每项标注所属 folder。"""
    if not train_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for sub in sorted(d for d in train_dir.iterdir() if d.is_dir()):
        out.extend(
            list_captions_in_folder(
                train_dir, sub.name, preview=preview, full=full
            )
        )
    return out


def read_one(train_dir: Path, folder: str, filename: str) -> dict[str, Any]:
    img = train_dir / folder / filename
    if not img.exists():
        raise FileNotFoundError(f"image not found: {folder}/{filename}")
    return {
        "name": img.name,
        "tags": read_tags(img),
        "format": (
            "json" if img.with_suffix(".json").exists()
            else "txt" if img.with_suffix(".txt").exists()
            else "none"
        ),
    }


def write_one(
    train_dir: Path, folder: str, filename: str, tags: list[str]
) -> dict[str, Any]:
    img = train_dir / folder / filename
    if not img.exists():
        raise FileNotFoundError(f"image not found: {folder}/{filename}")
    write_tags(img, tags)
    return read_one(train_dir, folder, filename)
