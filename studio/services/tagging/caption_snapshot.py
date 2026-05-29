"""Caption 快照（PP4）— 把 train/ 下全部 caption 文件打包成 zip 落 caption_snapshots/。

用户在 ④ 标签编辑页点「💾 备份」即生成；可列出历史 / 还原 / 删除。

快照路径：`<version_dir>/caption_snapshots/{ts}.zip`
zip 内部按 `<folder>/<filename>` 平铺，例如 `1_data/a.txt`、`5_face/b.json`。
还原时清空 train/ 下所有现存 *.txt / *.json，再解包写入。
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Any

from ...paths import safe_join, validate_path_component

CAPTION_EXTS = (".txt", ".json")
SNAPSHOT_DIRNAME = "caption_snapshots"


from studio.domain.errors import DomainError


class SnapshotError(DomainError):
    """Snapshot 业务错误。

    PR-2 C3 加 DomainError base — handler 自动翻 dual-write envelope。
    """
    default_code = "snapshot.error"


def snapshot_root(version_dir: Path) -> Path:
    return version_dir / SNAPSHOT_DIRNAME


def _iter_caption_files(train_dir: Path):
    """yield (rel_path_in_zip, abs_path) 的 caption 文件对。"""
    if not train_dir.exists():
        return
    for sub in sorted(d for d in train_dir.iterdir() if d.is_dir()):
        for f in sorted(sub.iterdir()):
            if f.is_file() and f.suffix.lower() in CAPTION_EXTS:
                yield f"{sub.name}/{f.name}", f


def _snapshot_meta(zip_path: Path) -> dict[str, Any]:
    sid = zip_path.stem
    stat = zip_path.stat()
    file_count = 0
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            file_count = sum(1 for n in z.namelist() if not n.endswith("/"))
    except zipfile.BadZipFile:
        file_count = -1
    return {
        "id": sid,
        "created_at": int(stat.st_mtime),
        "size": stat.st_size,
        "file_count": file_count,
    }


def create_snapshot(version_dir: Path) -> dict[str, Any]:
    """打包 train/ 下全部 caption 到一个新 zip。空 train 也允许（生成空 zip）。"""
    train_dir = version_dir / "train"
    out_dir = snapshot_root(version_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sid = str(int(time.time() * 1000))
    zip_path = out_dir / f"{sid}.zip"
    # 极小概率撞 ts，加后缀；防御性
    suffix = 0
    while zip_path.exists():
        suffix += 1
        zip_path = out_dir / f"{sid}_{suffix}.zip"
        if suffix > 9:
            raise SnapshotError("snapshot id 冲突过多")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for arcname, src in _iter_caption_files(train_dir):
            z.write(src, arcname=arcname)

    return _snapshot_meta(zip_path)


def list_snapshots(version_dir: Path) -> list[dict[str, Any]]:
    out_dir = snapshot_root(version_dir)
    if not out_dir.exists():
        return []
    items = [
        _snapshot_meta(p)
        for p in out_dir.iterdir()
        if p.is_file() and p.suffix == ".zip"
    ]
    items.sort(key=lambda x: x["created_at"], reverse=True)
    return items


def _resolve_snapshot(version_dir: Path, sid: str) -> Path:
    try:
        validate_path_component(sid)
    except ValueError as exc:
        raise SnapshotError(f"非法 id: {sid} ({exc})") from exc
    try:
        p = safe_join(snapshot_root(version_dir), f"{sid}.zip")
    except ValueError as exc:
        raise SnapshotError(f"非法 id: {sid} ({exc})") from exc
    if not p.exists():
        raise FileNotFoundError(f"snapshot not found: {sid}")
    return p


def restore_snapshot(version_dir: Path, sid: str) -> dict[str, Any]:
    """还原：先删 train/ 下全部 *.txt/*.json，再解包写入。图片文件保持不变。"""
    zip_path = _resolve_snapshot(version_dir, sid)
    train_dir = version_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # 1) 清旧 caption
    removed = 0
    for sub in train_dir.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.is_file() and f.suffix.lower() in CAPTION_EXTS:
                f.unlink()
                removed += 1

    # 2) 解包写新（仅在对应 folder 已存在 / 或会创建）
    written = 0
    skipped: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            name = member.filename
            if member.is_dir():
                continue
            if "/" not in name:
                skipped.append(name)
                continue
            folder, fname = name.split("/", 1)
            if Path(fname).suffix.lower() not in CAPTION_EXTS:
                skipped.append(name)
                continue
            try:
                target = safe_join(train_dir, folder, fname)
            except ValueError:
                skipped.append(name)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(target, "wb") as dst:
                dst.write(src.read())
            written += 1

    return {
        "id": sid,
        "removed_old": removed,
        "written": written,
        "skipped": skipped,
    }


def delete_snapshot(version_dir: Path, sid: str) -> None:
    zip_path = _resolve_snapshot(version_dir, sid)
    zip_path.unlink()
