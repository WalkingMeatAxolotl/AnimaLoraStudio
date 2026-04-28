"""Version 数据模型 + 物理目录 + fork 训练树 + activate。

Version 是 Pipeline 的「实验单元」：每个 version 独立维护 train/ reg/
output/ samples/ 与 monitor_state.json。label 由用户起（baseline /
high-lr 这种语义名），同 project 内唯一，且不可改（路径锚点）。

软删：目录搬到 `_trash/projects/{slug}/versions/{label}/`，db 行删除。
若被删的是 active version，自动 reassign 到「最新创建的剩余 version」。
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from . import projects
from .datasets import IMAGE_EXTS

VALID_STAGES: frozenset[str] = frozenset({
    "curating", "tagging", "regularizing",
    "ready", "training", "done",
})

# label 必须是路径安全的：字母 / 数字 / 下划线 / 连字符 / 点
_VALID_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


class VersionError(Exception):
    """Version 业务错误。"""


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def version_dir(project_id: int, slug: str, label: str) -> Path:
    return projects.project_dir(project_id, slug) / "versions" / label


def _write_version_json(v: dict[str, Any], pdir_label_path: Path) -> None:
    pdir_label_path.mkdir(parents=True, exist_ok=True)
    (pdir_label_path / "version.json").write_text(
        json.dumps(v, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# 默认训练子文件夹：Kohya 风格 N_label，repeat=1。
# 之所以默认建一个：用户进 Curation 页就能直接复制图，不需要先「+ 新建文件夹」。
DEFAULT_TRAIN_FOLDER = "1_data"


def _ensure_version_tree(vdir: Path) -> None:
    for sub in ("train", "reg", "output", "samples"):
        (vdir / sub).mkdir(parents=True, exist_ok=True)
    (vdir / "train" / DEFAULT_TRAIN_FOLDER).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def _row_to_version(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return dict(row) if row else None


def get_version(
    conn: sqlite3.Connection, version_id: int
) -> Optional[dict[str, Any]]:
    row = conn.execute(
        "SELECT * FROM versions WHERE id = ?", (version_id,)
    ).fetchone()
    return _row_to_version(row)


def _must_get(conn: sqlite3.Connection, version_id: int) -> dict[str, Any]:
    v = get_version(conn, version_id)
    if not v:
        raise VersionError(f"版本不存在: id={version_id}")
    return v


def list_versions(
    conn: sqlite3.Connection, project_id: int
) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM versions WHERE project_id = ? ORDER BY created_at ASC",
            (project_id,),
        )
    ]


def create_version(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    label: str,
    fork_from_version_id: Optional[int] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """label 校验：仅 [A-Za-z0-9_.-]+；同 project 内唯一。

    fork_from_version_id 给了：复制源 version 的 train/ 目录树（递归 copy）。
    config_name 也一起拷过来 —— 注意 PP6 才会真用 config_name；这里只是字段层
    继承，避免后续切预设时漏 fork。
    """
    p = projects.get_project(conn, project_id)
    if not p:
        raise VersionError(f"项目不存在: id={project_id}")
    if not _VALID_LABEL.fullmatch(label):
        raise VersionError(
            f"非法 label: {label!r}（仅允许字母/数字/下划线/连字符/点）"
        )
    # 唯一性
    if conn.execute(
        "SELECT 1 FROM versions WHERE project_id = ? AND label = ?",
        (project_id, label),
    ).fetchone():
        raise VersionError(f"label 已存在: {label!r}")

    src_config_name: Optional[str] = None
    if fork_from_version_id is not None:
        src = get_version(conn, fork_from_version_id)
        if not src or src["project_id"] != project_id:
            raise VersionError(
                f"fork 源不存在或不属于当前项目: id={fork_from_version_id}"
            )
        src_config_name = src["config_name"]

    now = time.time()
    cur = conn.execute(
        "INSERT INTO versions(project_id, label, config_name, stage, created_at, note) "
        "VALUES (?, ?, ?, 'curating', ?, ?)",
        (project_id, label, src_config_name, now, note),
    )
    conn.commit()
    vid = int(cur.lastrowid)

    vdir = version_dir(project_id, p["slug"], label)
    _ensure_version_tree(vdir)

    if fork_from_version_id is not None:
        src = _must_get(conn, fork_from_version_id)
        src_train = version_dir(project_id, p["slug"], src["label"]) / "train"
        if src_train.exists():
            _copytree_train(src_train, vdir / "train")

    v = _must_get(conn, vid)
    _write_version_json(v, vdir)

    # 项目里第一个 version → 自动设为 active
    if p.get("active_version_id") is None:
        projects.update_project(conn, project_id, active_version_id=vid)

    return v


def _copytree_train(src: Path, dst: Path) -> None:
    """复制训练树（含子文件夹与 .txt/.json 同名 metadata）。

    Win 上硬链接受限较多，统一走 copy（PP1 说明这点）。
    """
    dst.mkdir(parents=True, exist_ok=True)
    for sub in src.iterdir():
        target = dst / sub.name
        if sub.is_dir():
            _copytree_train(sub, target)
        else:
            shutil.copy2(sub, target)


_UPDATABLE = {"note", "stage", "config_name", "output_lora_path"}


def update_version(
    conn: sqlite3.Connection, version_id: int, **fields: Any
) -> dict[str, Any]:
    v = _must_get(conn, version_id)
    keep = {k: val for k, val in fields.items() if k in _UPDATABLE}
    if "stage" in keep and keep["stage"] not in VALID_STAGES:
        raise VersionError(f"非法 stage: {keep['stage']!r}")
    if not keep:
        return v
    cols = ", ".join(f"{k} = ?" for k in keep)
    params: list[Any] = list(keep.values()) + [version_id]
    conn.execute(f"UPDATE versions SET {cols} WHERE id = ?", params)
    conn.commit()
    v = _must_get(conn, version_id)
    p = projects.get_project(conn, v["project_id"])
    if p:
        _write_version_json(v, version_dir(p["id"], p["slug"], v["label"]))
    return v


def delete_version(conn: sqlite3.Connection, version_id: int) -> None:
    """目录搬到 _trash；db 删行；若是 active 自动 reassign。"""
    v = _must_get(conn, version_id)
    p = projects.get_project(conn, v["project_id"])
    if p:
        src = version_dir(p["id"], p["slug"], v["label"])
        if src.exists():
            trash = (
                projects.TRASH_DIR
                / f"{p['id']}-{p['slug']}"
                / "versions"
                / v["label"]
            )
            if trash.exists():
                trash = trash.with_name(f"{v['label']}-{int(time.time())}")
            trash.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(trash))

        if p.get("active_version_id") == version_id:
            # 选剩下里 created_at 最新的；都没了就清空
            row = conn.execute(
                "SELECT id FROM versions WHERE project_id = ? AND id != ? "
                "ORDER BY created_at DESC LIMIT 1",
                (v["project_id"], version_id),
            ).fetchone()
            new_active = int(row[0]) if row else None
            projects.update_project(
                conn, v["project_id"], active_version_id=new_active
            )

    conn.execute("DELETE FROM versions WHERE id = ?", (version_id,))
    conn.commit()


def activate_version(
    conn: sqlite3.Connection, version_id: int
) -> dict[str, Any]:
    """把当前 version 设为项目的 active_version。返回更新后的 version。"""
    v = _must_get(conn, version_id)
    projects.update_project(conn, v["project_id"], active_version_id=version_id)
    return v


def advance_stage(
    conn: sqlite3.Connection, version_id: int, target: str
) -> dict[str, Any]:
    if target not in VALID_STAGES:
        raise VersionError(f"非法 stage: {target!r}")
    return update_version(conn, version_id, stage=target)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def stats_for_version(p: dict[str, Any], v: dict[str, Any]) -> dict[str, Any]:
    """train 子文件夹与图片计数 / 已打标计数 / reg 计数 / output 是否存在。"""
    vdir = version_dir(p["id"], p["slug"], v["label"])
    train_dir = vdir / "train"
    train_folders: list[dict[str, Any]] = []
    train_total = 0
    tagged_total = 0
    if train_dir.exists():
        for sub in sorted(train_dir.iterdir()):
            if sub.is_dir():
                cnt = 0
                for f in sub.iterdir():
                    if not (f.is_file() and f.suffix.lower() in IMAGE_EXTS):
                        continue
                    cnt += 1
                    if f.with_suffix(".txt").exists() or f.with_suffix(".json").exists():
                        tagged_total += 1
                train_folders.append({"name": sub.name, "image_count": cnt})
                train_total += cnt
    reg_dir = vdir / "reg"
    reg_total = 0
    reg_meta_exists = False
    if reg_dir.exists():
        # reg/1_general/{train-subfolder}/{post_id}.png — 递归扫
        for f in reg_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                reg_total += 1
        reg_meta_exists = (reg_dir / "1_general" / "meta.json").exists()
    output_dir = vdir / "output"
    has_output = output_dir.exists() and any(output_dir.iterdir())
    return {
        "train_image_count": train_total,
        "tagged_image_count": tagged_total,
        "train_folders": train_folders,
        "reg_image_count": reg_total,
        "reg_meta_exists": reg_meta_exists,
        "has_output": has_output,
    }
