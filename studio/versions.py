"""Version 数据模型 + 物理目录 + fork 训练树 + activate。

Version 是 Pipeline 的「实验单元」：每个 version 独立维护 train/ reg/
output/ samples/ 与 monitor_state.json。label 由用户起（baseline /
high-lr 这种语义名），同 project 内唯一，且不可改（路径锚点）。

删除：直接 rmtree version 目录 + DELETE db 行。不可恢复。
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

# ADR-0007 §11.3-B：versions.stage 被拆成 status + phase 两个正交字段。
# 本节加 enum + readonly accessor；写入路径由 PR-3 双写过渡，v9 删 stage 同步清理。


class VersionStatus:
    """版本运行态状态机（5 enum，ADR-0007 §11.3-B）。"""

    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    VALUES: frozenset[str] = frozenset({
        PREPARING, TRAINING, COMPLETED, FAILED, CANCELED,
    })


class VersionPhase:
    """版本准备 cursor，仅 status=preparing 时有业务语义（ADR-0007 §11.3-B / §11.5-A）。

    顺序：curating → tagging → editing → regularizing → ready。
    regularizing 可跳过（SKIPPABLE），其余必经。
    """

    CURATING = "curating"
    TAGGING = "tagging"
    EDITING = "editing"
    REGULARIZING = "regularizing"
    READY = "ready"

    ORDER: tuple[str, ...] = (
        CURATING, TAGGING, EDITING, REGULARIZING, READY,
    )
    VALUES: frozenset[str] = frozenset(ORDER)
    SKIPPABLE: frozenset[str] = frozenset({REGULARIZING})


def get_status(v: dict[str, Any]) -> str:
    """读 version.status；None / 缺字段 fallback → preparing。"""
    return str(v.get("status") or VersionStatus.PREPARING)


def get_phase(v: dict[str, Any]) -> str:
    """读 version.phase；None / 缺字段 fallback → curating。"""
    return str(v.get("phase") or VersionPhase.CURATING)

# label 必须是路径安全的：字母 / 数字 / 下划线 / 连字符 / 点
_VALID_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


class VersionError(Exception):
    """Version 业务错误。"""


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def version_dir(project_id: int, slug: str, label: str) -> Path:
    return projects.project_dir(project_id, slug) / "versions" / label


def _natural_key(s: str) -> list[Any]:
    """自然序 key：字符串里的数字段当 int 比较，让 a_5 < a_60。

    re.split(r'(\\d+)', 'a_60') -> ['a_', '60', '']
    转换为 ['a_', 60, '']，与同样转换后的 'a_5' -> ['a_', 5, ''] 按位比较。
    """
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_lora_ckpts(vdir: Path) -> list[dict[str, Any]]:
    """扫 versions/{label}/output/*.safetensors，列所有 LoRA ckpt 文件。

    anima_train 输出命名约定（runtime/anima_train.py:2434, 2464）：
      - {output_name}_step{N}.safetensors    （按 step 保存）
      - {output_name}_epoch{N}.safetensors   （按 epoch 保存）
      - {output_name}_final.safetensors      （训练完毕）

    返回每个 ckpt 的 {kind, value, label, path, mtime}：
      - kind: 'step' | 'epoch' | 'final' | 'other'
      - value: int（step/epoch 数；final/other → 0）
      - label: 显示用，"step 2476" / "epoch 5" / "final" / 文件名
      - path: 绝对路径字符串
      - mtime: 修改时间戳（前端按时间倒序展示）
    排序：final 在前 → step 数字降序 → epoch 数字降序 → 其他按 label 自然序升序
    （让 a_5 < a_60，避免 lex 序把 a_60 排到 a_9 前面或 mtime 序乱掉用户预期）。
    """
    output_dir = vdir / "output"
    if not output_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for f in output_dir.glob("*.safetensors"):
        if not f.is_file():
            continue
        name = f.stem  # 去掉 .safetensors
        kind = "other"
        value = 0
        label = name
        # 匹配 *_step{N}
        m = re.search(r"_step(\d+)$", name)
        if m:
            kind = "step"
            value = int(m.group(1))
            label = f"step {value}"
        else:
            m = re.search(r"_epoch(\d+)$", name)
            if m:
                kind = "epoch"
                value = int(m.group(1))
                label = f"epoch {value}"
            elif name.endswith("_final"):
                kind = "final"
                label = "final"
        try:
            mtime = f.stat().st_mtime
        except OSError:
            mtime = 0.0
        items.append({
            "kind": kind, "value": value, "label": label,
            "path": str(f), "mtime": mtime,
        })

    # 排序：final 顶部；step/epoch 按 value 降序；other 按 label 自然序升序
    kind_order = {"final": 0, "step": 1, "epoch": 2, "other": 3}

    def _sort_key(x: dict[str, Any]) -> tuple[Any, ...]:
        ko = kind_order.get(x["kind"], 9)
        if x["kind"] in ("step", "epoch"):
            return (ko, -x["value"], [], -x["mtime"])
        # final / other：value 都是 0，按 label 自然序升序（other 主要受益）
        return (ko, 0, _natural_key(x["label"]), -x["mtime"])

    items.sort(key=_sort_key)
    return items


_STATE_FILE_RE = re.compile(r"training_state_(step|epoch)(\d+)\.pt$")


def list_state_ckpts(vdir: Path) -> list[dict[str, Any]]:
    """扫 version output/ 下所有断点续训 state 文件。

    扫描两个位置（ADR 0006 PR-1 路径迁移）：
      - 旧路径：``output/training_state_step{N}.pt``（pre-PR-1 残留）
      - 新路径：``output/state/task_<TID>/training_state_step{N}.pt``（PR-1+）

    两种粒度都看（PR-1 顺手修扫描漏 epoch 的旧 bug）：
      - step  →  ``training_state_step{N}.pt``    label "step N"
      - epoch →  ``training_state_epoch{N}.pt``   label "epoch N"

    pause 文件（PR-2+ 的 ``pause_step_<N>.pt``）**不在此列**——picker 不应
    暴露 pause 中间态。命名前缀天然过滤。

    返回 [{step, label, path, mtime}]，step 降序，epoch 单独按 step（int 部分）
    降序排在 step 项前后；UI 按 mtime/step 自己排即可。
    """
    output_dir = vdir / "output"
    if not output_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    # 同一相对路径不要进两次（理论上不会撞，但 glob 重叠 + symlink 兜底）。
    candidates: list[Path] = []
    candidates.extend(output_dir.glob("training_state_*.pt"))
    state_root = output_dir / "state"
    if state_root.exists():
        candidates.extend(state_root.glob("task_*/training_state_*.pt"))
    for f in candidates:
        if not f.is_file():
            continue
        m = _STATE_FILE_RE.search(f.name)
        if not m:
            continue
        key = str(f.resolve())
        if key in seen:
            continue
        seen.add(key)
        kind = m.group(1)  # "step" or "epoch"
        n = int(m.group(2))
        try:
            mtime = f.stat().st_mtime
        except OSError:
            mtime = 0.0
        items.append({
            "step": n if kind == "step" else 0,
            "label": f"{kind} {n}",
            "path": str(f),
            "mtime": mtime,
            "_kind": kind,  # 内部排序用，返回前剥掉
            "_n": n,
        })
    # 先 step 段（按 step 降序），后 epoch 段（按 epoch 降序）。
    items.sort(key=lambda x: (0 if x["_kind"] == "step" else 1, -x["_n"]))
    for it in items:
        it.pop("_kind", None)
        it.pop("_n", None)
    return items


def list_project_state_ckpts(
    conn: sqlite3.Connection, project: dict[str, Any]
) -> list[dict[str, Any]]:
    """列项目所有 versions 的 state.pt，按 version 分组（Train 页 resume_state picker 用）。

    返回 [{version_id, label, items: [{step, label, path, mtime}, ...]}]，按 version
    `created_at` 升序，items 按 step 降序。空 version（没产出 .pt）保留分组但 items 为空。
    """
    pid = int(project["id"])
    slug = str(project["slug"])
    groups: list[dict[str, Any]] = []
    for v in list_versions(conn, pid):
        vdir = version_dir(pid, slug, str(v["label"]))
        groups.append({
            "version_id": int(v["id"]),
            "label": str(v["label"]),
            "items": list_state_ckpts(vdir),
        })
    return groups


def list_project_lora_ckpts(
    conn: sqlite3.Connection, project: dict[str, Any]
) -> list[dict[str, Any]]:
    """列项目所有 versions 的 LoRA ckpt（.safetensors），按 version 分组（resume_lora picker 用）。

    返回 [{version_id, label, items: [{kind, value, label, path, mtime}, ...]}]，
    按 version `created_at` 升序；items 按 list_lora_ckpts 内置排序（final → step desc → epoch desc → other）。
    """
    pid = int(project["id"])
    slug = str(project["slug"])
    groups: list[dict[str, Any]] = []
    for v in list_versions(conn, pid):
        vdir = version_dir(pid, slug, str(v["label"]))
        groups.append({
            "version_id": int(v["id"]),
            "label": str(v["label"]),
            "items": list_lora_ckpts(vdir),
        })
    return groups


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


# PP10.1 — fork 时源 stage 落到新 version 的映射。
# done / training 的 version 已经训完或在训，新 version 重新进入待训练态；
# 其他 stage（curating / tagging / regularizing / ready）原样跟随，让用户从
# 副本所处的同一 step 接着干。
_FORK_STAGE_RESET: frozenset[str] = frozenset({"done", "training"})


def create_version(
    conn: sqlite3.Connection,
    *,
    project_id: int,
    label: str,
    fork_from_version_id: Optional[int] = None,
    note: Optional[str] = None,
) -> dict[str, Any]:
    """label 校验：仅 [A-Za-z0-9_.-]+；同 project 内唯一。

    fork_from_version_id 给了 → 全量复制源 version 的用户产物：
        train/、reg/、config.yaml、.unlocked.json（PP10.4）
    输出类（output/、samples/、monitor_state.json）一律不复制。
    复制 config.yaml 后立即重写一次，把 data_dir / reg_data_dir / output_dir /
    output_name 强制刷成新 version 的路径。
    stage 跟随源（done/training → ready；其他原样）。
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
    src_stage: str = "curating"
    if fork_from_version_id is not None:
        src = get_version(conn, fork_from_version_id)
        if not src or src["project_id"] != project_id:
            raise VersionError(
                f"fork 源不存在或不属于当前项目: id={fork_from_version_id}"
            )
        src_config_name = src["config_name"]
        src_stage = "ready" if src["stage"] in _FORK_STAGE_RESET else src["stage"]

    now = time.time()
    cur = conn.execute(
        "INSERT INTO versions(project_id, label, config_name, stage, created_at, note) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (project_id, label, src_config_name, src_stage, now, note),
    )
    conn.commit()
    vid = int(cur.lastrowid)

    vdir = version_dir(project_id, p["slug"], label)
    _ensure_version_tree(vdir)

    if fork_from_version_id is not None:
        src = _must_get(conn, fork_from_version_id)
        src_vdir = version_dir(project_id, p["slug"], src["label"])
        # train / reg：递归复制目录（存在才复制）
        for sub in ("train", "reg"):
            src_sub = src_vdir / sub
            if src_sub.exists():
                _copytree(src_sub, vdir / sub)
        # config.yaml + .unlocked.json：单文件复制
        for fname in ("config.yaml", ".unlocked.json"):
            src_file = src_vdir / fname
            if src_file.exists():
                shutil.copy2(src_file, vdir / fname)
        # config.yaml 复制过来后，data_dir / reg_data_dir / output_dir /
        # output_name 还指向源 version；用 force_project_overrides=True 重写
        # 一次刷成新 version 的路径。reg_data_dir 由 project_specific_overrides
        # 自动检测新 version 的 reg/meta.json 是否存在 → 跟随复制结果。
        v_for_rewrite = _must_get(conn, vid)
        new_cfg_path = vdir / "config.yaml"
        if new_cfg_path.exists():
            from .services import version_config as _vc  # 延迟避免循环
            try:
                cfg = _vc.read_version_config(p, v_for_rewrite)
                _vc.write_version_config(
                    p, v_for_rewrite, cfg, force_project_overrides=True
                )
            except _vc.VersionConfigError:
                # 源 config 损坏不阻断新建；用户去 Train 页换预设
                pass

    v = _must_get(conn, vid)
    _write_version_json(v, vdir)

    # 项目里第一个 version → 自动设为 active
    if p.get("active_version_id") is None:
        projects.update_project(conn, project_id, active_version_id=vid)

    return v


def _copytree(src: Path, dst: Path) -> None:
    """递归复制目录（含子文件夹与同名 metadata 文件）。

    Win 上硬链接受限较多，统一走 copy（PP1 说明这点）。
    PP10.1 起从 _copytree_train 通用化 — train / reg 都用这个。
    """
    dst.mkdir(parents=True, exist_ok=True)
    for sub in src.iterdir():
        target = dst / sub.name
        if sub.is_dir():
            _copytree(sub, target)
        else:
            shutil.copy2(sub, target)


_UPDATABLE = {"note", "stage", "config_name", "output_lora_path", "trigger_word"}


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
    """rmtree version 目录 + DELETE db 行；若是 active 自动 reassign。不可恢复。"""
    v = _must_get(conn, version_id)
    p = projects.get_project(conn, v["project_id"])
    if p:
        src = version_dir(p["id"], p["slug"], v["label"])
        if src.exists():
            shutil.rmtree(src, ignore_errors=True)

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
        # reg/{train-subfolder-mirror}/{post_id}.png — 递归扫（与源脚本一致）
        for f in reg_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                reg_total += 1
        reg_meta_exists = (reg_dir / "meta.json").exists()
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
