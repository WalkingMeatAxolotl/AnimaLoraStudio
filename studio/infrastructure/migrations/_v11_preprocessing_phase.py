"""v10 → v11: ADR 0010 加 `preprocessing` phase 到 VersionPhase.ORDER。

VersionPhase 从 5 → 6：

    curating → preprocessing → tagging → editing → regularizing → ready

新 phase 介于 curating / tagging 之间；可跳过（跟 regularizing 一致）。

回填策略（all silent, add-only — 跟 _v8 同 pattern）：

  - phase = curating + `versions/{label}/train/{sub-folder}/` 有图 →
    推进到 preprocessing（用户已经 curate 完，train 集合现存）
  - phase = curating + train 空 → 保持 curating
  - 其他 phase（tagging / editing / regularizing / ready） → 保持不变

依据：ADR 0010 §Migration —— 用户原话 "preprocessing 的图片已经复制到现有
train 的图片了"，train/ 非空意味着 curate 实质完成；提供"silent advance to
preprocessing"让升级后用户进入项目直接看到新 phase。

零 schema 改动：phase 列已是 TEXT（_v8 加的），新值无需扩字段；旧的
phase 字段约束在 backend code（VersionPhase.VALUES）维护，DB 层只是 TEXT。

跟 _v9 destructive 的关系：本 migration 不动 stage 列（_v9 已删）；只 UPDATE
phase 列。顺序保证 (`MIGRATIONS[10]` 在 `_v9` 之后) 不需要特殊处理。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


# 跟 dataset.scan.IMAGE_EXTS 一致；这里 inline 是为了避免 migration 模块依赖
# services/ 层（migrations 应该 self-contained，让 cold-start 顺序灵活）。
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})


def _train_has_image(project_dir: Path, version_label: str) -> bool:
    """跟 manifest._scan_train_images 一致逻辑：扫 train/{sub-folder}/{image}。
    train 根目录直接放的图忽略（LoRA 训练只读 sub-folder 内）。
    """
    train_dir = project_dir / "versions" / version_label / "train"
    if not train_dir.exists():
        return False
    for sub in train_dir.iterdir():
        if not sub.is_dir():
            continue
        for f in sub.iterdir():
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                return True
    return False


def migrate(conn: sqlite3.Connection) -> None:
    # Lazy import：避免 migration 模块加载时强依赖 services 层；测试 monkeypatch
    # services.projects.PROJECTS_DIR 后跑 migrate 会拿到正确路径。
    from ...services.projects import projects as _projects

    rows = conn.execute(
        "SELECT v.id, v.label, p.id, p.slug "
        "FROM versions v "
        "JOIN projects p ON v.project_id = p.id "
        "WHERE v.phase = 'curating'"
    ).fetchall()
    for vid, label, pid, slug in rows:
        try:
            project_dir = _projects.project_dir(int(pid), str(slug))
        except (TypeError, ValueError):
            continue
        if _train_has_image(project_dir, str(label)):
            conn.execute(
                "UPDATE versions SET phase = 'preprocessing' WHERE id = ?",
                (vid,),
            )
    conn.commit()
