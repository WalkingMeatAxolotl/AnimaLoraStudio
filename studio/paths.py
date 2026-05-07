"""Studio 内部使用的路径常量与目录初始化。"""
from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 训练侧已有。`OUTPUT_DIR` 仅给 `/samples/{name}` 端点（无 task_id 时）兜底用，
# CLI 用户用 `./output/samples/...`；Studio 模式样本落到
# `studio_data/projects/{id}-{slug}/versions/{label}/output/samples/`。
# PP6.1 后全局 `monitor_data/` 已退役（监控状态走 per-task），不再保留常量 / 创建目录。
OUTPUT_DIR = REPO_ROOT / "output"
LEGACY_MONITOR_HTML = REPO_ROOT / "monitor_smooth.html"

# Studio 持久化（SQLite + 用户保存的 preset + 任务日志）
STUDIO_DATA = REPO_ROOT / "studio_data"
STUDIO_DB = STUDIO_DATA / "studio.db"
USER_PRESETS_DIR = STUDIO_DATA / "presets"
USER_CONFIGS_DIR = USER_PRESETS_DIR  # 兼容别名（PP0 后将随 configs_io 一起移除）
LOGS_DIR = STUDIO_DATA / "logs"
THUMB_CACHE_DIR = STUDIO_DATA / "thumb_cache"
GENERATE_JOBS_DIR = STUDIO_DATA / "generate_jobs"
GENERATE_CONFIGS_DIR = STUDIO_DATA / "generate_configs"

# React 前端
WEB_DIR = REPO_ROOT / "studio" / "web"
WEB_DIST = WEB_DIR / "dist"


def migrate_configs_to_presets() -> None:
    """旧版本把 yaml 放在 studio_data/configs/，这里把目录原地重命名为 presets/。
    只在 presets/ 不存在时执行；不会覆盖用户的新数据。"""
    old = STUDIO_DATA / "configs"
    if old.exists() and not USER_PRESETS_DIR.exists():
        old.rename(USER_PRESETS_DIR)


def ensure_dirs() -> None:
    """首次运行时创建必要目录。"""
    STUDIO_DATA.mkdir(parents=True, exist_ok=True)
    migrate_configs_to_presets()
    for d in (USER_PRESETS_DIR, LOGS_DIR, GENERATE_JOBS_DIR, GENERATE_CONFIGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
