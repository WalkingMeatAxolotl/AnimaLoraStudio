"""Studio 内部使用的路径常量与目录初始化。"""
from __future__ import annotations

import json
import logging
from pathlib import Path

# PR-7：本文件从 studio/paths.py 搬到 studio/infrastructure/paths.py，多嵌一层；
# REPO_ROOT 要再上跳一层（__file__ → infrastructure/ → studio/ → repo root）。
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# 训练侧已有。`OUTPUT_DIR` 仅给 `/samples/{name}` 端点（无 task_id 时）兜底用，
# CLI 用户用 `./output/samples/...`；Studio 模式样本落到
# `studio_data/projects/{id}-{slug}/versions/{label}/output/samples/`。
# PP6.1 后全局 `monitor_data/` 已退役（监控状态走 per-task），不再保留常量 / 创建目录。
OUTPUT_DIR = REPO_ROOT / "output"

# 用户显式选择「导出到本机目录」时的落地目录。
DATA_EXPORTS = REPO_ROOT / "data_exports"


# Studio 持久化（SQLite + 用户保存的 preset + 任务日志）。
#
# 位置可自定义：仓库根的指针文件 `studio_data_location.json`（{"path": "..."}）
# 指向自定义目录。指针必须在 studio_data **外面** —— secrets.json / db 都在
# studio_data 里，位置本身的配置存里面就成了鸡生蛋。指针在模块 import 时读
# 一次，进程内不变；迁移（/api/studio-data/migrate）写完指针后需重启 server
# 生效（cli.py 重启循环用 subprocess 拉新进程，paths 会重新求值）。
DEFAULT_STUDIO_DATA = REPO_ROOT / "studio_data"
STUDIO_DATA_POINTER = REPO_ROOT / "studio_data_location.json"


def resolve_studio_data(pointer_file: Path | None = None) -> Path:
    """读指针文件解析 studio_data 位置；无指针 / 指针无效 → 默认位置。

    指针目标必须是已存在的绝对路径目录 —— 盘符未挂载 / 目录被删时回退默认
    （默认位置仍保留迁移前的旧数据，可用），只 log warning 不抛错。
    """
    ptr = pointer_file if pointer_file is not None else STUDIO_DATA_POINTER
    try:
        if ptr.is_file():
            raw = json.loads(ptr.read_text("utf-8"))
            target = Path(str(raw.get("path", "")))
            if target.is_absolute() and target.is_dir():
                return target
            logging.getLogger(__name__).warning(
                "studio_data 指针目标无效（不存在或非绝对路径），回退默认位置: %s", target,
            )
    except Exception:
        logging.getLogger(__name__).warning(
            "studio_data 指针文件解析失败，回退默认位置: %s", ptr, exc_info=True,
        )
    return DEFAULT_STUDIO_DATA


STUDIO_DATA = resolve_studio_data()
STUDIO_DB = STUDIO_DATA / "studio.db"
USER_PRESETS_DIR = STUDIO_DATA / "presets"
USER_CONFIGS_DIR = USER_PRESETS_DIR  # 兼容别名（PP0 后将随 configs_io 一起移除）
# LOGS_DIR：pre-task-scoped layout 时所有 task 日志的扁平目录。
# 新 task 走 `tasks/<id>/run.log`（见 task_log_path）；这个目录仅保留给
# 老 task 兼容读取（不写新）。
LOGS_DIR = STUDIO_DATA / "logs"
THUMB_CACHE_DIR = STUDIO_DATA / "thumb_cache"

# Task-scoped 档案根目录。每个 task 独立子目录，跟 version 解耦，
# 删 version 不会带走 task 历史（loss / 参数 / sample / 日志）。
# 子目录约定（snapshot/ 已由 task_snapshot.py 引入 ADR-0007 §11.7）：
#   tasks/<id>/snapshot/config.yaml   ← task 启动时 freeze 的 config
#   tasks/<id>/monitor/state.json     ← 训练监控状态（loss/LR/sample 索引）
#   tasks/<id>/samples/*.png          ← 训练采样图
#   tasks/<id>/run.log                ← worker 子进程 stdout/stderr
TASKS_DIR = STUDIO_DATA / "tasks"

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
    for d in (USER_PRESETS_DIR, LOGS_DIR, DATA_EXPORTS):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Task-scoped 路径 helper
# ---------------------------------------------------------------------------
#
# 所有 helper 都不 mkdir —— 调用方按需 mkdir(parents=True, exist_ok=True)。
# 跟 task_snapshot.snapshot_dir 已有约定保持一致：snapshot_dir(id) 也是
# `tasks/<id>/snapshot/` 一个子目录，本组 helper 提供 monitor/samples/log 三个
# sibling，组合起来就是 task 完整档案。

def task_dir(task_id: int) -> Path:
    """`studio_data/tasks/<task_id>/` —— task 档案根。"""
    return TASKS_DIR / str(int(task_id))


def task_monitor_state_path(task_id: int) -> Path:
    """`tasks/<task_id>/monitor/state.json` —— 训练监控状态文件。

    取代旧路径：
    - `versions/<v>/monitor/task_<id>/state.json`（PP6.1，v0.5.0+，仍兼容读）
    - `versions/<v>/monitor_state.json`（pre-PP6.1，仍兼容读）
    - `studio_data/monitors/task_<id>/state.json`（无 version_id 兜底，仍兼容读）
    """
    return task_dir(task_id) / "monitor" / "state.json"


def task_samples_dir(task_id: int) -> Path:
    """`tasks/<task_id>/samples/` —— 训练采样图。

    runtime 写：`runtime/training/phases/bootstrap.py` 把 ctx.sample_dir 指向这里。
    API 读：`studio/api/routers/samples.py` 候选首位。
    """
    return task_dir(task_id) / "samples"


def task_eval_dir(task_id: int) -> Path:
    """`tasks/<task_id>/eval/` —— training task scoped LoRA eval artifacts."""
    return task_dir(task_id) / "eval"


def task_log_path(task_id: int) -> Path:
    """`tasks/<task_id>/run.log` —— worker 子进程 stdout/stderr。

    取代旧路径 `studio_data/logs/<task_id>.log`（仍兼容读）。
    `project_jobs` 的日志（`studio_data/jobs/<job_id>.log`）是另一套，不动。
    """
    return task_dir(task_id) / "run.log"


# ---------------------------------------------------------------------------
# Path traversal 防护
# ---------------------------------------------------------------------------
#
# 历史教训：早期端点只用字面量黑名单（`"/" in name or "\\" in name or ".." in name`）
# 防穿越。`..` 字面量检查会误杀含 ASCII 省略号的合法文件名（Pixiv 标题 `「これ...」`），
# 删掉又会让 defense-in-depth 单薄。统一走 resolve() + relative_to() containment check：
# 既允许 `..` 作为文件名字符出现，又保证拼出来的最终路径不能逃出 base。

def validate_path_component(name: str) -> None:
    """校验单个路径片段：拒绝空串 / 含路径分隔符 / 绝对路径前缀。

    `..` 字面量本身放行（containment check 是真正防线），所以
    `「これでいっすか...」.txt` 这种合法文件名能通过。
    """
    if not name:
        raise ValueError("path component is empty")
    if "/" in name or "\\" in name:
        raise ValueError(f"path component contains separator: {name!r}")
    # Windows 盘符（C:\...）和 POSIX 绝对路径（/foo）一律视作非法片段
    if Path(name).is_absolute():
        raise ValueError(f"path component is absolute: {name!r}")


def safe_join(base: Path, *parts: str) -> Path:
    """把 parts 拼到 base 下并做 containment 校验。

    每个 part 走 `validate_path_component`；拼接 + resolve() 后必须仍在
    `base.resolve()` 子树内，否则 raise ValueError。

    返回 resolve() 后的绝对 Path。调用方按需做 exists() / is_file() 检查。
    """
    for p in parts:
        validate_path_component(p)
    base_resolved = base.resolve()
    candidate = base_resolved.joinpath(*parts).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"path escapes base: {candidate} not in {base_resolved}") from exc
    return candidate
