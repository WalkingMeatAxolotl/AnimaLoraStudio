"""Studio 内部使用的路径常量与目录初始化。"""
from __future__ import annotations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 训练侧已有。`OUTPUT_DIR` 仅给 `/samples/{name}` 端点（无 task_id 时）兜底用，
# CLI 用户用 `./output/samples/...`；Studio 模式样本落到
# `studio_data/projects/{id}-{slug}/versions/{label}/output/samples/`。
# PP6.1 后全局 `monitor_data/` 已退役（监控状态走 per-task），不再保留常量 / 创建目录。
OUTPUT_DIR = REPO_ROOT / "output"

# 用户显式选择「导出到本机目录」时的落地目录。
DATA_EXPORTS = REPO_ROOT / "data_exports"


# Studio 持久化（SQLite + 用户保存的 preset + 任务日志）
STUDIO_DATA = REPO_ROOT / "studio_data"
STUDIO_DB = STUDIO_DATA / "studio.db"
USER_PRESETS_DIR = STUDIO_DATA / "presets"
USER_CONFIGS_DIR = USER_PRESETS_DIR  # 兼容别名（PP0 后将随 configs_io 一起移除）
LOGS_DIR = STUDIO_DATA / "logs"
THUMB_CACHE_DIR = STUDIO_DATA / "thumb_cache"

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
