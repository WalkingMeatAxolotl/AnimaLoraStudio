"""依赖检测、YAML 配置加载、进度条初始化等启动期工具。

抽自原 runtime/anima_train.py L60-180（ADR 0003 PR-A）。

公开函数：
- ensure_dependencies — 检测并可选自动安装缺失依赖
- load_yaml_config / apply_yaml_config — YAML 配置 → args 合并
- init_progress — Rich 进度条初始化
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ensure_dependencies(auto_install: bool = False) -> None:
    """检测并可选自动安装缺失依赖。"""
    required = {
        "numpy": "numpy",
        "PIL": "Pillow",
        "safetensors": "safetensors",
        "transformers": "transformers",
        "einops": "einops",
        "torchvision": "torchvision",
        "yaml": "pyyaml",
    }
    missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except Exception:
            missing.append(pip_name)
    if not missing:
        return
    missing_list = ", ".join(sorted(set(missing)))
    print(f"Missing dependencies: {missing_list}")
    if not auto_install:
        print(f"Install them with:\n  {sys.executable} -m pip install {missing_list}")
        raise SystemExit(1)
    cmd = [sys.executable, "-m", "pip", "install", *sorted(set(missing))]
    print("Installing missing dependencies...")
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"Auto-install failed: {exc}")
        raise SystemExit(1)
    still_missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except Exception:
            still_missing.append(pip_name)
    if still_missing:
        still_list = ", ".join(sorted(set(still_missing)))
        print(f"Still missing: {still_list}")
        raise SystemExit(1)


def load_yaml_config(config_path):
    """加载 YAML 配置文件。"""
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed. Install with: pip install pyyaml")
        raise SystemExit(1)

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def apply_yaml_config(args, config):
    """将 YAML 配置应用到 args；命令行显式参数优先于 YAML。

    实现走 studio.argparse_bridge.merge_yaml_into_namespace —— 字段名 / 默认值
    都从 studio.schema.TrainingConfig 这一份单一权威源派生，避免与 parse_args
    脱节。未在 schema 中的 YAML 键会被忽略（拼写错误一目了然）。

    在 merge 前调用 migrate_legacy_attention 兜底老 yaml 的 xformers/flash_attn
    双 bool —— argparse_bridge 不走 pydantic validator，schema 层的迁移逻辑
    无法生效，所以这里显式做一次。
    """
    from studio.argparse_bridge import merge_yaml_into_namespace
    from studio.schema import TrainingConfig, migrate_legacy_attention
    config = migrate_legacy_attention(dict(config or {}))
    return merge_yaml_into_namespace(args, config, TrainingConfig)


def init_progress(show_progress, total_steps):
    """初始化 Rich 进度条。

    返回 `(progress, task_id, kind)`：
    - 关闭进度时返回 `(None, None, None)`
    - Rich 可用时返回 `(Progress 实例, task_id, "rich")`
    - Rich 缺失时返回 `("plain", None, None)`（main() 据此走纯文本进度）
    """
    if not show_progress:
        return None, None, None
    try:
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress, TextColumn,
            TimeElapsedColumn, TimeRemainingColumn,
        )
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("loss={task.fields[loss]:.4f}"),
            TextColumn("lr={task.fields[lr]:.2e}"),
            TextColumn("speed={task.fields[speed]:.2f} it/s"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=10,
        )
        task = progress.add_task("train", total=total_steps, loss=0.0, lr=0.0, speed=0.0)
        return progress, task, "rich"
    except Exception:
        return "plain", None, None
