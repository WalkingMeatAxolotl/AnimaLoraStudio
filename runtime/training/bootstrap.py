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
    """将 YAML 与 CLI 显式参数合并，经 TrainingConfig 完整构造后返回新 args。

    config 管线刀 1（R1，docs/design/config-pipeline-refactor.md）：trainer 与
    Studio 走同一条 pydantic 加载路径 —— 字段迁移 / FAMILY_CONFIG_DEFAULTS
    族默认 overlay / 互斥与能力校验全部单点生效，本函数不再手工重放迁移
    （旧 merge_yaml_into_namespace 绕过 validator 年代的产物）。

    命令行显式参数优先于 YAML：parse_args 以 suppress_defaults 构建 parser，
    args 只含显式键，优先级是精确判定而非「值==默认值」近似。

    校验失败逐条打印到 stderr 后 SystemExit(2) —— 与能力防线同款 fail-fast，
    supervisor 截 stderr 尾部作为任务错误信息。
    """
    from pydantic import ValidationError

    from studio.infrastructure.argparse_bridge import namespace_from_config
    from studio.schema import TrainingConfig

    try:
        return namespace_from_config(args, dict(config or {}), TrainingConfig)
    except ValidationError as exc:
        errors = exc.errors()
        print(f"配置校验失败（{len(errors)} 处）:", file=sys.stderr)
        for err in errors:
            loc = ".".join(str(p) for p in err["loc"]) or "config"
            print(f"  {loc}: {err['msg']}", file=sys.stderr)
        raise SystemExit(2) from exc


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
