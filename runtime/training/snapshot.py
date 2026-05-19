"""Pause / resume 用的 state snapshot helpers（ADR 0006 PR-2）。

落地三块：

  - `build_pause_state_path(state_dir, step)`：拼 pause `.pt` 路径
  - `write_config_snapshot(path, args, sample_prompts)`：把暂停那一刻
    训练实际在用的全部 args 序列化成 JSON（详见 ADR §5.7）
  - `emit_event(event_type, payload)`：往 stdout 写 `__EVENT__:...` 行，
    supervisor `_on_line` 识别并 publish 成 SSE typed event

这三个都没有训练流水线 import，故意独立成 module 让 supervisor / spike
脚本 / 测试都能复用，避免 context.py 越长越臃肿。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# 跟 studio/supervisor.py:49 _EVENT_MARKER 协议对齐；改这个常量 = 跨进程
# breaking change，所以挪到此模块顶部唯一字面量。
EVENT_MARKER = "__EVENT__:"


def build_pause_state_path(state_dir: Path, step: int) -> Path:
    """暂停 `.pt` 文件路径（ADR §5.1）。

    `pause_` 前缀跟 PR-1 周期 save 的 `training_state_step<N>.pt` 区分；
    同名 `.config.json` 是 snapshot（见 `write_config_snapshot`）。
    """
    return state_dir / f"pause_step_{step}.pt"


def build_pause_config_path(state_dir: Path, step: int) -> Path:
    """暂停 config snapshot 文件路径（与 state `.pt` 同前缀，`.config.json` 后缀）。"""
    return state_dir / f"pause_step_{step}.config.json"


def _jsonify(value: Any) -> Any:
    """把 args / sample_prompts 里的对象转成可 json.dump 的形式。

    覆盖范围：Path → str，set → list，其他保持原样。argparse.Namespace
    本身 vars() 出来都是 primitive + Path，遇到别的类型就 fallback repr()。
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonify(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    return repr(value)  # 兜底，不抛错


def write_config_snapshot(
    path: Path,
    args: Any,  # argparse.Namespace 或 dict
    sample_prompts: list[str] | None = None,
) -> None:
    """暂停时把当前训练实际在用的全部参数 freeze 成 JSON（ADR §5.7）。

    Resume 严格用 snapshot 拼新 args，跟用户后续改 config / preset / yaml
    完全解耦。snapshot 不含 wandb run id（已 finish）和 monitor live state
    （已 dump 在 `.pt` 内）。

    `sample_prompts` 来自 `ctx.sample_prompts`（运行时状态），不是 args
    字段，单独传。
    """
    if hasattr(args, "__dict__"):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        args_dict = {"_args_repr": repr(args)}

    payload = {
        "version": 1,  # 给 future schema migration 留触点
        "args": {k: _jsonify(v) for k, v in args_dict.items()},
        "sample_prompts": list(sample_prompts) if sample_prompts else [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def emit_event(event_type: str, payload: dict[str, Any] | None = None) -> None:
    """往 stdout 写 supervisor `__EVENT__:type:json` 协议行。

    flush=True 是关键 — 否则被子进程 stdout buffer 攒到几 KB 才送达，
    pause 链路 IO 反馈延迟，supervisor `_on_line` 抓不到事件就误判超时。
    """
    body = json.dumps(payload or {}, ensure_ascii=False)
    sys.stdout.write(f"{EVENT_MARKER}{event_type}:{body}\n")
    sys.stdout.flush()
