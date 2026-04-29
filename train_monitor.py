"""训练监控状态写入器（PP6.1 改造）。

历史：原本是 HTTP server + JSON 文件双轨。Studio 前端有自己的 monitor 页，
HTTP server 无用，已删除。本文件现在只负责把训练进度（loss / lr / samples）
写到一个 JSON 文件，由 `set_state_file(path)` 决定路径。

API：
- `set_state_file(path)` — 设置写入路径（应用启动一次）；不设置则 save_state 静默 no-op
- `update_monitor(...)` — 训练循环里调，更新 in-memory state 并落盘
- `restore_monitor_state(...)` — 断点续训恢复历史曲线
- `get_state()` — 读当前 state（拷贝，避免被外部修改）
- `_downsample_uniform(points, n)` — 工具：均匀降采样，给前端展示用

状态结构：losses / lr_history / samples / epoch / total_epochs / step /
total_steps / speed / start_time / config，与原来兼容（前端 monitor_smooth.html
仍能解析；total_epochs 是 PP6.x 后期补的，老 state 缺失时前端按 0 兜底）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


# 全局状态（in-memory）
MONITOR_STATE: dict[str, Any] = {
    "losses": [],
    "lr_history": [],
    "epoch": 0,
    "total_epochs": 0,
    "step": 0,
    "total_steps": 0,
    "speed": 0.0,
    "samples": [],
    "start_time": None,
    "config": {},
}

# 文件输出路径；None = 不写盘（save_state silent no-op）
_state_file: Optional[Path] = None


def set_state_file(path: Optional[Path | str]) -> None:
    """配置 state JSON 输出路径。None 表示不写盘。

    会确保父目录存在；若已有同路径状态文件则保留（断点续训由
    `restore_monitor_state` 负责加载历史，此处不读）。
    """
    global _state_file
    if path is None:
        _state_file = None
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _state_file = p


def save_state() -> None:
    """把当前 MONITOR_STATE 写到 _state_file（如果配置了）。失败静默吞。"""
    if _state_file is None:
        return
    try:
        with open(_state_file, "w", encoding="utf-8") as f:
            json.dump(MONITOR_STATE, f)
    except Exception:
        pass


def update_monitor(
    loss=None, lr=None, epoch=None, total_epochs=None, step=None,
    total_steps=None, speed=None, sample_path=None, config=None,
):
    """更新监控状态。先更新 step/epoch 等元信息，再追加 loss/lr 点位。"""
    if epoch is not None:
        MONITOR_STATE["epoch"] = epoch
    if total_epochs is not None:
        MONITOR_STATE["total_epochs"] = total_epochs
    if step is not None:
        MONITOR_STATE["step"] = step
    if total_steps is not None:
        MONITOR_STATE["total_steps"] = total_steps
    if speed is not None:
        MONITOR_STATE["speed"] = speed

    if loss is not None:
        MONITOR_STATE["losses"].append(
            {"step": MONITOR_STATE["step"], "loss": loss, "time": time.time()}
        )
        if len(MONITOR_STATE["losses"]) > 50000:
            MONITOR_STATE["losses"] = MONITOR_STATE["losses"][-50000:]

    if lr is not None:
        MONITOR_STATE["lr_history"].append(
            {"step": MONITOR_STATE["step"], "lr": lr}
        )
        if len(MONITOR_STATE["lr_history"]) > 50000:
            MONITOR_STATE["lr_history"] = MONITOR_STATE["lr_history"][-50000:]

    if sample_path is not None:
        MONITOR_STATE["samples"].append({
            "path": str(sample_path),
            "step": MONITOR_STATE["step"],
            "time": time.time(),
        })
        if len(MONITOR_STATE["samples"]) > 50:
            MONITOR_STATE["samples"] = MONITOR_STATE["samples"][-50:]

    if config is not None:
        MONITOR_STATE["config"] = config

    if MONITOR_STATE["start_time"] is None:
        MONITOR_STATE["start_time"] = time.time()

    save_state()


def get_state() -> dict[str, Any]:
    """读当前 state（浅拷贝；列表本体仍共享，调用方不要原地改）。"""
    return MONITOR_STATE.copy()


def restore_monitor_state(
    losses=None, lr_history=None, epoch=None, total_epochs=None, step=None,
    total_steps=None, start_time=None, config=None,
):
    """断点续训：把存档里的历史曲线灌回 in-memory state，再落盘。"""
    if losses is not None:
        MONITOR_STATE["losses"] = losses
    if lr_history is not None:
        MONITOR_STATE["lr_history"] = lr_history
    if epoch is not None:
        MONITOR_STATE["epoch"] = epoch
    if total_epochs is not None:
        MONITOR_STATE["total_epochs"] = total_epochs
    if step is not None:
        MONITOR_STATE["step"] = step
    if total_steps is not None:
        MONITOR_STATE["total_steps"] = total_steps
    if start_time is not None:
        MONITOR_STATE["start_time"] = start_time
    if config is not None:
        MONITOR_STATE["config"] = config
    save_state()


def _downsample_uniform(points: list[Any], target_points: int) -> list[Any]:
    """均匀降采样到 target_points（保留首尾），适合 loss/lr 长序列展示。"""
    if not isinstance(target_points, int) or target_points <= 0:
        return points
    n = len(points)
    if n <= target_points:
        return points
    if target_points == 1:
        return [points[-1]]
    step = (n - 1) / (target_points - 1)
    out = []
    for i in range(target_points):
        idx = round(i * step)
        out.append(points[idx])
    return out


def reset_state() -> None:
    """测试用：把 in-memory state 清回初始值。"""
    MONITOR_STATE.clear()
    MONITOR_STATE.update({
        "losses": [],
        "lr_history": [],
        "epoch": 0,
        "total_epochs": 0,
        "step": 0,
        "total_steps": 0,
        "speed": 0.0,
        "samples": [],
        "start_time": None,
        "config": {},
    })
