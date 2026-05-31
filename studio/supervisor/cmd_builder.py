"""默认 cmd builder + worker EVENT 协议常量（PR-4 从 supervisor.py 抽出）。

`Supervisor.__init__` 接受 `cmd_builder` / `job_cmd_builder` 注入参数，方便
测试替换；这里实现 supervisor 内默认走真实 runtime/anima_train.py / workers
模块的版本。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from ..paths import REPO_ROOT, task_monitor_state_path

# PP10.2.b：哪些 job kind 吃 GPU。这些 kind 在训练运行中默认会被推迟，
# 除非 secrets.queue.allow_gpu_during_train=True 显式允许并行。
# preprocess 走 spandrel super-resolution，加载权重到 GPU 推理。
GPU_BOUND_JOB_KINDS: frozenset[str] = frozenset({
    "preprocess",
    "tag",
    "reg_build",
    "eval_samples",
})

# Worker → supervisor 的结构化事件标记。worker 写
#   __EVENT__:my_event_type:{"foo":1,"bar":"x"}
# 到 stdout，supervisor 在 _on_line 里识别并 publish 成 typed SSE 事件
# （job_id / project_id 自动注入），不会进 job_log。比专门搭 IPC 轻。
_EVENT_MARKER = "__EVENT__:"


EventCallback = Callable[[dict[str, Any]], None]
CmdBuilder = Callable[[dict[str, Any], Path], list[str]]
JobCmdBuilder = Callable[[dict[str, Any]], list[str]]


def _default_cmd_builder(task: dict[str, Any], config_path: Path) -> list[str]:
    """根据 task_type 路由到对应脚本。

    train (默认 / 老 task): runtime/anima_train.py
    reg_ai: runtime/anima_reg_ai.py（先验生成）
    generate: 走 inference_daemon，**不**经这个 cmd_builder，supervisor
        在 _dispatch_generate 里直接派给 daemon。这里 fallback 到 anima_generate.py
        只是为了某天测试可能注入 cmd_builder 时不爆 KeyError —— 实际跑
        不到这条 path（_next_pending_task_in 在 dispatch_train 里只挑
        train/reg_ai）。
    """
    task_type = task.get("task_type") or "train"
    if task_type == "reg_ai":
        script = REPO_ROOT / "runtime" / "anima_reg_ai.py"
    elif task_type == "generate":
        script = REPO_ROOT / "runtime" / "anima_generate.py"  # 兜底，正常路径不来这
    else:
        script = REPO_ROOT / "runtime" / "anima_train.py"
    cmd = [
        sys.executable,
        str(script),
        "--config",
        str(config_path),
    ]
    msp = task.get("monitor_state_path")
    if msp:
        cmd.extend(["--monitor-state-file", str(msp)])
    # ADR 0006 PR-3: paused task 复活 → 注入 --resume-state，让 anima_train
    # 的 resume_phase 加载 state；旁边的 .config.json snapshot 由 bootstrap_phase
    # 自动检测并 freeze args（ADR §5.7）。
    paused_state = task.get("paused_state_path")
    if paused_state:
        cmd.extend(["--resume-state", str(paused_state)])
    return cmd


def _resolve_monitor_state_path(task: dict[str, Any]) -> Path:
    """决定 task 的 monitor_state.json 落盘路径。

    一律落 `studio_data/tasks/<id>/monitor/state.json`，跟 version 解耦：
    - 删 version 不再带走 task 历史（loss 曲线 / 采样图）
    - 同一 version 多次跑的 task 各自独立档案，用户可以拉出来对比

    历史路径仅保留**读**兼容（老 task DB monitor_state_path 列保留旧值，
    读端按值取，新 task 不再写这些路径）：
    - `versions/<label>/monitor/task_<id>/state.json` —— PP6.1（v0.5.0+）
    - `versions/<label>/monitor_state.json` —— pre-PP6.1
    - `studio_data/monitors/task_<id>/state.json` —— 无 version_id 兜底
    """
    return task_monitor_state_path(task["id"])


def _default_job_cmd_builder(job: dict[str, Any]) -> list[str]:
    """默认按 kind 选 worker 模块。"""
    kind = job["kind"]
    return [
        sys.executable,
        "-m",
        f"studio.workers.{kind}_worker",
        "--job-id",
        str(job["id"]),
    ]
