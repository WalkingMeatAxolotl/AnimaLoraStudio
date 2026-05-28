"""共享响应常量（PR-5 从 server.py 抽出）。"""
from __future__ import annotations

from typing import Any

# /api/state 在 task_id 不存在 / 没 task / state.json 缺失时返回的空 state，
# 保持前端 monitor 页能稳定渲染（不报错也不显示 "loading"）。
EMPTY_STATE: dict[str, Any] = {
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
