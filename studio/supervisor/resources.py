"""资源档位模型（0.17 R-1，设计见 docs/design/queue-resource-model-0.17.md §3）。

队列的第一性目标是**显存并发准入**：训练（或任何独占 GPU 的工作）运行时，
什么可以同时上卡。档位是工作项的静态属性，不是归属表的属性：

- ``exclusive``：底模级显存（数 GB），全系统同时最多 1 个。train / reg_ai /
  generate / eval_samples（评估出图与 generate 同款底模栈，独立子进程）。
  daemon 常驻模型视作持有一张可吊销的 exclusive 租约。
- ``light``：小模型（数十–数百 MB）。无 exclusive 运行时恒放行；有 exclusive
  运行时按 `secrets.queue.light_tasks_during_train` 开关放行（默认开）。
- ``io``：不上 GPU，恒放行（仅受 queue hold 约束）。

新增任务类型 / job kind 必须在此显式声明档位；未声明的 kind 保守按
``exclusive`` 处理（绝不与训练并行，宁慢勿崩）。
"""
from __future__ import annotations

RESOURCE_EXCLUSIVE = "exclusive"
RESOURCE_LIGHT = "light"
RESOURCE_IO = "io"

# tasks 表 task_type → 档位（当前三类全部独占）。
TASK_TYPE_RESOURCE_CLASS: dict[str, str] = {
    "train": RESOURCE_EXCLUSIVE,
    "reg_ai": RESOURCE_EXCLUSIVE,
    "generate": RESOURCE_EXCLUSIVE,
}

# project_jobs kind → 档位。
JOB_KIND_RESOURCE_CLASS: dict[str, str] = {
    "download": RESOURCE_IO,
    "preprocess": RESOURCE_LIGHT,   # spandrel 超分，小模型（D-R1）
    "tag": RESOURCE_LIGHT,          # WD14 / CLTagger ONNX
    "reg_build": RESOURCE_LIGHT,
    "eval_samples": RESOURCE_EXCLUSIVE,  # 与 generate 同款底模栈（D-R2）
    "eval_clip": RESOURCE_LIGHT,
    "eval_dino": RESOURCE_LIGHT,
    "eval_tag": RESOURCE_LIGHT,
    "eval_ccip": RESOURCE_LIGHT,
}


def job_resource_class(kind: str) -> str:
    """job kind → 档位；未知 kind 保守按 exclusive（绝不与训练并行）。"""
    return JOB_KIND_RESOURCE_CLASS.get(kind, RESOURCE_EXCLUSIVE)


def task_resource_class(task_type: str | None) -> str:
    """task_type → 档位；老行 NULL 兜底 train（=exclusive）。"""
    return TASK_TYPE_RESOURCE_CLASS.get(task_type or "train", RESOURCE_EXCLUSIVE)
