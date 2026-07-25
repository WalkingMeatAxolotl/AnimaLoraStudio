"""评估入队 —— 训练后自动与手动两个入口，统一创建一个 EvalSession。

0.21 及以前这里是「隐式任务链」的源头：逐 checkpoint 排出图作业，出图 worker 完成后
再回头排指标作业，一次评估 fan-out 成几百个 tasks 行 + 几百个日志目录（issue #465）。
现在两个入口都收口到 `eval_session.create_session`，一次评估 = 一个 `eval_session`
作业，阶段编排在 `workers/eval_session_worker` 里。

本模块只剩两件事：
1. **算出该评估哪些 checkpoint**（`checkpoint_skip_count` + `select_checkpoints`）；
2. 把入口参数归一后交给 `eval_session.create_session`。

历史保留：每次评估创建一个新 Session，旧的留档（A 方案，推翻 ADR 0011 Addendum §5
的「先清空上一轮、永远只显示当次」）—— 一次评估一个 task 之后历史本身就是干净可读的，
而「改了配置重训后指标怎么变」是评估的核心用途之一。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from studio.services import eval_session
from studio.services.projects import projects, versions

logger = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]


def _version_eval_config(
    project: dict[str, Any], version: dict[str, Any]
) -> dict[str, Any]:
    """读 version 训练配置。评估开关与 checkpoint 策略同源，一次读出。

    读的是**当前** version config（跟 `eval_validation_enabled` 一直以来的口径一致）。
    改读 task frozen snapshot 是后续的事 —— 那需要 EvalPlan 从快照取全部生成参数，
    不能只换这一处来源，否则两套口径混用更难解释。
    """
    from studio.services import version_config
    try:
        return version_config.read_version_config(project, version)
    except Exception:
        return {}


def _version_eval_enabled(project: dict[str, Any], version: dict[str, Any]) -> bool:
    """Per-version opt-in for post-training validation metrics (training config)."""
    return bool(_version_eval_config(project, version).get("eval_validation_enabled"))


# checkpoint 采样：见 TrainingConfig.eval_checkpoint_skip_count。只有一个旋钮 ——
# 「评一个跳几个」，0 = 全评。issue #465 的作业膨胀已经在 Session 层根治（一次评估
# 一个作业），所以这里不需要再靠限制 checkpoint 数来止血，默认就是全评。
DEFAULT_CHECKPOINT_SKIP_COUNT = 0


def checkpoint_skip_count(cfg: dict[str, Any]) -> int:
    """从训练配置解析 skip_count；缺失 / 非法值归一到 0（全评）。"""
    try:
        skip = int(cfg.get("eval_checkpoint_skip_count", DEFAULT_CHECKPOINT_SKIP_COUNT))
    except (TypeError, ValueError):
        skip = DEFAULT_CHECKPOINT_SKIP_COUNT
    return max(0, skip)


def select_checkpoints(
    ckpts: list[dict[str, Any]], *, skip_count: int
) -> list[dict[str, Any]]:
    """挑出要评估的 checkpoint 子集。

    - ``skip_count <= 0``：全部评估。
    - 否则：训练顺序取一个、跳 ``skip_count`` 个再取下一个；**最终权重始终在内**。

    `list_lora_ckpts` 给的是**展示序**（final 在前、step/epoch 降序）。采样按**训练
    顺序**（升序）走 —— 「评一个跳 N 个」才对得上 user 心智，且同一批 ckpt 的采样结果
    稳定，不随展示序变化。返回值仍按入参原序。
    """
    if not ckpts or skip_count <= 0:
        return list(ckpts)
    finals = [c for c in ckpts if c.get("kind") == "final"]
    rest = list(reversed([c for c in ckpts if c.get("kind") != "final"]))
    stride = int(skip_count) + 1
    chosen = [*finals, *rest[::stride]]
    keep = {str(c.get("path") or "") for c in chosen}
    return [c for c in ckpts if str(c.get("path") or "") in keep]


def _resolve_context(
    conn, task: dict[str, Any]
) -> Optional[tuple[dict[str, Any], dict[str, Any], Path]]:
    """task → (project, version, version_dir)；绑定缺失 / 不一致时 None。"""
    project_id = int(task.get("project_id") or 0)
    version_id = int(task.get("version_id") or 0)
    if not project_id or not version_id:
        return None
    project = projects.get_project(conn, project_id)
    version = versions.get_version(conn, version_id)
    if not project or not version or int(version["project_id"]) != project_id:
        return None
    vdir = versions.version_dir(
        project_id, str(project["slug"]), str(version["label"])
    )
    return project, version, vdir


def queue_training_finished_eval(
    conn,
    task: dict[str, Any],
    payload: dict[str, Any] | None = None,
) -> Optional[dict[str, Any]]:
    """训练完成 → 建一个 EvalSession。返回 session，未开启 / 无候选时 None。

    受 per-version 开关 `eval_validation_enabled` 门控；评估哪些 checkpoint 由
    `eval_checkpoint_skip_count` 决定（默认 0 = 全评，见 `select_checkpoints`）。
    """
    ctx = _resolve_context(conn, task)
    if ctx is None:
        return None
    project, version, vdir = ctx

    cfg = _version_eval_config(project, version)
    if not cfg.get("eval_validation_enabled"):
        return None
    skip_count = checkpoint_skip_count(cfg)

    all_ckpts = versions.list_lora_ckpts(vdir)
    selected = select_checkpoints(all_ckpts, skip_count=skip_count)
    if not selected:
        logger.info(
            "after-training eval skipped for task=%s: no checkpoint in output/",
            task.get("id"),
        )
        return None
    if len(selected) != len(all_ckpts):
        logger.info(
            "eval checkpoint sampling skip=%s: %s/%s checkpoints selected",
            skip_count, len(selected), len(all_ckpts),
        )

    try:
        return eval_session.create_session(
            conn, project, version, vdir,
            checkpoints=selected,
            trigger="after_training",
            parent_task_id=int(task.get("id") or 0) or None,
            skip_count=skip_count,
        )
    except Exception:
        logger.exception(
            "after-training eval session creation failed for task=%s", task.get("id")
        )
        return None


def queue_manual_eval(
    conn,
    project: dict[str, Any],
    version: dict[str, Any],
    vdir: Path,
    checkpoints: list[str],
    *,
    parent_task_id: int | None = None,
) -> Optional[dict[str, Any]]:
    """手动「运行评估」→ 建一个 EvalSession（显式 checkpoint 集）。

    与训练后入口的区别：**不看** per-version 开关（用户明确点了按钮），且 checkpoint
    是用户选的，不走策略。上一轮的 Session 不动 —— 历史保留（A 方案）。

    评估的对象是 version 下的一组 checkpoint，所以 `parent_task_id` 是可选的**溯源**
    信息，不是归属：从训练页发起时填上（面板据此只显示这次训练的评估），从版本的
    评估页发起时留空（比如评一个手动丢进 output/ 的 LoRA）。
    """
    selected = resolve_checkpoint_selection(project, version, vdir, checkpoints)
    if not selected:
        return None

    return eval_session.create_session(
        conn, project, version, vdir,
        checkpoints=selected,
        trigger="manual",
        parent_task_id=parent_task_id,
    )


def queue_manual_task_eval(
    conn,
    task: dict[str, Any],
    checkpoints: list[str],
) -> Optional[dict[str, Any]]:
    """训练页发起的手动评估 —— 同上，只是自动带上 parent_task_id 溯源。"""
    ctx = _resolve_context(conn, task)
    if ctx is None:
        return None
    project, version, vdir = ctx
    task_id = int(task.get("id") or 0)
    if not task_id:
        return None
    return queue_manual_eval(
        conn, project, version, vdir, checkpoints, parent_task_id=task_id
    )


def resolve_checkpoint_selection(
    project: dict[str, Any],
    version: dict[str, Any],
    vdir: Path,
    raw_paths: list[str],
) -> list[dict[str, Any]]:
    """用户传的 checkpoint 路径 → `list_lora_ckpts` 条目。

    去重、丢弃 output/ 之外的路径（防穿越），并保持 `list_lora_ckpts` 的展示序 ——
    候选顺序不该取决于用户点选的先后。
    """
    project_id = int(project["id"])
    slug = str(project.get("slug") or "")
    wanted: set[str] = set()
    for raw in raw_paths:
        rel = _checkpoint_relative_to_output(project_id, slug, version, str(raw or ""))
        if rel:
            wanted.add(rel)
    if not wanted:
        return []

    picked: list[dict[str, Any]] = []
    for ckpt in versions.list_lora_ckpts(vdir):
        rel = _checkpoint_relative_to_output(
            project_id, slug, version, str(ckpt.get("path") or "")
        )
        if rel and rel in wanted:
            picked.append(ckpt)
    return picked


def eval_scale(
    project: dict[str, Any],
    version: dict[str, Any],
    vdir: Path,
    *,
    selected_count: int | None = None,
) -> dict[str, Any]:
    """评估规模预估（创建前的成本可见性，issue #465）。

    `selected_count` 给定时按手动选中的 checkpoint 数算；为 None 时按 version 的
    checkpoint 策略算（训练后自动评估会评几个）。
    """
    if selected_count is None:
        cfg = _version_eval_config(project, version)
        skip_count = checkpoint_skip_count(cfg)
        all_ckpts = versions.list_lora_ckpts(vdir)
        count = len(select_checkpoints(all_ckpts, skip_count=skip_count))
        total = len(all_ckpts)
    else:
        skip_count = None
        count = max(0, int(selected_count))
        total = len(versions.list_lora_ckpts(vdir))

    summary = eval_session.resource_summary(
        project, version, vdir, selected_count=count
    )
    summary["checkpoints_total"] = total
    summary["skip_count"] = skip_count
    return summary


def _checkpoint_relative_to_output(
    project_id: int,
    project_slug: str,
    version: dict[str, Any],
    raw_path: str,
) -> str | None:
    """checkpoint 路径 → 相对 output/ 的 posix 路径；越界返回 None。"""
    if not raw_path or not project_slug:
        return None
    vdir = versions.version_dir(project_id, project_slug, str(version["label"]))
    output_dir = (vdir / "output").resolve()
    path = Path(raw_path)
    if not path.is_absolute():
        path = output_dir / raw_path.replace("\\", "/")
    try:
        rel = path.resolve().relative_to(output_dir)
    except ValueError:
        logger.warning("eval checkpoint outside output dir: %s", raw_path)
        return None
    return f"output/{rel.as_posix()}"
