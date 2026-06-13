"""Queue 任务生命周期（PR-6 commit 6 从 server.py 抽出）。

12 routes：
    GET   /api/queue                 list（默认隐藏 generate / reg_ai）
    POST  /api/queue                 enqueue（按 preset 名）
    GET   /api/queue/hold            查队列挂起状态 + 等待恢复 pending 数
    POST  /api/queue/hold            挂起队列（dispatcher 停拉新 task）
    POST  /api/queue/release         恢复调度
    POST  /api/queue/reorder         按 id 列表重排
    GET   /api/queue/{task_id}       task DB 行（含 is_pausable / is_resumable 信号）
    POST  /api/queue/{task_id}/cancel
    POST  /api/queue/{task_id}/pause   ADR 0006 §4.1
    POST  /api/queue/{task_id}/resume  ADR 0006 §6 路径 A + Addendum 2（paused/failed/canceled）
    POST  /api/queue/{task_id}/retry   复制 config_path / project_id / version_id 起新 task
    DELETE /api/queue/{task_id}       仅 terminal task 可删
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException

from ...deps import _supervisor
from ...schemas.queue import EnqueueRequest, ReorderRequest
from .... import db
from ....infrastructure.event_bus import bus
from ....paths import USER_PRESETS_DIR, task_dir

router = APIRouter()

# resume 允许的起点状态（ADR 0006 Addendum 2）。done 不进 — 语义是重训，
# 走 retry / ResumeFieldPicker。
RESUMABLE_STATUSES = ("paused", "failed", "canceled")


def _resume_source_paths(task: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """按 status 取恢复点 (state_path, config_path)。

    paused 用 paused_*（pause 流程写入，指向 auto_epoch_state.pt）；
    failed/canceled 用 last_*（supervisor 每 epoch 收 auto_epoch_backup_written
    事件落 DB，ADR Addendum 2）；其余状态无恢复点。
    """
    status = task.get("status")
    if status == "paused":
        return task.get("paused_state_path"), task.get("paused_config_path")
    if status in ("failed", "canceled"):
        return task.get("last_state_path"), task.get("last_config_path")
    return None, None


def _is_resumable(task: dict[str, Any]) -> bool:
    """UI "继续训练" 按钮显隐信号。跟 resume endpoint 的放行条件保持一致：
    state 文件必须在；config snapshot 路径有值时文件也必须在（严格 freeze）。
    """
    state_path, config_path = _resume_source_paths(task)
    if not state_path or not Path(state_path).exists():
        return False
    if config_path and not Path(config_path).exists():
        return False
    return True


@router.get("/api/queue")
def list_queue(
    status: Optional[str] = None,
    include_generate: bool = False,
) -> dict[str, Any]:
    """队列默认隐藏 generate 测试出图任务（commit 15 P0-2）。

    generate task 走 daemon 不占 train slot，且生命周期短（出完图就结束），
    出现在队列里只会让用户混淆"为什么队列卡住"。需要排查时加
    `?include_generate=true` 兜底。
    """
    if status and status not in db.VALID_STATUSES:
        raise HTTPException(400, f"unknown status: {status}")
    with db.connection_for() as conn:
        items = db.list_tasks(conn, status=status)
    if not include_generate:
        items = db.filter_out_task_types(items, ("generate", "reg_ai"))
    # ADR 0006 PR-4 — is_pausable 信号每行注入（§8.1 / 上面 get_queue_item 注释）
    try:
        sup = _supervisor()
        for it in items:
            it["is_pausable"] = sup.is_task_pausable(int(it["id"]))
    except HTTPException:
        for it in items:
            it["is_pausable"] = False
    # ADR Addendum 2 — is_resumable 信号（paused/failed/canceled + 恢复点文件在盘）
    for it in items:
        it["is_resumable"] = _is_resumable(it)
    return {"items": items}


@router.post("/api/queue")
def enqueue(body: EnqueueRequest) -> dict[str, Any]:
    cfg_path = USER_PRESETS_DIR / f"{body.config_name}.yaml"
    if not cfg_path.exists():
        raise HTTPException(404, f"preset not found: {body.config_name}")
    name = body.name or body.config_name
    with db.connection_for() as conn:
        task_id = db.create_task(
            conn, name=name, config_name=body.config_name, priority=body.priority
        )
        task = db.get_task(conn, task_id)
    bus.publish(
        {"type": "task_state_changed", "task_id": task_id, "status": "pending"}
    )
    return task or {"id": task_id}


@router.get("/api/queue/hold")
def get_queue_hold() -> dict[str, Any]:
    """查看当前队列挂起状态 + 等待恢复调度的 pending task 数（UI banner 用）。"""
    with db.connection_for() as conn:
        held = db.get_queue_held(conn)
        pending = db.list_tasks(conn, status="pending")
    return {"held": held, "pending_waiting": len(pending)}


@router.post("/api/queue/hold")
def hold_queue() -> dict[str, Any]:
    """挂起队列：dispatcher 不再拉新 task。已 running 的不受影响（ADR §3.2）。

    "同时暂停 running task" 由前端 modal 拆成两步：先调本 endpoint，再
    单独调 `/api/queue/{id}/pause`。后端不做合一操作。
    """
    with db.connection_for() as conn:
        db.set_queue_held(conn, True)
    bus.publish({"type": "queue_hold_changed", "held": True})
    return {"held": True}


@router.post("/api/queue/release")
def release_queue() -> dict[str, Any]:
    """恢复调度：dispatcher 重新按 priority + created_at 拉 pending。"""
    with db.connection_for() as conn:
        db.set_queue_held(conn, False)
    bus.publish({"type": "queue_hold_changed", "held": False})
    return {"held": False}


@router.post("/api/queue/reorder")
def reorder_queue(body: ReorderRequest) -> dict[str, Any]:
    with db.connection_for() as conn:
        db.reorder(conn, body.ordered_ids)
    return {"reordered": len(body.ordered_ids)}


@router.get("/api/queue/{task_id}")
def get_queue_item(task_id: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
    if not task:
        raise HTTPException(404)
    # ADR 0006 PR-4 — is_pausable 信号让 UI 决定是否显示暂停按钮（§8.1）。
    # 仅 supervisor 跑得起来时计算；空载（test / 启动期）默认 False。
    try:
        task["is_pausable"] = _supervisor().is_task_pausable(task_id)
    except HTTPException:
        task["is_pausable"] = False
    task["is_resumable"] = _is_resumable(task)
    return task


@router.post("/api/queue/{task_id}/cancel")
def cancel_task(task_id: int) -> dict[str, Any]:
    if not _supervisor().cancel(task_id):
        # 可能任务已结束 / 不在 supervisor 控制
        with db.connection_for() as conn:
            task = db.get_task(conn, task_id)
        if not task:
            raise HTTPException(404)
        if task["status"] in db.TERMINAL_STATUSES:
            raise HTTPException(400, f"task already {task['status']}")
        raise HTTPException(409, "cancel rejected (state mismatch)")
    return {"task_id": task_id, "canceled": True}


@router.post("/api/queue/{task_id}/pause")
def pause_task(task_id: int) -> dict[str, Any]:
    """暂停 running task（ADR §4.1 / §4.3）。

    异步：立即返回；UI 端 modal 订阅 SSE 看保存进度。supervisor 收到子进程
    `__EVENT__:pause_state` 后把 status 写为 paused 并 publish task_state_changed。
    """
    ok, reason = _supervisor().pause(task_id)
    if not ok:
        # 区分客户端错误（404/409）vs 状态机不允许（409）
        with db.connection_for() as conn:
            task = db.get_task(conn, task_id)
        if not task:
            raise HTTPException(404, "task not found")
        raise HTTPException(409, reason or "pause rejected")
    return {"task_id": task_id, "pause_pending": True}


@router.post("/api/queue/{task_id}/resume")
def resume_task(task_id: int) -> dict[str, Any]:
    """恢复 paused / failed / canceled task（ADR 0006 §6 路径 A + Addendum 2）。

    流程：
      1. 校验 status ∈ RESUMABLE_STATUSES + 恢复点文件存在
         （paused 读 paused_*；failed/canceled 读 last_*，即 epoch 末 auto backup）
      2. task → pending；failed/canceled 把 last_* 复制进 paused_* —— 复用
         paused 管道，cmd_builder（--resume-state 注入）/ bootstrap_phase
         （sibling .config.json snapshot freeze）整条下游零改动
      3. supervisor 下次 _tick 自然 pick up
      4. 子进程 load_training_state 成功后 emit `resume_state_loaded` →
         supervisor `_clear_pause_fields` 清 db 字段（文件保留，Addendum 2）
      5. failed/canceled 曾把 version finalize 成 failed/canceled —— 复活后
         reconcile 派生回 training 并 publish version_state_changed

    文件丢失 → 409（ADR §5.5：引导用户走 ResumeFieldPicker 起新 task）。
    done 不可 resume（语义是重训，走 retry / ResumeFieldPicker）。
    """
    corrected_version: Optional[dict[str, Any]] = None
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
        if not task:
            raise HTTPException(404, "task not found")
        status = task["status"]
        if status not in RESUMABLE_STATUSES:
            raise HTTPException(
                409,
                f"task status is {status!r}, not resumable "
                f"(expected one of {', '.join(RESUMABLE_STATUSES)})",
            )
        state_path, config_path = _resume_source_paths(task)
        if not state_path or not Path(state_path).exists():
            raise HTTPException(
                409,
                f"resume state file missing: {state_path!r}; "
                f"use ResumeFieldPicker to start a fresh task from another .pt",
            )
        if config_path and not Path(config_path).exists():
            # snapshot 缺失虽然不致命（bootstrap_phase 会沿用 args.config yaml），
            # 但 resume 语义会漂；按 ADR §5.7 严格 freeze 原则，拒绝继续。
            raise HTTPException(
                409,
                f"resume config snapshot missing: {config_path!r}; "
                f"cannot guarantee config freeze, refusing to resume",
            )
        fields: dict[str, Any] = dict(
            status="pending",
            started_at=None,
            finished_at=None,
            exit_code=None,
            error_msg=None,
        )
        if status in ("failed", "canceled"):
            # ADR Addendum 2 决策 3：把恢复点搬进 paused_* 走既有管道。
            fields["paused_state_path"] = state_path
            fields["paused_config_path"] = config_path
            fields["paused_step"] = task.get("last_state_step")
        db.update_task(conn, task_id, **fields)
        if status in ("failed", "canceled") and task.get("version_id"):
            # ADR Addendum 2 决策 5：version status 从 failed/canceled 派生回
            # training（task 已 pending）。dispatch 时 _write_task_running_to_db
            # 还会再写一次，这里先改是为了 UI 立即一致。
            from ....services.projects import versions as _versions
            v, was_corrected = _versions.reconcile_version_status(
                conn, int(task["version_id"])
            )
            if was_corrected and v:
                corrected_version = v
    bus.publish({"type": "task_state_changed", "task_id": task_id, "status": "pending"})
    if corrected_version is not None:
        from ....services.projects import versions as _versions
        bus.publish({
            "type": "version_state_changed",
            "project_id": corrected_version["project_id"],
            "version_id": corrected_version["id"],
            "status": _versions.get_status(corrected_version),
            "phase": _versions.get_phase(corrected_version),
        })
    return {"task_id": task_id, "status": "pending"}


@router.post("/api/queue/{task_id}/retry")
def retry_task(task_id: int) -> dict[str, Any]:
    """已结束任务重新入队：复制完整训练上下文创建新 task。

    需要复制的字段（PP6.1+ 引入；老的 retry 只复制 name/config_name/priority
    会让 supervisor 走老降级路径用全局 preset 而不是 version 私有 config，
    导致重试参数与原任务不一致）：
    - config_path：version 私有 config 的绝对路径
    - project_id / version_id：用于 monitor_state_path 解析与 stage 推进

    不复制：status / pid / *_at / exit_code / error_msg / monitor_state_path
    （都是「上次跑」的产物；新任务从 pending 开始，supervisor 会重新解析）。
    """
    with db.connection_for() as conn:
        original = db.get_task(conn, task_id)
        if not original:
            raise HTTPException(404)
        if original["status"] not in db.TERMINAL_STATUSES:
            raise HTTPException(400, "only terminal tasks can be retried")
        new_id = db.create_task(
            conn,
            name=original["name"],
            config_name=original["config_name"],
            priority=original["priority"],
        )
        copy_fields: dict[str, Any] = {}
        for k in ("config_path", "project_id", "version_id"):
            if original.get(k) is not None:
                copy_fields[k] = original[k]
        if copy_fields:
            db.update_task(conn, new_id, **copy_fields)
        new_task = db.get_task(conn, new_id)
    bus.publish(
        {"type": "task_state_changed", "task_id": new_id, "status": "pending"}
    )
    return new_task or {"id": new_id}


@router.delete("/api/queue/{task_id}")
def delete_queue_item(task_id: int) -> dict[str, Any]:
    with db.connection_for() as conn:
        task = db.get_task(conn, task_id)
        if not task:
            raise HTTPException(404)
        if task["status"] not in db.TERMINAL_STATUSES:
            raise HTTPException(400, "only terminal tasks can be deleted")
        db.delete_task(conn, task_id)
    # task-scoped 档案（snapshot/config.yaml / monitor/state.json / samples/ /
    # run.log）跟 task DB 行同生命周期 —— 删 task 一并清。老 task 散在
    # studio_data/logs/<id>.log / studio_data/monitors/task_<id>/ 的不动
    # （不写迁移脚本，留作老版本兼容性测试数据）。
    import shutil
    tdir = task_dir(task_id)
    if tdir.exists():
        shutil.rmtree(tdir, ignore_errors=True)
    # ADR Addendum 2 决策 4：恢复点文件（<output_dir>/state/task_<id>/）不在
    # task_dir 下，生命周期同样跟 task 行绑定，删 task 一并清。目录名必须
    # 精确等于 task_<id> 才动手 —— 防 DB 路径异常时误删别的目录。
    for col in ("last_state_path", "paused_state_path"):
        p = task.get(col)
        if not p:
            continue
        state_dir = Path(p).parent
        if state_dir.name == f"task_{task_id}" and state_dir.exists():
            shutil.rmtree(state_dir, ignore_errors=True)
    return {"deleted": task_id}
