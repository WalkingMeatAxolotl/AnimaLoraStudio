"""进程生命周期：重启 / 自更新 / 回滚 / 仓库状态（PR-6 commit 4 从 server.py 抽出）。

11 routes：
    POST /api/system/restart        无 pull 重启（写 tmp/restart + SIGINT）
    GET  /api/system/version        commit / tag / branch / dirty
    GET  /api/system/update_check   git fetch + 比对（master 24h cache / dev 总 fetch）
    POST /api/system/update         请求 update（写 .update_pending + 触发重启）
    POST /api/system/rollback       回滚到 .last_version（同 update 路径）
    GET  /api/system/update_status  最近一次 update 结构化结果 + rollback_target
    GET  /api/system/update_log     完整 .update_log 文本
    GET  /api/system/preflight      4 项前置检查 + requirements diff 摘要
    GET  /api/system/dev_commits    `git log origin/dev -N` 摘要
    POST /api/system/init_git       zip 用户初始化 git 仓库（幂等）

重启协议（参见 docs/adr/0002-webui-self-update.md）：
    1. server 写 REPO_ROOT/tmp/restart 标志
    2. server 通过 BackgroundTask 在响应发出后给自己发 SIGINT
    3. uvicorn 捕获 SIGINT 走 graceful shutdown（lifespan teardown + 在飞请求收尾）
    4. 进程退出 → cli.py 的 subprocess.call 返回
    5. cli.py 检测到 tmp/restart 存在 → 删除标志 → loop 回去重新 bootstrap + 起 server

跨平台 SIGINT：用 signal.raise_signal(SIGINT)（Python 3.8+），它在 Windows /
POSIX 都把当前进程置为收到 SIGINT，uvicorn 的内置 handler 会按 graceful
路径处理。os.kill(getpid, SIGINT) 在 Windows 上不工作。
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, BackgroundTasks

from ..schemas.system import UpdateRequest
from ... import db
from ...domain.errors import ConflictError, DomainError, ValidationError
from ...paths import REPO_ROOT
from ...services.runtime import updater

router = APIRouter()

_RESTART_FLAG = REPO_ROOT / "tmp" / "restart"
_SHUTDOWN_FORCE_EXIT_TIMEOUT = 5.0


def _raise_sigint_after_response() -> None:
    """在响应已经发完后给自己发 SIGINT，触发 uvicorn graceful shutdown。

    BackgroundTask 在 starlette 路径上是 response 完成后调度的；这里再 sleep
    一点点保险（防止某些代理 / keep-alive 情况下还有数据没冲走）。

    Force-exit 兜底（PR-D fix）：`/api/events` 是长 SSE，generator 内的
    `asyncio.wait_for(queue.get(), 15)` 不响应 uvicorn 关停信号，graceful
    shutdown 会等 client 主动断开 → 表现为「后端卡在 waiting for
    connection to close」，用户必须刷页让浏览器关 SSE 才能继续。给 graceful
    5 秒窗口后强退（正常 in-flight 1-2s 收尾够用）。BackgroundTask 跑在
    threadpool，graceful 成功路径主进程退出会带走此线程，os._exit 不会
    触达；只有 graceful 卡住时才真正强退。
    """
    import signal
    time.sleep(0.3)
    try:
        signal.raise_signal(signal.SIGINT)
    except Exception:
        # 兜底：raise_signal 抛错（极少见）→ 直接强退
        os._exit(0)
        return
    time.sleep(_SHUTDOWN_FORCE_EXIT_TIMEOUT)
    os._exit(0)


def _check_no_running_tasks() -> None:
    """重启 / 更新前置：所有 task 必须 done / failed / canceled / pending。

    有 running 直接 422 + task 列表，让前端给用户友好的提示（"先暂停以下任务"）。
    """
    with db.connection_for() as conn:
        running = db.list_tasks(conn, status="running")
    if running:
        raise ValidationError(
            "Tasks are running; cancel or wait for them to finish first",
            code="system.tasks_running",
            details={
                "tasks": [
                    {
                        "id": t["id"],
                        "name": t.get("name", ""),
                        "task_type": t.get("task_type", "train"),
                    }
                    for t in running
                ],
            },
            http_status=422,
        )


@router.post("/api/system/restart")
def system_restart(background: BackgroundTasks) -> dict[str, Any]:
    """重启 server（不 pull 代码）。

    流程：写 tmp/restart 标志 → 响应 200 → BackgroundTask 发 SIGINT 触发
    uvicorn graceful shutdown → cli.py loop 拾起 → 重新起新 server。

    PR-B 起加 running task 强制约束。
    """
    _check_no_running_tasks()
    _RESTART_FLAG.parent.mkdir(parents=True, exist_ok=True)
    _RESTART_FLAG.touch()
    background.add_task(_raise_sigint_after_response)
    return {"ok": True, "message": "restart scheduled"}


@router.get("/api/system/version")
def system_version() -> dict[str, Any]:
    """当前仓库状态：__version__ / commit / tag / branch / dirty。"""
    return asdict(updater.current_version())


@router.get("/api/system/update_check")
def system_update_check(channel: str = "master", force: bool = False) -> dict[str, Any]:
    """git fetch + 比对。master 通道用 24h cache（force=true 强制重 fetch）；
    dev 通道每次都 fetch，不缓存（开发者主动触发，避免污染 master 信号）。
    """
    if channel not in ("master", "dev"):
        raise ValidationError(
            f"Unsupported update channel: {channel}",
            code="system.channel_invalid",
            details={"channel": channel}, http_status=400,
        )
    return asdict(updater.check_update(channel=channel, use_cache=not force))


@router.post("/api/system/update")
def system_update(body: UpdateRequest, background: BackgroundTasks) -> dict[str, Any]:
    """请求 update：precondition 校验 + 写 .update_pending + 触发重启。

    实际 git pull 在 cli.py 启动期 updater.apply_pending() 完成（避免在 server
    进程里跑 git pull，规避 native module 已锁的问题）。
    """
    _check_no_running_tasks()

    cur = updater.current_version()
    # force=True：用户在 UI 上已确认"强制覆盖本地改动"，跳过 dirty 闸；启动期
    # apply_pending 的 git reset --hard 会丢弃这些未提交改动（不可恢复）。
    if cur.is_dirty and not body.force:
        raise ValidationError(
            "Local changes are uncommitted; commit or stash them before updating",
            code="system.working_tree_dirty", http_status=422,
        )

    updater.request_update(body.target, force=body.force)
    background.add_task(_raise_sigint_after_response)
    return {"ok": True, "message": f"update scheduled → {body.target}"}


@router.post("/api/system/rollback")
def system_rollback(background: BackgroundTasks) -> dict[str, Any]:
    """回滚到 .last_version 记录的上一版本（PR-C）。

    走与正向 update 完全一致的路径（写 .update_pending=<sha> + tmp/restart
    → cli.py 启动期 apply_pending 实际 reset），所以 dirty / running task
    precondition 一样适用，回滚成功后 .last_version 会被写成"回滚前的版本"
    （即正向)，支持来回切。
    """
    _check_no_running_tasks()

    cur = updater.current_version()
    if cur.is_dirty:
        raise ValidationError(
            "Local changes are uncommitted; commit or stash them before updating",
            code="system.working_tree_dirty", http_status=422,
        )

    target = updater.request_rollback()
    if target is None:
        raise ConflictError(
            "No previous version is available to roll back to",
            code="system.no_rollback_target",
        )

    background.add_task(_raise_sigint_after_response)
    return {"ok": True, "message": f"rollback scheduled → {target[:8]}", "target": target}


@router.get("/api/system/update_status")
def system_update_status() -> dict[str, Any]:
    """最近一次 update 的结构化结果 + rollback target（PR-C）。

    rollback_target 与 status 解耦：即使从未走过 update（.update_status 不存在），
    只要 .last_version 指向的 commit 还在仓库里，回滚按钮就应当能用（user 手动
    git reset 后想"还原到上一版"也是合法场景）。

    UI 上：
    - status=null：没有 update 历史，不展示 banner / 不展示"查看上次日志"按钮
    - status='ok'：可选展示"已更新到 X，X 秒前"
    - status='aborted' / 'failed' / 'partial'：红色 banner + reason + 跳日志
    - rollback_target 非 null（不依赖 status）：显示"切换到 sha"按钮
    """
    rollback_to = updater.rollback_target()
    rollback_tag = updater.exact_tag_for(rollback_to) if rollback_to else None
    st = updater.last_status()
    if st is None:
        return {
            "status": None,
            "rollback_target": rollback_to,
            "rollback_target_tag": rollback_tag,
        }
    return {
        **asdict(st),
        "rollback_target": rollback_to,
        "rollback_target_tag": rollback_tag,
    }


@router.get("/api/system/update_log")
def system_update_log() -> dict[str, Any]:
    """完整 .update_log 文本内容（PR-C 失败时 UI 弹 modal 用）。"""
    return {"content": updater.read_update_log()}


@router.get("/api/system/preflight")
def system_preflight(target: str = "origin/master") -> dict[str, Any]:
    """更新前置检查（chunk 4）— VersionSection preview 状态展开时拉取。

    返回 4 项结构化检查 + target_resolved sha + requirements.txt diff 摘要。
    每项含 level (ok / warn / err)；任一 err → blocking=true，前端禁用
    确认按钮。target 接受任意 git ref（tag / branch / commit sha）。
    """
    cur = updater.current_version()

    with db.connection_for() as conn:
        running = db.list_tasks(conn, status="running")

    target_resolved = updater.resolve_ref(target)
    req_diff = updater.requirements_diff(target) if target_resolved else updater.RequirementsDiff()
    req_total = len(req_diff.added) + len(req_diff.removed) + len(req_diff.changed)

    checks: list[dict[str, str]] = []

    if cur.is_dirty:
        # warn 而非 err：dirty 不再硬阻断，前端确认时弹"强制覆盖"modal，确认后
        # 带 force 提交 → reset --hard 覆盖本地改动。running task / self_update 等
        # 才是不可绕过的 err。（package-lock.json 等自动 churn 已在 updater 的
        # dirty 判定里剔除，不会走到这。）
        checks.append({"key": "dirty", "level": "warn",
                       "label": "工作树有未提交修改 — 确认更新会覆盖这些改动"})
    else:
        checks.append({"key": "dirty", "level": "ok",
                       "label": "工作树干净 · 无未提交改动"})

    if running:
        names = ", ".join((t.get("name") or f"#{t['id']}") for t in running[:3])
        more = f" + 还有 {len(running) - 3}" if len(running) > 3 else ""
        checks.append({"key": "running_tasks", "level": "err",
                       "label": f"{len(running)} 个任务正在运行：{names}{more}"})
    else:
        checks.append({"key": "running_tasks", "level": "ok",
                       "label": "当前 0 个训练 / 打标任务运行中"})

    if not target_resolved:
        checks.append({"key": "requirements_diff", "level": "err",
                       "label": f"target ref 解析失败：{target}"})
    elif req_total > 0:
        parts = []
        if req_diff.added:    parts.append(f"+{len(req_diff.added)}")
        if req_diff.removed:  parts.append(f"-{len(req_diff.removed)}")
        if req_diff.changed:  parts.append(f"~{len(req_diff.changed)}")
        checks.append({"key": "requirements_diff", "level": "warn",
                       "label": f"requirements.txt 变化 · {' / '.join(parts)} 包 · 预计 pip install 1-2 分钟"})
    else:
        checks.append({"key": "requirements_diff", "level": "ok",
                       "label": "requirements.txt 未变化 · 跳过 pip install"})

    checks.append({"key": "last_version", "level": "ok",
                   "label": f"更新后 .last_version = {cur.commit_short}（可一键切回）"})

    # Safety net：目标 ref 早于 self-update feature 引入 → 切过去就丢失 webui
    # 升级能力（只能 CLI git pull 救援）。err 级别阻断，前端 confirm 自动 disable。
    if target_resolved and not updater.target_has_self_update(target):
        checks.append({"key": "self_update_compat", "level": "err",
                       "label": "目标版本早于 webui 自更新 feature — 切过去后只能 CLI / shell 升级（webui 无救援能力）"})

    blocking = any(c["level"] == "err" for c in checks)

    return {
        "target": target,
        "target_resolved": target_resolved,
        "checks": checks,
        "blocking": blocking,
        # 工作树脏（真实改动；自动 churn 已剔除）。前端据此在确认时弹"强制覆盖"
        # modal —— dirty 是 warn 不进 blocking，但仍需用户显式确认才覆盖。
        "working_tree_dirty": cur.is_dirty,
        "requirements_diff": {
            "added": req_diff.added,
            "removed": req_diff.removed,
            "changed": req_diff.changed,
        },
    }


@router.get("/api/system/dev_commits")
def system_dev_commits(limit: int = 10) -> dict[str, Any]:
    """`git log origin/dev -N` 摘要（chunk 3）— VersionSection dev 卡时间线用。

    每次拉 git fetch + log；fetch 失败仍尝试用本地 origin/dev 缓存（带
    error 文案）。limit clamp 1-50。
    """
    result = updater.dev_commits(limit=limit)
    return {
        "commits": [asdict(c) for c in result.commits],
        "fetched": result.fetched,
        "error": result.error,
    }


@router.post("/api/system/init_git")
def system_init_git() -> dict[str, Any]:
    """zip 解压用户一键初始化 git 仓库（0.8.1 hotfix）。

    幂等：调用前 / 调用后都跑 `git_repo_status()`，如已是仓库直接返 ok=true。
    流程见 `updater.bootstrap_git_repo()`：init + remote add origin + fetch master
    + reset --mixed 到对应 release tag。

    失败状态码：
    - 500 + error 字符串：git binary 缺失 / fetch 网络问题 / 磁盘问题
    """
    pre = updater.git_repo_status()
    if pre.is_repo:
        return {"ok": True, "already_initialized": True}

    result = updater.bootstrap_git_repo()
    if not result.ok:
        raise DomainError(
            f"Failed to initialize the Git repository: {result.error or 'unknown error'}",
            code="system.git_init_failed",
            details={"reason": result.error or "unknown error"}, http_status=500,
        )

    return {"ok": True, "already_initialized": False, **asdict(result)}
