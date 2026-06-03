"""Phase 推进 / 完成校验 — ADR-0007 §11.5-A / §11.5-B。

Phase enum 见 ``versions.VersionPhase``：
``curating → tagging → editing → regularizing → ready``。

完成判定（§11.5-B）：
- ``curating``: ``train/ ≥ 1`` 张图
- ``tagging``: caption 100% 覆盖（每张 train 图都有同名 .txt）
- ``editing``: 同 tagging（兜底，防 user 删了 caption）
- ``regularizing``: 无 reg_build job 处于 pending/running（可跳过，§11.5-A SKIPPABLE）
- ``ready``: training config 文件存在 + schema 校验通过

cursor 推进规则（§11.5-A）：
- 单向（只前进）
- header "下一步" 按钮永远可点；校验失败给用户提示
- 必经 phase 校验失败 → 不推进
- 可跳过 phase 校验 = 无 concurrent job（regularizing 无 confirm dialog）
- cursor 不主动回退（§11.5-C；user 删数据后下次 next 校验时才提示）
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Optional

from . import projects as _projects
from . import versions as _versions
from .. import version_config as _version_config


@dataclass(frozen=True)
class CheckResult:
    """Phase 校验结果。ok=True 时 reason 可空；ok=False 时 reason 是 user 可读提示。"""
    ok: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# 单 phase 校验函数（参数取最小所需，便于 unit test 隔离）
# ---------------------------------------------------------------------------


def check_curating(stats: dict[str, Any]) -> CheckResult:
    """train/ ≥ 1 张图（§11.5-B）。"""
    if stats.get("train_image_count", 0) < 1:
        return CheckResult(False, "训练集为空，请先选择训练图")
    return CheckResult(True)


def check_tagging(stats: dict[str, Any]) -> CheckResult:
    """caption 100% 覆盖（§11.5-B）。"""
    total = int(stats.get("train_image_count", 0))
    tagged = int(stats.get("tagged_image_count", 0))
    if total < 1:
        return CheckResult(False, "训练集为空，请先选择训练图")
    if tagged < total:
        missing = total - tagged
        return CheckResult(False, f"还有 {missing} 张未生成 caption，请重跑或删除")
    return CheckResult(True)


def check_editing(stats: dict[str, Any]) -> CheckResult:
    """editing 同 tagging（兜底，§11.5-B；大多数情况自动通过）。"""
    return check_tagging(stats)


def check_regularizing(
    conn: sqlite3.Connection, version_id: int
) -> CheckResult:
    """无 reg_build job 处于 pending/running（§11.5-B；可跳过 = 不强求正则集非空）。"""
    row = conn.execute(
        "SELECT COUNT(*) FROM project_jobs "
        "WHERE version_id = ? "
        "  AND kind = 'reg_build' "
        "  AND status IN ('pending', 'running')",
        (version_id,),
    ).fetchone()
    if int(row[0]) > 0:
        return CheckResult(False, "正则任务进行中，请等待完成")
    return CheckResult(True)


def check_preprocessing(
    conn: sqlite3.Connection, version_id: int
) -> CheckResult:
    """无 preprocess job 处于 pending/running（ADR 0010；可跳过 = 不强求处理过）。

    跟 `check_regularizing` 同 pattern——预处理是可选 phase（upscale / crop /
    去重等都可跳过，接受训练时默认放大算法）；校验仅防 concurrent job 撞车。
    """
    row = conn.execute(
        "SELECT COUNT(*) FROM project_jobs "
        "WHERE version_id = ? "
        "  AND kind = 'preprocess' "
        "  AND status IN ('pending', 'running')",
        (version_id,),
    ).fetchone()
    if int(row[0]) > 0:
        return CheckResult(False, "预处理任务进行中，请等待完成")
    return CheckResult(True)


def check_ready(
    project: dict[str, Any], version: dict[str, Any]
) -> CheckResult:
    """training config 存在 + schema 校验通过（§11.5-B）。"""
    try:
        _version_config.read_version_config(project, version)
    except _version_config.VersionConfigError as exc:
        return CheckResult(False, f"请先完成训练配置：{exc}")
    return CheckResult(True)


# ---------------------------------------------------------------------------
# Dispatcher + 推进
# ---------------------------------------------------------------------------


def check_phase(
    conn: sqlite3.Connection, version_id: int, phase: str
) -> CheckResult:
    """根据 phase 选择对应 check 函数；version / project 不存在也以 CheckResult 返回。"""
    v = _versions.get_version(conn, version_id)
    if not v:
        return CheckResult(False, "版本不存在")
    p = _projects.get_project(conn, int(v["project_id"]))
    if not p:
        return CheckResult(False, "项目不存在")

    P = _versions.VersionPhase
    if phase == P.CURATING:
        return check_curating(_versions.stats_for_version(p, v))
    if phase == P.PREPROCESSING:
        return check_preprocessing(conn, version_id)
    if phase == P.TAGGING:
        return check_tagging(_versions.stats_for_version(p, v))
    if phase == P.EDITING:
        return check_editing(_versions.stats_for_version(p, v))
    if phase == P.REGULARIZING:
        return check_regularizing(conn, version_id)
    if phase == P.READY:
        return check_ready(p, v)
    return CheckResult(False, f"未知 phase: {phase}")


def advance_phase(
    conn: sqlite3.Connection, version_id: int
) -> tuple[bool, CheckResult, Optional[str]]:
    """尝试推进 phase cursor 到下一个（必经/可跳过都走这个）。

    返回 ``(advanced, result, new_phase)``:
    - ``advanced=True``: cursor 已推进；``new_phase`` 是新 phase 名
    - ``advanced=False``: ``result.reason`` 含失败原因；``new_phase=None``

    ``ready`` 是最后一个 phase — 调用方收到 ``(False, ok-but-end, None)`` 后
    应转入 status: ``preparing → training`` + submit task（即 enqueue 训练）。
    """
    v = _versions.get_version(conn, version_id)
    if not v:
        return False, CheckResult(False, "版本不存在"), None

    current_phase = _versions.get_phase(v)
    order = _versions.VersionPhase.ORDER

    # phase 不在已知集合 → fallback 拒绝
    if current_phase not in order:
        return False, CheckResult(False, f"未知 phase: {current_phase}"), None

    # 已到最后 phase（ready）→ phase 不再推进；由调用方触发 status 转换
    idx = order.index(current_phase)
    if idx >= len(order) - 1:
        result = check_phase(conn, version_id, current_phase)
        # 即使到 ready 也跑一次校验，让调用方决定是否进 training
        return False, result, None

    # 走校验
    result = check_phase(conn, version_id, current_phase)
    if not result.ok:
        return False, result, None

    next_phase = order[idx + 1]
    _versions.update_version(conn, version_id, phase=next_phase)
    return True, CheckResult(True), next_phase


def skip_phase(
    conn: sqlite3.Connection, version_id: int
) -> tuple[bool, CheckResult, Optional[str]]:
    """跳过当前 phase（仅 ``SKIPPABLE`` 集合允许；当前 = preprocessing /
    regularizing）。

    与 ``advance_phase`` 区别：不要求"完成条件"满足（如不要求生成正则集 /
    不要求每张图都预处理过），仅校验"无 concurrent job"防止状态错乱。
    """
    v = _versions.get_version(conn, version_id)
    if not v:
        return False, CheckResult(False, "版本不存在"), None

    current_phase = _versions.get_phase(v)
    if current_phase not in _versions.VersionPhase.SKIPPABLE:
        return False, CheckResult(False, f"phase {current_phase} 不可跳过"), None

    # skip 时仍校验"无 concurrent job"防止状态错乱
    if current_phase == _versions.VersionPhase.REGULARIZING:
        result = check_regularizing(conn, version_id)
        if not result.ok:
            return False, result, None
    elif current_phase == _versions.VersionPhase.PREPROCESSING:
        result = check_preprocessing(conn, version_id)
        if not result.ok:
            return False, result, None

    order = _versions.VersionPhase.ORDER
    idx = order.index(current_phase)
    if idx >= len(order) - 1:
        return False, CheckResult(False, "已到最后 phase"), None
    next_phase = order[idx + 1]
    _versions.update_version(conn, version_id, phase=next_phase)
    return True, CheckResult(True), next_phase
