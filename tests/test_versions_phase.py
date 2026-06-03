"""ADR-0007 §11.5-A / §11.5-B: phase check_completion + advance / skip 函数测试。"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from studio import db
from studio.services.projects import projects, versions, phase as versions_phase
from studio.services.projects.phase import CheckResult


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    pdir = tmp_path / "projects"
    monkeypatch.setattr(projects, "PROJECTS_DIR", pdir)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    return {"db": dbfile}


def _make_version(isolated, label: str = "v1") -> dict:
    with db.connection_for(isolated["db"]) as conn:
        p = projects.create_project(conn, title="P")
        v = versions.create_version(conn, project_id=p["id"], label=label)
    return v


def _put_image(folder: Path, name: str, with_caption: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.png").write_bytes(b"fake")
    if with_caption:
        (folder / f"{name}.txt").write_text("tag1, tag2", encoding="utf-8")


# ---------------------------------------------------------------------------
# Pure check function tests (stats dict 直接喂)
# ---------------------------------------------------------------------------


def test_check_curating_empty() -> None:
    result = versions_phase.check_curating({"train_image_count": 0})
    assert not result.ok
    assert "训练集为空" in result.reason


def test_check_curating_with_images() -> None:
    assert versions_phase.check_curating({"train_image_count": 1}).ok
    assert versions_phase.check_curating({"train_image_count": 100}).ok


def test_check_tagging_full_coverage() -> None:
    result = versions_phase.check_tagging({
        "train_image_count": 10, "tagged_image_count": 10,
    })
    assert result.ok


def test_check_tagging_partial_coverage() -> None:
    result = versions_phase.check_tagging({
        "train_image_count": 10, "tagged_image_count": 7,
    })
    assert not result.ok
    assert "3 张" in result.reason


def test_check_tagging_empty() -> None:
    result = versions_phase.check_tagging({"train_image_count": 0})
    assert not result.ok
    assert "训练集为空" in result.reason


def test_check_editing_same_as_tagging() -> None:
    """editing 是 tagging 的兜底，应该 100% 行为一致。"""
    stats = {"train_image_count": 10, "tagged_image_count": 5}
    assert (
        versions_phase.check_editing(stats).reason
        == versions_phase.check_tagging(stats).reason
    )


# ---------------------------------------------------------------------------
# Regularizing check (要 DB，因为查 project_jobs)
# ---------------------------------------------------------------------------


def test_check_regularizing_no_jobs(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        result = versions_phase.check_regularizing(conn, v["id"])
    assert result.ok


def test_check_regularizing_blocks_when_job_running(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        conn.execute(
            "INSERT INTO project_jobs(project_id, version_id, kind, params, status) "
            "VALUES (?, ?, 'reg_build', '{}', 'running')",
            (v["project_id"], v["id"]),
        )
        conn.commit()
        result = versions_phase.check_regularizing(conn, v["id"])
    assert not result.ok
    assert "正则" in result.reason


def test_check_regularizing_blocks_when_job_pending(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        conn.execute(
            "INSERT INTO project_jobs(project_id, version_id, kind, params, status) "
            "VALUES (?, ?, 'reg_build', '{}', 'pending')",
            (v["project_id"], v["id"]),
        )
        conn.commit()
        result = versions_phase.check_regularizing(conn, v["id"])
    assert not result.ok


def test_check_regularizing_done_job_doesnt_block(isolated) -> None:
    """已完成的 reg job 不阻塞 cursor 前进。"""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        conn.execute(
            "INSERT INTO project_jobs(project_id, version_id, kind, params, status) "
            "VALUES (?, ?, 'reg_build', '{}', 'done')",
            (v["project_id"], v["id"]),
        )
        conn.commit()
        result = versions_phase.check_regularizing(conn, v["id"])
    assert result.ok


# ---------------------------------------------------------------------------
# Ready check (要 file system，因为读 config.yaml)
# ---------------------------------------------------------------------------


def test_check_ready_no_config(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        p = projects.get_project(conn, v["project_id"])
        result = versions_phase.check_ready(p, v)
    assert not result.ok
    assert "配置" in result.reason


# ---------------------------------------------------------------------------
# advance_phase / skip_phase
# ---------------------------------------------------------------------------


def test_advance_phase_blocked_by_failed_check(isolated) -> None:
    """curating phase + train 为空 → next 失败 + cursor 不动。"""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        advanced, result, new_phase = versions_phase.advance_phase(conn, v["id"])
    assert not advanced
    assert not result.ok
    assert new_phase is None
    # cursor 应该还在 curating
    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.get_version(conn, v["id"])
    assert versions.get_phase(v2) == "curating"


def test_advance_phase_curating_to_preprocessing(isolated) -> None:
    """curating phase + train 有图 → 推进到 preprocessing（ADR 0010 加新 phase）。"""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        p = projects.get_project(conn, v["project_id"])
    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    _put_image(vdir / "train" / "5_concept", "001", with_caption=False)

    with db.connection_for(isolated["db"]) as conn:
        advanced, result, new_phase = versions_phase.advance_phase(conn, v["id"])
    assert advanced
    assert result.ok
    assert new_phase == "preprocessing"
    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.get_version(conn, v["id"])
    assert versions.get_phase(v2) == "preprocessing"


def test_advance_phase_tagging_blocked_by_missing_caption(isolated) -> None:
    """tagging phase + caption 覆盖不到 100% → 失败。"""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        p = projects.get_project(conn, v["project_id"])
        # 强制把 cursor 跳到 tagging
        versions.update_version(conn, v["id"], phase="tagging")
    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    _put_image(vdir / "train" / "5_concept", "001", with_caption=False)
    _put_image(vdir / "train" / "5_concept", "002", with_caption=True)

    with db.connection_for(isolated["db"]) as conn:
        advanced, result, _ = versions_phase.advance_phase(conn, v["id"])
    assert not advanced
    assert "1 张" in result.reason


def test_skip_phase_only_works_for_skippable(isolated) -> None:
    """curating 不允许 skip。"""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        advanced, result, _ = versions_phase.skip_phase(conn, v["id"])
    assert not advanced
    assert "不可跳过" in result.reason


def test_skip_phase_preprocessing_jumps_to_tagging(isolated) -> None:
    """ADR 0010：preprocessing 可跳过（无 preprocess job running） → cursor 跳到 tagging。"""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.update_version(conn, v["id"], phase="preprocessing")
        advanced, result, new_phase = versions_phase.skip_phase(conn, v["id"])
    assert advanced
    assert result.ok
    assert new_phase == "tagging"


def test_skip_phase_preprocessing_blocked_by_running_job(isolated) -> None:
    """有 preprocess job pending/running → 拒跳过（防 concurrent job 撞车）。"""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.update_version(conn, v["id"], phase="preprocessing")
        conn.execute(
            "INSERT INTO project_jobs(project_id, version_id, kind, params, status) "
            "VALUES (?, ?, 'preprocess', '{}', 'running')",
            (v["project_id"], v["id"]),
        )
        conn.commit()
        advanced, result, _ = versions_phase.skip_phase(conn, v["id"])
    assert not advanced
    assert "预处理" in result.reason


def test_check_preprocessing_ok_without_running_job(isolated) -> None:
    """无 concurrent preprocess job → OK（preprocessing 可跳过不强求完成度）。"""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        result = versions_phase.check_preprocessing(conn, v["id"])
    assert result.ok


def test_skip_phase_regularizing_jumps_to_ready(isolated) -> None:
    """regularizing skip → cursor 跳 ready。"""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.update_version(conn, v["id"], phase="regularizing")
        advanced, result, new_phase = versions_phase.skip_phase(conn, v["id"])
    assert advanced
    assert result.ok
    assert new_phase == "ready"


def test_skip_phase_regularizing_blocked_by_running_job(isolated) -> None:
    """regularizing skip 校验"无 job running"，有 job 时拒绝。"""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.update_version(conn, v["id"], phase="regularizing")
        conn.execute(
            "INSERT INTO project_jobs(project_id, version_id, kind, params, status) "
            "VALUES (?, ?, 'reg_build', '{}', 'running')",
            (v["project_id"], v["id"]),
        )
        conn.commit()
        advanced, result, _ = versions_phase.skip_phase(conn, v["id"])
    assert not advanced
    assert "正则" in result.reason


def test_advance_phase_at_ready_returns_check_result(isolated) -> None:
    """已到 ready（最后 phase）→ phase 不再前进，但返回 check 结果让调用方决定 status 转换。"""
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        versions.update_version(conn, v["id"], phase="ready")
        advanced, result, new_phase = versions_phase.advance_phase(conn, v["id"])
    assert not advanced
    assert new_phase is None
    # 无 config → check_ready 应失败
    assert not result.ok


# ---------------------------------------------------------------------------
# update_version 新字段校验
# ---------------------------------------------------------------------------


def test_update_version_accepts_valid_status(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.update_version(conn, v["id"], status="training")
    assert v2["status"] == "training"


def test_update_version_rejects_invalid_status(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(versions.VersionError, match="非法 status"):
            versions.update_version(conn, v["id"], status="bogus")


def test_update_version_accepts_valid_phase(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.update_version(conn, v["id"], phase="editing")
    assert v2["phase"] == "editing"


def test_update_version_rejects_invalid_phase(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        with pytest.raises(versions.VersionError, match="非法 phase"):
            versions.update_version(conn, v["id"], phase="bogus")


def test_update_version_accepts_last_failure_reason(isolated) -> None:
    v = _make_version(isolated)
    with db.connection_for(isolated["db"]) as conn:
        v2 = versions.update_version(conn, v["id"], last_failure_reason="OOM at step 500")
    assert v2["last_failure_reason"] == "OOM at step 500"
