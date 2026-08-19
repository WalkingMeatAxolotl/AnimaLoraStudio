"""诊断包（services/diagnostics + GET /api/diagnostics/bundle；logging-target-state §3.6）。

覆盖：zip 成员清单、run.log / 快照 / monitor 进包、studio.log 按任务起止时间窗切片
（含轮转文件、窗外行不进）、脱敏、无 task 的尾部模式、unknown task 404。
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.services import diagnostics as diag


def _ts(epoch: float) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _studio_line(epoch: float, msg: str) -> str:
    return json.dumps({"ts": _ts(epoch), "level": "INFO", "process": "webui", "logger": "x", "msg": msg}) + "\n"


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    from studio.infrastructure import paths as _paths
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    logs = tmp_path / "logs"
    logs.mkdir()
    monkeypatch.setattr(diag, "LOGS_DIR", logs)
    monkeypatch.setattr(_paths, "TASKS_DIR", tmp_path / "tasks")
    # 任务 42：started 1000s，finished 2000s
    with db.connection_for(dbfile) as conn:
        tid = db.create_task(conn, name="t", config_name="cfg")
        db.update_task(conn, tid, status="failed", started_at=1000.0, finished_at=2000.0,
                       error_msg="exit code 1")
    tdir = _paths.task_dir(tid)
    (tdir / "snapshot").mkdir(parents=True)
    (tdir / "monitor").mkdir(parents=True)
    (tdir / "run.log").write_text("2026-08-19 14:00:00.000 ERROR training.loop: boom api_key=SECRET123 done\n", encoding="utf-8")
    (tdir / "snapshot" / "config.yaml").write_text("epochs: 1\nwandb_api_key: abc\n", encoding="utf-8")
    (tdir / "monitor" / "state.json").write_text('{"step": 3}', encoding="utf-8")
    # studio.log：轮转文件里一条窗内、主文件里窗内两条 + 窗外两条
    (logs / "studio.log.1").write_text(_studio_line(950, "rotated-in-window") + _studio_line(10, "rotated-old"), encoding="utf-8")
    (logs / "studio.log").write_text(
        _studio_line(500, "before-window")
        + _studio_line(1500, "in-window Authorization: Bearer tok123")
        + _studio_line(2030, "tail-pad-in-window")
        + _studio_line(9000, "after-window hf_ABCDEFGHIJKLMNOPQRST"),
        encoding="utf-8",
    )
    return {"tid": tid, "logs": logs, "tdir": tdir}


def _open(data: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(data))


def test_bundle_members_and_window_slice(env: dict) -> None:
    data, name = diag.build_bundle(env["tid"], extra_env={"env_summary": {"driver_version": "1.2"}})
    assert name.startswith(f"anima-diag-task{env['tid']}-") and name.endswith(".zip")
    z = _open(data)
    names = set(z.namelist())
    assert names == {"README.txt", "env.json", "task.json", "task/run.log",
                     "task/snapshot/config.yaml", "task/monitor/state.json", "studio.log"}
    env_json = json.loads(z.read("env.json"))
    assert env_json["env_summary"] == {"driver_version": "1.2"}
    assert "studio_version" in env_json and "torch" in env_json or "torch_error" in env_json
    task_json = json.loads(z.read("task.json"))
    assert task_json["id"] == env["tid"] and task_json["error_msg"] == "exit code 1"
    slice_ = z.read("studio.log").decode("utf-8")
    assert "rotated-in-window" in slice_ and "in-window" in slice_ and "tail-pad-in-window" in slice_
    assert "before-window" not in slice_ and "after-window" not in slice_ and "rotated-old" not in slice_
    # 轮转文件在前（时间从旧到新）
    assert slice_.index("rotated-in-window") < slice_.index("in-window Authorization")


def test_bundle_redacts_secrets(env: dict) -> None:
    data, _ = diag.build_bundle(env["tid"])
    z = _open(data)
    run_log = z.read("task/run.log").decode("utf-8")
    assert "SECRET123" not in run_log and "api_key=***" in run_log and "done" in run_log
    assert "tok123" not in z.read("studio.log").decode("utf-8")
    cfg = z.read("task/snapshot/config.yaml").decode("utf-8")
    assert "abc" not in cfg and "wandb_api_key: ***" in cfg
    readme = z.read("README.txt").decode("utf-8")
    assert "脱敏" in readme and f"task_id: {env['tid']}" in readme


def test_redact_patterns() -> None:
    assert diag.redact("token=abc&x=1") == "token=***&x=1"
    assert diag.redact("password: hunter2") == "password: ***"
    assert diag.redact("Authorization: Bearer eyJ.abc") == "Authorization: Bearer ***"
    assert diag.redact("see hf_ABCDEFGHIJKLMNOPQRST here") == "see hf_*** here"
    assert diag.redact("sk-abcdefghijklmnopqrstuvwxyz") == "sk-***"
    assert diag.redact("epoch=3 lr=1e-4 loss=0.1") == "epoch=3 lr=1e-4 loss=0.1"


def test_bundle_without_task_is_tail_mode(env: dict) -> None:
    data, name = diag.build_bundle(None)
    assert name.startswith("anima-diag-") and "task" not in name
    z = _open(data)
    assert set(z.namelist()) == {"README.txt", "env.json", "studio.log"}
    assert "after-window" in z.read("studio.log").decode("utf-8")  # 尾部模式不按窗切


def test_bundle_unknown_task_raises(env: dict) -> None:
    with pytest.raises(LookupError):
        diag.build_bundle(999999)


def test_endpoint_streams_zip_and_404(env: dict) -> None:
    client = TestClient(server.app)
    r = client.get(f"/api/diagnostics/bundle?task_id={env['tid']}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/zip")
    assert f"anima-diag-task{env['tid']}-" in r.headers["content-disposition"]
    assert "task/run.log" in _open(r.content).namelist()
    r2 = client.get("/api/diagnostics/bundle")
    assert r2.status_code == 200 and "task.json" not in _open(r2.content).namelist()
    r3 = client.get("/api/diagnostics/bundle?task_id=999999")
    assert r3.status_code == 404 and r3.json()["error"]["code"] == "task.not_found"
