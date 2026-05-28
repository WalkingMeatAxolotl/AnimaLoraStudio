"""队列导入 / 导出测试。"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.services.presets import io as presets_io
from studio.services import queue_io
from studio.schema import TrainingConfig


def _payload() -> dict:
    return TrainingConfig().model_dump(mode="python")


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    presets = tmp_path / "presets"
    presets.mkdir()
    # 把所有模块的指针都换到 tmp_path
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", presets)
    monkeypatch.setattr(server, "USER_PRESETS_DIR", presets)
    return {"db": dbfile, "presets": presets}


@pytest.fixture
def client(isolated) -> TestClient:
    server.app.state.supervisor = None  # 端点用到时会 503，但导入导出不需要
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# queue_io 单元
# ---------------------------------------------------------------------------


def test_export_then_import_roundtrip(isolated) -> None:
    payload = _payload()
    payload["epochs"] = 7
    presets_io.write_preset("alpha", payload)

    with db.connection_for(isolated["db"]) as conn:
        tid = db.create_task(conn, name="run1", config_name="alpha", priority=3)

    exported = queue_io.export_tasks([tid], db_path=isolated["db"])
    assert exported["version"] == 1
    assert len(exported["tasks"]) == 1
    assert exported["tasks"][0]["config"]["epochs"] == 7

    # 把现有删掉，再 import
    presets_io.delete_preset("alpha")
    with db.connection_for(isolated["db"]) as conn:
        db.delete_task(conn, tid)

    result = queue_io.import_tasks(exported, db_path=isolated["db"])
    assert result["imported_count"] == 1
    # 配置和任务都被还原
    assert presets_io.read_preset("alpha")["epochs"] == 7
    with db.connection_for(isolated["db"]) as conn:
        all_tasks = db.list_tasks(conn)
    assert len(all_tasks) == 1
    assert all_tasks[0]["priority"] == 3


def test_import_renames_on_conflict(isolated) -> None:
    """同名 preset 已存在时自动加 _imported_N 后缀。"""
    base = _payload()
    base["epochs"] = 3
    presets_io.write_preset("alpha", base)  # 占位

    incoming = {
        "version": 1,
        "tasks": [
            {
                "name": "x",
                "config_name": "alpha",
                "priority": 0,
                "config": {**_payload(), "epochs": 99},
            }
        ],
    }
    result = queue_io.import_tasks(incoming, db_path=isolated["db"])
    assert result["imported_count"] == 1
    assert result["renamed"]["alpha"].startswith("alpha_imported_")
    new_name = result["renamed"]["alpha"]
    assert presets_io.read_preset(new_name)["epochs"] == 99
    assert presets_io.read_preset("alpha")["epochs"] == 3  # 原来的不变


def test_import_rejects_unknown_version(isolated) -> None:
    with pytest.raises(ValueError):
        queue_io.import_tasks({"version": 99, "tasks": []}, db_path=isolated["db"])


def test_import_skips_tasks_without_config_when_local_missing(isolated) -> None:
    """payload 没附带 config 且本地也没有，对应任务应被跳过。"""
    incoming = {
        "version": 1,
        "tasks": [{
            "name": "x", "config_name": "doesnt_exist", "priority": 0,
            "config": None,
        }],
    }
    result = queue_io.import_tasks(incoming, db_path=isolated["db"])
    assert result["imported_count"] == 0


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_api_export_all(client: TestClient, isolated) -> None:
    presets_io.write_preset("a", _payload())
    with db.connection_for(isolated["db"]) as conn:
        db.create_task(conn, name="t1", config_name="a")
        db.create_task(conn, name="t2", config_name="a")
    resp = client.get("/api/queue/export")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == 1
    assert len(body["tasks"]) == 2


def test_api_export_subset(client: TestClient, isolated) -> None:
    presets_io.write_preset("a", _payload())
    with db.connection_for(isolated["db"]) as conn:
        a = db.create_task(conn, name="t1", config_name="a")
        db.create_task(conn, name="t2", config_name="a")
    resp = client.get(f"/api/queue/export?ids={a}")
    assert resp.status_code == 200
    assert len(resp.json()["tasks"]) == 1


def test_api_export_invalid_ids_400(client: TestClient) -> None:
    resp = client.get("/api/queue/export?ids=abc")
    assert resp.status_code == 400


def test_api_import(client: TestClient, isolated) -> None:
    payload = {
        "version": 1,
        "tasks": [{
            "name": "imported",
            "config_name": "fresh",
            "priority": 5,
            "config": _payload(),
        }],
    }
    resp = client.post("/api/queue/import", json={"payload": payload})
    assert resp.status_code == 200, resp.text
    assert resp.json()["imported_count"] == 1
    assert presets_io.read_preset("fresh") is not None


def test_api_import_bad_version_400(client: TestClient) -> None:
    resp = client.post(
        "/api/queue/import",
        json={"payload": {"version": 999, "tasks": []}},
    )
    assert resp.status_code == 400
