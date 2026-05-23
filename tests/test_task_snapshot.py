"""ADR-0007 §11.7: task config snapshot module + endpoint 测试。"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, projects, server, task_snapshot


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    # task_snapshot 用 STUDIO_DATA：mock 到 tmp_path / studio_data
    monkeypatch.setattr(task_snapshot, "STUDIO_DATA", tmp_path / "studio_data")
    return {"db": dbfile, "data": tmp_path / "studio_data"}


# ---------------------------------------------------------------------------
# 纯 module tests
# ---------------------------------------------------------------------------


def test_snapshot_paths_under_studio_data(isolated) -> None:
    root = task_snapshot.snapshot_root()
    assert root == isolated["data"] / "tasks"
    assert task_snapshot.snapshot_dir(42) == root / "42" / "snapshot"
    assert task_snapshot.snapshot_config_path(42) == root / "42" / "snapshot" / "config.yaml"


def test_freeze_config_creates_file(isolated, tmp_path: Path) -> None:
    src = tmp_path / "src.yaml"
    src.write_text("lr: 0.001\nbatch_size: 4\n", encoding="utf-8")
    dst = task_snapshot.freeze_config(7, src)
    assert dst == task_snapshot.snapshot_config_path(7)
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == "lr: 0.001\nbatch_size: 4\n"


def test_freeze_config_overwrites_existing(isolated, tmp_path: Path) -> None:
    src1 = tmp_path / "v1.yaml"
    src1.write_text("lr: 0.001\n", encoding="utf-8")
    task_snapshot.freeze_config(7, src1)

    src2 = tmp_path / "v2.yaml"
    src2.write_text("lr: 0.0005\n", encoding="utf-8")
    task_snapshot.freeze_config(7, src2)

    final = task_snapshot.read_snapshot_config(7)
    assert final is not None
    assert final["config"]["lr"] == 0.0005


def test_freeze_config_missing_source_raises(isolated, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        task_snapshot.freeze_config(7, tmp_path / "nonexistent.yaml")


def test_has_snapshot_true_false(isolated, tmp_path: Path) -> None:
    assert not task_snapshot.has_snapshot(8)
    src = tmp_path / "src.yaml"
    src.write_text("x: 1\n", encoding="utf-8")
    task_snapshot.freeze_config(8, src)
    assert task_snapshot.has_snapshot(8)


def test_read_snapshot_returns_yaml_and_dict(isolated, tmp_path: Path) -> None:
    src = tmp_path / "src.yaml"
    src.write_text("lr: 0.001\nbatch_size: 4\n", encoding="utf-8")
    task_snapshot.freeze_config(7, src)

    data = task_snapshot.read_snapshot_config(7)
    assert data is not None
    assert data["yaml"] == "lr: 0.001\nbatch_size: 4\n"
    assert data["config"] == {"lr": 0.001, "batch_size": 4}


def test_read_snapshot_missing_returns_none(isolated) -> None:
    assert task_snapshot.read_snapshot_config(99999) is None


def test_read_snapshot_non_mapping_falls_back_to_empty(
    isolated, tmp_path: Path
) -> None:
    """如果 yaml 不是 mapping（如纯字符串），config 字段应是 {}。"""
    src = tmp_path / "weird.yaml"
    src.write_text("just_a_string\n", encoding="utf-8")
    task_snapshot.freeze_config(10, src)
    data = task_snapshot.read_snapshot_config(10)
    assert data is not None
    assert data["config"] == {}


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client(isolated) -> TestClient:
    return TestClient(server.app)


def _create_task(isolated, name: str = "t1") -> int:
    with db.connection_for(isolated["db"]) as conn:
        return db.create_task(conn, name=name, config_name="cfg")


def test_endpoint_404_when_task_missing(client: TestClient) -> None:
    resp = client.get("/api/queue/99999/snapshot/config")
    assert resp.status_code == 404
    assert "task" in resp.json()["detail"].lower()


def test_endpoint_404_when_snapshot_missing(client: TestClient, isolated) -> None:
    tid = _create_task(isolated)
    resp = client.get(f"/api/queue/{tid}/snapshot/config")
    assert resp.status_code == 404
    assert "snapshot" in resp.json()["detail"].lower()


def test_endpoint_returns_snapshot_data(
    client: TestClient, isolated, tmp_path: Path
) -> None:
    tid = _create_task(isolated)
    src = tmp_path / "cfg.yaml"
    src.write_text("lr: 0.001\noptimizer: adamw\n", encoding="utf-8")
    task_snapshot.freeze_config(tid, src)

    resp = client.get(f"/api/queue/{tid}/snapshot/config")
    assert resp.status_code == 200
    body = resp.json()
    assert body["yaml"] == "lr: 0.001\noptimizer: adamw\n"
    assert body["config"] == {"lr": 0.001, "optimizer": "adamw"}
