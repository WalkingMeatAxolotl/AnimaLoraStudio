"""Studio FastAPI 守护进程的端点冒烟测试（P1 范围）。

测试只覆盖 server.py 暴露的 5 个端点。每个用例通过 monkeypatch 把
`studio.server` 模块里指向运行时数据/HTML 的路径常量改写到 tmp_path，
避免污染仓库真实目录。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import server


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """把 server 模块里的路径全部指向 tmp_path 下的隔离目录。"""
    from studio import db
    output = tmp_path / "output"
    samples_dir = output / "samples"
    legacy_html = tmp_path / "monitor_smooth.html"
    web_dist = tmp_path / "web_dist"  # 不创建即模拟未构建
    samples_dir.mkdir(parents=True)

    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server, "OUTPUT_DIR", output)
    monkeypatch.setattr(server, "LEGACY_MONITOR_HTML", legacy_html)
    monkeypatch.setattr(server, "WEB_DIST", web_dist)
    return {
        "tmp": tmp_path,
        "db": dbfile,
        "output": output,
        "samples_dir": samples_dir,
        "legacy_html": legacy_html,
        "web_dist": web_dist,
    }


@pytest.fixture
def client(isolated_paths: dict[str, Path]) -> TestClient:
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

def test_health_returns_ok(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == server.app.version


# ---------------------------------------------------------------------------
# /api/state
# ---------------------------------------------------------------------------

def test_state_missing_returns_empty(client: TestClient, isolated_paths: dict[str, Path]) -> None:
    """没有 task_id 也没有 running 任务时返回空状态。"""
    resp = client.get("/api/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body["losses"] == []
    assert body["lr_history"] == []
    assert body["step"] == 0
    assert body["epoch"] == 0
    assert body["start_time"] is None


def _make_task_with_state(
    isolated_paths: dict[str, Path], payload: dict | str | None
) -> int:
    """建一个 task 并写 state 文件，返回 task_id。payload=None 表示不写文件。"""
    from studio import db as _db
    state_dir = isolated_paths["tmp"] / "states"
    state_dir.mkdir(exist_ok=True)
    state_file = state_dir / "state.json"
    if payload is not None:
        state_file.write_text(
            json.dumps(payload) if isinstance(payload, dict) else payload,
            encoding="utf-8",
        )
    with _db.connection_for(isolated_paths["db"]) as conn:
        tid = _db.create_task(conn, name="t", config_name="x")
        _db.update_task(conn, tid, monitor_state_path=str(state_file))
    return tid


def test_state_by_task_id_returns_parsed_json(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    payload = {
        "losses": [{"step": 1, "loss": 0.5, "time": 100.0}],
        "lr_history": [{"step": 1, "lr": 1e-4}],
        "epoch": 2,
        "step": 42,
        "total_steps": 1000,
        "speed": 1.23,
        "samples": [],
        "start_time": 1700000000.0,
        "config": {"lora_rank": 32},
    }
    tid = _make_task_with_state(isolated_paths, payload)
    resp = client.get(f"/api/state?task_id={tid}")
    assert resp.status_code == 200
    assert resp.json() == payload


def test_state_corrupt_returns_500(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    tid = _make_task_with_state(isolated_paths, "this is not json")
    resp = client.get(f"/api/state?task_id={tid}")
    assert resp.status_code == 500


def test_state_unknown_task_returns_empty(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    resp = client.get("/api/state?task_id=99999")
    assert resp.status_code == 200
    assert resp.json()["losses"] == []


def test_state_running_task_used_when_no_task_id(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """没给 task_id → 默认拉当前 running 的 task。"""
    payload = {"losses": [], "lr_history": [], "epoch": 0, "step": 7,
               "total_steps": 0, "speed": 0.0, "samples": [],
               "start_time": None, "config": {}}
    from studio import db as _db
    tid = _make_task_with_state(isolated_paths, payload)
    with _db.connection_for(isolated_paths["db"]) as conn:
        _db.update_task(conn, tid, status="running", started_at=1.0)
    resp = client.get("/api/state")
    assert resp.json()["step"] == 7


# ---------------------------------------------------------------------------
# /samples/{filename}
# ---------------------------------------------------------------------------

def test_sample_404_for_missing(client: TestClient) -> None:
    resp = client.get("/samples/does_not_exist.png")
    assert resp.status_code == 404


def test_sample_returns_file(client: TestClient, isolated_paths: dict[str, Path]) -> None:
    img_path = isolated_paths["samples_dir"] / "step_42.png"
    img_path.write_bytes(b"fake-png-bytes")
    resp = client.get("/samples/step_42.png")
    assert resp.status_code == 200
    assert resp.content == b"fake-png-bytes"


@pytest.mark.parametrize("bad", ["../secret.txt", "..\\secret.txt", "sub/dir.png", "sub\\dir.png"])
def test_sample_blocks_traversal(client: TestClient, bad: str) -> None:
    """`/samples/{name}` 不允许斜杠 / 反斜杠 / 上级路径。"""
    resp = client.get(f"/samples/{bad}")
    # 含 `/` 或 `\` 的会被路由层拆成多段（404），含 `..` 的被显式 400 拒绝；
    # 任何一种都不应该 200。
    assert resp.status_code != 200


def test_sample_with_task_id_finds_in_output_samples(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """回归 Q4：anima_train 把 sample 写到 `output_dir/samples/`，端点应在
    `monitor_state_path 同级 output/samples/` 也能命中（之前只查了同级 samples/）。"""
    from studio import db as _db
    state_path = isolated_paths["tmp"] / "v1" / "monitor_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{}", encoding="utf-8")
    out_samples = state_path.parent / "output" / "samples"
    out_samples.mkdir(parents=True)
    (out_samples / "step_0_baseline_0.png").write_bytes(b"sample-bytes")

    with _db.connection_for(isolated_paths["db"]) as conn:
        tid = _db.create_task(conn, name="t", config_name="x")
        _db.update_task(conn, tid, monitor_state_path=str(state_path))

    resp = client.get(f"/samples/step_0_baseline_0.png?task_id={tid}")
    assert resp.status_code == 200, resp.text
    assert resp.content == b"sample-bytes"


def test_sample_with_task_id_finds_in_state_dir_samples(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """旧约定路径（monitor_state.json 同级 samples/）仍兼容。"""
    from studio import db as _db
    state_path = isolated_paths["tmp"] / "v2" / "monitor_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{}", encoding="utf-8")
    samples = state_path.parent / "samples"
    samples.mkdir()
    (samples / "step_5.png").write_bytes(b"old-layout")

    with _db.connection_for(isolated_paths["db"]) as conn:
        tid = _db.create_task(conn, name="t", config_name="x")
        _db.update_task(conn, tid, monitor_state_path=str(state_path))

    resp = client.get(f"/samples/step_5.png?task_id={tid}")
    assert resp.status_code == 200
    assert resp.content == b"old-layout"


# ---------------------------------------------------------------------------
# /
# ---------------------------------------------------------------------------

def test_root_serves_legacy_html(client: TestClient, isolated_paths: dict[str, Path]) -> None:
    isolated_paths["legacy_html"].write_text("<!DOCTYPE html><h1>legacy</h1>", encoding="utf-8")
    resp = client.get("/")
    assert resp.status_code == 200
    assert "legacy" in resp.text


def test_root_fallback_when_no_legacy(client: TestClient, isolated_paths: dict[str, Path]) -> None:
    """旧监控页缺失时返回 JSON 提示，而不是 404。"""
    assert not isolated_paths["legacy_html"].exists()
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "AnimaStudio" in body["message"]
