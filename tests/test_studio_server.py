"""Studio FastAPI 守护进程的端点冒烟测试（P1 范围）。

测试只覆盖 server.py 暴露的 5 个端点。每个用例通过 monkeypatch 把
`studio.server` 模块里指向运行时数据的路径常量改写到 tmp_path，
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
    web_dist = tmp_path / "web_dist"  # 不创建即模拟未构建
    samples_dir.mkdir(parents=True)

    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server, "OUTPUT_DIR", output)
    monkeypatch.setattr(server, "WEB_DIST", web_dist)
    return {
        "tmp": tmp_path,
        "db": dbfile,
        "output": output,
        "samples_dir": samples_dir,
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

def test_torch_status_proxies_service(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/torch/status 把 torch_setup.current_status() 透传给前端。"""
    from studio.services import torch_setup
    monkeypatch.setattr(torch_setup, "current_status", lambda: {
        "installed": True,
        "version": "2.5.0+cpu",
        "cuda_build": "cpu",
        "cuda_available": False,
        "device_name": None,
        "cuda_detect": {"available": True, "driver_version": "555.86", "gpu_name": "RTX 5090"},
        "recommended_cu_tag": "cu128",
        "is_cpu_with_gpu": True,
        "is_cuda_build_unavailable": False,
    })
    resp = client.get("/api/torch/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_cpu_with_gpu"] is True
    assert body["recommended_cu_tag"] == "cu128"


def test_torch_reinstall_registers_marker_returns_pending(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """POST /api/torch/reinstall 不真装，写 marker 返回 pending。"""
    from studio.services import pending_install, torch_setup
    monkeypatch.setattr(pending_install, "STUDIO_DATA", tmp_path)
    monkeypatch.setattr(pending_install, "PENDING_MARKER", tmp_path / ".pending-pip-install.json")
    monkeypatch.setattr(torch_setup, "_decide_target_tag", lambda _t: "cu128")

    resp = client.post("/api/torch/reinstall", json={"target": "auto"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["pending"] is True
    assert body["tag"] == "cu128"
    assert body["target"] == "auto"
    assert "studio.bat" in body["message"]
    # marker 文件已写
    assert (tmp_path / ".pending-pip-install.json").exists()


def test_torch_reinstall_invalid_target_returns_400(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services import torch_setup
    monkeypatch.setattr(
        torch_setup, "_decide_target_tag",
        lambda t: (_ for _ in ()).throw(ValueError(f"非法 target: {t!r}")),
    )
    resp = client.post("/api/torch/reinstall", json={"target": "xpu"})
    assert resp.status_code == 400
    assert "非法 target" in resp.json()["detail"]


def test_flash_attention_status_returns_env_and_candidates(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/flash-attention/status 应返回 status + env + slim candidates + fetch_error。"""
    from studio.services import flash_attention_setup
    monkeypatch.setattr(flash_attention_setup, "current_status", lambda: {
        "installed": True, "version": "2.8.3"
    })
    monkeypatch.setattr(flash_attention_setup, "detect_env", lambda: {
        "python_tag": "cp311", "cuda_tag": "cu128", "cuda_ver": "12.8",
        "torch_tag": "torch2.5", "torch_ver": "2.5.0+cu128", "platform": "win_amd64",
    })
    monkeypatch.setattr(flash_attention_setup, "find_candidates", lambda _env: ([
        {
            "url": "https://x/wheel.whl",
            "name": "flash_attn-2.8.3+cu128torch2.5-cp311-cp311-win_amd64.whl",
            "score": 40,  # 应被剥掉
            "notes": [],
            "usable": True,
            "tags": {"cuda": "cu128"},  # 应被剥掉
        },
    ], None))

    resp = client.get("/api/flash-attention/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["installed"] is True
    assert body["version"] == "2.8.3"
    assert body["env"]["platform"] == "win_amd64"
    # candidates 只保留 url/name/notes/usable —— score / tags 不暴露给前端
    assert len(body["candidates"]) == 1
    c = body["candidates"][0]
    assert set(c.keys()) == {"url", "name", "notes", "usable"}
    assert body["fetch_error"] is None


def test_flash_attention_status_passes_fetch_error_through(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub 限流 / 网络异常时 fetch_error 透传给 UI。"""
    from studio.services import flash_attention_setup
    monkeypatch.setattr(flash_attention_setup, "current_status", lambda: {
        "installed": False, "version": None,
    })
    monkeypatch.setattr(flash_attention_setup, "detect_env", lambda: {
        "python_tag": "cp311", "cuda_tag": None, "cuda_ver": None,
        "torch_tag": None, "torch_ver": None, "platform": "linux_x86_64",
    })
    monkeypatch.setattr(
        flash_attention_setup, "find_candidates",
        lambda _env: ([], "GitHub API 错误: API rate limit exceeded"),
    )
    resp = client.get("/api/flash-attention/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["candidates"] == []
    assert "rate limit" in body["fetch_error"]


def test_flash_attention_install_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services import flash_attention_setup
    captured: dict = {}

    def fake_install(url):
        captured["url"] = url
        return {
            "installed": True, "version": "2.8.3",
            "url": url or "https://auto/wheel.whl",
            "stdout_tail": "Successfully installed",
            "restart_required": True,
        }

    monkeypatch.setattr(flash_attention_setup, "install", fake_install)
    resp = client.post("/api/flash-attention/install", json={"url": "https://x/manual.whl"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["installed"] is True
    assert body["restart_required"] is True
    assert captured["url"] == "https://x/manual.whl"


def test_flash_attention_install_url_null_uses_auto(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """前端不传 url（或显式 null）→ service 收到 None，走自动匹配。"""
    from studio.services import flash_attention_setup
    captured: dict = {}

    def fake_install(url):
        captured["url"] = url
        return {"installed": True, "version": "2.8.3", "url": "auto",
                "stdout_tail": "", "restart_required": True}

    monkeypatch.setattr(flash_attention_setup, "install", fake_install)
    resp = client.post("/api/flash-attention/install", json={"url": None})
    assert resp.status_code == 200
    assert captured["url"] is None


def test_flash_attention_install_failure_returns_500(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services import flash_attention_setup

    def boom(_url):
        raise RuntimeError("pip install 失败:\nERROR: bad wheel")

    monkeypatch.setattr(flash_attention_setup, "install", boom)
    resp = client.post("/api/flash-attention/install", json={"url": "https://x/bad.whl"})
    assert resp.status_code == 500
    assert "bad wheel" in resp.json()["detail"]


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


def test_state_max_points_downsamples_losses(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """PR #37：/api/state 兑现 max_points，losses/lr 长度超过时均匀降采样。"""
    losses = [{"step": i, "loss": 1.0 / (i + 1), "time": float(i)} for i in range(5000)]
    lr_history = [{"step": i, "lr": 1e-4} for i in range(5000)]
    payload = {
        "losses": losses, "lr_history": lr_history, "epoch": 0, "step": 4999,
        "total_steps": 5000, "speed": 0.0, "samples": [],
        "start_time": None, "config": {},
    }
    tid = _make_task_with_state(isolated_paths, payload)

    # max_points=500 → 都被压到 500
    resp = client.get(f"/api/state?task_id={tid}&max_points=500")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["losses"]) == 500
    assert len(body["lr_history"]) == 500
    # 首尾保留
    assert body["losses"][0]["step"] == 0
    assert body["losses"][-1]["step"] == 4999
    # 其他字段透传
    assert body["step"] == 4999
    assert body["total_steps"] == 5000


def test_state_max_points_zero_disables_downsample(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """max_points=0 (无穷) → 不降采样，原样返回。"""
    losses = [{"step": i, "loss": 0.0} for i in range(100)]
    payload = {"losses": losses, "lr_history": [], "epoch": 0, "step": 99,
               "total_steps": 100, "speed": 0.0, "samples": [],
               "start_time": None, "config": {}}
    tid = _make_task_with_state(isolated_paths, payload)
    resp = client.get(f"/api/state?task_id={tid}&max_points=0")
    assert resp.status_code == 200
    assert len(resp.json()["losses"]) == 100


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

def test_root_redirects_to_studio_when_built(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """前端 dist 存在时，/ 应 302 跳转到 /studio/。"""
    isolated_paths["web_dist"].mkdir(parents=True, exist_ok=True)
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/studio/"


def test_root_fallback_when_no_dist(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """前端未构建时返回 JSON 提示，而不是 404 / 跳转。"""
    assert not isolated_paths["web_dist"].exists()
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code == 200
    body = resp.json()
    assert "AnimaStudio" in body["message"]
