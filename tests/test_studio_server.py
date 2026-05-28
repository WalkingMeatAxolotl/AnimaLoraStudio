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
    """把 server 模块里的路径全部指向 tmp_path 下的隔离目录。

    PR-6 commit 1：/samples / / 等 routes 搬到 api/routers/，监控 OUTPUT_DIR /
    WEB_DIST 的真实绑名在新模块。新位置和 server.py 同时 patch，保 old
    `server.OUTPUT_DIR` patch 不丢、新 handler 也能看到 fake 值。
    """
    from studio import db
    from studio.api.routers import root as _root_router
    from studio.api.routers import samples as _samples_router
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
    monkeypatch.setattr(_samples_router, "OUTPUT_DIR", output)
    monkeypatch.setattr(_root_router, "WEB_DIST", web_dist)
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


def test_generate_sample_response_is_not_browser_cached(client: TestClient) -> None:
    from studio.services import generate_cache

    generate_cache.clear_all()
    try:
        generate_cache.cache_image(7, "sample.png", b"PNG")
        resp = client.get("/api/generate/7/sample/sample.png")
        assert resp.status_code == 200
        assert resp.content == b"PNG"
        assert resp.headers["cache-control"] == "no-store"
    finally:
        generate_cache.clear_all()


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


def test_state_default_returns_full_payload(
    client: TestClient, isolated_paths: dict[str, Path]
) -> None:
    """新默认（PR #43）：不传 max_points 等价于 max_points=0，返回全量历史。

    10k 步训练 cold start 时用户能拿到完整数据；想降采样的 caller 必须显式
    传具体数字。
    """
    losses = [{"step": i, "loss": 0.1} for i in range(10000)]
    payload = {
        "losses": losses, "lr_history": [], "epoch": 0, "step": 9999,
        "total_steps": 10000, "speed": 0.0, "samples": [],
        "start_time": None, "config": {},
    }
    tid = _make_task_with_state(isolated_paths, payload)
    # 不传 max_points 任何参数
    resp = client.get(f"/api/state?task_id={tid}")
    assert resp.status_code == 200
    assert len(resp.json()["losses"]) == 10000


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


# ---------------------------------------------------------------------------
# /api/system/restart (ADR 0002 / PR-A)
# ---------------------------------------------------------------------------

def test_system_restart_writes_flag_and_schedules_shutdown(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/system/restart 应该：
    1. 写 tmp/restart 标志文件
    2. 触发 BackgroundTask 走 _raise_sigint_after_response（实际发 SIGINT 会杀
       测试进程，所以这里 monkeypatch 成 no-op + 记录是否被调用）
    3. 返回 200 + {"ok": true}
    """
    # 把 SIGINT helper 替换为记录器，避免真发信号杀测试
    called: dict[str, bool] = {"ran": False}
    def _stub_shutdown() -> None:
        called["ran"] = True
    # PR-6 commit 4：system router 从 server.py 抽到 api/routers/system.py，
    # patch path 跟搬迁
    monkeypatch.setattr(
        "studio.api.routers.system._raise_sigint_after_response", _stub_shutdown
    )

    # 把 flag 路径指向 tmp，避免污染仓库 tmp/
    flag = isolated_paths["tmp"] / "restart"
    monkeypatch.setattr("studio.api.routers.system._RESTART_FLAG", flag)

    assert not flag.exists()
    resp = client.post("/api/system/restart")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert flag.exists(), "tmp/restart flag 应被写入"
    # BackgroundTask 在 starlette TestClient 上是同步执行的（response 走完后），
    # 所以这里 stub 一定被调用过
    assert called["ran"], "_raise_sigint_after_response BackgroundTask 应被调度"


# ---------------------------------------------------------------------------
# /api/system/version (ADR 0002 / PR-B)
# ---------------------------------------------------------------------------

def test_system_version_returns_fields(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /api/system/version 应返回 VersionInfo dataclass 的所有字段。"""
    from studio.services import updater

    fake = updater.VersionInfo(
        version="0.6.0", commit="abc123", commit_short="abc123",
        commit_time_iso="2026-05-13T10:00:00+00:00", branch="master",
        tag="v0.6.0", is_dirty=False,
    )
    monkeypatch.setattr(updater, "current_version", lambda: fake)

    resp = client.get("/api/system/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "0.6.0"
    assert body["branch"] == "master"
    assert body["tag"] == "v0.6.0"
    assert body["is_dirty"] is False


# ---------------------------------------------------------------------------
# /api/system/update_check (ADR 0002 / PR-B)
# ---------------------------------------------------------------------------

def test_system_update_check_master_default(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认 channel=master，返回 UpdateCheckResult 字段。"""
    from studio.services import updater

    fake = updater.UpdateCheckResult(
        channel="master", current_commit="abc", latest_commit="def",
        commits_ahead=2, has_update=True, latest_tag="v0.6.1",
        checked_at=1234567890.0,
    )
    captured: dict = {}
    def _fake_check(channel="master", use_cache=True):
        captured["channel"] = channel
        captured["use_cache"] = use_cache
        return fake
    monkeypatch.setattr(updater, "check_update", _fake_check)

    resp = client.get("/api/system/update_check")
    assert resp.status_code == 200
    body = resp.json()
    assert body["channel"] == "master"
    assert body["has_update"] is True
    assert body["latest_tag"] == "v0.6.1"
    assert captured == {"channel": "master", "use_cache": True}


def test_system_update_check_force_skips_cache(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force=true 应传 use_cache=False 给 updater.check_update。"""
    from studio.services import updater

    captured: dict = {}
    def _fake_check(channel="master", use_cache=True):
        captured["use_cache"] = use_cache
        return updater.UpdateCheckResult(
            channel=channel, current_commit="", latest_commit="",
            commits_ahead=0, has_update=False, latest_tag=None,
            checked_at=0.0,
        )
    monkeypatch.setattr(updater, "check_update", _fake_check)

    client.get("/api/system/update_check?force=true")
    assert captured["use_cache"] is False


def test_system_update_check_invalid_channel(client: TestClient) -> None:
    resp = client.get("/api/system/update_check?channel=feature%2Fwhatever")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /api/system/update (ADR 0002 / PR-B)
# ---------------------------------------------------------------------------

def test_system_update_rejects_when_running_task(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有 status='running' 任务时 update 应返 422 + 任务列表。"""
    from studio import db
    with db.connection_for(isolated_paths["db"]) as conn:
        tid = db.create_task(conn, name="炼丹中", config_name="train", priority=0)
        db.update_task(conn, tid, status="running", task_type="train")

    resp = client.post("/api/system/update", json={"target": "origin/master"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["detail"]["error"] == "running_tasks_present"
    assert len(body["detail"]["tasks"]) == 1
    assert body["detail"]["tasks"][0]["id"] == tid


def test_system_update_rejects_when_dirty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """working tree dirty 时 update 应返 422。"""
    from studio.services import updater
    dirty = updater.VersionInfo(
        version="0.6.0", commit="abc", commit_short="abc", commit_time_iso="",
        branch="master", tag=None, is_dirty=True,
    )
    monkeypatch.setattr(updater, "current_version", lambda: dirty)

    resp = client.post("/api/system/update", json={"target": "origin/master"})
    assert resp.status_code == 422
    body = resp.json()
    assert body["detail"]["error"] == "dirty_working_tree"


def test_system_update_writes_pending_and_restart_flag(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """正常路径：写 .update_pending（含 target）+ tmp/restart + 调度 SIGINT。"""
    from studio.services import updater

    # 干净 working tree
    clean = updater.VersionInfo(
        version="0.6.0", commit="abc", commit_short="abc", commit_time_iso="",
        branch="master", tag=None, is_dirty=False,
    )
    monkeypatch.setattr(updater, "current_version", lambda: clean)

    # 重定向 flag 路径到隔离 tmp
    pending = tmp_path / ".update_pending"
    restart_flag = tmp_path / "tmp" / "restart"
    monkeypatch.setattr(updater, "UPDATE_PENDING", pending)
    monkeypatch.setattr(updater, "RESTART_FLAG", restart_flag)

    # 拦截 SIGINT
    monkeypatch.setattr("studio.api.routers.system._raise_sigint_after_response", lambda: None)

    resp = client.post("/api/system/update", json={"target": "origin/dev"})
    assert resp.status_code == 200
    assert pending.exists()
    assert pending.read_text(encoding="utf-8") == "origin/dev"
    assert restart_flag.exists()


def test_system_restart_rejects_when_running_task(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR-B 新加：restart 也要查 running task。"""
    from studio import db
    with db.connection_for(isolated_paths["db"]) as conn:
        tid = db.create_task(conn, name="跑数据", config_name="tag", priority=0)
        db.update_task(conn, tid, status="running", task_type="tag")

    resp = client.post("/api/system/restart")
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "running_tasks_present"


# ---------------------------------------------------------------------------
# /api/system/rollback + update_status + update_log (ADR 0002 / PR-C)
# ---------------------------------------------------------------------------

def test_system_update_status_null_when_no_history(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """没有 update 历史时返 {status: null, rollback_target: null}。
    rollback_target 必须存在（即便 null），前端用它判断按钮显隐。"""
    from studio.services import updater
    monkeypatch.setattr(updater, "UPDATE_STATUS", tmp_path / ".update_status")
    monkeypatch.setattr(updater, "LAST_VERSION", tmp_path / ".last_version")
    resp = client.get("/api/system/update_status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] is None
    assert "rollback_target" in body
    assert body["rollback_target"] is None


def test_system_update_status_rollback_without_status(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """.update_status 不存在但 .last_version 存在（用户手动 git reset 后的场景）：
    应当返 status=null 但 rollback_target=sha，让 UI 显示回滚按钮。"""
    from studio.services import updater
    monkeypatch.setattr(updater, "last_status", lambda: None)
    monkeypatch.setattr(updater, "rollback_target", lambda: "cafebabe" * 5)

    resp = client.get("/api/system/update_status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] is None
    assert body["rollback_target"] == "cafebabe" * 5


def test_system_update_status_returns_recorded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """update_status 把 last_status() + rollback_target() 合并返回。"""
    from studio.services import updater
    fake = updater.UpdateStatus(
        status="failed", reason="git fetch: timeout", target="origin/master",
        from_commit="abc", to_commit="abc", started_at=1000.0, finished_at=1010.0,
        deps_changed=False, log_excerpt="[git fetch] FAILED",
    )
    monkeypatch.setattr(updater, "last_status", lambda: fake)
    monkeypatch.setattr(updater, "rollback_target", lambda: "deadbeef" * 5)

    resp = client.get("/api/system/update_status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["reason"] == "git fetch: timeout"
    assert body["rollback_target"] == "deadbeef" * 5


def test_system_update_log_empty(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services import updater
    monkeypatch.setattr(updater, "read_update_log", lambda: "")
    resp = client.get("/api/system/update_log")
    assert resp.status_code == 200
    assert resp.json() == {"content": ""}


def test_system_update_log_with_content(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.services import updater
    monkeypatch.setattr(updater, "read_update_log", lambda: "=== run ===\n[ok]\n")
    resp = client.get("/api/system/update_log")
    assert resp.json()["content"].startswith("=== run ===")


def test_system_rollback_409_when_no_target(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没 .last_version 或 commit 不在仓库时返 409。"""
    from studio.services import updater
    monkeypatch.setattr(updater, "current_version", lambda: updater.VersionInfo(
        version="0.0.0", commit="abc", commit_short="abc", commit_time_iso="",
        branch="master", tag=None, is_dirty=False,
    ))
    monkeypatch.setattr(updater, "request_rollback", lambda: None)

    resp = client.post("/api/system/rollback")
    assert resp.status_code == 409
    assert resp.json()["detail"]["error"] == "no_rollback_target"


def test_system_rollback_rejects_dirty(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rollback 共用 update 的 dirty precondition。"""
    from studio.services import updater
    monkeypatch.setattr(updater, "current_version", lambda: updater.VersionInfo(
        version="0.0.0", commit="abc", commit_short="abc", commit_time_iso="",
        branch="master", tag=None, is_dirty=True,
    ))

    resp = client.post("/api/system/rollback")
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "dirty_working_tree"


def test_system_rollback_rejects_running_task(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rollback 共用 update 的 running task precondition。"""
    from studio import db
    with db.connection_for(isolated_paths["db"]) as conn:
        tid = db.create_task(conn, name="炼丹", config_name="train", priority=0)
        db.update_task(conn, tid, status="running", task_type="train")

    resp = client.post("/api/system/rollback")
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "running_tasks_present"


def test_system_rollback_success_writes_flag(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正常路径：reads .last_version → request_update(target=<sha>) → 200。"""
    from studio.services import updater
    monkeypatch.setattr(updater, "current_version", lambda: updater.VersionInfo(
        version="0.0.0", commit="abc", commit_short="abc", commit_time_iso="",
        branch="master", tag=None, is_dirty=False,
    ))
    captured: dict = {}
    def _fake_rollback() -> str:
        captured["called"] = True
        return "feedbeef" * 5
    monkeypatch.setattr(updater, "request_rollback", _fake_rollback)
    monkeypatch.setattr("studio.api.routers.system._raise_sigint_after_response", lambda: None)

    resp = client.post("/api/system/rollback")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["target"] == "feedbeef" * 5
    assert captured["called"]


# ---------------------------------------------------------------------------
# /api/system/preflight (chunk 4)
# ---------------------------------------------------------------------------


def test_preflight_clean_no_running_no_diff(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """工作树干净 + 无运行任务 + requirements 无变化 → 3 ok + 1 ok（last_version 预览），blocking=False。"""
    from studio.services import updater
    monkeypatch.setattr(updater, "current_version", lambda: updater.VersionInfo(
        version="0.6.0", commit="abc123def456", commit_short="abc123de", commit_time_iso="",
        branch="master", tag="v0.6.0", is_dirty=False,
    ))
    monkeypatch.setattr(updater, "resolve_ref", lambda _ref: "deadbeef" * 5)
    monkeypatch.setattr(updater, "requirements_diff", lambda _ref: updater.RequirementsDiff())
    monkeypatch.setattr(updater, "target_has_self_update", lambda _ref: True)

    resp = client.get("/api/system/preflight?target=origin/master")
    assert resp.status_code == 200
    body = resp.json()
    assert body["blocking"] is False
    assert body["target_resolved"] == "deadbeef" * 5
    levels = [c["level"] for c in body["checks"]]
    assert "err" not in levels
    keys = [c["key"] for c in body["checks"]]
    assert keys == ["dirty", "running_tasks", "requirements_diff", "last_version"]


def test_preflight_dirty_blocks(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """is_dirty=True → dirty check level=err，blocking=True。"""
    from studio.services import updater
    monkeypatch.setattr(updater, "current_version", lambda: updater.VersionInfo(
        version="0.6.0", commit="x", commit_short="x", commit_time_iso="",
        branch="master", tag=None, is_dirty=True,
    ))
    monkeypatch.setattr(updater, "resolve_ref", lambda _ref: "x")
    monkeypatch.setattr(updater, "requirements_diff", lambda _ref: updater.RequirementsDiff())
    monkeypatch.setattr(updater, "target_has_self_update", lambda _ref: True)

    body = client.get("/api/system/preflight?target=origin/master").json()
    assert body["blocking"] is True
    dirty_check = next(c for c in body["checks"] if c["key"] == "dirty")
    assert dirty_check["level"] == "err"


def test_preflight_running_tasks_block(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有 running task → running_tasks check level=err，blocking=True，label 含任务名。"""
    from studio import db
    from studio.services import updater
    monkeypatch.setattr(updater, "current_version", lambda: updater.VersionInfo(
        version="0.6.0", commit="x", commit_short="x", commit_time_iso="",
        branch="master", tag=None, is_dirty=False,
    ))
    monkeypatch.setattr(updater, "resolve_ref", lambda _ref: "x")
    monkeypatch.setattr(updater, "requirements_diff", lambda _ref: updater.RequirementsDiff())
    monkeypatch.setattr(updater, "target_has_self_update", lambda _ref: True)

    # 写一条 running task（含所有 NOT NULL 字段）
    import time as _time
    with db.connection_for() as conn:
        conn.execute(
            "INSERT INTO tasks (name, config_name, status, task_type, created_at) VALUES (?, ?, ?, ?, ?)",
            ("training-XL-v3", "fake-config", "running", "train", int(_time.time())),
        )
        conn.commit()

    body = client.get("/api/system/preflight?target=origin/master").json()
    assert body["blocking"] is True
    rt = next(c for c in body["checks"] if c["key"] == "running_tasks")
    assert rt["level"] == "err"
    assert "training-XL-v3" in rt["label"]


def test_preflight_requirements_diff_warn(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """requirements diff 非空 → warn 级别，blocking 不受影响。label 含 +N/-N/~N 计数。"""
    from studio.services import updater
    monkeypatch.setattr(updater, "current_version", lambda: updater.VersionInfo(
        version="0.6.0", commit="x", commit_short="x", commit_time_iso="",
        branch="master", tag=None, is_dirty=False,
    ))
    monkeypatch.setattr(updater, "resolve_ref", lambda _ref: "x")
    monkeypatch.setattr(updater, "requirements_diff",
                       lambda _ref: updater.RequirementsDiff(
                           added=["newpkg1", "newpkg2"],
                           removed=["oldpkg"],
                           changed=[{"name": "torch", "from": "torch==2.0", "to": "torch==2.4"}],
                       ))
    monkeypatch.setattr(updater, "target_has_self_update", lambda _ref: True)

    body = client.get("/api/system/preflight?target=origin/master").json()
    assert body["blocking"] is False
    req = next(c for c in body["checks"] if c["key"] == "requirements_diff")
    assert req["level"] == "warn"
    assert "+2" in req["label"]
    assert "-1" in req["label"]
    assert "~1" in req["label"]
    assert body["requirements_diff"]["added"] == ["newpkg1", "newpkg2"]


def test_preflight_unresolved_target(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resolve_ref 返回 None（target ref 不存在）→ requirements 项 err，blocking。"""
    from studio.services import updater
    monkeypatch.setattr(updater, "current_version", lambda: updater.VersionInfo(
        version="0.6.0", commit="x", commit_short="x", commit_time_iso="",
        branch="master", tag=None, is_dirty=False,
    ))
    monkeypatch.setattr(updater, "resolve_ref", lambda _ref: None)

    body = client.get("/api/system/preflight?target=invalid").json()
    assert body["blocking"] is True
    assert body["target_resolved"] is None
    req = next(c for c in body["checks"] if c["key"] == "requirements_diff")
    assert req["level"] == "err"
    assert "invalid" in req["label"]


def test_preflight_target_missing_self_update_blocks(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目标 ref 早于 self-update feature → 加 err 行 + blocking=True；
    前端 confirm 按钮 disable，防止用户切到无 webui 救援能力的版本。"""
    from studio.services import updater
    monkeypatch.setattr(updater, "current_version", lambda: updater.VersionInfo(
        version="0.6.0", commit="x", commit_short="x", commit_time_iso="",
        branch="master", tag=None, is_dirty=False,
    ))
    monkeypatch.setattr(updater, "resolve_ref", lambda _ref: "deadbeef" * 5)
    monkeypatch.setattr(updater, "requirements_diff", lambda _ref: updater.RequirementsDiff())
    monkeypatch.setattr(updater, "target_has_self_update", lambda _ref: False)

    body = client.get("/api/system/preflight?target=ancient-commit").json()
    assert body["blocking"] is True
    compat = next(c for c in body["checks"] if c["key"] == "self_update_compat")
    assert compat["level"] == "err"
    assert "自更新" in compat["label"]


def test_preflight_target_with_self_update_passes(
    client: TestClient,
    isolated_paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目标 ref 带 self-update feature → 不加 self_update_compat 行，不影响 blocking。"""
    from studio.services import updater
    monkeypatch.setattr(updater, "current_version", lambda: updater.VersionInfo(
        version="0.6.0", commit="x", commit_short="x", commit_time_iso="",
        branch="master", tag=None, is_dirty=False,
    ))
    monkeypatch.setattr(updater, "resolve_ref", lambda _ref: "deadbeef" * 5)
    monkeypatch.setattr(updater, "requirements_diff", lambda _ref: updater.RequirementsDiff())
    monkeypatch.setattr(updater, "target_has_self_update", lambda _ref: True)

    body = client.get("/api/system/preflight?target=origin/master").json()
    assert body["blocking"] is False
    keys = [c["key"] for c in body["checks"]]
    assert "self_update_compat" not in keys
