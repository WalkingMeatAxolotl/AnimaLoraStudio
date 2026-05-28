"""PP6.2 — /api/projects/{pid}/versions/{vid}/config/* HTTP。"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import db, server
from studio.services.projects import projects, versions
from studio.schema import TrainingConfig


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir()
    from studio import presets_io
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", presets_dir)
    return {"db": dbfile, "presets": presets_dir}


@pytest.fixture
def client(env) -> TestClient:
    server.app.state.supervisor = None
    return TestClient(server.app)


def _make(client: TestClient) -> tuple[int, int]:
    p = client.post("/api/projects", json={"title": "P"}).json()
    return p["id"], p["versions"][0]["id"]


def _seed_preset(env, name: str, **overrides) -> None:
    from studio import presets_io
    base = TrainingConfig().model_dump()
    base.update(overrides)
    presets_io.write_preset(name, base)


# ---------------------------------------------------------------------------
# GET /config
# ---------------------------------------------------------------------------


def test_get_config_returns_no_config_initially(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/{vid}/config")
    assert r.status_code == 200
    body = r.json()
    assert body["has_config"] is False
    assert body["config"] is None
    assert "data_dir" in body["project_specific_fields"]
    assert "output_name" in body["project_specific_fields"]


def test_get_config_no_config_returns_project_specific_defaults(
    client: TestClient,
) -> None:
    """0.8.2 hotfix：has_config=False 时返回 project_specific_defaults，新建预设
    预览表单用它展示「fork 后会得到的值」。"""
    pid, vid = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/{vid}/config")
    body = r.json()
    assert body["has_config"] is False
    defaults = body.get("project_specific_defaults")
    assert defaults is not None, "缺 project_specific_defaults"
    # 项目特定路径填好了（reg 不存在时 reg_data_dir 是 None）
    assert defaults["data_dir"].endswith("train")
    assert defaults["output_dir"].endswith("output")
    assert defaults["output_name"], "output_name 不能为空"
    assert defaults["reg_data_dir"] is None  # reg 还没 build
    assert defaults["resume_lora"] is None
    # 全局模型路径也填好了（绝对路径，from secrets.models.root）
    assert defaults["transformer_path"], "transformer_path 应填默认模型路径"
    assert defaults["vae_path"], "vae_path 应填"
    assert defaults["text_encoder_path"]
    assert defaults["t5_tokenizer_path"]


def test_get_config_defaults_picks_up_reg_when_meta_exists(
    client: TestClient, env
) -> None:
    """reg/meta.json 存在时，project_specific_defaults.reg_data_dir 应指向 reg 目录。"""
    pid, vid = _make(client)
    # 模拟跑过 reg build → reg/meta.json 写好
    with db.connection_for(env["db"]) as conn:
        p = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    assert p is not None and v is not None
    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    (vdir / "reg").mkdir(parents=True, exist_ok=True)
    (vdir / "reg" / "meta.json").write_text('{"target_count": 50}', encoding="utf-8")

    r = client.get(f"/api/projects/{pid}/versions/{vid}/config")
    defaults = r.json()["project_specific_defaults"]
    assert defaults["reg_data_dir"] == str(vdir / "reg")


def test_get_config_returns_defaults_when_has_config(
    client: TestClient, env
) -> None:
    """has_config=True 时 project_specific_defaults 也要返回 —— 用户在已 fork
    过预设的 version 上点「+ 新建预设」，前端预览仍需要这个 hint 显示项目路径。"""
    pid, vid = _make(client)
    _seed_preset(env, "tpl", lora_rank=64)
    # 先 fork → has_config=True
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    # 加 reg meta，模拟 fork 后跑了 reg build 的常见场景
    with db.connection_for(env["db"]) as conn:
        p = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    assert p is not None and v is not None
    vdir = versions.version_dir(p["id"], p["slug"], v["label"])
    (vdir / "reg" / "meta.json").write_text('{"target_count": 50}', encoding="utf-8")

    r = client.get(f"/api/projects/{pid}/versions/{vid}/config")
    body = r.json()
    assert body["has_config"] is True
    defaults = body.get("project_specific_defaults")
    assert defaults is not None, "has_config=True 时也应返回 project_specific_defaults"
    assert defaults["reg_data_dir"] == str(vdir / "reg")
    assert defaults["transformer_path"]  # 模型路径也在


def test_get_config_for_unknown_version_404(client: TestClient) -> None:
    pid, _ = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/9999/config")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /config/from_preset
# ---------------------------------------------------------------------------


def test_fork_preset_writes_version_config(client: TestClient, env) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl", lora_rank=64)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["from_preset"] == "tpl"
    cfg = body["config"]
    assert cfg["lora_rank"] == 64
    # 项目特定字段被强制覆盖
    assert cfg["data_dir"].endswith("train")
    # version.config_name 同步成 informational 来源
    v = client.get(f"/api/projects/{pid}/versions/{vid}").json()
    assert v["config_name"] == "tpl"


def test_fork_preset_unknown_404(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "nope"},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# PUT /config
# ---------------------------------------------------------------------------


def test_put_config_keeps_user_values(client: TestClient, env) -> None:
    """PP10.4：PUT 端点不强制覆盖项目特定字段，让用户改 resume_lora /
    output_name 等。fork preset 时仍预填项目路径（在 from_preset 端点测覆盖）。"""
    pid, vid = _make(client)
    _seed_preset(env, "tpl")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    cfg = client.get(f"/api/projects/{pid}/versions/{vid}/config").json()["config"]
    cfg["output_name"] = "custom_lora"
    cfg["resume_lora"] = "/tmp/some/lora.safetensors"
    cfg["lora_rank"] = 96
    r = client.put(f"/api/projects/{pid}/versions/{vid}/config", json=cfg)
    assert r.status_code == 200
    body = r.json()
    # 用户改的项目特定字段被保留（PP10.4）
    assert body["config"]["output_name"] == "custom_lora"
    assert body["config"]["resume_lora"] == "/tmp/some/lora.safetensors"
    # 用户改的非项目字段也保留
    assert body["config"]["lora_rank"] == 96


def test_fork_preset_still_forces_project_overrides(
    client: TestClient, env
) -> None:
    """PP10.4：换预设时项目特定字段仍然被预填成项目路径（force=True 路径未动）。"""
    pid, vid = _make(client)
    # preset 故意带错误路径
    _seed_preset(env, "tpl", data_dir="/wrong/path", output_name="wrong_name")
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    assert r.status_code == 200
    cfg = client.get(f"/api/projects/{pid}/versions/{vid}/config").json()["config"]
    assert cfg["data_dir"].endswith("train")
    assert cfg["output_name"] != "wrong_name"


def test_put_config_invalid_data_400(client: TestClient, env) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    cfg = client.get(f"/api/projects/{pid}/versions/{vid}/config").json()["config"]
    cfg["lora_rank"] = 0  # ge=4 → 下限越界（lora_rank 故意没设上限，LoKr 全维度走大数）
    r = client.put(f"/api/projects/{pid}/versions/{vid}/config", json=cfg)
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# POST /config/save_as_preset
# ---------------------------------------------------------------------------


def test_save_as_preset_clears_project_fields(client: TestClient, env) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl", lora_rank=128)
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/save_as_preset",
        json={"name": "my-tuned"},
    )
    assert r.status_code == 200, r.text
    saved = r.json()
    assert saved["saved_preset"] == "my-tuned"
    assert saved["config"]["data_dir"] == "./dataset"  # schema 默认
    assert saved["config"]["lora_rank"] == 128

    # 全局 preset 池里出现 my-tuned
    from studio import presets_io
    presets = {p["name"] for p in presets_io.list_presets()}
    assert "my-tuned" in presets


def test_save_as_preset_existing_409(client: TestClient, env) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/save_as_preset",
        json={"name": "tpl"},  # 同名 + 不带 overwrite
    )
    assert r.status_code == 400  # presets_io.PresetError("已存在") → _err_code 400
    # overwrite=True 应允许
    r2 = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/save_as_preset",
        json={"name": "tpl", "overwrite": True},
    )
    assert r2.status_code == 200


def test_save_as_preset_without_config_400(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/config/save_as_preset",
        json={"name": "x"},
    )
    assert r.status_code == 400  # version 还没 fork 过任何 preset


# ---------------------------------------------------------------------------
# POST /queue (PP6.3 入队)
# ---------------------------------------------------------------------------


def test_enqueue_without_config_400(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(f"/api/projects/{pid}/versions/{vid}/queue")
    assert r.status_code == 400


def test_enqueue_creates_task_with_ids_and_config_path(
    client: TestClient, env
) -> None:
    pid, vid = _make(client)
    _seed_preset(env, "tpl", lora_rank=64)
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    r = client.post(f"/api/projects/{pid}/versions/{vid}/queue")
    assert r.status_code == 200, r.text
    task = r.json()
    assert task["status"] == "pending"
    assert task["project_id"] == pid
    assert task["version_id"] == vid
    assert task["config_path"] and task["config_path"].endswith("config.yaml")
    # ADR-0007 PR-5: version.status 由 supervisor 在 spawn 时推 training；enqueue 时仍 preparing


def test_enqueue_rejects_active_task(client: TestClient, env) -> None:
    """同 version 已有 pending/running task → 409。"""
    pid, vid = _make(client)
    _seed_preset(env, "tpl")
    client.post(
        f"/api/projects/{pid}/versions/{vid}/config/from_preset",
        json={"name": "tpl"},
    )
    r1 = client.post(f"/api/projects/{pid}/versions/{vid}/queue")
    assert r1.status_code == 200
    r2 = client.post(f"/api/projects/{pid}/versions/{vid}/queue")
    assert r2.status_code == 409
