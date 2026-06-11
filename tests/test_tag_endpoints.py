"""PP4 — /api/tagger/check + /tag + /captions/* HTTP。"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from studio import db, secrets, server
from studio.services.projects import jobs as project_jobs, projects, versions
from studio.services.tagging import llm as llm_tagger
from studio.services.tagging import base as tagger_mod


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dbfile = tmp_path / "studio.db"
    db.init_db(dbfile)
    monkeypatch.setattr(projects, "PROJECTS_DIR", tmp_path / "projects")
    monkeypatch.setattr(project_jobs, "JOB_LOGS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(server.db, "STUDIO_DB", dbfile)
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    return {"db": dbfile}


@pytest.fixture
def client(env) -> TestClient:
    server.app.state.supervisor = None
    return TestClient(server.app)


def _make(client: TestClient) -> tuple[int, int]:
    p = client.post("/api/projects", json={"title": "P"}).json()
    return p["id"], p["versions"][0]["id"]


def _seed_train(client: TestClient, pid: int, vid: int, folder: str, files: dict[str, str]) -> Path:
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    train = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "train"
    d = train / folder
    d.mkdir(parents=True, exist_ok=True)
    for name, tags in files.items():
        (d / name).write_bytes(b"x")
        (d / name).with_suffix(".txt").write_text(tags, encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# /api/tagger/{name}/check
# ---------------------------------------------------------------------------


def test_check_unknown_tagger(client: TestClient) -> None:
    r = client.get("/api/tagger/bogus/check")
    assert r.status_code == 400


def test_check_wd14(client: TestClient, env, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = MagicMock()
    fake.is_available.return_value = (True, "ready")
    fake.requires_service = False
    # server 内部 import 是 from .services.tagging.base import get_tagger，
    # 模块级 binding 在 server 命名空间，需要在那打补丁。
    # PR-6 commit 1：/api/tagger/{name}/check 搬到 api/routers/tagger.py
    from studio.api.routers import tagger as _tagger_router
    monkeypatch.setattr(_tagger_router, "get_tagger", lambda name: fake)
    r = client.get("/api/tagger/wd14/check").json()
    assert r == {"name": "wd14", "ok": True, "msg": "ready", "requires_service": False}


# ---------------------------------------------------------------------------
# /api/projects/{pid}/versions/{vid}/tag
# ---------------------------------------------------------------------------


def test_start_tag_creates_job(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/tag",
        json={"tagger": "wd14", "output_format": "txt"},
    )
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["kind"] == "tag"
    assert job["status"] == "pending"
    # 不再有 folders 入参；params 只剩 tagger / version_id / output_format
    import json as _json
    p_dict = _json.loads(job["params"])
    assert "folders" not in p_dict
    # ADR-0007 PR-5: tag job 不再自动推 stage；phase cursor 由用户推进


def test_start_tag_unknown_tagger_400(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/tag",
        json={"tagger": "x"},
    )
    assert r.status_code == 400


def test_start_tag_bad_format_400(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/tag",
        json={"tagger": "wd14", "output_format": "yaml"},
    )
    assert r.status_code == 400


def test_start_tag_with_wd14_overrides(client: TestClient) -> None:
    """传 wd14_overrides 时，端点应把它落进 params['wd14_overrides']。"""
    import json as _json
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/tag",
        json={
            "tagger": "wd14",
            "output_format": "txt",
            "wd14_overrides": {
                "threshold_general": 0.2,
                "blacklist_tags": ["solo"],
            },
        },
    )
    assert r.status_code == 200, r.text
    params = _json.loads(r.json()["params"])
    assert params["wd14_overrides"] == {
        "threshold_general": 0.2,
        "blacklist_tags": ["solo"],
    }


def test_start_tag_with_cltagger_overrides(client: TestClient) -> None:
    """传 cltagger_overrides 时，端点应把它落进 params。"""
    import json as _json
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/tag",
        json={
            "tagger": "cltagger",
            "output_format": "txt",
            "cltagger_overrides": {
                "threshold_general": 0.25,
                "threshold_character": 0.55,
                "add_rating_tag": True,
                "blacklist_tags": ["signature"],
            },
        },
    )
    assert r.status_code == 200, r.text
    params = _json.loads(r.json()["params"])
    assert params["cltagger_overrides"] == {
        "threshold_general": 0.25,
        "threshold_character": 0.55,
        "add_rating_tag": True,
        "blacklist_tags": ["signature"],
    }


def test_start_tag_with_llm_overrides(client: TestClient) -> None:
    """传 llm_overrides 时，端点应把它落进 params。`api_key` 不在 schema 里被忽略。"""
    import json as _json
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/tag",
        json={
            "tagger": "llm",
            "output_format": "json",
            "llm_overrides": {
                "current_preset": "joycaption",
                "endpoint": "responses",
                "model": "vision-model",
                "temperature": 0.1,
                "concurrency": 3,
                "requests_per_second": 1.5,
                "max_requests_per_minute": 30,
            },
        },
    )
    assert r.status_code == 200, r.text
    params = _json.loads(r.json()["params"])
    assert params["llm_overrides"] == {
        "current_preset": "joycaption",
        "endpoint": "responses",
        "model": "vision-model",
        "temperature": 0.1,
        "concurrency": 3,
        "requests_per_second": 1.5,
        "max_requests_per_minute": 30,
    }


def test_refresh_llm_models_saves_masked_config(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def _fake_fetch(base_url: str, api_key: str, *, timeout: int, session=None):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        captured["timeout"] = timeout
        return ["vision-a", "vision-b"]

    monkeypatch.setattr(llm_tagger, "fetch_openai_compatible_models", _fake_fetch)
    r = client.post(
        "/api/llm-tagger/models/refresh",
        json={
            "preset_id": "style_json",
            "base_url": "http://x/v1",
            "api_key": "secret",
            "timeout": 12,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["items"] == ["vision-a", "vision-b"]
    assert body["preset_id"] == "style_json"
    style_masked = next(
        p for p in body["secrets"]["llm_tagger"]["presets"] if p["id"] == "style_json"
    )
    assert style_masked["api_key"] == secrets.MASK
    assert style_masked["model"] == "vision-a"
    assert style_masked["model_ids"] == ["vision-a", "vision-b"]
    assert captured == {
        "base_url": "http://x/v1",
        "api_key": "secret",
        "timeout": 12,
    }
    style_loaded = next(p for p in secrets.load().llm_tagger.presets if p.id == "style_json")
    assert style_loaded.api_key == "secret"


def test_llm_connection_test_uses_masked_saved_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets.update(
        {
            "llm_tagger": {
                "current_preset": "style_json",
                "presets": [
                    {
                        "id": "style_json",
                        "base_url": "http://saved/v1",
                        "api_key": "saved-secret",
                        "model": "saved-model",
                        "endpoint": "chat_completions",
                        "timeout": 22,
                    }
                ],
            }
        }
    )
    captured: dict = {}

    def _fake_test(base_url: str, api_key: str, model: str, **kwargs):
        captured.update(
            {
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
                **kwargs,
            }
        )
        return {
            "ok": True,
            "endpoint": kwargs["endpoint"],
            "endpoint_url": "http://saved/v1/chat/completions",
            "model": model,
            "elapsed_ms": 12,
            "status_code": 200,
            "response_preview": "ok",
            "error": "",
            "request_shape": "chat_completions_text",
        }

    monkeypatch.setattr(llm_tagger, "test_openai_compatible_connection", _fake_test)
    r = client.post(
        "/api/llm-tagger/test",
        json={"api_key": secrets.MASK, "model": "draft-model", "timeout": 9},
    )

    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True
    assert captured["base_url"] == "http://saved/v1"
    assert captured["api_key"] == "saved-secret"
    assert captured["model"] == "draft-model"
    assert captured["timeout"] == 9
    # 测试只是 dry-run，不应改 secrets 里的 model
    style_loaded = next(p for p in secrets.load().llm_tagger.presets if p.id == "style_json")
    assert style_loaded.model == "saved-model"


def test_start_tag_drops_empty_wd14_overrides(client: TestClient) -> None:
    """全部字段都是 None 时不要写空 dict 进 params。"""
    import json as _json
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/tag",
        json={
            "tagger": "wd14",
            "output_format": "txt",
            "wd14_overrides": {
                "threshold_general": None,
                "threshold_character": None,
            },
        },
    )
    params = _json.loads(r.json()["params"])
    assert "wd14_overrides" not in params


def test_start_tag_on_existing_skip(client: TestClient) -> None:
    """on_existing=skip 时端点把它落进 params。"""
    import json as _json
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/tag",
        json={"tagger": "wd14", "output_format": "txt", "on_existing": "skip"},
    )
    assert r.status_code == 200, r.text
    params = _json.loads(r.json()["params"])
    assert params["on_existing"] == "skip"


def test_start_tag_on_existing_append(client: TestClient) -> None:
    import json as _json
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/tag",
        json={"tagger": "wd14", "output_format": "txt", "on_existing": "append"},
    )
    assert r.status_code == 200, r.text
    params = _json.loads(r.json()["params"])
    assert params["on_existing"] == "append"


def test_start_tag_on_existing_overwrite_not_persisted(client: TestClient) -> None:
    """默认 overwrite 不写入 params（worker 端默认即 overwrite，减小 payload）。"""
    import json as _json
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/tag",
        json={"tagger": "wd14", "output_format": "txt", "on_existing": "overwrite"},
    )
    assert r.status_code == 200, r.text
    params = _json.loads(r.json()["params"])
    assert "on_existing" not in params


def test_start_tag_on_existing_bad_value_400(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/tag",
        json={"tagger": "wd14", "output_format": "txt", "on_existing": "bogus"},
    )
    assert r.status_code == 400


def test_start_tag_ignores_overrides_for_joycaption(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tagger != wd14 时即便传了 overrides 也不应入 params。"""
    import json as _json
    fake = MagicMock()
    fake.is_available.return_value = (True, "ok")
    fake.requires_service = True
    # PR-6 commit 1：/api/tagger/{name}/check 搬到 api/routers/tagger.py
    from studio.api.routers import tagger as _tagger_router
    monkeypatch.setattr(_tagger_router, "get_tagger", lambda name: fake)
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/tag",
        json={
            "tagger": "joycaption",
            "wd14_overrides": {"threshold_general": 0.1},
        },
    )
    params = _json.loads(r.json()["params"])
    assert "wd14_overrides" not in params


# ---------------------------------------------------------------------------
# /captions
# ---------------------------------------------------------------------------


def test_list_captions(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_a", {"1.png": "a, b", "2.png": "x"})
    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/captions?folder=5_a"
    ).json()
    names = sorted(i["name"] for i in r["items"])
    assert names == ["1.png", "2.png"]
    by_name = {i["name"]: i for i in r["items"]}
    assert by_name["1.png"]["tag_count"] == 2
    assert by_name["1.png"]["folder"] == "5_a"


def test_list_captions_all_folders(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "1_data", {"a.png": "x"})
    _seed_train(client, pid, vid, "5_face", {"b.png": "y, z"})
    r = client.get(f"/api/projects/{pid}/versions/{vid}/captions").json()
    assert r["folder"] is None
    by_name = {i["name"]: i for i in r["items"]}
    assert by_name["a.png"]["folder"] == "1_data"
    assert by_name["b.png"]["folder"] == "5_face"
    assert by_name["b.png"]["tag_count"] == 2


def test_get_and_put_caption(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_a", {"1.png": "a, b"})
    r = client.get(f"/api/projects/{pid}/versions/{vid}/captions/5_a/1.png").json()
    assert r["tags"] == ["a", "b"]
    r = client.put(
        f"/api/projects/{pid}/versions/{vid}/captions/5_a/1.png",
        json={"tags": ["x", "y"]},
    ).json()
    assert r["tags"] == ["x", "y"]


def test_get_caption_404(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.get(f"/api/projects/{pid}/versions/{vid}/captions/5_a/ghost.png")
    assert r.status_code == 404


def test_batch_add_remove_replace(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "5_a", {"1.png": "a, b", "2.png": "a, c"})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/captions/batch",
        json={
            "op": "add",
            "scope": {"kind": "folder", "name": "5_a"},
            "tags": ["new"],
        },
    ).json()
    assert r == {"op": "add", "affected": 2}

    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/captions/batch",
        json={
            "op": "replace",
            "scope": {"kind": "all"},
            "old": "a",
            "new": "AA",
        },
    ).json()
    assert r["affected"] == 2

    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/captions/batch",
        json={
            "op": "stats",
            "scope": {"kind": "folder", "name": "5_a"},
            "top": 5,
        },
    ).json()
    items = dict(r["items"])
    assert items.get("AA") == 2
    assert items.get("new") == 2


def test_batch_files_cross_folder(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "1_data", {"a.png": "x"})
    _seed_train(client, pid, vid, "5_face", {"b.png": "y"})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/captions/batch",
        json={
            "op": "add",
            "scope": {
                "kind": "files",
                "items": [
                    {"folder": "1_data", "name": "a.png"},
                    {"folder": "5_face", "name": "b.png"},
                ],
            },
            "tags": ["mark"],
        },
    ).json()
    assert r == {"op": "add", "affected": 2}


def test_list_captions_full_includes_tags(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "1_data", {"a.png": "x, y"})
    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/captions?full=1"
    ).json()
    by_name = {i["name"]: i for i in r["items"]}
    assert by_name["a.png"]["tags"] == ["x", "y"]
    assert by_name["a.png"]["format"] == "txt"


def test_commit_writes_and_snapshots(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "1_data", {"a.png": "old"})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/captions/commit",
        json={"items": [{"folder": "1_data", "name": "a.png", "tags": ["NEW", "TAG"]}]},
    ).json()
    assert r["written"] == 1
    assert "snapshot" in r and r["snapshot"]["id"]
    # caption 实际写入
    cap = client.get(
        f"/api/projects/{pid}/versions/{vid}/captions/1_data/a.png"
    ).json()
    assert cap["tags"] == ["NEW", "TAG"]
    # 快照能 restore 回 old
    sid = r["snapshot"]["id"]
    client.post(
        f"/api/projects/{pid}/versions/{vid}/captions/snapshots/{sid}/restore"
    )
    cap = client.get(
        f"/api/projects/{pid}/versions/{vid}/captions/1_data/a.png"
    ).json()
    assert cap["tags"] == ["old"]


def test_commit_skips_path_traversal(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "1_data", {"a.png": "x"})
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/captions/commit",
        json={
            "items": [
                {"folder": "../evil", "name": "a.png", "tags": ["x"]},
                {"folder": "1_data", "name": "../evil.png", "tags": ["x"]},
                {"folder": "1_data", "name": "a.png", "tags": ["ok"]},
            ]
        },
    ).json()
    assert r["written"] == 1
    assert len(r["skipped"]) == 2


def test_caption_snapshot_create_list_restore(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "1_data", {"a.png": "old"})
    # 创建快照
    s = client.post(
        f"/api/projects/{pid}/versions/{vid}/captions/snapshot"
    ).json()
    sid = s["id"]
    assert s["file_count"] == 1
    # 改 caption 模拟编辑
    client.put(
        f"/api/projects/{pid}/versions/{vid}/captions/1_data/a.png",
        json={"tags": ["new"]},
    )
    # list
    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/captions/snapshots"
    ).json()
    assert any(it["id"] == sid for it in r["items"])
    # restore
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/captions/snapshots/{sid}/restore"
    ).json()
    assert r["written"] == 1
    cap = client.get(
        f"/api/projects/{pid}/versions/{vid}/captions/1_data/a.png"
    ).json()
    assert cap["tags"] == ["old"]
    # delete
    client.delete(
        f"/api/projects/{pid}/versions/{vid}/captions/snapshots/{sid}"
    )
    r = client.get(
        f"/api/projects/{pid}/versions/{vid}/captions/snapshots"
    ).json()
    assert all(it["id"] != sid for it in r["items"])


def test_version_stats_includes_tagged_count(client: TestClient) -> None:
    pid, vid = _make(client)
    _seed_train(client, pid, vid, "1_data", {"a.png": "x"})
    # 加一张没 caption 的图
    with db.connection_for() as conn:
        proj = projects.get_project(conn, pid)
        v = versions.get_version(conn, vid)
    train = versions.version_dir(proj["id"], proj["slug"], v["label"]) / "train" / "1_data"
    (train / "b.png").write_bytes(b"x")
    detail = client.get(f"/api/projects/{pid}/versions/{vid}").json()
    stats = detail.get("stats")
    assert stats is not None
    assert stats["train_image_count"] == 2
    assert stats["tagged_image_count"] == 1


def test_batch_replace_requires_old_new(client: TestClient) -> None:
    pid, vid = _make(client)
    r = client.post(
        f"/api/projects/{pid}/versions/{vid}/captions/batch",
        json={"op": "replace", "scope": {"kind": "all"}},
    )
    assert r.status_code == 400
