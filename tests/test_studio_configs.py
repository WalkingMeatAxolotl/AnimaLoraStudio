"""Schema + /api/presets HTTP（PP0 之前是 /api/configs，保留 308 redirect）。

PP0 把 IO 单元测试拆到 test_presets_io.py；这里专注 HTTP 表面 + schema + 兼容。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from studio import presets_io, server
from studio.schema import TrainingConfig


@pytest.fixture
def presets_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pdir = tmp_path / "presets"
    pdir.mkdir()
    monkeypatch.setattr(presets_io, "USER_PRESETS_DIR", pdir)
    return pdir


@pytest.fixture
def client(presets_dir: Path) -> TestClient:  # noqa: ARG001
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_schema_is_complete() -> None:
    fields = TrainingConfig.model_fields
    for name in (
        "transformer_path", "data_dir", "lora_type", "lora_rank", "epochs",
        "optimizer_type", "prodigy_d_coef", "prodigy_safeguard_warmup",
        "sample_prompt", "sample_prompts", "no_monitor",
    ):
        assert name in fields, f"missing: {name}"
    assert "wandb_enabled" not in fields


def test_schema_endpoint_returns_groups(client: TestClient) -> None:
    resp = client.get("/api/schema")
    assert resp.status_code == 200
    body = resp.json()
    assert "schema" in body
    assert "properties" in body["schema"]
    assert {g["key"] for g in body["groups"]} >= {
        "model", "dataset", "lora", "training", "output", "sample", "monitor"
    }


def test_schema_carries_ui_metadata(client: TestClient) -> None:
    resp = client.get("/api/schema")
    props = resp.json()["schema"]["properties"]
    assert props["transformer_path"]["group"] == "model"
    assert props["transformer_path"]["control"] == "path"
    assert "show_when" in props["prodigy_d_coef"]
    assert "wandb_enabled" not in props


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(Exception):
        TrainingConfig.model_validate({"learning_ratee": 1e-4})


# ---------------------------------------------------------------------------
# /api/presets HTTP
# ---------------------------------------------------------------------------


def _payload() -> dict:
    return TrainingConfig().model_dump(mode="python")


def test_api_lifecycle(client: TestClient, presets_dir: Path) -> None:
    payload = _payload()
    payload["epochs"] = 7

    assert client.get("/api/presets").json()["items"] == []

    resp = client.put("/api/presets/myrun", json=payload)
    assert resp.status_code == 200, resp.text

    got = client.get("/api/presets/myrun").json()
    assert got["epochs"] == 7

    items = client.get("/api/presets").json()["items"]
    assert any(i["name"] == "myrun" for i in items)

    resp = client.post("/api/presets/myrun/duplicate", json={"new_name": "myrun_copy"})
    assert resp.status_code == 200
    assert client.get("/api/presets/myrun_copy").json()["epochs"] == 7

    assert client.delete("/api/presets/myrun").status_code == 200
    assert client.get("/api/presets/myrun").status_code == 404


def test_api_put_rejects_unknown_field(client: TestClient) -> None:
    bad = _payload()
    bad["nonexistent_field"] = 123
    resp = client.put("/api/presets/bad", json=bad)
    assert resp.status_code == 422


def test_api_get_invalid_name(client: TestClient) -> None:
    resp = client.get("/api/presets/has..dot")
    assert resp.status_code in (400, 422)


def test_api_duplicate_conflict(client: TestClient) -> None:
    payload = _payload()
    client.put("/api/presets/x", json=payload)
    client.put("/api/presets/y", json=payload)
    resp = client.post("/api/presets/x/duplicate", json={"new_name": "y"})
    assert resp.status_code == 400


def test_api_delete_missing(client: TestClient) -> None:
    resp = client.delete("/api/presets/ghost")
    assert resp.status_code == 404


def test_yaml_on_disk_is_human_readable(client: TestClient, presets_dir: Path) -> None:
    client.put("/api/presets/readable", json=_payload())
    text = (presets_dir / "readable.yaml").read_text(encoding="utf-8")
    assert "transformer_path:" in text
    assert not text.startswith("{")
    parsed = yaml.safe_load(text)
    assert parsed["lora_type"] == "lokr"


# ---------------------------------------------------------------------------
# /api/configs/* 兼容（308 redirect → /api/presets/*）
# ---------------------------------------------------------------------------


def test_legacy_configs_endpoint_redirects(client: TestClient) -> None:
    """旧 /api/configs 端点 308 跳转到 /api/presets，外部脚本不应直接断裂。"""
    # follow_redirects=False 让我们直接看到 308 + Location
    resp = client.get("/api/configs", follow_redirects=False)
    assert resp.status_code == 308
    assert resp.headers["location"].endswith("/api/presets")

    resp = client.get("/api/configs/foo", follow_redirects=False)
    assert resp.status_code == 308
    assert resp.headers["location"].endswith("/api/presets/foo")
