"""PP0 — secrets.json 读写、deep-merge、敏感字段掩码。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studio import secrets, server


@pytest.fixture
def secrets_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """所有读写都落到 tmp_path/secrets.json。"""
    sf = tmp_path / "secrets.json"
    monkeypatch.setattr(secrets, "SECRETS_FILE", sf)
    return sf


@pytest.fixture
def client(secrets_file: Path) -> TestClient:  # noqa: ARG001 (fixture chains the patch)
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------


def test_defaults_when_file_missing(secrets_file: Path) -> None:
    assert not secrets_file.exists()
    s = secrets.load()
    assert s.gelbooru.user_id == ""
    assert s.gelbooru.api_key == ""
    assert s.wd14.threshold_general == pytest.approx(0.35)
    assert s.joycaption.base_url.startswith("http://")


def test_load_corrupt_json_returns_defaults(secrets_file: Path) -> None:
    secrets_file.write_text("{not valid json", encoding="utf-8")
    # 不应抛错；返回默认实例
    s = secrets.load()
    assert s.gelbooru.user_id == ""


def test_wd14_defaults_include_candidate_list(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.wd14.model_id in s.wd14.model_ids
    assert set(secrets.DEFAULT_WD14_MODELS).issubset(set(s.wd14.model_ids))


def test_wd14_legacy_file_without_model_ids_gets_defaults(
    secrets_file: Path,
) -> None:
    """旧 secrets.json 没有 model_ids 字段时，加载后用默认列表填充并把
    当前 model_id 也保证在内。"""
    secrets_file.write_text(
        json.dumps({"wd14": {"model_id": "Custom/my-tagger"}}),
        encoding="utf-8",
    )
    s = secrets.load()
    assert "Custom/my-tagger" in s.wd14.model_ids
    # 默认 4 项也仍在列表里
    for m in secrets.DEFAULT_WD14_MODELS:
        assert m in s.wd14.model_ids


def test_wd14_empty_model_ids_falls_back_to_defaults(
    secrets_file: Path,
) -> None:
    secrets.update({"wd14": {"model_ids": []}})
    s = secrets.load()
    assert list(s.wd14.model_ids) == list(secrets.DEFAULT_WD14_MODELS)


def test_wd14_cannot_drop_current_model_id(secrets_file: Path) -> None:
    """删除候选时如果删掉当前 model_id，validator 自动加回去。"""
    secrets.update(
        {"wd14": {"model_id": "SmilingWolf/wd-vit-tagger-v3"}}
    )
    # 用户提交一个不含当前 model_id 的候选列表
    s = secrets.update({"wd14": {"model_ids": ["A/m1", "B/m2"]}})
    assert s.wd14.model_id == "SmilingWolf/wd-vit-tagger-v3"
    assert s.wd14.model_id in s.wd14.model_ids


def test_wd14_user_can_replace_current_then_drop(secrets_file: Path) -> None:
    """先切到另一个再删，才能真正从候选中移除原 model_id。"""
    s = secrets.update({"wd14": {"model_id": "A/m1"}})
    assert "A/m1" in s.wd14.model_ids
    # 切到一个新 id（model_validator 会把它加进列表）
    s = secrets.update({"wd14": {"model_id": "B/m2"}})
    assert s.wd14.model_id == "B/m2"
    # 现在 patch 列表把 A/m1 去掉
    s = secrets.update({"wd14": {"model_ids": [m for m in s.wd14.model_ids if m != "A/m1"]}})
    assert "A/m1" not in s.wd14.model_ids
    assert "B/m2" in s.wd14.model_ids


# ---------------------------------------------------------------------------
# update / mask round-trip
# ---------------------------------------------------------------------------


def test_update_writes_file(secrets_file: Path) -> None:
    secrets.update({"gelbooru": {"user_id": "alice", "api_key": "k1"}})
    on_disk = json.loads(secrets_file.read_text(encoding="utf-8"))
    assert on_disk["gelbooru"]["user_id"] == "alice"
    assert on_disk["gelbooru"]["api_key"] == "k1"


def test_update_deep_merge_preserves_other_sections(secrets_file: Path) -> None:
    secrets.update({"huggingface": {"token": "hf_x"}})
    secrets.update({"gelbooru": {"user_id": "bob"}})
    s = secrets.load()
    assert s.huggingface.token == "hf_x"
    assert s.gelbooru.user_id == "bob"


def test_update_mask_keeps_existing_value(secrets_file: Path) -> None:
    secrets.update({"gelbooru": {"api_key": "real-key"}})
    # 模拟前端把 "***" 回传：表示「保持原值」
    secrets.update({"gelbooru": {"api_key": secrets.MASK, "user_id": "bob"}})
    s = secrets.load()
    assert s.gelbooru.api_key == "real-key"
    assert s.gelbooru.user_id == "bob"


def test_to_masked_dict_replaces_sensitive(secrets_file: Path) -> None:
    secrets.update(
        {
            "gelbooru": {"user_id": "alice", "api_key": "secret"},
            "huggingface": {"token": "hf_secret"},
        }
    )
    masked = secrets.to_masked_dict(secrets.load())
    assert masked["gelbooru"]["user_id"] == "alice"  # 非敏感字段保留
    assert masked["gelbooru"]["api_key"] == secrets.MASK
    assert masked["huggingface"]["token"] == secrets.MASK


def test_to_masked_dict_keeps_empty_sensitive_empty(secrets_file: Path) -> None:
    """没有值的敏感字段不应该显示为 "***"，否则前端无法判断「真的为空」。"""
    masked = secrets.to_masked_dict(secrets.load())
    assert masked["gelbooru"]["api_key"] == ""
    assert masked["huggingface"]["token"] == ""


# ---------------------------------------------------------------------------
# get() 点路径
# ---------------------------------------------------------------------------


def test_get_dot_path(secrets_file: Path) -> None:
    secrets.update({"wd14": {"threshold_general": 0.5}})
    assert secrets.get("wd14.threshold_general") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_get_secrets_endpoint(client: TestClient) -> None:
    resp = client.get("/api/secrets")
    assert resp.status_code == 200
    body = resp.json()
    assert "gelbooru" in body
    assert "wd14" in body
    assert body["gelbooru"]["api_key"] == ""  # 默认为空，不掩码


def test_put_secrets_round_trip(client: TestClient, secrets_file: Path) -> None:
    resp = client.put(
        "/api/secrets",
        json={"gelbooru": {"user_id": "alice", "api_key": "k"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["gelbooru"]["user_id"] == "alice"
    assert body["gelbooru"]["api_key"] == secrets.MASK  # GET 形式：掩码

    # 真实值已落盘
    on_disk = json.loads(secrets_file.read_text(encoding="utf-8"))
    assert on_disk["gelbooru"]["api_key"] == "k"


def test_put_secrets_mask_keeps_value(client: TestClient) -> None:
    client.put("/api/secrets", json={"gelbooru": {"api_key": "first"}})
    # 客户端「不改 api_key 只改 user_id」时回传 MASK
    client.put(
        "/api/secrets",
        json={"gelbooru": {"api_key": secrets.MASK, "user_id": "alice"}},
    )
    s = secrets.load()
    assert s.gelbooru.api_key == "first"
    assert s.gelbooru.user_id == "alice"


def test_has_gelbooru_credentials(secrets_file: Path) -> None:
    assert secrets.has_gelbooru_credentials() is False
    secrets.update({"gelbooru": {"user_id": "u", "api_key": "k"}})
    assert secrets.has_gelbooru_credentials() is True
