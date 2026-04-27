"""配置 schema + CRUD 端点测试（P2-A）。

直接测 `studio.configs_io`（纯函数）和 server.py 的 /api/schema, /api/configs/*。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from studio import configs_io, server
from studio.schema import TrainingConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def configs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """所有 CRUD 落到 tmp_path/configs，避免污染 studio_data/configs。"""
    cfgdir = tmp_path / "configs"
    cfgdir.mkdir()
    monkeypatch.setattr(configs_io, "USER_CONFIGS_DIR", cfgdir)
    return cfgdir


@pytest.fixture
def client(configs_dir: Path) -> TestClient:
    return TestClient(server.app)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_schema_is_complete() -> None:
    """TrainingConfig 字段足以表达 train_template.yaml 全部已知字段。"""
    fields = TrainingConfig.model_fields
    # 抽样几个关键字段都必须在
    for name in (
        "transformer_path", "data_dir", "lora_type", "lora_rank", "epochs",
        "optimizer_type", "prodigy_d_coef", "prodigy_safeguard_warmup",
        "sample_prompt", "sample_prompts", "no_monitor",
    ):
        assert name in fields, f"missing: {name}"


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
    """前端需要 group/control，检查至少一个字段把它们带出来了。"""
    resp = client.get("/api/schema")
    props = resp.json()["schema"]["properties"]
    assert props["transformer_path"]["group"] == "model"
    assert props["transformer_path"]["control"] == "path"
    # show_when 应当存在
    assert "show_when" in props["prodigy_d_coef"]


def test_extra_fields_are_forbidden() -> None:
    """未知字段（拼写错误）必须直接报错，不允许悄悄落盘。"""
    with pytest.raises(Exception):
        TrainingConfig.model_validate({"learning_ratee": 1e-4})


# ---------------------------------------------------------------------------
# configs_io 单元
# ---------------------------------------------------------------------------


def _minimal_payload() -> dict:
    return TrainingConfig().model_dump(mode="python")


def test_write_then_read_roundtrip(configs_dir: Path) -> None:
    payload = _minimal_payload()
    payload["lora_rank"] = 64
    configs_io.write_config("alpha", payload)
    assert (configs_dir / "alpha.yaml").exists()
    got = configs_io.read_config("alpha")
    assert got["lora_rank"] == 64


def test_write_invalid_rejected(configs_dir: Path) -> None:
    with pytest.raises(configs_io.ConfigError):
        configs_io.write_config("bad", {"lora_rank": "not-an-int"})
    assert not list(configs_dir.glob("*.yaml"))


def test_name_validation(configs_dir: Path) -> None:
    for bad in ("../escape", "name with space", "name/sub", "name.dot"):
        with pytest.raises(configs_io.ConfigError, match="非法配置名"):
            configs_io.write_config(bad, _minimal_payload())


def test_list_sorted_by_mtime(configs_dir: Path) -> None:
    import time
    configs_io.write_config("first", _minimal_payload())
    time.sleep(0.05)
    configs_io.write_config("second", _minimal_payload())
    items = configs_io.list_configs()
    assert [x["name"] for x in items[:2]] == ["second", "first"]


def test_delete(configs_dir: Path) -> None:
    configs_io.write_config("to_delete", _minimal_payload())
    configs_io.delete_config("to_delete")
    assert not (configs_dir / "to_delete.yaml").exists()


def test_delete_missing_raises(configs_dir: Path) -> None:
    with pytest.raises(configs_io.ConfigError, match="不存在"):
        configs_io.delete_config("ghost")


def test_duplicate(configs_dir: Path) -> None:
    payload = _minimal_payload()
    payload["lora_rank"] = 16
    configs_io.write_config("src", payload)
    configs_io.duplicate_config("src", "src_copy")
    assert (configs_dir / "src_copy.yaml").exists()
    assert configs_io.read_config("src_copy")["lora_rank"] == 16


def test_duplicate_conflict(configs_dir: Path) -> None:
    configs_io.write_config("a", _minimal_payload())
    configs_io.write_config("b", _minimal_payload())
    with pytest.raises(configs_io.ConfigError, match="已存在"):
        configs_io.duplicate_config("a", "b")


# ---------------------------------------------------------------------------
# /api/configs HTTP
# ---------------------------------------------------------------------------


def test_api_lifecycle(client: TestClient, configs_dir: Path) -> None:
    payload = _minimal_payload()
    payload["epochs"] = 7

    # 空列表
    assert client.get("/api/configs").json()["items"] == []

    # PUT 创建
    resp = client.put("/api/configs/myrun", json=payload)
    assert resp.status_code == 200, resp.text

    # GET 读取
    got = client.get("/api/configs/myrun").json()
    assert got["epochs"] == 7

    # 列表里能看到
    items = client.get("/api/configs").json()["items"]
    assert any(i["name"] == "myrun" for i in items)

    # 复制
    resp = client.post("/api/configs/myrun/duplicate", json={"new_name": "myrun_copy"})
    assert resp.status_code == 200
    assert client.get("/api/configs/myrun_copy").json()["epochs"] == 7

    # 删除
    assert client.delete("/api/configs/myrun").status_code == 200
    assert client.get("/api/configs/myrun").status_code == 404


def test_api_put_rejects_unknown_field(client: TestClient) -> None:
    bad = _minimal_payload()
    bad["nonexistent_field"] = 123
    resp = client.put("/api/configs/bad", json=bad)
    assert resp.status_code == 422


def test_api_get_invalid_name(client: TestClient) -> None:
    """URL 含非法字符的请求落到 400（FastAPI 路由若放行，配 io 层会拒）。"""
    resp = client.get("/api/configs/has..dot")
    assert resp.status_code in (400, 422)


def test_api_duplicate_conflict(client: TestClient) -> None:
    payload = _minimal_payload()
    client.put("/api/configs/x", json=payload)
    client.put("/api/configs/y", json=payload)
    resp = client.post("/api/configs/x/duplicate", json={"new_name": "y"})
    assert resp.status_code == 400


def test_api_delete_missing(client: TestClient) -> None:
    resp = client.delete("/api/configs/ghost")
    assert resp.status_code == 404


def test_yaml_on_disk_is_human_readable(client: TestClient, configs_dir: Path) -> None:
    """落盘的 YAML 必须是可读 block 风格（不是 flow），方便手编辑。"""
    client.put("/api/configs/readable", json=_minimal_payload())
    text = (configs_dir / "readable.yaml").read_text(encoding="utf-8")
    # block 风格的 mapping 至少有 'key:' 在行尾
    assert "transformer_path:" in text
    # flow 风格会形如 "{key: value, ...}"，我们禁用了 default_flow_style
    assert not text.startswith("{")
    # 能被 yaml 完整解析
    parsed = yaml.safe_load(text)
    assert parsed["lora_type"] == "lokr"
