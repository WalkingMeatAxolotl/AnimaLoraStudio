"""Schema + /api/presets HTTP（PP0 之前是 /api/configs，保留 308 redirect）。

PP0 把 IO 单元测试拆到 test_presets_io.py；这里专注 HTTP 表面 + schema + 兼容。
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from studio import server
from studio.services.presets import io as presets_io
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
        "tlora_min_rank", "tlora_alpha_rank_scale", "tlora_use_ortho",
        "optimizer_type", "prodigy_d_coef", "prodigy_safeguard_warmup",
        # ProdigyPlusScheduleFree 字段
        "ppsf_d_coef", "ppsf_prodigy_steps", "ppsf_beta1", "ppsf_beta2",
        "ppsf_split_groups", "ppsf_split_groups_mean", "ppsf_use_speed",
        "ppsf_fused_back_pass", "ppsf_use_stableadamw",
        "sample_prompt", "sample_prompts", "no_monitor",
    ):
        assert name in fields, f"missing: {name}"
    assert "wandb_enabled" in fields
    lora_annotation = fields["lora_type"].annotation
    lora_options = getattr(lora_annotation, "__args__", ())
    assert "ortho" in lora_options
    assert "tlora" in lora_options
    # optimizer_type Literal 包含 PPSF
    optimizer_annotation = fields["optimizer_type"].annotation
    # Literal 的 __args__ 包含所有合法值
    assert "prodigy_plus_schedulefree" in getattr(optimizer_annotation, "__args__", ())


def test_lokr_rank_allows_full_dimension_trigger() -> None:
    payload = TrainingConfig().model_dump(mode="python")
    payload["lora_type"] = "lokr"
    payload["lora_rank"] = 50000
    cfg = TrainingConfig.model_validate(payload)
    assert cfg.lora_rank == 50000
    assert "maximum" not in TrainingConfig.model_json_schema()["properties"]["lora_rank"]


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
    assert props["tlora_min_rank"]["show_when"] == "lora_type==tlora"
    assert props["tlora_alpha_rank_scale"]["show_when"] == "lora_type==tlora"
    assert props["tlora_use_ortho"]["show_when"] == "lora_type==tlora"
    assert props["wandb_enabled"]["group"] == "wandb"
    # PPSF 字段都按 optimizer_type==prodigy_plus_schedulefree 显示
    for ppsf_field in (
        "ppsf_d_coef", "ppsf_prodigy_steps", "ppsf_beta1", "ppsf_beta2",
        "ppsf_split_groups", "ppsf_use_speed", "ppsf_fused_back_pass",
    ):
        assert "show_when" in props[ppsf_field], f"{ppsf_field} missing show_when"
        assert "prodigy_plus_schedulefree" in props[ppsf_field]["show_when"]


def test_extra_fields_are_silently_ignored() -> None:
    cfg = TrainingConfig.model_validate({"learning_ratee": 1e-4})
    assert not hasattr(cfg, "learning_ratee")


def test_ppsf_rejects_non_none_scheduler() -> None:
    """PPSF + lr_scheduler != none 应该在 pydantic 层就被拒。"""
    payload = TrainingConfig().model_dump(mode="python")
    payload["optimizer_type"] = "prodigy_plus_schedulefree"
    payload["lr_scheduler"] = "cosine"
    with pytest.raises(Exception):  # pydantic ValidationError
        TrainingConfig.model_validate(payload)


def test_ppsf_accepts_none_scheduler() -> None:
    """PPSF + lr_scheduler=none 是合法组合。"""
    payload = TrainingConfig().model_dump(mode="python")
    payload["optimizer_type"] = "prodigy_plus_schedulefree"
    payload["lr_scheduler"] = "none"
    cfg = TrainingConfig.model_validate(payload)
    assert cfg.optimizer_type == "prodigy_plus_schedulefree"


def test_prodigy_rejects_non_none_scheduler() -> None:
    """普通 Prodigy 也固定常数学习率，不允许外部 scheduler。"""
    payload = TrainingConfig().model_dump(mode="python")
    payload["optimizer_type"] = "prodigy"
    payload["lr_scheduler"] = "cosine"
    with pytest.raises(Exception):
        TrainingConfig.model_validate(payload)


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


def test_api_put_ignores_unknown_field(client: TestClient) -> None:
    bad = _payload()
    bad["nonexistent_field"] = 123
    resp = client.put("/api/presets/bad", json=bad)
    assert resp.status_code == 200
    got = client.get("/api/presets/bad").json()
    assert "nonexistent_field" not in got


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
# 端到端文件 I/O: /api/presets/{name}/download + /api/presets/import
# ---------------------------------------------------------------------------


def test_download_returns_raw_yaml(client: TestClient, presets_dir: Path) -> None:
    """下载端点字节级一致：磁盘上 yaml 文件原封不动透传给客户端。"""
    client.put("/api/presets/dl", json=_payload())
    on_disk = (presets_dir / "dl.yaml").read_bytes()

    resp = client.get("/api/presets/dl/download")
    assert resp.status_code == 200
    assert resp.content == on_disk
    assert resp.headers["content-type"].startswith("application/yaml")
    assert 'filename="dl.yaml"' in resp.headers["content-disposition"]


def test_download_missing(client: TestClient) -> None:
    assert client.get("/api/presets/ghost/download").status_code == 404


def test_download_invalid_name(client: TestClient) -> None:
    assert client.get("/api/presets/has..dot/download").status_code in (400, 422)


def test_import_yaml_roundtrip(client: TestClient, presets_dir: Path) -> None:
    """上传 yaml → 直接落盘到 suggested_name,返回 {name, path}。"""
    payload = _payload()
    payload["epochs"] = 9
    yaml_bytes = yaml.safe_dump(payload, allow_unicode=True).encode("utf-8")

    resp = client.post(
        "/api/presets/import",
        files={"file": ("my-run.yaml", yaml_bytes, "application/yaml")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["name"] == "my-run"
    # 落盘：磁盘上应出现 my-run.yaml,内容可读回
    assert (presets_dir / "my-run.yaml").exists()
    assert client.get("/api/presets/my-run").json()["epochs"] == 9


def test_import_json_also_works(client: TestClient, presets_dir: Path) -> None:
    """yaml.safe_load 是 JSON superset，旧的 .json 导出也能直接 import。"""
    import json as json_mod
    payload = _payload()
    payload["epochs"] = 3
    resp = client.post(
        "/api/presets/import",
        files={"file": ("legacy.json", json_mod.dumps(payload).encode(), "application/json")},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["name"] == "legacy"
    assert client.get("/api/presets/legacy").json()["epochs"] == 3


def test_import_ignores_unknown_field(client: TestClient, presets_dir: Path) -> None:
    bad = _payload()
    bad["nonexistent_field"] = 123
    yaml_bytes = yaml.safe_dump(bad).encode("utf-8")
    resp = client.post(
        "/api/presets/import",
        files={"file": ("bad.yaml", yaml_bytes, "application/yaml")},
    )
    assert resp.status_code == 200
    got = client.get("/api/presets/bad").json()
    assert "nonexistent_field" not in got


def test_import_rejects_malformed_yaml(client: TestClient) -> None:
    resp = client.post(
        "/api/presets/import",
        files={"file": ("trash.yaml", b"this: : not yaml\n  invalid", "application/yaml")},
    )
    assert resp.status_code in (400, 422)


def test_import_sanitizes_suggested_name(client: TestClient, presets_dir: Path) -> None:
    """文件名带空格 / 中文 / 特殊字符 → 名字走 [A-Za-z0-9_-] 白名单后落盘。"""
    yaml_bytes = yaml.safe_dump(_payload()).encode("utf-8")
    resp = client.post(
        "/api/presets/import",
        files={"file": ("我的 preset (v2).yaml", yaml_bytes, "application/yaml")},
    )
    assert resp.status_code == 200
    name = resp.json()["name"]
    # 白名单：[A-Za-z0-9_-]+，非匹配字符压成 '-'，首尾 strip
    assert all(c.isalnum() or c in "_-" for c in name)
    assert "v2" in name
    assert (presets_dir / f"{name}.yaml").exists()


def test_import_returns_409_on_conflict(
    client: TestClient, presets_dir: Path
) -> None:
    """同名 preset 已存在 → 409 + body 含 config / suggested_name 给前端重用。

    不写盘 —— 让 ImportConflictDialog 让用户选覆盖/另存为再走 PUT /api/presets/{name}。
    """
    yaml_bytes = yaml.safe_dump(_payload()).encode("utf-8")

    # 先占名
    resp1 = client.post(
        "/api/presets/import",
        files={"file": ("clash.yaml", yaml_bytes, "application/yaml")},
    )
    assert resp1.status_code == 200
    on_disk_mtime = (presets_dir / "clash.yaml").stat().st_mtime

    # 再上传同名 → 409
    payload2 = _payload()
    payload2["epochs"] = 42  # 内容不同,确保识别"未覆盖"
    yaml_bytes2 = yaml.safe_dump(payload2).encode("utf-8")
    resp2 = client.post(
        "/api/presets/import",
        files={"file": ("clash.yaml", yaml_bytes2, "application/yaml")},
    )
    assert resp2.status_code == 409, resp2.text
    detail = resp2.json()["detail"]
    assert isinstance(detail, dict)
    assert detail["suggested_name"] == "clash"
    assert detail["config"]["epochs"] == 42
    # 没覆盖原文件
    assert (presets_dir / "clash.yaml").stat().st_mtime == on_disk_mtime
    assert client.get("/api/presets/clash").json()["epochs"] != 42


# ---------------------------------------------------------------------------
# /api/presets/{name}?warnings=true — 兼容性警告
# ---------------------------------------------------------------------------


def test_get_preset_with_warnings_reports_dropped(
    client: TestClient, presets_dir: Path
) -> None:
    payload = _payload()
    raw = {**payload, "future_field_xyz": 42, "another_unknown": "hi"}
    yaml_bytes = yaml.safe_dump(raw, allow_unicode=True).encode("utf-8")
    (presets_dir / "compat.yaml").write_bytes(yaml_bytes)

    resp = client.get("/api/presets/compat?warnings=true")
    assert resp.status_code == 200
    body = resp.json()
    assert "config" in body
    assert set(body["dropped_fields"]) == {"future_field_xyz", "another_unknown"}
    assert body["defaulted_fields"] == []


def test_get_preset_with_warnings_reports_defaulted(
    client: TestClient, presets_dir: Path
) -> None:
    """字段值不合法时回退默认值并列入 defaulted_fields。"""
    payload = _payload()
    payload["optimizer_type"] = "lion"  # 不在 Literal 里
    yaml_bytes = yaml.safe_dump(payload, allow_unicode=True).encode("utf-8")
    (presets_dir / "badval.yaml").write_bytes(yaml_bytes)

    resp = client.get("/api/presets/badval?warnings=true")
    assert resp.status_code == 200
    body = resp.json()
    assert "optimizer_type" in body["defaulted_fields"]
    assert body["config"]["optimizer_type"] != "lion"


def test_get_preset_without_warnings_returns_flat(
    client: TestClient, presets_dir: Path
) -> None:
    client.put("/api/presets/flat", json=_payload())
    resp = client.get("/api/presets/flat")
    assert resp.status_code == 200
    body = resp.json()
    assert "config" not in body
    assert "transformer_path" in body


def test_tolerant_load_invalid_values(
    client: TestClient, presets_dir: Path
) -> None:
    """跨分支预设：未知字段 + 非法值 → 都能加载，不会 500。"""
    payload = _payload()
    payload["optimizer_type"] = "lion"
    payload["infonoise_K"] = 0
    raw = {**payload, "nonexistent_thing": True}
    yaml_bytes = yaml.safe_dump(raw, allow_unicode=True).encode("utf-8")
    (presets_dir / "cross.yaml").write_bytes(yaml_bytes)

    resp = client.get("/api/presets/cross")
    assert resp.status_code == 200


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
