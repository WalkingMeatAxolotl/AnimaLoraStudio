"""模型族切换动作的纯计算 + 端点（多模型 P4-3，C5——根除 C2/C3）。"""

from __future__ import annotations

import pytest

from studio.domain.family_switch import switch_family
from studio.schema import TrainingConfig
from studio.services.models import default_paths_for_new_version


def _paths(family: str) -> dict[str, str]:
    return default_paths_for_new_version(family=family)


def test_switch_anima_to_krea2_rewrites_paths_and_flavor():
    cfg = TrainingConfig(navit_packing=True, lora_rank=64).model_dump(mode="python")
    new, changes = switch_family(cfg, "krea2", _paths("krea2"))

    assert new["model_family"] == "krea2"
    assert new["transformer_path"].endswith("krea2-raw-bf16.safetensors")
    assert new["t5_tokenizer_path"] == ""
    assert new["sample_sampler_name"] == "euler"
    assert new["sample_scheduler"] == "krea2_shift"
    assert new["timestep_sampling"] == "krea2_shift"
    assert new["shuffle_caption"] is False
    # 目标族不支持的能力字段被关回，否则 validator 拒绝整份 config
    assert new["navit_packing"] is False
    # 用户的非风味字段原样保留
    assert new["lora_rank"] == 64
    # 切换产物必须能通过完整校验（capability + sampler 白名单 + Literal）
    TrainingConfig(**new)
    changed_fields = {c["field"] for c in changes}
    assert {"model_family", "transformer_path", "text_encoder_path",
            "t5_tokenizer_path", "sample_sampler_name", "navit_packing",
            } <= changed_fields


def test_switch_krea2_to_anima_restores_schema_defaults():
    cfg = TrainingConfig(model_family="krea2").model_dump(mode="python")
    cfg["t5_tokenizer_path"] = ""  # krea2 config 的现实形态
    new, changes = switch_family(cfg, "anima", _paths("anima"))

    assert new["model_family"] == "anima"
    assert new["sample_sampler_name"] == "er_sde"
    assert new["sample_scheduler"] == "simple"
    assert new["timestep_sampling"] == "logit_normal"
    assert (new["sample_infer_steps"], new["sample_cfg_scale"]) == (25, 4.0)
    # t5 路径从空恢复为全局默认（C3 的静默丢失变成显式重算）
    assert new["t5_tokenizer_path"].endswith("t5_tokenizer")
    TrainingConfig(**new)
    assert any(c["field"] == "t5_tokenizer_path" for c in changes)


def test_switch_same_family_reports_no_path_drift():
    """同族「切换」= 把 config 归位到当前 Settings 路径与族默认；已归位则零变更。"""
    cfg = TrainingConfig().model_dump(mode="python")
    cfg.update(_paths("anima"))
    new, changes = switch_family(cfg, "anima", _paths("anima"))
    assert changes == []
    assert new == cfg


def test_switch_partial_config_fills_missing():
    """config 允许部分字段缺失（preset 池里裁剪过的 yaml），缺失键 from=None。"""
    new, changes = switch_family(
        {"model_family": "anima", "lora_rank": 32}, "krea2", _paths("krea2"))
    assert new["sample_sampler_name"] == "euler"
    assert new["lora_rank"] == 32
    by_field = {c["field"]: c for c in changes}
    assert by_field["sample_sampler_name"]["from"] is None


def test_endpoint_switches_and_rejects_unknown():
    from fastapi.testclient import TestClient
    from studio.server import app

    client = TestClient(app)
    cfg = TrainingConfig().model_dump(mode="json")
    r = client.post("/api/models/family-switch",
                    json={"target": "krea2", "config": cfg})
    assert r.status_code == 200
    body = r.json()
    assert body["config"]["model_family"] == "krea2"
    assert body["config"]["sample_sampler_name"] == "euler"
    assert any(c["field"] == "transformer_path" for c in body["changes"])

    r = client.post("/api/models/family-switch",
                    json={"target": "no-such", "config": cfg})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "model.family_invalid"
