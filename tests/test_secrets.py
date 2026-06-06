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
    # joycaption 已合并为 llm_tagger 的 builtin preset
    joy = next(p for p in s.llm_tagger.presets if p.id == "joycaption")
    assert joy.base_url.startswith("http://")
    assert s.wandb.project == "AnimaLoraStudio"


def test_load_corrupt_json_returns_defaults(secrets_file: Path) -> None:
    secrets_file.write_text("{not valid json", encoding="utf-8")
    # 不应抛错；返回默认实例
    s = secrets.load()
    assert s.gelbooru.user_id == ""


def test_wd14_defaults_include_candidate_list(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.wd14.model_id in s.wd14.model_ids
    assert set(secrets.DEFAULT_WD14_MODELS).issubset(set(s.wd14.model_ids))


def test_cltagger_defaults_use_1_02(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.cltagger.model_id == "cella110n/cl_tagger"
    assert s.cltagger.model_path == "cl_tagger_1_02/model.onnx"
    assert s.cltagger.tag_mapping_path == "cl_tagger_1_02/tag_mapping.json"
    assert s.cltagger.threshold_character == pytest.approx(0.6)

def test_legacy_local_dir_dropped_on_load(secrets_file: Path) -> None:
    """旧 secrets.json 里残留的 local_dir / variant_local_dirs 字段：load 时被
    pydantic（extra=ignore）静默丢弃，不报错、不再出现在模型上。"""
    secrets_file.write_text(
        json.dumps({
            "wd14": {"local_dir": "/old/wd14"},
            "cltagger": {
                "local_dir": "/old/cltagger",
                "variant_local_dirs": {"cl_tagger_1_02": "/old/cltagger"},
            },
        }),
        encoding="utf-8",
    )
    s = secrets.load()
    assert not hasattr(s.wd14, "local_dir")
    assert not hasattr(s.cltagger, "local_dir")
    assert not hasattr(s.cltagger, "variant_local_dirs")


def test_eval_metrics_defaults_and_persistence(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.eval_metrics.clip_model_name == "openai/clip-vit-base-patch32"
    assert s.eval_metrics.dino_model_name == "facebook/dinov2-small"
    assert s.eval_metrics.auto_eval_on_checkpoint is False
    assert s.eval_metrics.auto_eval_max_items == 1

    secrets.update({
        "eval_metrics": {
            "clip_model_name": "/models/clip",
            "dino_model_name": "/models/dino",
            "auto_eval_on_checkpoint": True,
            "auto_eval_max_items": 999,
        }
    })
    saved = secrets.load()
    assert saved.eval_metrics.clip_model_name == "/models/clip"
    assert saved.eval_metrics.dino_model_name == "/models/dino"
    assert saved.eval_metrics.auto_eval_on_checkpoint is True
    assert saved.eval_metrics.auto_eval_max_items == 256


def test_llm_tagger_defaults(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.llm_tagger.current_preset == "style_json"
    assert [p.id for p in s.llm_tagger.presets] == [
        "style_json",
        "general_json",
        "txt_tags",
        "joycaption",
    ]
    assert all(p.builtin for p in s.llm_tagger.presets)
    # joycaption builtin preset 预填了 vLLM 推荐配置
    joy = next(p for p in s.llm_tagger.presets if p.id == "joycaption")
    assert joy.base_url == "http://localhost:8000/v1"
    assert joy.model.endswith("joycaption-beta-one-hf-llava")
    assert joy.endpoint == "chat_completions"
    assert joy.output_format == "text"
    assert joy.temperature == pytest.approx(0.6)
    assert joy.max_tokens == 300
    assert joy.concurrency == 1
    assert joy.requests_per_second == pytest.approx(0.0)
    assert joy.max_requests_per_minute == 0


def test_llm_preset_normalizes_request_pool_settings(secrets_file: Path) -> None:
    s = secrets.update(
        {
            "llm_tagger": {
                "presets": [
                    {
                        "id": "style_json",
                        "concurrency": 99,
                        "requests_per_second": -5,
                        "max_requests_per_minute": 9999,
                    }
                ]
            }
        }
    )
    style = next(p for p in s.llm_tagger.presets if p.id == "style_json")
    assert style.concurrency == 8
    assert style.requests_per_second == pytest.approx(0.0)
    assert style.max_requests_per_minute == 3600


def test_llm_preset_keeps_model_in_model_ids(secrets_file: Path) -> None:
    s = secrets.update(
        {
            "llm_tagger": {
                "presets": [{"id": "joycaption", "model": "vision-a", "model_ids": []}]
            }
        }
    )
    joy = next(p for p in s.llm_tagger.presets if p.id == "joycaption")
    assert joy.model == "vision-a"
    assert joy.model_ids == ["vision-a"]


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


def test_download_sources_default_seeds_huggingface(secrets_file: Path) -> None:
    """默认（无旧全局源）→ 三个双源类型都种子为 huggingface。"""
    s = secrets.load()
    assert s.download_sources == {
        "training": "huggingface",
        "wd14": "huggingface",
        "upscaler": "huggingface",
    }


def test_download_sources_migrate_from_legacy_global(secrets_file: Path) -> None:
    """旧 secrets.json 只有全局 download_source=modelscope → 各类型继承 MS，不静默回退 HF。"""
    secrets_file.write_text(
        json.dumps({"download_source": "modelscope"}),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.download_sources["training"] == "modelscope"
    assert s.download_sources["wd14"] == "modelscope"
    assert s.download_sources["upscaler"] == "modelscope"


def test_download_sources_explicit_override_not_clobbered_by_legacy(secrets_file: Path) -> None:
    """显式设过的类型不被旧全局种子覆盖；未设的才继承。"""
    secrets_file.write_text(
        json.dumps({
            "download_source": "modelscope",
            "download_sources": {"training": "huggingface"},
        }),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.download_sources["training"] == "huggingface"  # 显式保留
    assert s.download_sources["wd14"] == "modelscope"        # 未设 → 继承旧全局


def test_download_sources_persist_and_normalize(secrets_file: Path) -> None:
    secrets.update({"download_sources": {"wd14": "modelscope", "upscaler": "garbage"}})
    s = secrets.load()
    assert s.download_sources["wd14"] == "modelscope"
    assert s.download_sources["upscaler"] == "huggingface"  # 非法值归一


def test_download_image_settings_default(secrets_file: Path) -> None:
    """save_tags/convert_to_png/remove_alpha_channel 现在挂在 download 下。"""
    s = secrets.load()
    assert s.download.save_tags is False
    assert s.download.convert_to_png is True
    assert s.download.remove_alpha_channel is True


def test_migrate_gelbooru_image_settings_to_download(secrets_file: Path) -> None:
    """旧 secrets.json 把这三个挂在 gelbooru 下 → 迁移到 download.*，老值不丢。"""
    secrets_file.write_text(
        json.dumps({
            "gelbooru": {
                "user_id": "u",
                "save_tags": True,
                "convert_to_png": False,
                "remove_alpha_channel": False,
            }
        }),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.download.save_tags is True
    assert s.download.convert_to_png is False
    assert s.download.remove_alpha_channel is False
    assert s.gelbooru.user_id == "u"  # 凭据保留


def test_models_root_default_none(secrets_file: Path) -> None:
    """默认 secrets 里 models.root = None；下游应自行回退默认路径。"""
    s = secrets.load()
    assert s.models.root is None


def test_models_root_persists(secrets_file: Path) -> None:
    """patch 保存后能从磁盘读回；空字符串视作 None（下游回退默认）。"""
    secrets.update({"models": {"root": "/data/anima"}})
    assert secrets.load().models.root == "/data/anima"
    secrets.update({"models": {"root": None}})
    assert secrets.load().models.root is None


def test_model_downloader_uses_secrets_root(
    secrets_file: Path, tmp_path: Path
) -> None:
    """model_downloader.models_root() 优先读 secrets；未设回退 REPO_ROOT/models。

    （与 schema.py 默认 + WD14 已用的 `models/wd14/` 对齐）
    """
    from studio.services import models as model_downloader
    # 未设
    secrets.update({"models": {"root": None}})
    fallback = model_downloader.models_root()
    assert fallback.name == "models"
    # 设了
    custom = tmp_path / "custom_models"
    secrets.update({"models": {"root": str(custom)}})
    assert model_downloader.models_root() == custom


def test_find_anima_main_picks_latest(secrets_file: Path, tmp_path: Path) -> None:
    """多版本并存时按 ANIMA_VARIANTS 顺序（latest 优先）返回第一个存在的。"""
    from studio.services import models as model_downloader
    secrets.update({"models": {"root": str(tmp_path)}})
    dm = tmp_path / "diffusion_models"
    dm.mkdir(parents=True)

    # 一个都没 → None
    assert model_downloader.find_anima_main() is None

    # 只有 preview2 → 返回 preview2
    (dm / "anima-preview2.safetensors").write_bytes(b"x")
    assert model_downloader.find_anima_main().name == "anima-preview2.safetensors"

    # preview3-base 装上 → latest 优先返回 preview3-base（preview3-base 在 1.0 缺席时是次新）
    (dm / "anima-preview3-base.safetensors").write_bytes(b"y")
    assert (
        model_downloader.find_anima_main().name == "anima-preview3-base.safetensors"
    )

    # 1.0 装上 → latest 优先返回 1.0
    (dm / "anima-base-v1.0.safetensors").write_bytes(b"z")
    assert (
        model_downloader.find_anima_main().name == "anima-base-v1.0.safetensors"
    )


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
# reg.default_excluded_tags（正则集全局默认排除）
# ---------------------------------------------------------------------------


def test_reg_default_excluded_empty_by_default(secrets_file: Path) -> None:
    s = secrets.load()
    assert s.reg.default_excluded_tags == []


def test_reg_default_excluded_round_trip(secrets_file: Path) -> None:
    """patch 保存后能从磁盘读回（正则集页进新 build 时按此 seed）。"""
    secrets.update(
        {"reg": {"default_excluded_tags": ["white background", "signature"]}}
    )
    assert secrets.load().reg.default_excluded_tags == [
        "white background",
        "signature",
    ]


def test_reg_legacy_file_without_reg_field(secrets_file: Path) -> None:
    """老 secrets.json 没有 reg 字段时，加载用默认空列表，其它字段不受影响。"""
    secrets_file.write_text(
        json.dumps({"gelbooru": {"user_id": "alice"}}),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.reg.default_excluded_tags == []
    assert s.gelbooru.user_id == "alice"


def test_reg_default_excluded_in_masked_dict(secrets_file: Path) -> None:
    """default_excluded_tags 不是敏感字段，掩码后保留原值（前端需读它做 seed）。"""
    secrets.update({"reg": {"default_excluded_tags": ["lowres"]}})
    masked = secrets.to_masked_dict(secrets.load())
    assert masked["reg"]["default_excluded_tags"] == ["lowres"]


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
            "wandb": {"api_key": "wandb_secret"},
            "llm_tagger": {
                "presets": [{"id": "joycaption", "api_key": "llm_secret"}]
            },
        }
    )
    masked = secrets.to_masked_dict(secrets.load())
    assert masked["gelbooru"]["user_id"] == "alice"  # 非敏感字段保留
    assert masked["gelbooru"]["api_key"] == secrets.MASK
    assert masked["huggingface"]["token"] == secrets.MASK
    assert masked["wandb"]["api_key"] == secrets.MASK
    # llm_tagger.presets.*.api_key 通配
    joy_masked = next(p for p in masked["llm_tagger"]["presets"] if p["id"] == "joycaption")
    assert joy_masked["api_key"] == secrets.MASK


def test_to_masked_dict_keeps_empty_sensitive_empty(secrets_file: Path) -> None:
    """没有值的敏感字段不应该显示为 "***"，否则前端无法判断「真的为空」。"""
    masked = secrets.to_masked_dict(secrets.load())
    assert masked["gelbooru"]["api_key"] == ""
    assert masked["huggingface"]["token"] == ""
    assert masked["wandb"]["api_key"] == ""
    for preset in masked["llm_tagger"]["presets"]:
        assert preset["api_key"] == ""


def test_llm_tagger_legacy_schema_migration(secrets_file: Path) -> None:
    """老 secrets.json (PR #18 schema) → preset-unified 自动迁移。"""
    secrets_file.write_text(
        json.dumps(
            {
                "joycaption": {
                    "base_url": "http://my-vllm:9000/v1",
                    "model": "my-custom-joycaption",
                    "prompt_template": "My custom prompt",
                },
                "llm_tagger": {
                    "base_url": "https://api.openai.com/v1",
                    "api_key": "sk-xxx",
                    "model": "gpt-4o-mini",
                    "model_ids": ["gpt-4o-mini", "gpt-4o"],
                    "endpoint": "chat_completions",
                    "prompt_preset": "style_json",
                    "prompt_presets": [
                        {"id": "style_json", "label": "画风", "prompt": "P1", "builtin": True, "output_format": "json"},
                    ],
                    "custom_prompt": "",
                    "temperature": 0.3,
                    "max_tokens": 800,
                },
            }
        ),
        encoding="utf-8",
    )
    s = secrets.load()
    # 顶层 endpoint+生成参数下沉到每个 preset
    style = next(p for p in s.llm_tagger.presets if p.id == "style_json")
    assert style.base_url == "https://api.openai.com/v1"
    assert style.api_key == "sk-xxx"
    assert style.model == "gpt-4o-mini"
    assert style.endpoint == "chat_completions"
    assert style.temperature == pytest.approx(0.3)
    assert style.max_tokens == 800
    assert style.concurrency == 1
    assert style.requests_per_second == pytest.approx(0.0)
    assert style.max_requests_per_minute == 0
    # JoyCaption 卡片字段写到 joycaption preset
    joy = next(p for p in s.llm_tagger.presets if p.id == "joycaption")
    assert joy.base_url == "http://my-vllm:9000/v1"
    assert joy.model == "my-custom-joycaption"
    # 用户自定义 prompt_template 单独建一个 user_joycaption preset
    user_joy = next(p for p in s.llm_tagger.presets if p.id == "user_joycaption")
    assert user_joy.messages[0].type == "text"
    assert user_joy.messages[0].role == "system"
    assert user_joy.messages[0].content == "My custom prompt"
    assert user_joy.messages[-1].type == "image"
    assert user_joy.output_format == "text"


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


# ---------------------------------------------------------------------------
# PR-D / ADR 0005 — system.update_channel（用户视图偏好持久化）
# ---------------------------------------------------------------------------


def test_system_defaults_update_channel_stable(secrets_file: Path) -> None:
    """新装默认通道偏好 = stable，绝大多数用户只看稳定版。"""
    s = secrets.load()
    assert s.system.update_channel == "stable"
    assert s.system.show_dev_channel is False  # legacy 字段默认也是 False


def test_system_update_channel_round_trip(secrets_file: Path) -> None:
    """update + load 持久化（webui 切 toggle 后刷页应保留）。"""
    secrets.update({"system": {"update_channel": "dev"}})
    assert secrets.load().system.update_channel == "dev"
    secrets.update({"system": {"update_channel": "stable"}})
    assert secrets.load().system.update_channel == "stable"


def test_system_legacy_file_without_system_field(secrets_file: Path) -> None:
    """老 secrets.json 没有 system 字段时，加载用默认值 stable。"""
    secrets_file.write_text(
        json.dumps({"gelbooru": {"user_id": "alice"}}),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.system.update_channel == "stable"
    assert s.gelbooru.user_id == "alice"  # 其它字段不受影响


def test_system_update_channel_in_masked_dict(secrets_file: Path) -> None:
    """update_channel 不是敏感字段，掩码后应保留原值。"""
    secrets.update({"system": {"update_channel": "dev"}})
    masked = secrets.to_masked_dict(secrets.load())
    assert masked["system"]["update_channel"] == "dev"


def test_system_show_dev_channel_migrated_to_update_channel(
    secrets_file: Path,
) -> None:
    """ADR 0005：老 secrets.json 里 show_dev_channel=true 一次性迁移成
    update_channel='dev'，让升级用户保留之前的 dev 视图偏好。"""
    secrets_file.write_text(
        json.dumps({"system": {"show_dev_channel": True}}),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.system.update_channel == "dev"


def test_system_show_dev_channel_migration_does_not_overwrite_explicit_pref(
    secrets_file: Path,
) -> None:
    """update_channel 已显式设过 → 迁移函数不覆盖（幂等）。"""
    secrets_file.write_text(
        json.dumps({"system": {"show_dev_channel": True, "update_channel": "stable"}}),
        encoding="utf-8",
    )
    s = secrets.load()
    assert s.system.update_channel == "stable"  # 显式设过不被 legacy 覆盖
