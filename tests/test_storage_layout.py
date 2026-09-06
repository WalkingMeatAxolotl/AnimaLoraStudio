from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.infrastructure import credentials
from studio.infrastructure import llm_model_cache
from studio.infrastructure import llm_preset_store
from studio.infrastructure import secrets
from studio.infrastructure import settings_store
from studio.infrastructure import storage_layout


def _isolate(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(secrets, "SECRETS_FILE", root / "secrets.json")
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", root / "credentials.json")
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", root / "settings.json")
    monkeypatch.setattr(llm_preset_store, "LLM_PRESETS_DIR", root / "llm_presets")
    monkeypatch.setattr(llm_model_cache, "LLM_MODEL_CACHE_DIR", root / "cache" / "llm_models")
    monkeypatch.setattr(storage_layout, "STORAGE_LAYOUT_FILE", root / "storage-layout.json")


def _legacy() -> secrets.Secrets:
    value = secrets.Secrets()
    value.gelbooru.api_key = "gel-secret"
    value.huggingface.token = "hf-secret"
    value.llm_tagger.current_preset = "custom_caption"
    custom = value.llm_tagger.presets[0].model_copy(deep=True)
    custom.id = "custom_caption"
    custom.label = "Custom caption"
    custom.builtin = False
    custom.api_key = "llm-secret"
    custom.messages[0].content = "custom unicode 提示"
    value.llm_tagger.presets.append(custom)
    return value


def test_split_completion_is_scoped_to_one_storage_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate(monkeypatch, tmp_path)
    storage_layout.ensure_storage_layout()
    assert storage_layout.is_split_complete() is True

    other_root = tmp_path / "restored-legacy-root"
    monkeypatch.setattr(secrets, "SECRETS_FILE", other_root / "secrets.json")

    assert storage_layout.is_split_complete() is False


def test_split_migration_preserves_values_and_removes_secrets_from_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate(monkeypatch, tmp_path)
    legacy = _legacy()
    secrets.SECRETS_FILE.write_text(legacy.model_dump_json(indent=2), encoding="utf-8")
    original = secrets.SECRETS_FILE.read_bytes()

    record = storage_layout.ensure_storage_layout()

    assert record.state == "complete"
    assert secrets.SECRETS_FILE.read_bytes() == original
    public_settings = settings_store.load().values
    assert public_settings["gelbooru"]["api_key"] == ""
    assert public_settings["huggingface"]["token"] == ""
    assert public_settings["llm_tagger"] == {"default_preset_id": "custom_caption"}
    credential_values = {item.id: item.secret for item in credentials.load().items.values()}
    assert "gel-secret" in credential_values.values()
    assert "hf-secret" in credential_values.values()
    assert "llm-secret" in credential_values.values()

    stored = llm_preset_store.get("custom_caption")
    assert stored.config.messages[0].content == "custom unicode 提示"
    assert stored.config.api_key == ""
    assert stored.credential_ref
    raw_preset = (llm_preset_store.LLM_PRESETS_DIR / "custom_caption.json").read_text("utf-8")
    assert "llm-secret" not in raw_preset

    composed = secrets.load()
    assert composed.gelbooru.api_key == "gel-secret"
    assert composed.huggingface.token == "hf-secret"
    assert composed.llm_tagger.current_preset == "custom_caption"
    assert next(p for p in composed.llm_tagger.presets if p.id == "custom_caption").api_key == "llm-secret"

    assert storage_layout.ensure_storage_layout() == record

    updated = secrets.update({"gelbooru": {"user_id": "new-user"}})
    assert updated.gelbooru.user_id == "new-user"
    assert updated.gelbooru.api_key == "gel-secret"
    assert settings_store.load().values["gelbooru"]["api_key"] == ""


def test_prepared_migration_resumes_without_rewriting_legacy_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate(monkeypatch, tmp_path)
    legacy = _legacy()
    secrets.SECRETS_FILE.write_text(legacy.model_dump_json(indent=2), encoding="utf-8")
    original = secrets.SECRETS_FILE.read_bytes()
    real_validate = storage_layout._validate_final
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise storage_layout.StorageMigrationError("injected crash")
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(storage_layout, "_validate_final", fail_once)
    with pytest.raises(storage_layout.StorageMigrationError, match="injected crash"):
        storage_layout.ensure_storage_layout()

    prepared = json.loads(storage_layout.STORAGE_LAYOUT_FILE.read_text("utf-8"))
    assert prepared["migrations"]["split_llm_presets_v1"]["state"] == "prepared"
    assert secrets.SECRETS_FILE.read_bytes() == original

    resumed = storage_layout.ensure_storage_layout()
    assert resumed.state == "complete"
    assert secrets.SECRETS_FILE.read_bytes() == original
