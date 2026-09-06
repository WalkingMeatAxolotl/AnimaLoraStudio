from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from studio.infrastructure import credentials
from studio.infrastructure import llm_preset_store as preset_store
from studio.infrastructure import secrets
from studio.infrastructure import settings_store
from studio.infrastructure import storage_layout


@pytest.fixture
def isolated_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", tmp_path / "credentials.json")
    monkeypatch.setattr(preset_store, "LLM_PRESETS_DIR", tmp_path / "llm_presets")
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(
        storage_layout, "STORAGE_LAYOUT_FILE", tmp_path / "storage-layout.json"
    )
    return tmp_path


def _legacy_with_custom() -> secrets.Secrets:
    document = secrets.Secrets()
    style = next(item for item in document.llm_tagger.presets if item.id == "style_json")
    style.label = "Local style"
    style.api_key = "style-secret"
    custom = style.model_copy(
        deep=True,
        update={
            "id": "XL",
            "label": "Two letter legacy ID",
            "api_key": "custom-secret",
            "builtin": False,
        },
    )
    document.llm_tagger.presets.append(custom)
    document.llm_tagger.current_preset = "XL"
    return secrets.Secrets.model_validate(document.model_dump())


def test_migration_preserves_legacy_source_and_splits_presets_and_keys(
    isolated_layout: Path,
) -> None:
    legacy = _legacy_with_custom()
    secrets.save(legacy)
    source_bytes = secrets.SECRETS_FILE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()

    result = storage_layout.ensure_storage_layout()

    assert result.state == "complete"
    assert result.source_sha256 == source_hash
    assert secrets.SECRETS_FILE.read_bytes() == source_bytes
    backup = Path(result.backup_path)
    assert backup.read_bytes() == source_bytes
    assert result.id_map["style_json"] == "style_json"
    custom_id = result.id_map["XL"]
    assert custom_id.startswith("usr_")
    assert preset_store.get("style_json").config.label == "Local style"
    assert preset_store.get(custom_id).config.label == "Two letter legacy ID"
    assert preset_store.get(custom_id).credential_ref
    assert credentials.resolve(preset_store.get("style_json").credential_ref) == "style-secret"
    assert credentials.resolve(preset_store.get(custom_id).credential_ref) == "custom-secret"
    assert settings_store.get_default_llm_preset_id() == custom_id
    # Untouched builtins remain code-owned and do not create user overrides.
    assert not (preset_store.LLM_PRESETS_DIR / "general_json.json").exists()

    second = storage_layout.ensure_storage_layout()
    assert second == result
    assert len(list((isolated_layout / "backups" / "legacy").glob("*.json"))) == 1


def test_migration_with_no_legacy_file_seeds_only_default_state(
    isolated_layout: Path,
) -> None:
    result = storage_layout.ensure_storage_layout()

    assert result.state == "complete"
    assert result.source_sha256 == "missing"
    assert result.backup_path == ""
    assert settings_store.get_default_llm_preset_id() == "style_json"
    assert not credentials.CREDENTIALS_FILE.exists()
    assert not preset_store.LLM_PRESETS_DIR.exists()


def test_prepared_migration_resumes_idempotently_after_partial_install(
    isolated_layout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets.save(_legacy_with_custom())
    original = preset_store.install_migrated
    calls = 0

    def _fail_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 1:
            raise RuntimeError("injected crash")
        return result

    monkeypatch.setattr(preset_store, "install_migrated", _fail_after_first)
    with pytest.raises(RuntimeError, match="injected crash"):
        storage_layout.ensure_storage_layout()

    prepared = storage_layout.StorageLayout.model_validate_json(
        storage_layout.STORAGE_LAYOUT_FILE.read_bytes()
    ).migrations["split_llm_presets_v1"]
    assert prepared.state == "prepared"
    assert not settings_store.SETTINGS_FILE.exists()

    monkeypatch.setattr(preset_store, "install_migrated", original)
    completed = storage_layout.ensure_storage_layout()

    assert completed.state == "complete"
    assert settings_store.get_default_llm_preset_id() == completed.default_preset_id
    assert len(credentials.load().items) == 2


def test_prepared_migration_rejects_changed_legacy_source(
    isolated_layout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secrets.save(_legacy_with_custom())
    original = preset_store.install_migrated

    def _fail(*args, **kwargs):
        raise RuntimeError("injected crash")

    monkeypatch.setattr(preset_store, "install_migrated", _fail)
    with pytest.raises(RuntimeError):
        storage_layout.ensure_storage_layout()

    monkeypatch.setattr(preset_store, "install_migrated", original)
    changed = _legacy_with_custom()
    changed.llm_tagger.current_preset = "style_json"
    secrets.save(changed)

    with pytest.raises(
        storage_layout.StorageMigrationError,
        match="changed while storage migration was prepared",
    ):
        storage_layout.ensure_storage_layout()
