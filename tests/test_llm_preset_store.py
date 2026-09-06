from __future__ import annotations

import json
from pathlib import Path

import pytest

from studio.infrastructure import credentials
from studio.infrastructure import llm_preset_store as store


@pytest.fixture
def isolated_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(store, "LLM_PRESETS_DIR", tmp_path / "llm_presets")
    monkeypatch.setattr(credentials, "CREDENTIALS_FILE", tmp_path / "credentials.json")
    return tmp_path


def _payload(label: str = "My preset") -> dict:
    return {
        "label": label,
        "base_url": "https://private.example/v1",
        "model": "vision-model",
        "messages": [
            {"type": "text", "role": "system", "content": "Describe it"},
            {"type": "image", "role": "user", "content": ""},
        ],
        "output_format": "text",
    }


def test_builtin_list_uses_read_only_templates(isolated_stores: Path) -> None:
    items, invalid = store.list_all()

    assert invalid == []
    assert [item.config.id for item in items[:6]] == list(store.BUILTIN_PRESET_ORDER)
    assert all(item.origin == "builtin" for item in items[:6])
    assert not (isolated_stores / "llm_presets").exists()


def test_create_persists_one_secret_free_document(isolated_stores: Path) -> None:
    created = store.create(
        {**_payload(), "api_key": "must-not-persist", "model_ids": ["cached"]},
        credential_ref="cred_openai_main",
        preset_id="usr_recipe",
    )

    raw = json.loads(
        (isolated_stores / "llm_presets" / "usr_recipe.json").read_text(
            encoding="utf-8"
        )
    )
    assert raw["kind"] == "anima-llm-preset"
    assert raw["schema_version"] == 1
    assert raw["credential_ref"] == "cred_openai_main"
    assert "api_key" not in raw
    assert "model_ids" not in raw
    assert "builtin" not in raw
    assert created.origin == "custom"


def test_update_changes_only_target_and_rejects_stale_etag(
    isolated_stores: Path,
) -> None:
    first = store.create(_payload("First"), preset_id="usr_first")
    second = store.create(_payload("Second"), preset_id="usr_second")
    second_path = isolated_stores / "llm_presets" / "usr_second.json"
    second_before = second_path.read_bytes()

    updated = store.update(
        first.config.id,
        {"temperature": 0.7},
        expected_etag=first.etag,
    )

    assert updated.config.temperature == pytest.approx(0.7)
    assert second_path.read_bytes() == second_before
    with pytest.raises(store.LLMPresetConflictError) as caught:
        store.update(
            first.config.id,
            {"temperature": 0.9},
            expected_etag=first.etag,
        )
    assert caught.value.current_etag == updated.etag
    assert store.get(second.config.id).config.label == "Second"


def test_builtin_override_and_reset(isolated_stores: Path) -> None:
    builtin = store.get("style_json")
    overridden = store.update(
        "style_json", {"label": "My Style"}, expected_etag=builtin.etag
    )

    assert overridden.origin == "builtin_override"
    assert overridden.config.label == "My Style"
    assert (isolated_stores / "llm_presets" / "style_json.json").exists()

    reset = store.reset_builtin("style_json", expected_etag=overridden.etag)

    assert reset.origin == "builtin"
    assert reset.config.label != "My Style"
    assert not (isolated_stores / "llm_presets" / "style_json.json").exists()
    assert store.history("style_json")


def test_delete_custom_can_restore_archived_version(isolated_stores: Path) -> None:
    created = store.create(_payload(), preset_id="usr_restore")
    store.delete(created.config.id, expected_etag=created.etag)

    with pytest.raises(store.LLMPresetNotFoundError):
        store.get(created.config.id)
    versions = store.history(created.config.id)
    assert len(versions) == 1

    restored = store.restore(
        created.config.id,
        versions[0]["backup_id"],
        expected_etag=versions[0]["etag"],
    )

    assert restored.config.label == created.config.label
    assert restored.credential_ref == created.credential_ref


def test_portable_export_strips_local_connection(isolated_stores: Path) -> None:
    created = store.create(
        _payload(), credential_ref="cred_openai_main", preset_id="usr_export"
    )

    exported = json.loads(store.export_portable(created.config.id))

    assert exported["credential_ref"] == ""
    assert exported["base_url"] == ""
    assert "api_key" not in exported
    assert "model_ids" not in exported


def test_invalid_custom_file_is_isolated_in_list(isolated_stores: Path) -> None:
    preset_dir = isolated_stores / "llm_presets"
    preset_dir.mkdir()
    (preset_dir / "usr_broken.json").write_text("{broken", encoding="utf-8")
    store.create(_payload("Healthy"), preset_id="usr_healthy")

    items, invalid = store.list_all()

    assert "usr_healthy" in {item.config.id for item in items}
    assert [item.id for item in invalid] == ["usr_broken"]


def test_invalid_primary_read_is_side_effect_free_and_explicitly_restorable(
    isolated_stores: Path,
) -> None:
    created = store.create(_payload("Original"), preset_id="usr_corrupt")
    updated = store.update(
        created.config.id,
        {"label": "Updated"},
        expected_etag=created.etag,
    )
    path = isolated_stores / "llm_presets" / "usr_corrupt.json"
    path.write_text("{broken", encoding="utf-8")
    broken_bytes = path.read_bytes()

    items, invalid = store.list_all()

    assert "usr_corrupt" not in {item.config.id for item in items}
    assert [item.id for item in invalid] == ["usr_corrupt"]
    assert path.read_bytes() == broken_bytes
    versions = store.history("usr_corrupt")
    restored = store.restore(
        "usr_corrupt",
        versions[0]["backup_id"],
        expected_etag=versions[0]["etag"],
    )
    assert restored.config.label == "Original"
    assert restored.etag != updated.etag
    assert list(
        (isolated_stores / "backups" / "llm_presets" / "usr_corrupt").glob(
            "*.corrupt"
        )
    )


def test_invalid_primary_without_backup_can_be_discarded_or_reset(
    isolated_stores: Path,
) -> None:
    preset_dir = isolated_stores / "llm_presets"
    preset_dir.mkdir()
    custom = preset_dir / "usr_discard.json"
    builtin = preset_dir / "style_json.json"
    custom.write_text("{broken-custom", encoding="utf-8")
    builtin.write_text("{broken-builtin", encoding="utf-8")
    _, invalid = store.list_all()
    etags = {item.id: item.etag for item in invalid}

    store.delete("usr_discard", expected_etag=etags["usr_discard"])
    reset = store.reset_builtin("style_json", expected_etag=etags["style_json"])

    assert not custom.exists()
    assert not builtin.exists()
    assert reset.origin == "builtin"
    assert len(
        list(
            (isolated_stores / "backups" / "llm_presets" / "usr_discard").glob(
                "*.corrupt"
            )
        )
    ) == 1
    assert len(
        list(
            (isolated_stores / "backups" / "llm_presets" / "style_json").glob(
                "*.corrupt"
            )
        )
    ) == 1


def test_ids_cannot_normalize_to_another_file_or_use_windows_devices(
    isolated_stores: Path,
) -> None:
    existing = store.create(_payload("Safe"), preset_id="foo")
    before = (isolated_stores / "llm_presets" / "foo.json").read_bytes()

    with pytest.raises(store.LLMPresetInvalidError):
        store.create(_payload("Must not overwrite"), preset_id="foo_")
    with pytest.raises(store.LLMPresetInvalidError):
        store.create(_payload("Device"), preset_id="nul")

    assert store.get("foo").etag == existing.etag
    assert (isolated_stores / "llm_presets" / "foo.json").read_bytes() == before


def test_migration_installer_uses_explicit_normal_identity(
    isolated_stores: Path,
) -> None:
    legacy = store.LLMPresetConfig.model_validate({**_payload("Legacy"), "id": "XL"})

    migrated = store.install_migrated(legacy, preset_id="usr_xl")

    assert migrated.config.id == "usr_xl"
    on_disk = json.loads(
        (isolated_stores / "llm_presets" / "usr_xl.json").read_text("utf-8")
    )
    assert on_disk["id"] == "usr_xl"


def test_credential_metadata_never_returns_secret(isolated_stores: Path) -> None:
    created = credentials.create(label="Primary", secret="super-secret")

    assert "secret" not in created
    listed = credentials.list_metadata()
    assert listed == [created]
    assert "super-secret" not in json.dumps(listed)
    assert credentials.resolve(created["id"]) == "super-secret"


def test_credential_etag_rejects_stale_update(isolated_stores: Path) -> None:
    created = credentials.create(label="Primary", secret="one")
    updated = credentials.replace_secret(
        created["id"], "two", expected_etag=created["etag"]
    )

    with pytest.raises(credentials.CredentialConflictError):
        credentials.replace_secret(
            created["id"], "three", expected_etag=created["etag"]
        )
    assert credentials.resolve(created["id"]) == "two"
    assert updated["configured"] is True


def test_credential_import_is_idempotent_without_backup_churn(
    isolated_stores: Path,
) -> None:
    record = credentials.CredentialRecord(
        id="cred_migrated", label="Migrated", secret="secret"
    )
    credentials.import_records([record])
    first_bytes = credentials.CREDENTIALS_FILE.read_bytes()

    credentials.import_records([record])

    assert credentials.CREDENTIALS_FILE.read_bytes() == first_bytes
    assert not list((isolated_stores / "backups" / "credentials").glob("*.bak"))


def test_credential_corruption_recovers_previous_valid_version(
    isolated_stores: Path,
) -> None:
    first = credentials.create(label="Primary", secret="one")
    credentials.replace_secret(first["id"], "two", expected_etag=first["etag"])
    credentials.CREDENTIALS_FILE.write_text("{broken", encoding="utf-8")

    recovered = credentials.load()

    assert recovered.items[first["id"]].secret == "one"
    corrupt = list(
        (isolated_stores / "backups" / "credentials").glob("*.corrupt")
    )
    assert len(corrupt) == 1
