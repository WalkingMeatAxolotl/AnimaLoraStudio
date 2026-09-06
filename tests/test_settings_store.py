from __future__ import annotations

from pathlib import Path

import pytest

from studio.infrastructure import settings_store


def test_settings_recovers_last_valid_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", tmp_path / "settings.json")
    settings_store.save(settings_store.AppSettingsDocument(values={"system": {"gpu_index": 0}}))
    settings_store.save(settings_store.AppSettingsDocument(values={"system": {"gpu_index": 1}}))
    settings_store.SETTINGS_FILE.write_text("{broken", encoding="utf-8")

    recovered = settings_store.load()

    assert recovered.values["system"]["gpu_index"] == 0
    assert list((tmp_path / "backups" / "settings").glob("settings.json.*.corrupt"))


def test_settings_refuses_to_replace_unrecoverable_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings_store, "SETTINGS_FILE", tmp_path / "settings.json")
    settings_store.SETTINGS_FILE.write_text("{broken", encoding="utf-8")

    with pytest.raises(settings_store.SettingsStoreCorruptError):
        settings_store.save(settings_store.AppSettingsDocument(values={"ok": True}))
    assert settings_store.SETTINGS_FILE.read_text("utf-8") == "{broken"
