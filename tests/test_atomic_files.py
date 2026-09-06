from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from studio.infrastructure import atomic_files


def test_replace_failure_keeps_original_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "settings.json"
    target.write_bytes(b"old")

    def _fail_replace(src: Path, dst: Path) -> None:  # noqa: ARG001
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomic_files.os, "replace", _fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_files.atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_invalid_existing_file_is_not_replaced(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_bytes(b"corrupt")

    def _validate(data: bytes) -> None:
        if data != b"valid":
            raise ValueError("invalid")

    with pytest.raises(atomic_files.InvalidExistingFileError):
        atomic_files.atomic_write_bytes(
            target,
            b"new",
            backup_dir=tmp_path / "backups",
            keep_backups=2,
            validate_existing=_validate,
        )

    assert target.read_bytes() == b"corrupt"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_validator_protects_existing_file_without_backups(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_bytes(b"corrupt")

    def _invalid(data: bytes) -> None:  # noqa: ARG001
        raise ValueError("invalid")

    with pytest.raises(atomic_files.InvalidExistingFileError):
        atomic_files.atomic_write_bytes(
            target,
            b"new",
            validate_existing=_invalid,
        )

    assert target.read_bytes() == b"corrupt"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_backup_rotation_keeps_newest_versions(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    backup_dir = tmp_path / "backups"

    def _validate(data: bytes) -> None:
        assert data.startswith(b"v")

    for version in range(5):
        atomic_files.atomic_write_bytes(
            target,
            f"v{version}".encode(),
            backup_dir=backup_dir,
            keep_backups=2,
            validate_existing=_validate,
        )

    assert target.read_bytes() == b"v4"
    backups = atomic_files.list_backups(backup_dir, target.name)
    assert len(backups) == 2
    assert {item.read_bytes() for item in backups} == {b"v2", b"v3"}


def test_recover_preserves_corrupt_primary(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    backup_dir = tmp_path / "backups"
    target.write_bytes(b"broken")
    backup_dir.mkdir()
    backup = backup_dir / "settings.json.20260906T000000.000000Z.a.bak"
    backup.write_bytes(b"valid")

    def _validate(data: bytes) -> None:
        if data != b"valid":
            raise ValueError("invalid")

    result = atomic_files.recover_latest_valid_backup(
        target, backup_dir, validate=_validate
    )

    assert result is not None
    assert result.backup_path == backup
    assert target.read_bytes() == b"valid"
    assert result.corrupt_path.read_bytes() == b"broken"
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_secrets_concurrent_updates_keep_distinct_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Imported here so the low-level tests above do not depend on Secrets startup.
    from studio import secrets

    monkeypatch.setattr(secrets, "SECRETS_FILE", tmp_path / "secrets.json")

    def _update(index: int) -> None:
        secrets.update(
            {
                "model_sources": {
                    f"test_domain_{index}": [
                        {"kind": "local", "path": str(tmp_path / f"model-{index}")}
                    ]
                }
            }
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_update, range(12)))

    stored = secrets.load()
    assert {
        key for key in stored.model_sources if key.startswith("test_domain_")
    } == {f"test_domain_{index}" for index in range(12)}
