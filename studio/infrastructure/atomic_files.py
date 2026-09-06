"""Crash-safe file replacement primitives for Studio-owned state.

Writers serialize and fsync a same-directory temporary file before ``os.replace``.
Callers remain responsible for holding the lock that covers their complete
read-modify-write transaction; this module deliberately does not hide domain
locking or validation semantics.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

BytesValidator = Callable[[bytes], None]


class AtomicFileError(RuntimeError):
    """Base class for durable file-operation failures."""


class InvalidExistingFileError(AtomicFileError):
    """The current target failed validation and was not overwritten."""


@dataclass(frozen=True)
class RecoveryResult:
    """A primary file restored from a validated backup."""

    data: bytes
    backup_path: Path
    corrupt_path: Path


def _unique_sibling(path: Path, suffix: str) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}{suffix}")


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync (unsupported by normal Windows handles)."""
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_temp(path: Path, data: bytes, *, mode: Optional[int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _unique_sibling(path, ".tmp")
    try:
        # xb protects against the already-negligible UUID collision and ensures
        # a stale temp file can never be reused.
        with temp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            try:
                os.chmod(temp, mode)
            except OSError:
                # Permission hardening is best-effort across platforms. The
                # durability guarantee must not depend on chmod support.
                pass
        return temp
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _backup_name(path: Path) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{path.name}.{stamp}.{uuid.uuid4().hex}.bak"


def list_backups(backup_dir: Path, target_name: str) -> list[Path]:
    """Return newest-first backups belonging to ``target_name``."""
    if not backup_dir.exists():
        return []
    return sorted(
        backup_dir.glob(f"{target_name}.*.bak"),
        key=lambda item: item.name,
        reverse=True,
    )


def _prune_backups(backup_dir: Path, target_name: str, keep: int) -> None:
    for stale in list_backups(backup_dir, target_name)[max(0, keep):]:
        try:
            stale.unlink()
        except OSError:
            # A successful primary commit must not be reported as failed only
            # because antivirus/indexing temporarily held an old backup.
            pass


def atomic_write_bytes(
    path: Path,
    data: bytes,
    *,
    backup_dir: Optional[Path] = None,
    keep_backups: int = 0,
    validate_existing: Optional[BytesValidator] = None,
    mode: Optional[int] = None,
) -> Optional[Path]:
    """Durably replace ``path`` and optionally archive its valid old bytes.

    ``validate_existing`` is evaluated before the primary is touched. A failed
    validation raises :class:`InvalidExistingFileError`; this prevents callers
    from turning a recoverable corrupt file into an unrecoverable default.

    The caller must hold the relevant domain lock for the entire surrounding
    read-modify-write operation.
    """
    path = Path(path)
    temp = _write_temp(path, data, mode=mode)
    backup_path: Optional[Path] = None
    try:
        previous: Optional[bytes] = None
        if path.exists():
            previous = path.read_bytes()
            if validate_existing is not None:
                try:
                    validate_existing(previous)
                except Exception as exc:
                    raise InvalidExistingFileError(
                        f"Refusing to replace invalid existing file: {path}"
                    ) from exc

        if previous is not None and backup_dir is not None and keep_backups > 0:
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / _backup_name(path)
            backup_temp = _write_temp(backup_path, previous, mode=mode)
            try:
                os.replace(backup_temp, backup_path)
                _fsync_directory(backup_dir)
            finally:
                backup_temp.unlink(missing_ok=True)

        os.replace(temp, path)
        _fsync_directory(path.parent)
        if backup_dir is not None and keep_backups > 0:
            _prune_backups(backup_dir, path.name, keep_backups)
        return backup_path
    finally:
        temp.unlink(missing_ok=True)


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    backup_dir: Optional[Path] = None,
    keep_backups: int = 0,
    validate_existing: Optional[BytesValidator] = None,
    mode: Optional[int] = None,
) -> Optional[Path]:
    """Text wrapper around :func:`atomic_write_bytes`."""
    return atomic_write_bytes(
        path,
        text.encode(encoding),
        backup_dir=backup_dir,
        keep_backups=keep_backups,
        validate_existing=validate_existing,
        mode=mode,
    )


def archive_file(
    path: Path,
    archive_dir: Path,
    *,
    validate_existing: Optional[BytesValidator] = None,
    mode: Optional[int] = None,
    suffix: str = "bak",
) -> Path:
    """Atomically move an existing valid file into an archive directory."""
    path = Path(path)
    data = path.read_bytes()
    if validate_existing is not None:
        try:
            validate_existing(data)
        except Exception as exc:
            raise InvalidExistingFileError(
                f"Refusing to archive invalid existing file: {path}"
            ) from exc
    archive_dir.mkdir(parents=True, exist_ok=True)
    clean_suffix = suffix.strip(".") or "bak"
    if clean_suffix == "bak":
        archive_name = _backup_name(path)
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive_name = f"{path.name}.{stamp}.{uuid.uuid4().hex}.{clean_suffix}"
    archive_path = archive_dir / archive_name
    if mode is not None:
        try:
            os.chmod(path, mode)
        except OSError:
            pass
    os.replace(path, archive_path)
    _fsync_directory(path.parent)
    _fsync_directory(archive_dir)
    return archive_path


def replace_invalid_bytes(
    path: Path,
    data: bytes,
    archive_dir: Path,
    *,
    mode: Optional[int] = None,
) -> Path:
    """Preserve invalid primary bytes, then atomically install replacement bytes."""
    path = Path(path)
    if not path.exists():
        atomic_write_bytes(path, data, mode=mode)
        return path
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    corrupt_path = archive_dir / (
        f"{path.name}.{stamp}.{uuid.uuid4().hex}.corrupt"
    )
    atomic_write_bytes(corrupt_path, path.read_bytes(), mode=mode)
    atomic_write_bytes(path, data, mode=mode)
    return corrupt_path


def recover_latest_valid_backup(
    path: Path,
    backup_dir: Path,
    *,
    validate: BytesValidator,
    mode: Optional[int] = None,
) -> Optional[RecoveryResult]:
    """Restore ``path`` from the newest valid backup, preserving corrupt bytes.

    Returns ``None`` when no backup validates. The corrupt primary remains
    untouched in that case.
    """
    path = Path(path)
    if not path.exists():
        return None

    selected_path: Optional[Path] = None
    selected_data: Optional[bytes] = None
    for candidate in list_backups(backup_dir, path.name):
        try:
            candidate_data = candidate.read_bytes()
            validate(candidate_data)
        except Exception:
            continue
        selected_path = candidate
        selected_data = candidate_data
        break
    if selected_path is None or selected_data is None:
        return None

    corrupt_path = replace_invalid_bytes(
        path, selected_data, backup_dir, mode=mode
    )
    return RecoveryResult(
        data=selected_data,
        backup_path=selected_path,
        corrupt_path=corrupt_path,
    )
