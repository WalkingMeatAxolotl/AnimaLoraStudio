"""模型根目录迁移服务（studio.services.models_storage）单测。

只测纯函数（scan / validate_target / _run_migration 直调，避免线程 flaky）：
复制 + 更新 secrets.models.root + 失败回滚 + 校验规则 + 单飞。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio.services import models_storage as ms


def _make_src(root: Path) -> Path:
    """造一个有几个文件 + 子目录的源模型根目录。"""
    (root / "diffusion_models").mkdir(parents=True)
    (root / "diffusion_models" / "anima.safetensors").write_bytes(b"x" * 100)
    (root / "vae").mkdir()
    (root / "vae" / "vae.safetensors").write_bytes(b"y" * 50)
    (root / "top.txt").write_text("hi", encoding="utf-8")
    return root


@pytest.fixture
def reset_status():
    ms._set_status(state="idle", target="", error="")
    yield
    ms._set_status(state="idle", target="", error="")


def _fake_secrets_store(monkeypatch, old_root: Path):
    from studio import secrets
    state = {"s": secrets.Secrets(models={"root": str(old_root)})}
    monkeypatch.setattr(secrets, "load", lambda: state["s"])
    monkeypatch.setattr(secrets, "save", lambda s: state.update(s=s))
    return state


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

def test_scan_empty_or_missing_dir(tmp_path: Path) -> None:
    assert ms.scan_models_root(tmp_path / "nope") == {
        "total_files": 0, "total_bytes": 0, "entries": []
    }


def test_scan_counts_files_and_bytes(tmp_path: Path) -> None:
    src = _make_src(tmp_path / "models")
    res = ms.scan_models_root(src)
    assert res["total_files"] == 3
    assert res["total_bytes"] == 100 + 50 + 2
    by_name = {e["name"]: e for e in res["entries"]}
    assert by_name["diffusion_models"]["is_dir"] is True
    assert by_name["diffusion_models"]["files"] == 1
    assert by_name["diffusion_models"]["bytes"] == 100
    assert by_name["top.txt"]["is_dir"] is False


# ---------------------------------------------------------------------------
# validate_target
# ---------------------------------------------------------------------------

def test_validate_returns_models_subdir(tmp_path: Path) -> None:
    src = tmp_path / "cur"
    src.mkdir()
    target = tmp_path / "newroot"
    assert ms.validate_target(target, source=src) == target.resolve() / "models"


def test_validate_rejects_relative(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ms.validate_target(Path("relative/dir"), source=tmp_path / "cur")


def test_validate_rejects_same_as_current(tmp_path: Path) -> None:
    # 当前 root = parent/models；target=parent → dst=parent/models == src
    parent = tmp_path / "p"
    src = parent / "models"
    src.mkdir(parents=True)
    with pytest.raises(ValueError):
        ms.validate_target(parent, source=src)


def test_validate_rejects_nested(tmp_path: Path) -> None:
    src = tmp_path / "cur" / "models"
    src.mkdir(parents=True)
    with pytest.raises(ValueError):
        ms.validate_target(src / "sub", source=src)


def test_validate_rejects_nonempty_dst(tmp_path: Path) -> None:
    src = tmp_path / "cur"
    src.mkdir()
    target = tmp_path / "newroot"
    dst = target / "models"
    dst.mkdir(parents=True)
    (dst / "existing.bin").write_bytes(b"z")
    with pytest.raises(ValueError):
        ms.validate_target(target, source=src)


# ---------------------------------------------------------------------------
# _run_migration（直调，同步）
# ---------------------------------------------------------------------------

def test_run_migration_copies_and_updates_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_status
) -> None:
    src = _make_src(tmp_path / "old" / "models")
    dst = tmp_path / "new" / "models"
    state = _fake_secrets_store(monkeypatch, tmp_path / "old" / "models")
    events: list[dict] = []

    ms._run_migration(src, dst, publish=events.append)

    assert (dst / "diffusion_models" / "anima.safetensors").read_bytes() == b"x" * 100
    assert (dst / "vae" / "vae.safetensors").exists()
    assert (dst / "top.txt").read_text(encoding="utf-8") == "hi"
    # secret 切到新 root（立即生效，无需重启）
    assert state["s"].models.root == str(dst)
    assert any(e["type"] == "models_root_migrate_done" and e["ok"] for e in events)
    assert ms.migration_status()["state"] == "done"


def test_run_migration_rollback_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_status
) -> None:
    src = _make_src(tmp_path / "old" / "models")
    dst = tmp_path / "new" / "models"
    state = _fake_secrets_store(monkeypatch, tmp_path / "old" / "models")
    orig_root = state["s"].models.root

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(ms.shutil, "copy2", boom)

    events: list[dict] = []
    ms._run_migration(src, dst, publish=events.append)

    # dst 整树清掉（回滚），secret 未动
    assert not dst.exists()
    assert state["s"].models.root == orig_root
    assert any(e["type"] == "models_root_migrate_done" and not e["ok"] for e in events)
    assert ms.migration_status()["state"] == "error"


def test_start_migration_single_flight(tmp_path: Path, reset_status) -> None:
    ms._set_status(state="running")
    with pytest.raises(RuntimeError):
        ms.start_migration(tmp_path / "newroot", source=tmp_path / "cur")
