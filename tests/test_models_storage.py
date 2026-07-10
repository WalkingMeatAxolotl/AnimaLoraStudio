"""模型根目录迁移服务（studio.services.models_storage）单测。

只测纯函数（scan / validate_target / _run_migration 直调，避免线程 flaky）：
复制 + 更新 secrets.models.root + 失败回滚 + 校验规则 + 单飞 + 合并模式
（issue #351：目标非空冲突三选，skip/overwrite 及其失败不回滚语义）。
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


def test_validate_conflict_error_with_details(tmp_path: Path) -> None:
    """非空落地目录 + 未指定策略 → TargetConflictError，details 带三项统计。"""
    src = _make_src(tmp_path / "cur")
    target = tmp_path / "newroot"
    dst = target / "models"
    (dst / "diffusion_models").mkdir(parents=True)
    # 与 src 同名（会被 overwrite 覆盖）
    (dst / "diffusion_models" / "anima.safetensors").write_bytes(b"other" * 4)
    # 目标独有
    (dst / "extra.bin").write_bytes(b"e" * 10)
    with pytest.raises(ms.TargetConflictError) as ei:
        ms.validate_target(target, source=src)
    d = ei.value.details
    assert d["target"] == str(target.resolve() / "models")
    assert d["existing_files"] == 2
    assert d["existing_bytes"] == 20 + 10
    assert d["same_name_files"] == 1


def test_validate_allows_nonempty_dst_with_on_conflict(tmp_path: Path) -> None:
    src = tmp_path / "cur"
    src.mkdir()
    target = tmp_path / "newroot"
    dst = target / "models"
    dst.mkdir(parents=True)
    (dst / "existing.bin").write_bytes(b"z")
    for mode in ("skip", "overwrite"):
        assert ms.validate_target(target, source=src, on_conflict=mode) == dst.resolve()


def test_validate_rejects_unknown_on_conflict(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ms.validate_target(tmp_path / "newroot", source=tmp_path / "cur", on_conflict="merge")


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


def test_run_migration_skip_keeps_existing_and_fills_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_status
) -> None:
    """skip：目标同名文件保留原内容，缺的补齐；进度总量只算真复制的。"""
    src = _make_src(tmp_path / "old" / "models")
    dst = tmp_path / "new" / "models"
    (dst / "diffusion_models").mkdir(parents=True)
    (dst / "diffusion_models" / "anima.safetensors").write_bytes(b"theirs")
    state = _fake_secrets_store(monkeypatch, tmp_path / "old" / "models")
    events: list[dict] = []

    ms._run_migration(src, dst, publish=events.append, on_conflict="skip")

    # 同名文件保留目标原有版本
    assert (dst / "diffusion_models" / "anima.safetensors").read_bytes() == b"theirs"
    # 缺失文件补齐
    assert (dst / "vae" / "vae.safetensors").read_bytes() == b"y" * 50
    assert (dst / "top.txt").exists()
    # secret 切换 + 完成事件
    assert state["s"].models.root == str(dst)
    assert any(e["type"] == "models_root_migrate_done" and e["ok"] for e in events)
    status = ms.migration_status()
    assert status["state"] == "done"
    # src 共 3 个文件，1 个被跳过 → 只复制 2 个
    assert status["total_files"] == 2
    # 无临时文件残留
    assert not list(dst.rglob("*.part"))


def test_run_migration_overwrite_replaces_same_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_status
) -> None:
    """overwrite：同名文件换成当前根的副本，目标独有文件不动。"""
    src = _make_src(tmp_path / "old" / "models")
    dst = tmp_path / "new" / "models"
    (dst / "diffusion_models").mkdir(parents=True)
    (dst / "diffusion_models" / "anima.safetensors").write_bytes(b"theirs")
    (dst / "their-extra.bin").write_bytes(b"keep me")
    state = _fake_secrets_store(monkeypatch, tmp_path / "old" / "models")
    events: list[dict] = []

    ms._run_migration(src, dst, publish=events.append, on_conflict="overwrite")

    assert (dst / "diffusion_models" / "anima.safetensors").read_bytes() == b"x" * 100
    assert (dst / "their-extra.bin").read_bytes() == b"keep me"
    assert state["s"].models.root == str(dst)
    assert ms.migration_status()["state"] == "done"
    assert not list(dst.rglob("*.part"))


def test_run_migration_merge_failure_keeps_preexisting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reset_status
) -> None:
    """合并模式失败：目标既有数据绝不 rmtree，secret 不动，无 .part 残留。"""
    src = _make_src(tmp_path / "old" / "models")
    dst = tmp_path / "new" / "models"
    dst.mkdir(parents=True)
    (dst / "their-extra.bin").write_bytes(b"keep me")
    state = _fake_secrets_store(monkeypatch, tmp_path / "old" / "models")
    orig_root = state["s"].models.root

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(ms.shutil, "copy2", boom)

    events: list[dict] = []
    ms._run_migration(src, dst, publish=events.append, on_conflict="skip")

    # 目标目录和既有文件原样保留
    assert (dst / "their-extra.bin").read_bytes() == b"keep me"
    assert state["s"].models.root == orig_root
    assert not list(dst.rglob("*.part"))
    assert any(e["type"] == "models_root_migrate_done" and not e["ok"] for e in events)
    assert ms.migration_status()["state"] == "error"


def test_start_migration_single_flight(tmp_path: Path, reset_status) -> None:
    ms._set_status(state="running")
    with pytest.raises(RuntimeError):
        ms.start_migration(tmp_path / "newroot", source=tmp_path / "cur")
