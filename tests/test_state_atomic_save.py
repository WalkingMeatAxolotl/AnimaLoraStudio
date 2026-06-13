"""ADR 0006 Addendum 2 — save_training_state / write_config_snapshot 原子写盘。

auto_epoch_state.pt 是覆盖式单文件恢复点；直接 torch.save 在断电 / 强杀砸中
写盘窗口时会把唯一恢复点写成半截。改 tmp + os.replace 后的不变式：

1. 成功路径：目标文件完整可 load，目录里不残留 .tmp。
2. 写盘中途失败：旧文件原样保留（不被半截新文件污染），tmp 清掉。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from runtime.training.snapshot import write_config_snapshot
from runtime.training.state import save_training_state


class _FakeInjector:
    def state_dict(self):
        return {"w": torch.zeros(2)}


class _FakeOptimizer:
    def state_dict(self):
        return {"lr": 1e-4}


def _save(path: Path) -> None:
    save_training_state(
        path, _FakeInjector(), _FakeOptimizer(), epoch=1, global_step=100,
    )


# ---------------------------------------------------------------------------
# save_training_state
# ---------------------------------------------------------------------------


def test_save_success_no_tmp_leftover(tmp_path: Path) -> None:
    pt = tmp_path / "auto_epoch_state.pt"
    _save(pt)
    assert pt.exists()
    state = torch.load(pt, map_location="cpu", weights_only=False)
    assert state["global_step"] == 100
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_overwrites_previous_atomically(tmp_path: Path) -> None:
    pt = tmp_path / "auto_epoch_state.pt"
    _save(pt)
    save_training_state(
        pt, _FakeInjector(), _FakeOptimizer(), epoch=2, global_step=200,
    )
    state = torch.load(pt, map_location="cpu", weights_only=False)
    assert state["global_step"] == 200
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_failure_keeps_old_file(tmp_path: Path, monkeypatch) -> None:
    """torch.save 半途炸（模拟断电 / 磁盘满）→ 旧恢复点原样保留 + tmp 清掉。"""
    pt = tmp_path / "auto_epoch_state.pt"
    _save(pt)
    before = pt.read_bytes()

    import runtime.training.state as state_mod

    def _boom(obj, path):
        Path(path).write_bytes(b"half-written garbage")
        raise OSError("disk full")

    monkeypatch.setattr(state_mod.torch, "save", _boom)
    with pytest.raises(OSError):
        save_training_state(
            pt, _FakeInjector(), _FakeOptimizer(), epoch=2, global_step=200,
        )

    assert pt.read_bytes() == before  # 旧文件未被污染
    assert list(tmp_path.glob("*.tmp")) == []  # tmp 清掉


# ---------------------------------------------------------------------------
# write_config_snapshot
# ---------------------------------------------------------------------------


def test_snapshot_success_no_tmp_leftover(tmp_path: Path) -> None:
    snap = tmp_path / "auto_epoch_state.config.json"
    write_config_snapshot(snap, {"lr": 1e-4}, ["prompt"])
    payload = json.loads(snap.read_text(encoding="utf-8"))
    assert payload["args"]["lr"] == 1e-4
    assert list(tmp_path.glob("*.tmp")) == []


def test_snapshot_failure_keeps_old_file(tmp_path: Path, monkeypatch) -> None:
    snap = tmp_path / "auto_epoch_state.config.json"
    write_config_snapshot(snap, {"lr": 1e-4})
    before = snap.read_text(encoding="utf-8")

    import runtime.training.snapshot as snap_mod

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(snap_mod.os, "replace", _boom)
    with pytest.raises(OSError):
        write_config_snapshot(snap, {"lr": 5e-5})

    assert snap.read_text(encoding="utf-8") == before
    assert list(tmp_path.glob("*.tmp")) == []
