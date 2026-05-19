"""runtime/training/snapshot.py — pause helpers 单测（ADR 0006 PR-2）。"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import pytest

from runtime.training.snapshot import (  # type: ignore[import-not-found]
    EVENT_MARKER,
    build_pause_config_path,
    build_pause_state_path,
    emit_event,
    write_config_snapshot,
    _jsonify,
)


# ---- 路径 build --------------------------------------------------------------


def test_build_pause_state_path_uses_pause_prefix() -> None:
    sd = Path("/x/state/task_42")
    p = build_pause_state_path(sd, 100)
    assert p.name == "pause_step_100.pt"
    assert p.parent == sd


def test_build_pause_config_path_pairs_state() -> None:
    sd = Path("/x/state/task_42")
    assert build_pause_config_path(sd, 100).name == "pause_step_100.config.json"


def test_state_and_config_path_share_stem_step() -> None:
    """同 step 的 state 和 config 必须在同目录、文件名同前缀。"""
    sd = Path("/x/state/task_7")
    state = build_pause_state_path(sd, 500)
    cfg = build_pause_config_path(sd, 500)
    assert state.parent == cfg.parent
    assert state.stem == cfg.stem.split(".")[0]


# ---- _jsonify ----------------------------------------------------------------


def test_jsonify_path_to_str() -> None:
    p = Path("a/b/c")
    assert _jsonify(p) == str(p)


def test_jsonify_set_to_sorted_list() -> None:
    assert _jsonify({3, 1, 2}) == [1, 2, 3]


def test_jsonify_nested_dict_with_paths() -> None:
    out = _jsonify({"k": Path("v"), "n": 1, "ls": [Path("x")]})
    assert out["k"] == str(Path("v"))
    assert out["n"] == 1
    assert out["ls"] == [str(Path("x"))]


def test_jsonify_unknown_type_falls_back_to_repr() -> None:
    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    assert _jsonify(Weird()) == "<weird>"


def test_jsonify_none_passes_through() -> None:
    assert _jsonify(None) is None


# ---- write_config_snapshot ---------------------------------------------------


def test_write_config_snapshot_serializes_args_namespace(tmp_path: Path) -> None:
    args = argparse.Namespace(
        lr=1e-4, optimizer="AdamW", batch_size=4,
        output_dir=tmp_path,
        loss_type="mse",
    )
    target = tmp_path / "snap.config.json"
    write_config_snapshot(target, args, ["prompt1", "prompt2"])

    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["args"]["lr"] == 1e-4
    assert data["args"]["optimizer"] == "AdamW"
    assert data["args"]["batch_size"] == 4
    assert isinstance(data["args"]["output_dir"], str)  # Path → str
    assert data["sample_prompts"] == ["prompt1", "prompt2"]


def test_write_config_snapshot_handles_dict_args(tmp_path: Path) -> None:
    args = {"lr": 1.0, "name": "test"}
    target = tmp_path / "snap.json"
    write_config_snapshot(target, args, None)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["args"]["lr"] == 1.0
    assert data["sample_prompts"] == []


def test_write_config_snapshot_creates_parent_dir(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "snap.json"
    args = argparse.Namespace(x=1)
    write_config_snapshot(target, args, [])
    assert target.exists()


def test_write_config_snapshot_utf8_encoding(tmp_path: Path) -> None:
    args = argparse.Namespace(name="中文_名字", prompt="anime, 1girl")
    target = tmp_path / "snap.json"
    write_config_snapshot(target, args, ["1girl，masterpiece"])
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["args"]["name"] == "中文_名字"
    assert data["sample_prompts"] == ["1girl，masterpiece"]


# ---- emit_event --------------------------------------------------------------


def test_emit_event_writes_marker_protocol(capsys: pytest.CaptureFixture) -> None:
    emit_event("pause_state", {"step": 100, "state_path": "/x"})
    out = capsys.readouterr().out.strip()
    assert out.startswith(EVENT_MARKER)
    assert out.startswith(f"{EVENT_MARKER}pause_state:")
    payload = json.loads(out[len(f"{EVENT_MARKER}pause_state:"):])
    assert payload == {"step": 100, "state_path": "/x"}


def test_emit_event_empty_payload(capsys: pytest.CaptureFixture) -> None:
    emit_event("train_loop_started")
    out = capsys.readouterr().out.strip()
    assert out == f"{EVENT_MARKER}train_loop_started:{{}}"


def test_event_marker_constant_matches_supervisor() -> None:
    """跨进程 IPC 协议字面量；改 = breaking change。"""
    assert EVENT_MARKER == "__EVENT__:"
