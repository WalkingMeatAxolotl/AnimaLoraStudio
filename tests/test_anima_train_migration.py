"""anima_train.py 迁移到 schema 驱动后的端到端回归测试。

覆盖：
    - parse_args 通过 bridge 生成的 parser 接受所有历史 CLI 别名
    - apply_yaml_config 把 YAML 字段写入 args 时遵循「CLI 显式优先」语义
    - config/train_template.yaml 能被完整加载，所有字段类型正确

注：这些测试导入 anima_train 模块，会触发 torch import (~3s)，因此用 module
fixture 只导一次。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def at():
    """import anima_train 一次复用。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import anima_train  # noqa: PLC0415
    return anima_train


# ---------------------------------------------------------------------------
# CLI 别名
# ---------------------------------------------------------------------------


def test_legacy_cli_aliases_still_work(at, monkeypatch: pytest.MonkeyPatch) -> None:
    """老脚本里 --transformer / --vae / --qwen / --t5-tokenizer / --lr 必须仍然能用。"""
    monkeypatch.setattr(sys, "argv", [
        "anima_train.py",
        "--transformer", "/x/t.safetensors",
        "--vae", "/x/v.safetensors",
        "--qwen", "/x/q",
        "--t5-tokenizer", "/x/t5",
        "--lr", "5e-5",
    ])
    args = at.parse_args()
    assert args.transformer_path == "/x/t.safetensors"
    assert args.vae_path == "/x/v.safetensors"
    assert args.text_encoder_path == "/x/q"
    assert args.t5_tokenizer_path == "/x/t5"
    assert args.learning_rate == 5e-5


def test_cli_only_flags_present(at, monkeypatch: pytest.MonkeyPatch) -> None:
    """schema 之外的 CLI-only 开关必须保留。"""
    monkeypatch.setattr(sys, "argv", [
        "anima_train.py",
        "--auto-install",
        "--interactive",
        "--no-live-curve",
    ])
    args = at.parse_args()
    assert args.auto_install is True
    assert args.interactive is True
    assert args.no_live_curve is True


def test_deprecated_repeats_flags_silently_accepted(at, monkeypatch: pytest.MonkeyPatch) -> None:
    """--repeats / --reg-repeats 已弃用但仍接受（不破坏旧脚本）。"""
    monkeypatch.setattr(sys, "argv", [
        "anima_train.py", "--repeats", "5", "--reg-repeats", "3",
    ])
    args = at.parse_args()
    assert args.repeats == 5
    assert args.reg_repeats == 3


def test_no_prefer_json_flips_default(at, monkeypatch: pytest.MonkeyPatch) -> None:
    """bridge 自动从 prefer_json: bool=True 派生 --prefer-json / --no-prefer-json。"""
    monkeypatch.setattr(sys, "argv", ["anima_train.py", "--no-prefer-json"])
    assert at.parse_args().prefer_json is False
    monkeypatch.setattr(sys, "argv", ["anima_train.py"])
    assert at.parse_args().prefer_json is True


# ---------------------------------------------------------------------------
# YAML 合并
# ---------------------------------------------------------------------------


def test_apply_yaml_overrides_defaults(at, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["anima_train.py"])
    args = at.parse_args()
    yaml_data = {"epochs": 99, "lora_rank": 64}
    at.apply_yaml_config(args, yaml_data)
    assert args.epochs == 99
    assert args.lora_rank == 64


def test_cli_wins_over_yaml(at, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["anima_train.py", "--epochs", "3"])
    args = at.parse_args()
    at.apply_yaml_config(args, {"epochs": 99})
    assert args.epochs == 3


def test_unknown_yaml_keys_ignored(at, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["anima_train.py"])
    args = at.parse_args()
    at.apply_yaml_config(args, {"this_key_doesnt_exist": 42})
    assert not hasattr(args, "this_key_doesnt_exist")


# ---------------------------------------------------------------------------
# config/train_template.yaml 已随 CLI 工作流一并删除（Studio 改走 preset 池），
# 这两条 fixture 测试同步移除。Studio 模式下 yaml 由 fork_preset_for_version 注入
# 绝对路径，覆盖在 test_version_config / test_presets_io 等里。
