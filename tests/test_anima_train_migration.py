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
    import importlib.util  # noqa: PLC0415
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "_anima_train_for_test", repo_root / "runtime" / "anima_train.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def test_args_has_t5_tokenizer_path_not_legacy_t5_tokenizer(
    at, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回归：args 字段名是 t5_tokenizer_path，绝不能再出现 args.t5_tokenizer。

    Why: anima_train.py 路径解析阶段曾写成
    `getattr(args, "t5_tokenizer", "")`，访问已迁移走的旧名，恒返回 ""，
    把 yaml/CLI 填好的 t5_tokenizer_path 覆盖成空，导致 T5Tokenizer 静默
    fallback 到联网下载 google/t5-v1_1-xxl —— 离线/弱网环境直接挂。
    """
    monkeypatch.setattr(sys, "argv", [
        "anima_train.py",
        "--t5-tokenizer", "/x/t5",
    ])
    args = at.parse_args()
    assert args.t5_tokenizer_path == "/x/t5"
    assert not hasattr(args, "t5_tokenizer"), (
        "args 不应有 t5_tokenizer 属性 —— schema 字段是 t5_tokenizer_path，"
        "任何 getattr(args, 't5_tokenizer', ...) 都会拿到默认值并清空路径"
    )


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
    """bridge 自动从 prefer_json: bool=True 派生 --prefer-json / --no-prefer-json。

    刀 1 / R1 起 parse_args 是 sparse namespace（只含显式键）；schema 默认值
    由 apply_yaml_config 经 TrainingConfig 统一补齐。
    """
    monkeypatch.setattr(sys, "argv", ["anima_train.py", "--no-prefer-json"])
    assert at.parse_args().prefer_json is False
    monkeypatch.setattr(sys, "argv", ["anima_train.py"])
    sparse = at.parse_args()
    assert not hasattr(sparse, "prefer_json")  # 未显式传 → 不在 sparse namespace
    assert at.apply_yaml_config(sparse, {}).prefer_json is True


# ---------------------------------------------------------------------------
# YAML 合并（apply_yaml_config 返回归一后的新 namespace，不再原地改）
# ---------------------------------------------------------------------------


def test_apply_yaml_overrides_defaults(at, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["anima_train.py"])
    args = at.apply_yaml_config(at.parse_args(), {"epochs": 99, "lora_rank": 64})
    assert args.epochs == 99
    assert args.lora_rank == 64


def test_cli_wins_over_yaml(at, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["anima_train.py", "--epochs", "3"])
    args = at.apply_yaml_config(at.parse_args(), {"epochs": 99})
    assert args.epochs == 3


def test_cli_explicit_default_wins_over_yaml(at, monkeypatch: pytest.MonkeyPatch) -> None:
    """显式传等于 schema 默认的值也算显式（SUPPRESS 精确判定；旧「值==默认值」
    近似的盲区，epochs 默认 10）。"""
    monkeypatch.setattr(sys, "argv", ["anima_train.py", "--epochs", "10"])
    args = at.apply_yaml_config(at.parse_args(), {"epochs": 99})
    assert args.epochs == 10


def test_unknown_yaml_keys_ignored(at, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["anima_train.py"])
    args = at.apply_yaml_config(at.parse_args(), {"this_key_doesnt_exist": 42})
    assert not hasattr(args, "this_key_doesnt_exist")


def test_invalid_yaml_combo_exits(at, monkeypatch: pytest.MonkeyPatch) -> None:
    """互斥组合经统一加载路径 fail-fast（刀 1 前 trainer 静默放行）。"""
    monkeypatch.setattr(sys, "argv", ["anima_train.py"])
    with pytest.raises(SystemExit):
        at.apply_yaml_config(
            at.parse_args(),
            {"infonoise_enabled": True, "loss_weighting": "min_snr"},
        )


# ---------------------------------------------------------------------------
# config/train_template.yaml 已随 CLI 工作流一并删除（Studio 改走 preset 池），
# 这两条 fixture 测试同步移除。Studio 模式下 yaml 由 fork_preset_for_version 注入
# 绝对路径，覆盖在 test_version_config / test_presets_io 等里。
