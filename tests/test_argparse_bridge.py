"""argparse_bridge —— pydantic 模型 → argparse parser 的反向生成器测试。"""
from __future__ import annotations

from typing import Literal, Optional

import pytest
from pydantic import BaseModel, Field

from studio import argparse_bridge as bridge
from studio.schema import TrainingConfig


# ---------------------------------------------------------------------------
# 类型映射 —— 用最小 fixture 模型逐一检查
# ---------------------------------------------------------------------------


class _Sample(BaseModel):
    """覆盖 bridge 支持的所有类型分支。"""
    name: str = Field("alpha")
    count: int = Field(3, ge=0)
    rate: float = Field(0.5)
    enabled: bool = Field(True)
    mode: Literal["a", "b", "c"] = Field("a")
    tags: list[str] = Field(default_factory=list)
    optional_path: Optional[str] = Field(None)


def test_int_field_parsed_as_int() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    ns = parser.parse_args(["--count", "42"])
    assert ns.count == 42 and isinstance(ns.count, int)


def test_float_field_parsed_as_float() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    ns = parser.parse_args(["--rate", "0.125"])
    assert ns.rate == 0.125


def test_bool_uses_paired_flag() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    # 默认 True
    assert parser.parse_args([]).enabled is True
    # --no-enabled 翻转
    assert parser.parse_args(["--no-enabled"]).enabled is False
    # --enabled 显式开启
    assert parser.parse_args(["--enabled"]).enabled is True


def test_literal_emits_choices() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    assert parser.parse_args(["--mode", "b"]).mode == "b"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "x"])


def test_list_uses_nargs_star() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    assert parser.parse_args([]).tags == []
    assert parser.parse_args(["--tags", "a", "b", "c"]).tags == ["a", "b", "c"]


def test_optional_str_default_is_none() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    assert parser.parse_args([]).optional_path is None
    assert parser.parse_args(["--optional-path", "/x"]).optional_path == "/x"


def test_dest_matches_field_name() -> None:
    """dest 必须用下划线（YAML 字段名），而非 CLI 的连字符形式。"""
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    ns = parser.parse_args(["--optional-path", "x"])
    assert hasattr(ns, "optional_path") and not hasattr(ns, "optional-path")


# ---------------------------------------------------------------------------
# CLI alias —— 通过 json_schema_extra 显式声明
# ---------------------------------------------------------------------------


class _Aliased(BaseModel):
    learning_rate: float = Field(1e-4, json_schema_extra={"cli_alias": "--lr"})


def test_cli_alias_overrides_default_flag() -> None:
    parser = bridge.build_parser(_Aliased, add_config_arg=False)
    ns = parser.parse_args(["--lr", "5e-5"])
    assert ns.learning_rate == 5e-5
    # 默认 flag --learning-rate 不应该再存在（不支持）
    with pytest.raises(SystemExit):
        parser.parse_args(["--learning-rate", "5e-5"])


# ---------------------------------------------------------------------------
# YAML 合并语义
# ---------------------------------------------------------------------------


def test_yaml_overrides_when_value_is_default() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    args = parser.parse_args([])
    # count 仍是默认 3，YAML 应当覆盖
    bridge.merge_yaml_into_namespace(args, {"count": 99}, _Sample)
    assert args.count == 99


def test_cli_wins_when_user_set_non_default() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    args = parser.parse_args(["--count", "7"])
    bridge.merge_yaml_into_namespace(args, {"count": 99}, _Sample)
    assert args.count == 7  # CLI 优先


def test_yaml_unknown_keys_ignored() -> None:
    parser = bridge.build_parser(_Sample, add_config_arg=False)
    args = parser.parse_args([])
    bridge.merge_yaml_into_namespace(args, {"this_does_not_exist": 123}, _Sample)
    assert not hasattr(args, "this_does_not_exist")


# ---------------------------------------------------------------------------
# 真实 TrainingConfig 全量自检
# ---------------------------------------------------------------------------


def test_training_config_builds_without_collisions() -> None:
    """全量字段都能注册到一个 parser，没有 dest / flag 冲突。"""
    parser = bridge.build_parser(TrainingConfig)
    # --config 由 add_config_arg=True 自动加
    ns = parser.parse_args([])
    assert hasattr(ns, "config")
    # 抽样字段都能解析出正确类型
    assert ns.lora_rank == 32
    assert ns.lora_type == "lokr"
    assert ns.cache_latents is True
    assert ns.sample_prompts == []
    assert ns.optimizer_type == "adamw"


def test_training_config_cli_smoke() -> None:
    """模拟用户从 CLI 改几个字段。"""
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--lora-rank", "64",
        "--optimizer-type", "prodigy",
        "--no-shuffle-caption",
        "--sample-prompts", "p1", "p2",
    ])
    assert ns.lora_rank == 64
    assert ns.optimizer_type == "prodigy"
    assert ns.shuffle_caption is False
    assert ns.sample_prompts == ["p1", "p2"]


def test_training_config_yaml_round_trip() -> None:
    """走完 CLI → YAML 合并这一条路径，确认 yaml_dict 字段都能被读进 args。"""
    parser = bridge.build_parser(TrainingConfig)
    args = parser.parse_args([])
    yaml_data = {
        "lora_rank": 16,
        "epochs": 3,
        "optimizer_type": "prodigy",
        "prodigy_d_coef": 0.5,
    }
    bridge.merge_yaml_into_namespace(args, yaml_data, TrainingConfig)
    assert args.lora_rank == 16
    assert args.epochs == 3
    assert args.optimizer_type == "prodigy"
    assert args.prodigy_d_coef == 0.5


def test_training_config_help_does_not_crash() -> None:
    """生成 help 文本时不应触发任何 NoneType / 格式错误。"""
    parser = bridge.build_parser(TrainingConfig)
    text = parser.format_help()
    assert "--lora-rank" in text
    assert "--no-cache-latents" in text  # bool 字段的反向 flag
