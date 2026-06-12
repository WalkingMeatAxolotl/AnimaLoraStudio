"""argparse_bridge —— pydantic 模型 → argparse parser 的反向生成器测试。"""
from __future__ import annotations

from typing import Literal, Optional

import pytest
from pydantic import BaseModel, Field

from studio.infrastructure import argparse_bridge as bridge
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


def test_bool_field_named_no_x_uses_paired_store_actions() -> None:
    """字段名以 no_ 开头时退化成两个互斥 store_true/store_false flag。

    py3.13+ argparse 拒绝把 --no-X 塞给 BooleanOptionalAction（issue #170）。
    本用例 codify 退化路径：默认值保留 / --no-X 设 True / --X 设 False。
    """
    from studio.schema import TrainingConfig

    parser = bridge.build_parser(TrainingConfig, add_config_arg=False)
    # 默认 True（schema 里 no_progress 默认 True）
    assert parser.parse_args([]).no_progress is True
    # --no-progress 显式打开（向后兼容旧 CLI 习惯）
    assert parser.parse_args(["--no-progress"]).no_progress is True
    # --progress 关闭 → no_progress=False
    assert parser.parse_args(["--progress"]).no_progress is False


def test_help_tolerates_percent_in_description() -> None:
    """description 含裸 `%` 时 format_help 不应崩，且输出仍是单个 `%`。

    argparse 把 description 当 printf 模板做 `% params` 展开，未 escape 的
    `%` 会触发 ValueError。Schema description 同时供 Web UI / i18n 使用，
    不应被 argparse 语义污染 —— bridge 一层兜底转义。
    """
    class _PctSample(BaseModel):
        ratio: float = Field(0.5, description="新值占 90%；越大响应越快")

    parser = bridge.build_parser(_PctSample, add_config_arg=False)
    text = parser.format_help()  # 修复前在 py3.10+ 直接 ValueError
    # 用户看到的仍是单个 %（不是 %%）
    assert "占 90%；" in text
    assert "%%" not in text


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
    assert ns.vae_cache_batch_size == 0
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


def test_training_config_cli_tlora() -> None:
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--lora-type", "tlora",
        "--tlora-min-rank", "12",
        "--tlora-alpha-rank-scale", "1.5",
        "--tlora-use-ortho",
    ])
    assert ns.lora_type == "tlora"
    assert ns.tlora_min_rank == 12
    assert ns.tlora_alpha_rank_scale == 1.5
    assert ns.tlora_use_ortho is True


def test_training_config_cli_ortho() -> None:
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args(["--lora-type", "ortho"])
    assert ns.lora_type == "ortho"


def test_training_config_cli_lion() -> None:
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--optimizer-type", "lion",
        "--lion-beta1", "0.95",
        "--lion-beta2", "0.98",
    ])
    assert ns.optimizer_type == "lion"
    assert ns.lion_beta1 == 0.95
    assert ns.lion_beta2 == 0.98


def test_training_config_cli_automagic() -> None:
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--optimizer-type", "automagic",
        "--automagic-min-lr", "1e-8",
        "--automagic-max-lr", "0.001",
        "--automagic-lr-bump", "2e-6",
        "--automagic-beta2", "0.998",
        "--automagic-clip-threshold", "0.8",
    ])
    assert ns.optimizer_type == "automagic"
    assert ns.automagic_min_lr == 1e-8
    assert ns.automagic_max_lr == 0.001
    assert ns.automagic_lr_bump == 2e-6
    assert ns.automagic_beta2 == 0.998
    assert ns.automagic_clip_threshold == 0.8


def test_training_config_cli_cosine_with_warmup() -> None:
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--lr-scheduler", "cosine_with_warmup",
        "--lr-scheduler-warmup-steps", "25",
        "--lr-scheduler-eta-min", "1e-7",
    ])
    assert ns.lr_scheduler == "cosine_with_warmup"
    assert ns.lr_scheduler_warmup_steps == 25
    assert ns.lr_scheduler_eta_min == 1e-7


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


def test_training_config_cli_ppsf() -> None:
    """ProdigyPlusScheduleFree CLI 字段都能解析出来。"""
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--optimizer-type", "prodigy_plus_schedulefree",
        "--ppsf-d-coef", "0.5",
        "--ppsf-prodigy-steps", "500",
        "--ppsf-beta2", "0.95",
        "--ppsf-use-speed",
        "--no-ppsf-split-groups-mean",
    ])
    assert ns.optimizer_type == "prodigy_plus_schedulefree"
    assert ns.ppsf_d_coef == 0.5
    assert ns.ppsf_prodigy_steps == 500
    assert ns.ppsf_beta2 == 0.95
    assert ns.ppsf_use_speed is True
    assert ns.ppsf_split_groups_mean is False


def test_training_config_yaml_ppsf() -> None:
    """PPSF 字段能通过 YAML 合并到 namespace。"""
    parser = bridge.build_parser(TrainingConfig)
    args = parser.parse_args([])
    yaml_data = {
        "optimizer_type": "prodigy_plus_schedulefree",
        "ppsf_d_coef": 0.3,
        "ppsf_beta1": 0.9,
        "ppsf_beta2": 0.99,
        "ppsf_prodigy_steps": 1000,
        "ppsf_use_stableadamw": True,
    }
    bridge.merge_yaml_into_namespace(args, yaml_data, TrainingConfig)
    assert args.optimizer_type == "prodigy_plus_schedulefree"
    assert args.ppsf_d_coef == 0.3
    assert args.ppsf_beta1 == 0.9
    assert args.ppsf_beta2 == 0.99
    assert args.ppsf_prodigy_steps == 1000
    assert args.ppsf_use_stableadamw is True


# ---------------------------------------------------------------------------
# SqrtZ3 三 PR 新增字段：bridge 自动生成 CLI flag 验证
# ---------------------------------------------------------------------------


def test_training_config_emits_detail_inv_t_flags() -> None:
    """PR #72 引入 detail_inv_t_min/max；bridge 应自动生成 --detail-inv-t-min/--max。"""
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args([
        "--detail-inv-t-min", "1.5",
        "--detail-inv-t-max", "8.0",
    ])
    assert ns.detail_inv_t_min == 1.5
    assert ns.detail_inv_t_max == 8.0


def test_training_config_emits_timestep_mix_low_prob_flag() -> None:
    """PR #73 引入 timestep_mix_low_prob；bridge 应自动生成 --timestep-mix-low-prob。"""
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args(["--timestep-mix-low-prob", "0.25"])
    assert ns.timestep_mix_low_prob == 0.25


def test_training_config_emits_timestep_schedule_shift_flag() -> None:
    """PR #73 引入 timestep_schedule_shift（PR-A 重命名后名）；bridge 应自动生成
    --timestep-schedule-shift。"""
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args(["--timestep-schedule-shift", "1.5"])
    assert ns.timestep_schedule_shift == 1.5


def test_training_config_emits_mixed_uniform_modes() -> None:
    """PR #73 引入两个新 mode；Literal choices 应被 bridge 接受。"""
    parser = bridge.build_parser(TrainingConfig)
    ns_low = parser.parse_args(["--timestep-sampling", "mixed_uniform_low"])
    assert ns_low.timestep_sampling == "mixed_uniform_low"
    ns_logit = parser.parse_args(["--timestep-sampling", "mixed_uniform_logit"])
    assert ns_logit.timestep_sampling == "mixed_uniform_logit"


def test_training_config_emits_loss_type_and_huber_flags() -> None:
    """PR #75 引入 loss_type / huber_c；bridge 应自动生成 --loss-type / --huber-c。"""
    parser = bridge.build_parser(TrainingConfig)
    ns = parser.parse_args(["--loss-type", "huber", "--huber-c", "0.2"])
    assert ns.loss_type == "huber"
    assert ns.huber_c == 0.2
