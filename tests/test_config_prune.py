"""config_prune —— evalShowWhen 的 Python 镜像语义 + show_when 落盘裁剪。

求值器语义必须与 studio/web/src/lib/schema.ts 的 evalShowWhen 逐字一致
（含 JS String() 字符串化差异），两边测试用例保持对照。
"""
from __future__ import annotations

from studio.domain.config_prune import eval_show_when, prune_inactive_fields
from studio.schema import TrainingConfig


# ---------------------------------------------------------------------------
# eval_show_when —— 与 schema.test.ts 的 evalShowWhen 用例对照
# ---------------------------------------------------------------------------


def test_empty_expression_is_true() -> None:
    assert eval_show_when(None, {}) is True
    assert eval_show_when("", {}) is True


def test_eq_matching() -> None:
    assert eval_show_when("mode == prodigy", {"mode": "prodigy"}) is True
    assert eval_show_when("mode == prodigy", {"mode": "adamw"}) is False


def test_ne_matching() -> None:
    assert eval_show_when("lr_scheduler != none", {"lr_scheduler": "cosine"}) is True
    assert eval_show_when("lr_scheduler != none", {"lr_scheduler": "none"}) is False


def test_or_branches() -> None:
    expr = "optimizer_type==prodigy||optimizer_type==prodigy_plus_schedulefree"
    assert eval_show_when(expr, {"optimizer_type": "prodigy"}) is True
    assert eval_show_when(expr, {"optimizer_type": "prodigy_plus_schedulefree"}) is True
    assert eval_show_when(expr, {"optimizer_type": "adamw"}) is False


def test_and_conjunction() -> None:
    expr = "optimizer_type==automagic&&automagic_variant==v2"
    assert eval_show_when(expr, {"optimizer_type": "automagic", "automagic_variant": "v2"}) is True
    assert eval_show_when(expr, {"optimizer_type": "automagic", "automagic_variant": "v1"}) is False
    assert eval_show_when(expr, {"optimizer_type": "adamw", "automagic_variant": "v2"}) is False


def test_or_binds_looser_than_and() -> None:
    # JS 先按 || 切分支，每个分支内再按 && 合取
    expr = "a==1&&b==2||c==3"
    assert eval_show_when(expr, {"a": 1, "b": 2, "c": 0}) is True
    assert eval_show_when(expr, {"a": 1, "b": 0, "c": 3}) is True
    assert eval_show_when(expr, {"a": 1, "b": 0, "c": 0}) is False


def test_unparseable_expression_is_true_failsafe() -> None:
    assert eval_show_when("garbage", {}) is True


def test_js_bool_stringification() -> None:
    assert eval_show_when("enabled==true", {"enabled": True}) is True
    assert eval_show_when("enabled==true", {"enabled": False}) is False
    assert eval_show_when("enabled!=true", {"enabled": False}) is True


def test_js_integer_valued_float_stringification() -> None:
    # JS String(1.0) === "1"（training.py 里 timestep_schedule_shift!=1 依赖）
    assert eval_show_when("shift!=1", {"shift": 1.0}) is False
    assert eval_show_when("shift!=1", {"shift": 2.0}) is True
    assert eval_show_when("shift!=1", {"shift": 1.5}) is True


def test_missing_key_is_undefined() -> None:
    # JS values[key] === undefined → String() === "undefined"
    assert eval_show_when("ghost==x", {}) is False
    assert eval_show_when("ghost!=x", {}) is True


# ---------------------------------------------------------------------------
# prune_inactive_fields
# ---------------------------------------------------------------------------


def _dump(**overrides) -> dict:
    return TrainingConfig(**overrides).model_dump(mode="python")


def test_prune_drops_inactive_optimizer_params() -> None:
    pruned = prune_inactive_fields(_dump(optimizer_type="adamw"))
    assert "came_beta1" not in pruned
    assert "lion_beta1" not in pruned
    assert "automagic_min_lr" not in pruned
    assert "prodigy_d_coef" not in pruned


def test_prune_keeps_active_optimizer_params() -> None:
    pruned = prune_inactive_fields(_dump(optimizer_type="came"))
    assert "came_beta1" in pruned
    assert "came_clip_threshold" in pruned
    assert "lion_beta1" not in pruned


def test_prune_drops_stale_hidden_value() -> None:
    # 用户场景：开 came 调了 came_beta1，切回 adamw 后字段不可见 —— 落盘也不该有
    dumped = _dump(optimizer_type="adamw")
    dumped["came_beta1"] = 0.5  # 模拟隐藏字段残留的非默认值
    pruned = prune_inactive_fields(dumped)
    assert "came_beta1" not in pruned


def test_prune_follows_lora_type() -> None:
    assert "lokr_factor" not in prune_inactive_fields(_dump(lora_type="lora"))
    assert "lokr_factor" in prune_inactive_fields(_dump(lora_type="lokr"))


def test_prune_infonoise_subparams() -> None:
    off = prune_inactive_fields(_dump(infonoise_enabled=False))
    assert "infonoise_K" not in off
    assert "infonoise_gate_pivot_c" not in off
    on = prune_inactive_fields(_dump(infonoise_enabled=True))
    assert "infonoise_K" in on


def test_prune_transitive_and_condition() -> None:
    # automagic_agreement_threshold: optimizer_type==automagic&&automagic_variant==v2
    v1 = prune_inactive_fields(_dump(optimizer_type="automagic"))
    assert "automagic_variant" in v1
    assert "automagic_agreement_threshold" not in v1
    v2 = prune_inactive_fields(_dump(optimizer_type="automagic", automagic_variant="v2"))
    assert "automagic_agreement_threshold" in v2


def test_prune_keeps_disable_when_and_plain_fields() -> None:
    """disable_when 字段（值可能被钉在 disable_value ≠ default）与无条件字段不裁。"""
    dumped = _dump(optimizer_type="prodigy")
    pruned = prune_inactive_fields(dumped)
    # lr_scheduler 是 disable_when 字段：Prodigy 下前端钉成 "none"，必须保留
    assert "lr_scheduler" in pruned
    # 无 show_when 的字段全部保留（含 hidden 字段）
    no_show_when = {
        name
        for name, field in TrainingConfig.model_fields.items()
        if not (
            isinstance(field.json_schema_extra, dict)
            and field.json_schema_extra.get("show_when")
        )
    }
    assert no_show_when <= set(pruned)


def test_prune_roundtrip_is_stable() -> None:
    """裁剪后的 yaml 读回（缺失补默认）再裁剪，结果不变 —— 不会越裁越多。

    传 dict 副本：migrate_noise_enhancement_type（mode="before" validator）
    会原地往传入 dict 写互斥清零字段，直接传 pruned 会把比较基准改掉。
    """
    pruned = prune_inactive_fields(_dump(optimizer_type="came", lora_type="lokr"))
    reinflated = TrainingConfig.model_validate(dict(pruned)).model_dump(mode="python")
    assert prune_inactive_fields(reinflated) == pruned
