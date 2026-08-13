"""Generate / RegAI 按族接线（多模型 P4-4）：sampler 白名单 + Turbo 检测。"""

from __future__ import annotations

import pytest

from studio.domain.generate import GenerateConfig
from studio.domain.reg import RegAiConfig
from studio.services.models.families import get_assets


def test_generate_config_family_sampler_whitelist():
    """每任务临时 config 无 legacy 语料——越族 sampler 直接报错（422）。"""
    GenerateConfig(model_family="krea2", sampler_name="euler",
                   scheduler="simple")
    with pytest.raises(ValueError, match="er_sde"):
        GenerateConfig(model_family="krea2", sampler_name="er_sde",
                       scheduler="simple")
    with pytest.raises(ValueError, match="euler"):
        GenerateConfig(model_family="anima", sampler_name="euler")
    with pytest.raises(ValueError, match="sgm_uniform"):
        GenerateConfig(model_family="krea2", sampler_name="euler",
                       scheduler="sgm_uniform")


def test_reg_config_family_sampler_whitelist():
    RegAiConfig(model_family="krea2", sampler_name="euler",
                scheduler="simple")
    with pytest.raises(ValueError, match="sgm_uniform"):
        RegAiConfig(model_family="krea2", sampler_name="euler",
                    scheduler="sgm_uniform")


def test_generate_config_carries_family_and_distilled_to_daemon():
    cfg = GenerateConfig(model_family="krea2", distilled=True,
                         sampler_name="euler", scheduler="simple")
    dumped = cfg.model_dump()
    assert dumped["model_family"] == "krea2"
    assert dumped["distilled"] is True


def test_is_distilled_path_by_official_variant():
    """Turbo 与 Raw 结构全等，loader 指纹无法区分——只能按 catalog 文件名判。"""
    krea2 = get_assets("krea2")
    assert krea2.is_distilled_path("G:/models/diffusion_models/krea2-turbo-bf16.safetensors")
    # 官方 fp8 Turbo 同为蒸馏推理靶子——测试页选中自动应用 8 步/无 CFG
    assert krea2.is_distilled_path("G:/models/diffusion_models/krea2-turbo-fp8-scaled.safetensors")
    assert not krea2.is_distilled_path("G:/models/diffusion_models/krea2-raw-bf16.safetensors")
    # fp8 Raw 是非蒸馏训练/推理底模——绝不能被 purpose 逻辑误判成 Turbo
    assert not krea2.is_distilled_path("G:/models/diffusion_models/krea2-raw-fp8-scaled.safetensors")
    # custom 权重无 purpose 元数据 → 非蒸馏（A1：不加白名单，参数用户控制）
    assert not krea2.is_distilled_path("G:/models/my-community-turbo-mix.safetensors")
    assert not krea2.is_distilled_path("")
    assert not get_assets("anima").is_distilled_path(
        "G:/models/diffusion_models/anima-base-v1.0.safetensors")


# ── 族条件的运行时旋钮门控 ──────────────────────────────────────────────────
# blocks_to_swap 来自**全局**出图设置（用户为某个模型调的），但每次请求可以是另一
# 个族的底模。原样透传时 runtime 会 fail-fast（"model_family=X 不支持 block swap"）。
# schema 的写时门控走 cap_gate，运行时这条走 supports_capability —— 同一张
# FAMILY_CAPABILITIES 表，不是第二份镜像。

def test_supports_capability_matches_the_family_table():
    from studio.domain.common import FAMILY_CAPABILITIES, supports_capability

    assert supports_capability("krea2", "block_swap") is True
    assert supports_capability("anima", "block_swap") is True
    # 反例：anima 不支持 text_cache（online 族），机制必须能说「不」
    assert supports_capability("anima", "text_cache") is False
    # 未知族保守拒绝，不放行族条件旋钮
    assert supports_capability("no_such_family", "block_swap") is False
    # 与表本身同源（防有人再抄一份镜像）
    for family, caps in FAMILY_CAPABILITIES.items():
        for cap in caps:
            assert supports_capability(family, cap) is True


def test_eval_and_generate_gate_block_swap_the_same_way(monkeypatch: pytest.MonkeyPatch):
    """评估和测试出图共用同一条规则 —— 两处各写一份迟早会漂。"""
    from studio.domain.common import supports_capability
    from studio.services import eval_generation

    class _Gen:
        vae_precision = "bf16"
        lora_merge_precision = "fp32"
        vram_policy = "auto"
        ram_guard = True
        blocks_to_swap = 14

    class _Secrets:
        generate = _Gen()

    monkeypatch.setattr("studio.secrets.load", lambda: _Secrets())
    eval_generation._BLOCK_SWAP_NOTICED.discard("no_such_family")

    # 两族现在都支持 block swap，全局层数原样透传
    assert eval_generation._generate_settings("anima")["blocks_to_swap"] == 14
    assert eval_generation._generate_settings("krea2")["blocks_to_swap"] == 14
    # 过滤机制本身仍要能说「不」：未知族保守置 0
    assert eval_generation._generate_settings("no_such_family")["blocks_to_swap"] == 0
    # generate 路由用的是同一个判据
    assert supports_capability("anima", "block_swap") is True
    assert supports_capability("krea2", "block_swap") is True
