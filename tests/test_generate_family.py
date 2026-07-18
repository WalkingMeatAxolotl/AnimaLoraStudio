"""Generate / RegAI 按族接线（多模型 P4-4）：sampler 白名单 + Turbo 检测。"""

from __future__ import annotations

import pytest

from studio.domain.generate import GenerateConfig
from studio.domain.reg import RegAiConfig
from studio.services.models.families import get_assets


def test_generate_config_family_sampler_whitelist():
    """每任务临时 config 无 legacy 语料——越族 sampler 直接报错（422）。"""
    GenerateConfig(model_family="krea2", sampler_name="euler",
                   scheduler="krea2_shift")
    with pytest.raises(ValueError, match="er_sde"):
        GenerateConfig(model_family="krea2", sampler_name="er_sde",
                       scheduler="krea2_shift")
    with pytest.raises(ValueError, match="euler"):
        GenerateConfig(model_family="anima", sampler_name="euler")
    with pytest.raises(ValueError, match="krea2_shift"):
        GenerateConfig(model_family="anima", scheduler="krea2_shift")


def test_reg_config_family_sampler_whitelist():
    RegAiConfig(model_family="krea2", sampler_name="euler",
                scheduler="krea2_shift")
    with pytest.raises(ValueError, match="simple"):
        RegAiConfig(model_family="krea2", sampler_name="euler",
                    scheduler="simple")


def test_generate_config_carries_family_and_distilled_to_daemon():
    cfg = GenerateConfig(model_family="krea2", distilled=True,
                         sampler_name="euler", scheduler="krea2_shift")
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
