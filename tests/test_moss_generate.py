"""MOSS-Image 出图 / 正则接入（PR-4c）。

- GenerateConfig / RegAiConfig 接受 moss；采样白名单按族
- Skip Layer Guidance 三个旋钮：只有声明 ``skip_layer_guidance`` 能力位的族能带
- 出图请求的 TE variant 覆盖与 token 计数走族资产声明（不按族名分支）
- MossFamily.sample_image 接受统一协议的 ``distilled``，SLG 透传到采样循环
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from studio.domain import GenerateConfig, RegAiConfig
from studio.domain.common import supports_capability
from studio.domain.generate import skip_layer_guidance_kwargs


def _gen(**kw) -> GenerateConfig:
    base = {"model_family": "moss", "sampler_name": "euler", "scheduler": "simple"}
    return GenerateConfig(**{**base, **kw})


# ── schema ──────────────────────────────────────────────────────────────────

def test_generate_config_accepts_moss_with_family_samplers() -> None:
    assert _gen().model_family == "moss"
    assert _gen(sampler_name="er_sde").sampler_name == "er_sde"
    with pytest.raises(ValidationError, match="model_family='moss'"):
        _gen(sampler_name="dpmpp_3m_sde")


def test_reg_config_accepts_moss() -> None:
    cfg = RegAiConfig(model_family="moss", sampler_name="er_sde", scheduler="simple")
    assert cfg.model_family == "moss"


def test_skip_layer_guidance_only_for_capable_family() -> None:
    assert supports_capability("moss", "skip_layer_guidance")
    cfg = _gen(uncond_skip_layers=[22], uncond_skip_start_percent=0.0, uncond_skip_end_percent=0.5)
    assert cfg.uncond_skip_layers == [22]
    for family, sampler in (("anima", "er_sde"), ("krea2", "euler")):
        assert not supports_capability(family, "skip_layer_guidance")
        with pytest.raises(ValidationError, match="Skip Layer Guidance"):
            GenerateConfig(model_family=family, sampler_name=sampler, uncond_skip_layers=[22])


def test_skip_layer_guidance_value_checks() -> None:
    with pytest.raises(ValidationError, match="不能大于"):
        _gen(uncond_skip_start_percent=0.6, uncond_skip_end_percent=0.5)
    with pytest.raises(ValidationError, match="非负"):
        _gen(uncond_skip_layers=[-1])
    with pytest.raises(ValidationError):
        _gen(uncond_skip_end_percent=1.5)
    # 空列表 = 显式关闭，合法
    assert _gen(uncond_skip_layers=[]).uncond_skip_layers == []


def test_skip_layer_guidance_kwargs_drops_unset() -> None:
    cfg = _gen(uncond_skip_layers=[], uncond_skip_end_percent=0.5).model_dump()
    assert skip_layer_guidance_kwargs(cfg) == {
        "uncond_skip_layers": [], "uncond_skip_end_percent": 0.5,
    }
    assert skip_layer_guidance_kwargs(GenerateConfig().model_dump()) == {}


# ── 路由：TE variant 覆盖 / token 计数 ────────────────────────────────────────

@pytest.fixture
def models_root(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr("studio.services.models.paths.models_root", lambda: tmp_path)
    return tmp_path


def test_text_encoder_override_moss(models_root) -> None:
    from studio.api.routers.generate import _text_encoder_override
    from studio.services.models.families.moss import moss_te_dir

    with pytest.raises(HTTPException) as e:
        _text_encoder_override("moss", "int8")
    assert e.value.status_code == 409 and "MOSS-Image" in e.value.detail

    te = moss_te_dir(models_root, "int8")
    te.mkdir(parents=True)
    (te / "config.json").write_text("{}", encoding="utf-8")
    assert _text_encoder_override("moss", "int8") == str(te)


@pytest.mark.parametrize("family,variant", [("moss", "fp8"), ("krea2", "int8"), ("anima", "bf16")])
def test_text_encoder_override_rejects_foreign_variant(models_root, family, variant) -> None:
    from studio.api.routers.generate import _text_encoder_override

    with pytest.raises(HTTPException) as e:
        _text_encoder_override(family, variant)
    assert e.value.status_code == 422


def test_text_encoder_override_krea2_fp8_still_checks_download(models_root) -> None:
    from studio.api.routers.generate import _text_encoder_override
    from studio.services.models.families.krea2 import qwen3_vl_dir_for

    with pytest.raises(HTTPException) as e:
        _text_encoder_override("krea2", "fp8")
    assert e.value.status_code == 409
    te = qwen3_vl_dir_for(models_root, "fp8")
    te.mkdir(parents=True)
    (te / "config.json").write_text("{}", encoding="utf-8")
    assert _text_encoder_override("krea2", "fp8") == str(te)


def test_token_count_uses_family_selected_text_encoder(models_root, monkeypatch) -> None:
    from studio.api.routers import generate as router
    from studio.services.models.families import get_assets

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    seen: list[str] = []
    for family in ("anima", "krea2", "moss"):
        assets = get_assets(family)
        tok_dir = str(assets.text_encoder_dir(models_root, assets.selected_text_encoder()))
        seen.append(tok_dir)
        monkeypatch.setitem(router._TOKENIZER_CACHE, tok_dir, _Tok())
        assert router.count_prompt_tokens({"text": "a b c", "model_family": family}) == {"tokens": 3}
    assert len(set(seen)) == 3  # 三族各取自己的 TE 目录
    assert seen[2].endswith("moss-qwen3.5-9b-int8")


# ── runtime：MossFamily.sample_image ────────────────────────────────────────

def test_moss_sample_image_accepts_distilled_and_forwards_slg(monkeypatch) -> None:
    from training.families import moss as moss_family

    captured: dict = {}

    def fake_prepare(text, prompt, **kw):
        return object()

    def fake_sample_latents(model, condition, **kw):
        captured.update(kw)
        return "latents"

    monkeypatch.setattr(moss_family, "prepare_sampling_condition", fake_prepare)
    monkeypatch.setattr(moss_family, "sample_latents", fake_sample_latents)
    monkeypatch.setattr(moss_family, "decode_to_pil", lambda vae, latents: ("img", latents))

    out = moss_family.MossFamily().sample_image(
        None, None, object(), "prompt",
        steps=4, cfg_scale=3.0, sampler_name="euler", scheduler="simple",
        distilled=False, vram_policy="auto", device="cpu",
        uncond_skip_layers=[7, 8], uncond_skip_start_percent=0.0, uncond_skip_end_percent=0.5,
    )
    assert out == ("img", "latents")
    assert captured["uncond_skip_layers"] == [7, 8]
    assert captured["uncond_skip_end_percent"] == 0.5
    assert captured["steps"] == 4


# ── 出图挂 LoRA：int8 底模只放行 bypass 能正确前向的算法 ──────────────────────

def _int8_model():
    import torch
    from torch import nn

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            lin = nn.Linear(4, 4, bias=False)
            lin.weight = nn.Parameter(torch.zeros(4, 4, dtype=torch.int8), requires_grad=False)
            self.layers = nn.ModuleList([nn.ModuleDict({"proj": lin})])

    return _Tiny()


def _write_lora(tmp_path, network_args: dict) -> str:
    import json

    import torch
    from safetensors.torch import save_file

    prefix = "lora_unet_layers_0_proj"
    sd = {
        f"{prefix}.lora_up.weight": torch.zeros(4, 2),
        f"{prefix}.lora_down.weight": torch.zeros(2, 4),
        f"{prefix}.alpha": torch.tensor(2.0),
    }
    path = tmp_path / "moss_lora.safetensors"
    save_file(sd, str(path), metadata={
        "ss_network_dim": "2", "ss_network_alpha": "2.0",
        "ss_network_args": json.dumps({"model_family": "moss", **network_args}),
    })
    return str(path)


@pytest.mark.parametrize("network_args,needle", [
    ({"algo": "lora", "weight_decompose": True}, r"lora \+ DoRA"),
    ({"algo": "full"}, "是 full"),
])
def test_apply_loras_int8_base_rejects_non_bypass_algos(tmp_path, network_args, needle) -> None:
    import torch

    from studio.services.inference.core import LoRASpec, apply_loras

    path = _write_lora(tmp_path, network_args)
    with pytest.raises(ValueError, match=needle):
        apply_loras(_int8_model(), [LoRASpec(path=path, scale=1.0)], "cpu", torch.float32,
                    family_id="moss")
