"""ModelFamily registry / 派发 / 防回归（多模型 PR-2b）。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from training.families import get_family, get_spec, resolve_family
from training.families.anima import ANIMA_SPEC
from training.families.krea2 import KREA2_SPEC
from training.families.protocol import ModelFamily

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_get_family_returns_singleton_with_spec():
    fam = get_family("anima")
    assert fam is get_family("anima")
    assert fam.spec is ANIMA_SPEC is get_spec("anima")
    assert isinstance(fam, ModelFamily)


def test_get_family_unknown_lists_registered():
    with pytest.raises(ValueError, match="anima"):
        get_family("no-such-family")


def test_resolve_family_defaults_and_carriers():
    # Namespace 缺字段 / dict 缺键 / dict 显式指定 三种载体（D7 零迁移）
    assert resolve_family(SimpleNamespace()).spec.family_id == "anima"
    assert resolve_family({}).spec.family_id == "anima"
    assert resolve_family({"model_family": "anima"}).spec.family_id == "anima"
    assert resolve_family({"model_family": "krea2"}).spec is KREA2_SPEC


def test_family_lora_contract():
    fam = get_family("anima")
    assert fam.lora_metadata() == {
        "model_family": "anima",
        "preset": ANIMA_SPEC.lora.preset_name,
    }
    sd = {"k": torch.zeros(1)}
    assert fam.convert_lora_state_dict(sd) is sd  # 恒等（04 §7.1）
    assert fam.lora_preset()["lora_prefix"] == ANIMA_SPEC.lora.prefix
    assert fam.prepare_text_cache([], []) is None  # online 族 no-op

    krea2 = get_family("krea2")
    assert krea2 is get_family("krea2")
    assert krea2.spec is KREA2_SPEC is get_spec("krea2")
    assert isinstance(krea2, ModelFamily)
    assert krea2.lora_preset()["lora_prefix"] == "lora_unet"
    assert krea2.lora_metadata() == {
        "model_family": "krea2",
        "preset": "krea2_full",
    }


def test_krea2_and_anima_share_latent_space_identity():
    # 同一性而非相等性：两族引用 latent_spaces.WAN21_F8C16 同一实例，
    # D6 的缓存跨族共享是结构事实，不靠副本 + 相等断言维持。
    assert KREA2_SPEC.latent is ANIMA_SPEC.latent


def test_krea2_forward_train_passes_varlen_mask_and_checkpoint():
    class _Condition:
        context = torch.zeros(2, 7, 12, 2560)
        attention_mask = torch.ones(2, 7, dtype=torch.bool)

    class _FakeDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, noisy, t, context, *, attention_mask, use_checkpoint):
            self.seen = (t, context, attention_mask, use_checkpoint)
            return noisy

    fam = get_family("krea2")
    dit = _FakeDiT()
    noisy = torch.zeros(2, 16, 1, 8, 8)
    t = torch.rand(2)
    out = fam.forward_train(dit, noisy, t, _Condition(), use_checkpoint=True)
    assert out.shape == noisy.shape
    assert dit.seen[0] is t
    assert dit.seen[1] is _Condition.context
    assert dit.seen[2] is _Condition.attention_mask
    assert dit.seen[3] is True


def test_forward_train_shapes_and_padding_mask():
    """pad_mask 构造与 t 形状按摩收进族内（03-③）。"""

    class _FakeDiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = {}

        def forward(self, latents, timesteps, cross, padding_mask=None):
            self.seen["t_shape"] = tuple(timesteps.shape)
            self.seen["pad_shape"] = tuple(padding_mask.shape)
            return latents

    fam = get_family("anima")
    dit = _FakeDiT()
    noisy = torch.zeros(2, 16, 1, 8, 6)
    t = torch.rand(2)
    out = fam.forward_train(dit, noisy, t, cond=None, use_checkpoint=False)
    assert out.shape == noisy.shape  # v_pred 同形（不变量 #3）
    assert dit.seen["t_shape"] == (2, 1)
    assert dit.seen["pad_shape"] == (2, 1, 8, 6)


def test_no_direct_loader_literals_in_dispatch_sites():
    """防回归（01 §9）：派发点不得再出现绕过 family 的直调字面量。"""
    phases_models = (REPO_ROOT / "runtime/training/phases/models.py").read_text(encoding="utf-8")
    assert "load_anima_model(" not in phases_models
    assert "load_text_encoders(" not in phases_models

    loop = (REPO_ROOT / "runtime/training/loop.py").read_text(encoding="utf-8")
    assert "preprocess_text_embeds" not in loop  # 文本块已下沉 family
    assert "forward_with_optional_checkpoint(" not in loop  # 前向经 family


def test_krea2_load_text_generate_fixes_te_fp16_storage_fp32_compute(monkeypatch):
    """generate 场景 TE 固定 fp16 存储 + fp32 compute（ComfyUI sd.py:258
    口径，忽略调用方 dtype、无旋钮）；训练路径维持调用方 dtype，不设
    compute_dtype。"""
    import training.families.krea2 as krea2_module

    calls: list[dict] = []
    monkeypatch.setattr(
        krea2_module, "load_krea2_text_stack",
        lambda path, **kwargs: calls.append(kwargs) or "stack",
    )
    family = krea2_module.Krea2Family()

    assert family.load_text(
        "te-dir", "cpu", torch.bfloat16, purpose="generate", cache_enabled=False,
    ) == "stack"
    assert calls[0]["dtype"] == torch.float16
    assert calls[0]["compute_dtype"] == torch.float32
    assert calls[0]["cache_enabled"] is False

    assert family.load_text("te-dir", "cpu", torch.bfloat16) == "stack"
    assert calls[1]["dtype"] == torch.bfloat16
    assert "compute_dtype" not in calls[1]


def test_anima_load_text_purpose_never_overrides_backend_choice(monkeypatch):
    """purpose 对 Anima 接受并忽略（load_dit 同款）：generate 调用面传
    purpose 不得把用户显式选择的 hf backend 静默切成 comfy_qwen。"""
    import training.families.anima.loader as anima_loader

    calls: list[dict] = []
    monkeypatch.setattr(
        anima_loader, "load_text_encoders",
        lambda *args, **kwargs: calls.append(kwargs) or "encoders",
    )

    fam = get_family("anima")
    assert fam.load_text(
        "te-dir", "cpu", torch.bfloat16, comfy_qwen=False, purpose="generate",
    ) == "encoders"
    assert calls[0]["comfy_qwen"] is False
