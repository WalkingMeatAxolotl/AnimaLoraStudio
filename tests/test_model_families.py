"""ModelFamily registry / 派发 / 防回归（多模型 PR-2b）。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from training.families import get_family, get_spec, resolve_family
from training.families.anima import ANIMA_SPEC
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
    with pytest.raises(ValueError):
        resolve_family({"model_family": "krea2"})  # Phase 3 前未注册


def test_family_lora_contract():
    fam = get_family("anima")
    assert fam.lora_metadata() == {"model_family": "anima"}
    sd = {"k": torch.zeros(1)}
    assert fam.convert_lora_state_dict(sd) is sd  # 恒等（04 §7.1）
    assert fam.lora_preset()["lora_prefix"] == ANIMA_SPEC.lora.prefix
    assert fam.prepare_text_cache([], []) is None  # online 族 no-op


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
