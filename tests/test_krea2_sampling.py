"""Krea2 FlowMatchEuler schedule, CFG convention, and image orchestration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modeling.krea2 import Krea2Config, SingleStreamDiT
from training.families.krea2.sampling import (
    KREA2_RAW_GUIDANCE,
    KREA2_RAW_STEPS,
    KREA2_TURBO_GUIDANCE,
    KREA2_TURBO_STEPS,
    Krea2SamplingCondition,
    align_krea2_resolution,
    build_krea2_sigmas,
    calculate_krea2_mu,
    prepare_sampling_condition,
    resolve_sampling_settings,
    sample_image,
    sample_latents,
)
from training.families.krea2.text_encoding import Krea2TextCondition


def _condition(value: float, *, layers=2, width=4) -> Krea2TextCondition:
    return Krea2TextCondition(
        context=torch.full((1, 3, layers, width), value),
        attention_mask=torch.tensor([[True, True, False]]),
    )


class _VelocityModel(nn.Module):
    def __init__(self, *, channels=2, patch=2):
        super().__init__()
        self.config = SimpleNamespace(channels=channels, patch=patch)
        self.calls = []

    def forward(self, x, timesteps, context, attention_mask=None):
        value = float(context[0, 0, 0, 0])
        self.calls.append((value, timesteps.detach().clone(), attention_mask.clone()))
        return torch.full_like(x, value)


class _FakeVAE:
    def decode(self, latents):
        batch, _channels, frames, height, width = latents.shape
        return torch.zeros(batch, 3, frames, height * 8, width * 8)


def test_mu_matches_reference_endpoints_and_1024_anchor():
    assert calculate_krea2_mu(256) == pytest.approx(0.5)
    assert calculate_krea2_mu(4096) == pytest.approx(0.90625)
    assert calculate_krea2_mu(6400) == pytest.approx(1.15)
    assert calculate_krea2_mu(8000) > 1.15  # upstream does not endpoint-clamp


def test_default_sigmas_match_comfy_flux_time_shift():
    """默认 sigma = Comfy parity 口径：固定 mu=1.15，与 ComfyUI
    ModelSamplingFlux 的 flux_time_shift(1.15, 1.0, t) 逐位恒等，且与
    分辨率无关（Raw / Turbo / 任何 image_seq_len 同一张表）。"""
    sigmas = build_krea2_sigmas(4096, 4)
    t = torch.linspace(1.0, 0.0, 5)
    mu = torch.tensor(1.15)
    # flux_time_shift: exp(mu) / (exp(mu) + (1/t - 1)^1)；t=0 时定义为 0
    expected = torch.where(
        t > 0, mu.exp() / (mu.exp() + (1.0 / t.clamp_min(1e-12) - 1.0)),
        torch.zeros(()),
    )
    torch.testing.assert_close(sigmas, expected)
    assert sigmas[0] == 1
    assert sigmas[-1] == 0
    assert torch.all(sigmas[:-1] > sigmas[1:])
    # 分辨率无关 + distilled 不再影响 sigma
    torch.testing.assert_close(sigmas, build_krea2_sigmas(256, 4))
    torch.testing.assert_close(sigmas, build_krea2_sigmas(4096, 4, distilled=True))


def test_dynamic_mu_preserved_as_non_default():
    """diffusers/musubi 的分辨率感知动态 mu 保留为 opt-in 对照路径。"""
    dynamic = build_krea2_sigmas(4096, 4, dynamic_mu=True)
    base = torch.linspace(1.0, 0.0, 5)
    shift = torch.tensor(0.90625).exp()  # calculate_krea2_mu(4096)
    expected = shift * base / (1 + (shift - 1) * base)
    torch.testing.assert_close(dynamic, expected)
    # 动态路径分辨率相关；与默认固定口径不同
    assert not torch.equal(dynamic, build_krea2_sigmas(256, 4, dynamic_mu=True))
    assert not torch.equal(dynamic, build_krea2_sigmas(4096, 4))


def test_raw_and_turbo_defaults_and_sampler_validation():
    assert resolve_sampling_settings(
        distilled=False, steps=None, cfg_scale=None,
        sampler_name=None, scheduler=None,
    ) == (KREA2_RAW_STEPS, KREA2_RAW_GUIDANCE)
    assert resolve_sampling_settings(
        distilled=True, steps=None, cfg_scale=None,
        sampler_name=None, scheduler=None,
    ) == (KREA2_TURBO_STEPS, KREA2_TURBO_GUIDANCE)

    with pytest.raises(ValueError, match="仅支持"):
        resolve_sampling_settings(
            distilled=False, steps=28, cfg_scale=4.5,
            sampler_name="er_sde", scheduler="simple",
        )


def test_prepare_condition_only_encodes_negative_when_guidance_enabled():
    calls = []
    phases = []

    class FakeTextStack:
        def encode_text_for_batch(self, captions, *, device, dtype):
            calls.append(list(captions))
            batch = len(captions)
            return Krea2TextCondition(
                context=torch.arange(batch, dtype=dtype).view(batch, 1, 1, 1),
                attention_mask=torch.ones(batch, 1, dtype=torch.bool),
            )

    guided = prepare_sampling_condition(
        FakeTextStack(), "positive", negative_prompt=None, cfg_scale=4.5,
        device="cpu", dtype=torch.float32, phase_callback=phases.append,
    )
    unguided = prepare_sampling_condition(
        FakeTextStack(), "turbo", negative_prompt="ignored", cfg_scale=0,
        device="cpu", dtype=torch.float32,
    )

    assert calls == [["positive", ""], ["turbo"]]
    assert phases == ["clip"]
    assert guided.negative is not None
    assert unguided.negative is None


def test_euler_uses_krea_guidance_and_reports_denoised_preview():
    model = _VelocityModel()
    condition = Krea2SamplingCondition(
        positive=_condition(2.0),
        negative=_condition(1.0),
    )
    initial = torch.zeros(1, 2, 1, 2, 2)
    previews = []

    output = sample_latents(
        model,
        condition,
        height=16,
        width=16,
        steps=1,
        cfg_scale=4.5,
        device="cpu",
        dtype=torch.float32,
        latents=initial,
        step_callback=lambda step, total, denoised: previews.append(
            (step, total, denoised.clone())
        ),
    )

    # cond + 4.5 * (cond - uncond) = 2 + 4.5 * (2 - 1) = 6.5
    torch.testing.assert_close(output, torch.full_like(initial, -6.5))
    torch.testing.assert_close(previews[0][2], torch.full_like(initial, -6.5))
    assert previews[0][:2] == (0, 1)
    assert [call[0] for call in model.calls] == [2.0, 1.0]
    assert model.training  # caller's original mode is restored


def test_turbo_default_disables_negative_forward_and_runs_eight_steps():
    model = _VelocityModel()
    model.eval()
    condition = Krea2SamplingCondition(
        positive=_condition(2.0),
        negative=_condition(99.0),
    )

    output = sample_latents(
        model,
        condition,
        height=16,
        width=16,
        distilled=True,
        device="cpu",
        dtype=torch.float32,
        latents=torch.zeros(1, 2, 1, 2, 2),
    )

    # Krea g=0 means conditional-only (not unconditional and not standard CFG=0).
    torch.testing.assert_close(output, torch.full_like(output, -2.0))
    assert len(model.calls) == KREA2_TURBO_STEPS
    assert all(call[0] == 2.0 for call in model.calls)
    assert not model.training


def test_sample_image_rounds_up_to_16_and_emits_phases():
    model = _VelocityModel(channels=2, patch=2)
    phases = []

    image = sample_image(
        model,
        _FakeVAE(),
        Krea2SamplingCondition(positive=_condition(0.0)),
        height=17,
        width=19,
        steps=1,
        cfg_scale=0,
        device="cpu",
        dtype=torch.float32,
        phase_callback=phases.append,
        seed=123,
    )

    assert align_krea2_resolution(17) == 32
    assert align_krea2_resolution(32) == 32
    assert image.size == (32, 32)
    assert phases == ["sample", "vae"]


def test_sampling_runs_through_real_tiny_krea2_forward():
    config = Krea2Config(
        features=64,
        tdim=16,
        txtdim=16,
        heads=4,
        multiplier=2,
        layers=1,
        patch=2,
        channels=2,
        bias=False,
        theta=1000.0,
        kvheads=2,
        txtlayers=2,
        txtheads=2,
        txtkvheads=1,
    )
    model = SingleStreamDiT(config)
    condition = Krea2SamplingCondition(
        positive=Krea2TextCondition(
            context=torch.randn(1, 3, 2, 16),
            attention_mask=torch.tensor([[True, True, False]]),
        )
    )

    output = sample_latents(
        model,
        condition,
        height=16,
        width=16,
        steps=1,
        cfg_scale=0,
        device="cpu",
        dtype=torch.float32,
        latents=torch.zeros(1, 2, 1, 2, 2),
    )

    assert output.shape == (1, 2, 1, 2, 2)
    assert torch.isfinite(output).all()
