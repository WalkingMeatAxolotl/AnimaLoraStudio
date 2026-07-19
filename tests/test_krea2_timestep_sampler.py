"""Phase 3 Krea2 dynamic-shift timestep sampler tests."""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import pytest
import torch

from training.loop import _sample_timesteps, run
from training.timestep_samplers import BUILDERS, build_timestep_sampler
from training.timestep_samplers.krea2_shift import (
    Krea2ShiftTimestepSampler,
    apply_krea2_dynamic_shift,
    krea2_mu,
)
from training.timestep_samplers.protocol import TimestepSamplerProtocol


def test_krea2_mu_matches_upstream_endpoints_and_1024_anchor() -> None:
    mu = krea2_mu([256, 4096, 6400])
    torch.testing.assert_close(
        mu,
        torch.tensor([0.5, 0.90625, 1.15]),
        rtol=0,
        atol=1e-6,
    )


def test_krea2_mu_does_not_clamp_outside_reference_endpoints() -> None:
    mu = krea2_mu([128, 8000])
    assert mu[0] < 0.5
    assert mu[1] > 1.15


def test_krea2_shift_matches_known_1024_midpoint() -> None:
    out = apply_krea2_dynamic_shift(torch.tensor([0.5]), [4096])
    shift = math.exp(0.90625)
    expected = shift / (1 + shift)
    assert out.item() == pytest.approx(expected, abs=1e-6)
    assert out.item() == pytest.approx(0.7122, abs=1e-4)


def test_krea2_shift_is_per_sample_and_resolution_monotonic() -> None:
    out = apply_krea2_dynamic_shift(
        torch.full((3,), 0.5),
        [256, 4096, 6400],
    )
    assert out[0] < out[1] < out[2]


def test_krea2_sampler_uses_logit_normal_then_dynamic_shift(monkeypatch) -> None:
    monkeypatch.setattr(
        torch,
        "randn",
        lambda bs, *, device, dtype: torch.zeros(bs, device=device, dtype=dtype),
    )
    sampler = Krea2ShiftTimestepSampler()
    out = sampler.sample(3, torch.device("cpu"), token_counts=[256, 4096, 6400])
    expected = apply_krea2_dynamic_shift(torch.full((3,), 0.5), [256, 4096, 6400])
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize(
    ("token_counts", "match"),
    [
        (None, "需要每个样本"),
        ([256], "数量必须等于 batch size"),
        ([256, 0], "有限正数"),
    ],
)
def test_krea2_sampler_rejects_invalid_batch_context(token_counts, match) -> None:
    sampler = Krea2ShiftTimestepSampler()
    with pytest.raises(ValueError, match=match):
        sampler.sample(2, torch.device("cpu"), token_counts=token_counts)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_krea2_sampler_rejects_invalid_sigmoid_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="sigmoid_scale"):
        Krea2ShiftTimestepSampler(sigmoid_scale=scale)


def test_krea2_sampler_satisfies_protocol_and_is_stateless() -> None:
    sampler = Krea2ShiftTimestepSampler()
    assert isinstance(sampler, TimestepSamplerProtocol)
    assert sampler.state_dict() == {}
    assert sampler.status() == {
        "kind": "krea2_shift",
        "sigmoid_scale": 1.0,
        "base_image_seq_len": 256,
        "max_image_seq_len": 6400,
        "base_shift": 0.5,
        "max_shift": 1.15,
    }


def test_registry_selects_krea2_shift_without_schema_exposure() -> None:
    assert "krea2_shift" in BUILDERS
    args = argparse.Namespace(timestep_sampling="krea2_shift", infonoise_enabled=False)
    sampler = build_timestep_sampler(args, total_steps=100)
    assert isinstance(sampler, Krea2ShiftTimestepSampler)


def test_registry_rejects_krea2_shift_with_infonoise() -> None:
    args = argparse.Namespace(timestep_sampling="krea2_shift", infonoise_enabled=True)
    with pytest.raises(ValueError, match="不能同时启用"):
        build_timestep_sampler(args, total_steps=100)


def test_loop_injects_token_counts_for_capability_sampler() -> None:
    class RecordingSampler:
        requires_token_counts = True

        def sample(self, bs, device, *, token_counts=None):
            self.token_counts = token_counts
            return torch.full((bs,), 0.25, device=device)

    sampler = RecordingSampler()
    latents = torch.zeros(2, 16, 1, 128, 128)
    out = _sample_timesteps(sampler, 2, torch.device("cpu"), latents)
    assert sampler.token_counts == [4096, 4096]
    torch.testing.assert_close(out, torch.tensor([0.25, 0.25]))


def test_loop_keeps_ordinary_sampler_on_context_free_path(monkeypatch) -> None:
    class OrdinarySampler:
        def sample(self, bs, device):
            return torch.full((bs,), 0.75, device=device)

    monkeypatch.setattr(
        "training.loop.latent_token_counts",
        lambda latents: pytest.fail("普通 sampler 不应计算 token_counts"),
    )
    out = _sample_timesteps(
        OrdinarySampler(),
        2,
        torch.device("cpu"),
        torch.zeros(2, 16, 1, 128, 128),
    )
    torch.testing.assert_close(out, torch.tensor([0.75, 0.75]))


def test_loop_rejects_double_resolution_shift_before_training() -> None:
    ctx = SimpleNamespace(
        args=argparse.Namespace(
            timestep_shift_resolution_aware=True,
            resolution=1024,
        ),
        dataloader=[],
        timestep_sampler=Krea2ShiftTimestepSampler(),
    )
    with pytest.raises(ValueError, match="不能与自带分辨率 shift"):
        run(ctx)
