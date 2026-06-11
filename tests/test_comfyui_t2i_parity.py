"""ComfyUI KSampler parity checks for the single-image txt2img path.

Aligned with the dev architecture after rebuild:
- `_flow_sigmas_simple` returns the raw ComfyUI `simple_scheduler` sequence.
  The sampler layer applies `offset_first_sigma_for_snr`; initial noise still
  uses the raw first sigma.
- Initial noise uses `_prepare_comfy_t2i_noise`, matching ComfyUI CPU-seeded
  txt2img noise and CONST `noise_scaling`.
- Generate-time Comfy parity uses raw Anima prompt text for Qwen and passes
  T5 token weights into `model.preprocess_text_embeds`.
"""

from __future__ import annotations

import torch

import training.sampling as sampling
from training.inference_samplers import BUILDERS
from training.sampling import (
    _flow_sigmas_sgm_uniform,
    _flow_sigmas_simple,
    _prepare_comfy_t2i_noise,
    _time_snr_shift,
)
from training.text_encoding import build_comfy_anima_conditioning_inputs


def _ref_sgm_uniform(steps: int, *, shift: float = 3.0, timesteps: int = 1000) -> torch.Tensor:
    """Reference for ComfyUI normal_scheduler(model_sampling, steps, sgm=True)."""
    sigma_max = _time_snr_shift(float(shift), torch.tensor(1.0, dtype=torch.float32))
    sigma_min = _time_snr_shift(float(shift), torch.tensor(1.0 / timesteps, dtype=torch.float32))
    # Anima uses multiplier=1.0 in ComfyUI, but it cancels out for this flow formula.
    timesteps_lin = torch.linspace(float(sigma_max), float(sigma_min), steps + 1, dtype=torch.float32)[:-1]
    sigmas = _time_snr_shift(float(shift), timesteps_lin)
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def test_simple_scheduler_returns_raw_comfyui_first_sigma() -> None:
    sigmas = _flow_sigmas_simple(25, shift=3.0, device="cpu")

    # ComfyUI simple_scheduler returns 1.0 here. KSampler applies
    # offset_first_sigma_for_snr inside the sampler, after initial noise scaling.
    assert sigmas[0].item() == 1.0
    assert sigmas[-1].item() == 0.0


def test_prepare_comfy_t2i_noise_matches_cpu_seeded_const_scaling() -> None:
    sigmas = _flow_sigmas_simple(25, shift=3.0, device="cpu")

    actual = _prepare_comfy_t2i_noise((1, 16, 1, 4, 4), sigmas, device="cpu", seed=123)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    expected = torch.randn((1, 16, 1, 4, 4), dtype=torch.float32, generator=generator) * sigmas[0]
    assert torch.equal(actual, expected)


def test_comfy_empty_latent_fix_repeats_resolution_master_4ch_latent() -> None:
    latent = torch.zeros((1, 4, 240, 160), dtype=torch.float32)

    fixed = sampling._fix_comfy_empty_latent_channels(
        latent,
        latent_channels=16,
        latent_dimensions=3,
    )

    assert fixed.shape == (1, 16, 1, 240, 160)
    assert fixed.dtype == torch.float32
    assert torch.count_nonzero(fixed).item() == 0


def test_sgm_uniform_matches_comfyui_reference() -> None:
    for steps in (1, 2, 5, 25, 50):
        actual = _flow_sigmas_sgm_uniform(steps, shift=3.0, device="cpu")
        expected = _ref_sgm_uniform(steps, shift=3.0)

        assert torch.allclose(actual, expected, atol=2e-7, rtol=0.0)


def test_inference_registry_exposes_dpmpp_3m_sde() -> None:
    assert "er_sde" in BUILDERS
    assert "dpmpp_3m_sde" in BUILDERS


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        return {"input_ids": [ord(ch) % 251 + 2 for ch in str(text)] + [self.eos_token_id]}


def test_comfy_anima_conditioning_keeps_raw_qwen_prompt() -> None:
    prompt = "1girl, (masterpiece:1.2), [flat color]"

    qwen_text, _t5_ids, _t5_attn, _t5_weights = build_comfy_anima_conditioning_inputs(_FakeTokenizer(), prompt)

    assert qwen_text == prompt
