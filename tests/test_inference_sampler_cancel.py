from __future__ import annotations

import builtins

import pytest
import torch

from runtime.anima_daemon import GenerationCanceled
from training.inference_samplers import dpmpp_3m_sde, er_sde


@pytest.mark.parametrize("sampler", [er_sde.sample, dpmpp_3m_sde.sample])
def test_sampler_does_not_swallow_generation_cancel(sampler) -> None:
    x = torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.0], dtype=torch.float32)

    def denoise_fn(x_in, _sigma):
        return x_in

    def cancel_callback(_step, _total, _denoised):
        raise GenerationCanceled()

    with pytest.raises(GenerationCanceled):
        sampler(
            denoise_fn,
            x,
            sigmas,
            seed=1,
            step_callback=cancel_callback,
        )


def test_dpmpp_3m_sde_strict_brownian_tree_requires_torchsde(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torchsde":
            raise ImportError("missing torchsde")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    x = torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32)
    sigmas = torch.tensor([1.0, 0.0], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="torchsde"):
        dpmpp_3m_sde.sample(
            lambda x_in, _sigma: x_in,
            x,
            sigmas,
            seed=1,
            require_brownian_tree=True,
        )
