"""SRA v2 regression tests.

These tests use a tiny fake block stack instead of the real Anima model, so they
cover SRA's shape/state contracts without loading model weights.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from training.sra_align import SRAAligner


class _IdentityBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_IdentityBlock()])


class _StubInjector:
    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, _state: dict, strict: bool = True):
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])


def _new_aligner(seed: int = 0, normalize: bool = True) -> tuple[_FakeModel, SRAAligner]:
    torch.manual_seed(seed)
    model = _FakeModel()
    aligner = SRAAligner(
        model=model,
        block_idx=0,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=8,
        vae_channels=16,
        device="cpu",
        dtype=torch.float32,
        normalize=normalize,
    )
    return model, aligner


def test_sra_compute_accepts_native_flatten_block_output() -> None:
    model, aligner = _new_aligner()
    target = torch.zeros(2, 16, 1, 4, 4)

    # torch.compile native-flatten mode presents block output as
    # (B, 1, seq_len, 1, D); SRA should rebuild the target patch grid.
    hidden = torch.randn(2, 1, 4, 1, 8)
    model.blocks[0](hidden)

    loss = aligner.compute(target)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_sra_normalize_keeps_loss_order_one_for_large_targets() -> None:
    # A large-magnitude target (e.g. a video-VAE latent) blows the un-normalized
    # smooth-L1 align loss several orders of magnitude above the denoise loss.
    # Standardizing both sides keeps it on an O(1) structural footing.
    big_target = torch.randn(2, 16, 1, 4, 4) * 50.0
    hidden = torch.randn(2, 1, 4, 1, 8)

    model_n, aligner_norm = _new_aligner(normalize=True)
    model_n.blocks[0](hidden)
    loss_norm = aligner_norm.compute(big_target)

    model_r, aligner_raw = _new_aligner(normalize=False)
    model_r.blocks[0](hidden)
    loss_raw = aligner_raw.compute(big_target)

    assert torch.isfinite(loss_norm)
    assert loss_norm.item() < 5.0
    assert loss_raw.item() > 10.0 * loss_norm.item()


def test_sra_compute_applies_sample_weight() -> None:
    model, aligner = _new_aligner()
    target = torch.zeros(2, 16, 1, 4, 4)
    hidden = torch.randn(2, 1, 4, 1, 8)
    model.blocks[0](hidden)

    loss = aligner.compute(target, sample_weight=torch.zeros(2))
    assert loss.item() == 0.0


def test_sra_weight_zero_is_preserved() -> None:
    from training.loop import _resolve_sra_weight

    assert _resolve_sra_weight(SimpleNamespace(sra_weight=0.0)) == 0.0
    assert _resolve_sra_weight(SimpleNamespace(sra_weight=None)) == 0.2
    assert _resolve_sra_weight(SimpleNamespace()) == 0.2


def test_sra_effective_weight_decay_modes() -> None:
    from training.loop import _resolve_sra_effective_weight

    base = dict(
        sra_weight=0.1,
        sra_decay_start_ratio=0.2,
        sra_decay_end_ratio=0.4,
    )

    assert _resolve_sra_effective_weight(
        SimpleNamespace(**base, sra_decay_type="none"), 50, 100,
    ) == pytest.approx(0.1)
    assert _resolve_sra_effective_weight(
        SimpleNamespace(**base, sra_decay_type="linear"), 30, 100,
    ) == pytest.approx(0.05)
    assert _resolve_sra_effective_weight(
        SimpleNamespace(**base, sra_decay_type="cosine"), 30, 100,
    ) == pytest.approx(0.05)
    assert _resolve_sra_effective_weight(
        SimpleNamespace(**base, sra_decay_type="jump"), 19, 100,
    ) == pytest.approx(0.1)
    assert _resolve_sra_effective_weight(
        SimpleNamespace(**base, sra_decay_type="jump"), 20, 100,
    ) == pytest.approx(0.0)


def test_sra_decay_schema_rejects_inverted_linear_range() -> None:
    from studio.schema import TrainingConfig

    with pytest.raises(Exception):
        TrainingConfig(
            sra_enabled=True,
            sra_decay_type="linear",
            sra_decay_start_ratio=0.5,
            sra_decay_end_ratio=0.25,
        )


def test_sra_state_roundtrip(tmp_path: Path) -> None:
    from training.state import load_training_state, save_training_state

    _model1, aligner1 = _new_aligner(seed=1)
    with torch.no_grad():
        for p in aligner1.proj.parameters():
            p.uniform_(-0.25, 0.25)
    optimizer1 = torch.optim.AdamW(aligner1.get_param_groups(), lr=1e-3)

    state_path = tmp_path / "state.pt"
    save_training_state(
        state_path,
        _StubInjector(),
        optimizer1,
        epoch=2,
        global_step=42,
        sra_aligner=aligner1,
    )

    _model2, aligner2 = _new_aligner(seed=2)
    optimizer2 = torch.optim.AdamW(aligner2.get_param_groups(), lr=1e-3)
    load_training_state(
        state_path,
        _StubInjector(),
        optimizer2,
        sra_aligner=aligner2,
    )

    sd1 = aligner1.state_dict()["proj"]
    sd2 = aligner2.state_dict()["proj"]
    assert sd1.keys() == sd2.keys()
    for key in sd1:
        assert torch.equal(sd1[key], sd2[key]), key
