from __future__ import annotations

import torch

from runtime.training.personalization import (
    AptAdaptiveController,
    apply_adaptive_affine,
    build_region_spatial_weight,
    region_schedule_scale,
)


def test_region_schedule_has_exact_whole_image_tail() -> None:
    assert region_schedule_scale(0, 100) == 1.0
    assert region_schedule_scale(45, 100) == 1.0
    assert 0.0 < region_schedule_scale(50, 100) < 1.0
    assert region_schedule_scale(55, 100) == 0.0
    assert region_schedule_scale(100, 100) == 0.0


def test_region_weight_combines_focus_and_ignore_then_turns_off() -> None:
    region = torch.zeros(1, 4, 4)
    region[:, 1:3, 1:3] = 1
    ignore = torch.ones(1, 4, 4)
    ignore[:, 0, 0] = 0
    spatial, stats = build_region_spatial_weight(
        region, ignore, torch.tensor([1.0]), global_step=0, total_steps=100,
        max_weight=3.0, hold_ratio=0.45, end_ratio=0.55,
    )
    assert spatial is not None
    assert spatial[0, 1, 1].item() == 3.0
    assert spatial[0, 0, 0].item() == 0.0
    assert stats.coverage == 0.25

    tail, tail_stats = build_region_spatial_weight(
        region, None, None, global_step=55, total_steps=100,
        max_weight=3.0, hold_ratio=0.45, end_ratio=0.55,
    )
    assert tail is None
    assert tail_stats.scale == 0.0


def test_apt_controller_updates_only_main_and_round_trips() -> None:
    controller = AptAdaptiveController(bins=2, ema_alpha=1.0)
    t = torch.tensor([0.2, 0.8])
    controller.update(
        t,
        base_loss=torch.tensor([2.0, 100.0]),
        tuned_loss=torch.tensor([1.0, 0.0]),
        main_mask=torch.tensor([True, False]),
    )
    status = controller.status()
    assert status["initialized_bins"] == 1
    assert status["gamma_max"] > 0.99
    assert controller.loss_weights(t, torch.tensor([True, False]))[1].item() == 1.0

    restored = AptAdaptiveController(bins=2, ema_alpha=0.1)
    restored.load_state_dict(controller.state_dict())
    assert restored.status() == status


def test_adaptive_affine_zero_gamma_is_bit_exact() -> None:
    latents = torch.randn(2, 3, 1, 4, 4)
    regions = torch.rand(2, 4, 4)
    out, out_regions, _, fraction = apply_adaptive_affine(
        latents, regions, None, torch.zeros(2),
    )
    assert fraction == 0.0
    assert torch.equal(out, latents)
    assert torch.equal(out_regions, regions)
