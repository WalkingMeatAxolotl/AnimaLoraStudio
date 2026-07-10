"""timestep shift 分辨率修正（timestep_shift_resolution_aware）单测。

覆盖：
  1) apply_resolution_shift：基准档恒等 / 大图升 t、小图降 t / 公式数值点。
  2) Möbius 乘法复合律：先全局 shift 再分辨率修正 == 一次乘积 shift（正交性依据）。
  3) per-sample 向量：一个 batch 内不同 token 数各得各的修正。
  4) 端点 clamp：极端 token 比不越出 (1e-4, 1-1e-4)。
  5) latent_token_counts：批量网格 tensor 与 NaViT 异构 latent 列表两种输入。
  6) config：默认 off、可开。

CPU-only、无 GPU：CI（Linux 无 GPU）可跑。
"""
from __future__ import annotations

import math

import pytest
import torch

from training.timestep_sampling import (
    apply_resolution_shift,
    latent_token_counts,
)


# --------------------------------------------------------- 公式性质
def test_identity_at_base_tokens():
    t = torch.linspace(0.05, 0.95, 10)
    out = apply_resolution_shift(t, [4096] * 10, 4096)
    assert torch.allclose(out, t, atol=1e-6)


def test_direction_and_known_values():
    # 4× token → s=2；1/4 token → s=0.5。t=0.5 处解析值 2/3 与 1/3。
    t = torch.full((4,), 0.5)
    up = apply_resolution_shift(t, [4096 * 4] * 4, 4096)
    down = apply_resolution_shift(t, [4096 // 4] * 4, 4096)
    assert torch.all(up > t) and torch.all(down < t)
    assert torch.allclose(up, torch.full((4,), 2.0 / 3.0), atol=1e-5)
    assert torch.allclose(down, torch.full((4,), 1.0 / 3.0), atol=1e-5)


def test_composes_multiplicatively_with_global_shift():
    """Möbius 偏移按 s 乘法复合：全局 shift(s1) 后再分辨率修正(s2) == shift(s1·s2)。

    这是"全局 timestep_shift 仍是基准档校准值、本修正只补分辨率差"的数学依据。
    """
    t = torch.linspace(0.05, 0.95, 17)
    s1, n, base = 3.0, 16384, 4096  # s2 = sqrt(16384/4096) = 2
    a = (t * s1) / (1 + (s1 - 1) * t)
    a = apply_resolution_shift(a, [n] * len(t), base)
    s12 = s1 * math.sqrt(n / base)
    b = ((t * s12) / (1 + (s12 - 1) * t)).clamp(1e-4, 1 - 1e-4)
    assert torch.allclose(a, b, atol=1e-5)


def test_per_sample_vector():
    t = torch.tensor([0.5, 0.5, 0.5])
    out = apply_resolution_shift(t, [4096, 16384, 1024], 4096)
    assert out[0] == pytest.approx(0.5, abs=1e-6)
    assert out[1] > 0.5 > out[2]


def test_bounds_clamped():
    t = torch.tensor([1e-4, 1 - 1e-4])
    out = apply_resolution_shift(t, [1_000_000, 1], 4096)
    assert torch.all(out >= 1e-4) and torch.all(out <= 1 - 1e-4)


# --------------------------------------------------------- token 计数口径
def test_latent_token_counts_batched_tensor():
    # 1024px → latent 128×128 → 64×64 token = 4096（与 CachedLatentDataset 口径一致）
    lat = torch.zeros(3, 16, 1, 128, 128)
    assert latent_token_counts(lat) == [4096, 4096, 4096]


def test_latent_token_counts_navit_list():
    # NaViT 异构列表：5D [1,C,T,h,w] 与 4D [C,T,h,w] 都按最后两维算
    lst = [torch.zeros(1, 16, 1, 62, 93), torch.zeros(16, 1, 48, 48)]
    assert latent_token_counts(lst) == [31 * 46, 24 * 24]


# --------------------------------------------------------- config switch
def test_config_default_off():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.timestep_shift_resolution_aware is False


def test_config_can_enable():
    from studio.domain import TrainingConfig
    cfg = TrainingConfig(timestep_shift_resolution_aware=True)
    assert cfg.timestep_shift_resolution_aware is True
