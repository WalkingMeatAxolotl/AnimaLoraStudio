"""VAEWrapper OOM-fallback tiled decode 单测（issue #200）。

测：
- `_tile_starts` 边界 case（含 1024 latent / 128 latent 实际 issue 规格）
- `_cosine_blend_mask` 对称性 + clamp_min 避免角落除 0
- `VAEWrapper.decode` 走 try-full 路径不触发 tile
- `VAEWrapper.decode` 在 model.decode 抛 OOM 时走 tile fallback 并产出形状一致的结果
- `_tiled_decode` 拼接：mock 一个线性 decode（z 上 nearest-upsample 8× 取前 3 通道），
  tile 重建结果与整图重建数值一致（验证 cosine blend + accumulator 正确性）
"""
from __future__ import annotations

import pytest
import torch

from training.models import (
    VAEWrapper,
    _cosine_blend_mask,
    _tile_starts,
)


# ---------------------------------------------------------------------------
# _tile_starts
# ---------------------------------------------------------------------------


def test_tile_starts_size_le_tile_returns_zero_only() -> None:
    assert _tile_starts(64, 64, 48) == [0]
    assert _tile_starts(32, 64, 48) == [0]


def test_tile_starts_appends_boundary_when_stride_doesnt_cover() -> None:
    """128 latent (1024 px reg) 的实际规格：tile=64 stride=48 → [0, 48, 64]。"""
    assert _tile_starts(128, 64, 48) == [0, 48, 64]


def test_tile_starts_exact_stride_no_duplicate_tail() -> None:
    """stride 恰好整除时尾部不重复追加。"""
    # size=160 tile=64 stride=48 → range(0, 97, 48) = [0, 48, 96]，96+64=160 命中尾部
    assert _tile_starts(160, 64, 48) == [0, 48, 96]


# ---------------------------------------------------------------------------
# _cosine_blend_mask
# ---------------------------------------------------------------------------


def test_cosine_blend_mask_shape_and_dtype() -> None:
    m = _cosine_blend_mask(512, 512, fade=128, device="cpu")
    assert m.shape == (1, 1, 1, 512, 512)
    assert m.dtype == torch.float32


def test_cosine_blend_mask_center_one_edges_clamped() -> None:
    """中心 = 1，边缘 ≈ 1e-6（防 wsum 除 0；fp32 clamp 实测略低于 1e-6）。"""
    m = _cosine_blend_mask(512, 512, fade=128, device="cpu")[0, 0, 0]
    # 中心
    assert m[256, 256].item() == pytest.approx(1.0, abs=1e-5)
    # 四角：clamp 在 fp32 精度下 ≈ 1e-6（容差 1e-7）
    assert m[0, 0].item() == pytest.approx(1e-6, abs=1e-7)
    assert m[-1, -1].item() == pytest.approx(1e-6, abs=1e-7)
    assert m[0, -1].item() == pytest.approx(1e-6, abs=1e-7)


def test_cosine_blend_mask_symmetric() -> None:
    m = _cosine_blend_mask(256, 256, fade=64, device="cpu")[0, 0, 0]
    # 上下对称
    assert torch.allclose(m, m.flip(0), atol=1e-6)
    # 左右对称
    assert torch.allclose(m, m.flip(1), atol=1e-6)


def test_cosine_blend_mask_zero_fade_all_ones() -> None:
    m = _cosine_blend_mask(64, 64, fade=0, device="cpu")
    assert torch.all(m == 1.0)


# ---------------------------------------------------------------------------
# VAEWrapper.decode 走 try-full 路径
# ---------------------------------------------------------------------------


class _RecordingModel:
    """记录 decode 调用次数 + 形状的 mock；行为是 nearest upsample 8× + 取前 3 通道。"""

    def __init__(self, oom_on_full: bool = False):
        self.calls: list[tuple[int, ...]] = []
        self.oom_on_full = oom_on_full

    def decode(self, z: torch.Tensor, scale) -> torch.Tensor:
        self.calls.append(tuple(z.shape))
        b, _c, t, h, w = z.shape
        # 模拟 WanVAE_ 的 8× spatial upsample + 16ch → 3ch
        # 故意用 z 的前 3 通道 nearest upsample，方便 tile / full 对比
        upsampled = torch.nn.functional.interpolate(
            z[:, :3].reshape(b * t, 3, h, w),
            scale_factor=8,
            mode="nearest",
        ).reshape(b, 3, t, h * 8, w * 8)
        if self.oom_on_full and h >= 128 and len(self.calls) == 1:
            raise torch.cuda.OutOfMemoryError("simulated OOM on full decode")
        return upsampled


def _make_wrapper(model) -> VAEWrapper:
    mean = torch.zeros(16)
    std = torch.ones(16)
    return VAEWrapper(model, mean, std)


def test_decode_full_path_calls_model_once() -> None:
    """大显存路径：try full 成功 → 不进 tile，model.decode 只调一次。"""
    model = _RecordingModel(oom_on_full=False)
    wrapper = _make_wrapper(model)
    z = torch.randn(1, 16, 1, 32, 32)  # 256×256 输出
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 256, 256)
    assert len(model.calls) == 1


# ---------------------------------------------------------------------------
# VAEWrapper.decode OOM fallback
# ---------------------------------------------------------------------------


def test_decode_oom_fallback_invokes_tiled_decode() -> None:
    """整图 OOM → catch + empty_cache + tile 多次调 model.decode。

    1024 reg 规格：z = [1,16,1,128,128]；tile=64 stride=48 →
    tile 起点 hs=ws=[0,48,64] → 3×3 = 9 个 tile decode 调用，
    加上第一次失败的整图调用 = 10。
    """
    model = _RecordingModel(oom_on_full=True)
    wrapper = _make_wrapper(model)
    z = torch.zeros(1, 16, 1, 128, 128)  # 1024×1024 reg
    out = wrapper.decode(z)
    assert out.shape == (1, 3, 1, 1024, 1024)
    # 1 次失败 full + 9 个 tile
    assert len(model.calls) == 1 + 9
    # 第一次是 full
    assert model.calls[0] == (1, 16, 1, 128, 128)
    # 后续都是 tile 大小
    for shape in model.calls[1:]:
        assert shape == (1, 16, 1, 64, 64)


# ---------------------------------------------------------------------------
# Tile 拼接数值正确性
# ---------------------------------------------------------------------------


def test_tiled_decode_reconstructs_full_for_linear_decoder() -> None:
    """对一个线性 decode（nearest upsample 8×），tile + cosine blend 拼接
    结果与整图 decode 数值一致（accumulator 归一化正确）。

    用 latent[:, :3] 作 "image"，nearest 8× 上采。每个像素的实际值与位置无
    关，blend mask 加权累加后归一化必须恢复原值。
    """
    model = _RecordingModel(oom_on_full=False)
    wrapper = _make_wrapper(model)
    torch.manual_seed(0)
    z = torch.randn(1, 16, 1, 128, 128)

    full = wrapper.decode(z)
    tiled = wrapper._tiled_decode(z)

    # 数值应几乎一致；mask clamp_min 1e-6 + fp32 累加只引入极小数值误差
    assert tiled.shape == full.shape
    assert torch.allclose(tiled, full, atol=1e-4)


def test_tiled_decode_handles_small_input_without_tiling() -> None:
    """size ≤ tile 时 tile_starts 返回 [0]，等同整图 decode。"""
    model = _RecordingModel(oom_on_full=False)
    wrapper = _make_wrapper(model)
    z = torch.randn(1, 16, 1, 32, 32)  # < tile=64
    out = wrapper._tiled_decode(z)
    assert out.shape == (1, 3, 1, 256, 256)
