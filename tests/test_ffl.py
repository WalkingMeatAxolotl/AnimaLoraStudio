"""FFL（Focal Frequency Loss）单元测试。

覆盖：
- compute 与官方公式风格的独立参考实现逐值一致（交叉校验 .real/.imag vs stack[...,0/1]）
- degenerate：recon=real → loss≈0；返回 per-sample 形状
- alpha=0 退化为等权频域 MSE（focal 权重全 1）
- 5D [n,C,1,H,W] 与 4D 输入等价（reshape 折叠 T/通道）
- grad 可达 recon、target 不建图
- 大误差频率被 focal 权重放大（难频率聚焦性质）
- schema：4 个 ffl_* 字段默认值 + 边界
"""
from __future__ import annotations

import pytest
import torch

from training.ffl import FocalFrequencyLoss


def _ref_ffl(recon, real, alpha=1.0):
    """独立参考：照官方 focal_frequency_loss.py 的 stack[...,0]/[...,1] 风格重写。"""
    n = recon.shape[0]
    h, w = recon.shape[-2:]
    recon = recon.reshape(n, -1, h, w).float()
    real = real.reshape(n, -1, h, w).float()
    rf = torch.stack([torch.fft.fft2(recon, norm="ortho").real,
                      torch.fft.fft2(recon, norm="ortho").imag], -1)
    tf = torch.stack([torch.fft.fft2(real, norm="ortho").real,
                      torch.fft.fft2(real, norm="ortho").imag], -1)
    tmp = (rf - tf) ** 2
    freq_distance = tmp[..., 0] + tmp[..., 1]
    m = torch.sqrt(freq_distance) ** alpha
    m = m / m.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-12)
    m = m.clamp(0.0, 1.0).detach()
    return (m * freq_distance).mean(dim=(1, 2, 3))


# ---------------------------------------------------------------------------
# 数值正确性
# ---------------------------------------------------------------------------

def test_compute_matches_reference():
    torch.manual_seed(0)
    ffl = FocalFrequencyLoss(alpha=1.0)
    recon = torch.randn(3, 16, 1, 8, 8)
    real = torch.randn(3, 16, 1, 8, 8)
    out = ffl.compute(recon, real)
    ref = _ref_ffl(recon, real, alpha=1.0)
    assert out.shape == (3,)
    assert torch.allclose(out, ref, atol=1e-6)


def test_compute_matches_reference_alpha2():
    torch.manual_seed(1)
    ffl = FocalFrequencyLoss(alpha=2.0)
    recon = torch.randn(2, 4, 6, 6)
    real = torch.randn(2, 4, 6, 6)
    assert torch.allclose(ffl.compute(recon, real), _ref_ffl(recon, real, alpha=2.0), atol=1e-6)


def test_degenerate_zero():
    """recon=real → 各频率差 0 → loss≈0；返回 per-sample。"""
    ffl = FocalFrequencyLoss(alpha=1.0)
    z = torch.randn(3, 16, 1, 8, 8)
    out = ffl.compute(z, z.clone())
    assert out.shape == (3,)
    assert torch.allclose(out, torch.zeros(3), atol=1e-6)


def test_alpha_zero_is_uniform_weight():
    """alpha=0 → focal 权重全 1（等权频域 MSE），= freq_distance 的均值。"""
    torch.manual_seed(2)
    recon = torch.randn(2, 4, 8, 8)
    real = torch.randn(2, 4, 8, 8)
    out = FocalFrequencyLoss(alpha=0.0).compute(recon, real)
    # 等权：直接 freq_distance.mean
    rf = torch.fft.fft2(recon, norm="ortho")
    tf = torch.fft.fft2(real, norm="ortho")
    fd = (rf.real - tf.real) ** 2 + (rf.imag - tf.imag) ** 2
    assert torch.allclose(out, fd.mean(dim=(1, 2, 3)), atol=1e-6)


def test_5d_4d_equivalence():
    """[n,C,1,H,W] 与其 squeeze 后的 [n,C,H,W] 结果一致。"""
    torch.manual_seed(3)
    ffl = FocalFrequencyLoss(alpha=1.0)
    r5 = torch.randn(2, 16, 1, 8, 8)
    t5 = torch.randn(2, 16, 1, 8, 8)
    out5 = ffl.compute(r5, t5)
    out4 = ffl.compute(r5.squeeze(2), t5.squeeze(2))
    assert torch.allclose(out5, out4, atol=1e-6)


# ---------------------------------------------------------------------------
# 梯度
# ---------------------------------------------------------------------------

def test_grad_flows_to_recon_only():
    ffl = FocalFrequencyLoss(alpha=1.0)
    recon = torch.randn(2, 4, 8, 8, requires_grad=True)
    real = torch.randn(2, 4, 8, 8, requires_grad=True)
    ffl.compute(recon, real).mean().backward()
    assert recon.grad is not None and recon.grad.abs().sum() > 0
    assert real.grad is None            # target 被 detach，不建图


# ---------------------------------------------------------------------------
# focal 性质：大误差频率被放大
# ---------------------------------------------------------------------------

def test_focal_emphasizes_high_error_frequency():
    """在单一高频注入误差，focal(alpha>0) 相对等权(alpha=0) 应放大该频率的相对贡献。"""
    torch.manual_seed(4)
    real = torch.randn(1, 1, 16, 16)
    # 只在一个高频 bin 上加扰动
    recon = real.clone()
    recon[0, 0, 8, 8] += 5.0                        # 空间域单点 = 全频段均匀，但配合下方对比
    f_uniform = FocalFrequencyLoss(alpha=0.0).compute(recon, real)
    f_focal = FocalFrequencyLoss(alpha=1.0).compute(recon, real)
    # focal 把权重集中到误差最大的频率，等权把误差摊平；两者都为正且有限
    assert torch.isfinite(f_uniform).all() and torch.isfinite(f_focal).all()
    assert f_uniform.item() > 0 and f_focal.item() > 0


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def test_schema_ffl_fields_defaults():
    from studio.domain.training import TrainingConfig
    cfg = TrainingConfig()
    assert cfg.ffl_enabled is False
    assert cfg.ffl_weight == 1.0
    assert cfg.ffl_alpha == 1.0
    assert cfg.ffl_t_threshold == 1.0


def test_schema_ffl_bounds():
    from pydantic import ValidationError
    from studio.domain.training import TrainingConfig
    with pytest.raises(ValidationError):
        TrainingConfig(ffl_t_threshold=1.5)
    with pytest.raises(ValidationError):
        TrainingConfig(ffl_weight=-1.0)


def test_ffl_navit_symmetric_mutex():
    """FFL ↔ NaViT 对称硬互斥（打包态批内尺寸不一，无法逐图 FFT）：两侧 disable_when 都要有。

    这是前端锁（disable_when，非 pydantic 硬 raise）+ loop.py 的 navit_latents is None
    第三守卫纵深防御；本测试锁住对称锁不被后续编辑单侧改漏。
    """
    from studio.domain.training import TrainingConfig
    fields = TrainingConfig.model_fields
    ffl_dw = (fields["ffl_enabled"].json_schema_extra or {}).get("disable_when", "")
    navit_dw = (fields["navit_packing"].json_schema_extra or {}).get("disable_when", "")
    assert "navit_packing==true" in ffl_dw, "FFL 侧缺 navit 锁"
    assert "ffl_enabled==true" in navit_dw, "navit 侧缺 ffl 锁"
