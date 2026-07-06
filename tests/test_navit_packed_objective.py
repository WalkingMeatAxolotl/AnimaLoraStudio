"""NaViT / Patch-n-Pack 训练步骤单测：navit_packed_forward_and_loss + pack_cross_embeddings。

验证：
  1) pack_cross_embeddings：trim / no-trim 两路径的拼接长度与 text_seqlens 正确。
  2) navit_packed_forward_and_loss：loss 有限、带梯度、per_image_loss 形状 [G]。
  3) grad_checkpoint 路径输出 ≡ 非检查点路径，且仍可反向。

Needs CUDA + xformers（varlen kernels GPU-only）；缺则 skip。
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xformers")

from modeling.anima_modeling import Anima
from modeling.cosmos_predict2_modeling import set_xformers_enabled
from training.losses.mse import MseLoss
from training.navit import navit_packed_forward_and_loss, pack_cross_embeddings

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA + xformers"
)


def _model(dtype):
    torch.manual_seed(0)
    m = Anima(
        max_img_h=64,
        max_img_w=64,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=128,
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2.0,
        crossattn_emb_channels=128,
        pos_emb_cls="rope3d",
    )
    return m.to(device="cuda", dtype=dtype).eval()


@requires_cuda
def test_pack_cross_embeddings_no_trim():
    """Legacy 512-pad: 每图带完整 caption，ΣL = G × 512。"""
    G, L, D = 3, 512, 128
    cross = torch.randn(G, L, D)
    t5_attn = torch.ones(G, L)
    packed, seqlens = pack_cross_embeddings(cross, t5_attn, navit_text_trim_padding=False)
    assert packed.shape == (1, G * L, D)
    assert seqlens == [L] * G
    assert sum(seqlens) == packed.shape[1]


@requires_cuda
def test_pack_cross_embeddings_trim():
    """Trim padding: 只拼有效 token，ΣL = sum(valid per image)。"""
    G, L, D = 3, 512, 128
    cross = torch.randn(G, L, D)
    # 每图不同有效长度
    valid_lens = [10, 77, 256]
    t5_attn = torch.zeros(G, L)
    for i, n in enumerate(valid_lens):
        t5_attn[i, :n] = 1.0
    packed, seqlens = pack_cross_embeddings(cross, t5_attn, navit_text_trim_padding=True)
    assert seqlens == valid_lens
    assert sum(seqlens) == packed.shape[1]
    # 验证拼接顺序：第一段 == cross[0, :10]
    torch.testing.assert_close(packed[0, :10, :], cross[0, :10, :])


@requires_cuda
def test_navit_forward_and_loss_basic():
    """navit_packed_forward_and_loss: loss 有限、带梯度、per_image 形状 [G]。"""
    set_xformers_enabled(True)
    dtype = torch.float16
    model = _model(dtype)
    D = 128

    latent_shapes = [(4, 4), (6, 8), (4, 6)]
    text_lens = [5, 9, 3]
    timesteps = [0.2, 0.6, 0.9]

    latents_list = [
        torch.randn(1, 16, 1, h, w, device="cuda", dtype=dtype)
        for h, w in latent_shapes
    ]
    cross_list = [torch.randn(1, L, D, device="cuda", dtype=dtype) for L in text_lens]
    cross_packed = torch.cat(cross_list, dim=1)
    t = torch.tensor(timesteps, device="cuda", dtype=dtype)

    loss_fn = MseLoss()
    loss, pred, info = navit_packed_forward_and_loss(
        model, latents_list, t, cross_packed, text_lens, loss_fn,
    )

    assert torch.isfinite(loss)
    assert loss.requires_grad
    assert info["per_image_loss"].shape == (len(latent_shapes),)
    assert all(torch.isfinite(v) for v in info["per_image_loss"])
    # 预测 token 数 == ΣN
    assert pred.shape[1] == sum(info["visual_seqlens"])

    # 梯度可反向
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


@requires_cuda
def test_navit_checkpoint_equivalence():
    """grad_checkpoint 路径 ≡ 非检查点路径，且仍可反向。"""
    set_xformers_enabled(True)
    dtype = torch.float16
    model = _model(dtype)
    D = 128

    latent_shapes = [(4, 4), (6, 8)]
    text_lens = [5, 9]
    timesteps = [0.3, 0.7]

    latents_list = [
        torch.randn(1, 16, 1, h, w, device="cuda", dtype=dtype)
        for h, w in latent_shapes
    ]
    cross_list = [torch.randn(1, L, D, device="cuda", dtype=dtype) for L in text_lens]
    cross_packed = torch.cat(cross_list, dim=1)
    t = torch.tensor(timesteps, device="cuda", dtype=dtype)
    loss_fn = MseLoss()

    with torch.no_grad():
        torch.manual_seed(42)
        _, pred_plain, _ = navit_packed_forward_and_loss(
            model, latents_list, t, cross_packed, text_lens, loss_fn,
            use_checkpoint=False,
        )
        torch.manual_seed(42)
        _, pred_ckpt, _ = navit_packed_forward_and_loss(
            model, latents_list, t, cross_packed, text_lens, loss_fn,
            use_checkpoint=True,
        )
    torch.testing.assert_close(pred_ckpt, pred_plain, rtol=2e-3, atol=2e-3)

    # 检查点路径仍可反向
    torch.manual_seed(42)
    loss, _, _ = navit_packed_forward_and_loss(
        model, latents_list, t, cross_packed, text_lens, loss_fn,
        use_checkpoint=True,
    )
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all()
        for p in model.parameters() if p.grad is not None
    )


@requires_cuda
def test_navit_per_image_weights():
    """per_image_weights（正则集 loss_weight × loss_weighting）按 per-image 缩放 loss。

    覆盖 navit 路径补全的两项加权：传入 [G] 权重后，per-image loss 与总 loss 均按权重
    缩放（与标准路径 per-sample 加权对称）。
    """
    set_xformers_enabled(True)
    dtype = torch.float16
    model = _model(dtype)
    D = 128

    latent_shapes = [(4, 4), (6, 8)]
    text_lens = [5, 9]
    timesteps = [0.3, 0.7]
    latents_list = [
        torch.randn(1, 16, 1, h, w, device="cuda", dtype=dtype)
        for h, w in latent_shapes
    ]
    cross_list = [torch.randn(1, L, D, device="cuda", dtype=dtype) for L in text_lens]
    cross_packed = torch.cat(cross_list, dim=1)
    t = torch.tensor(timesteps, device="cuda", dtype=dtype)
    loss_fn = MseLoss()

    # 基线（无 per-image 权重）
    torch.manual_seed(42)
    _, _, info0 = navit_packed_forward_and_loss(
        model, latents_list, t, cross_packed, text_lens, loss_fn,
    )
    base = info0["per_image_loss"].clone().to(torch.float32)  # [G]

    # per-image 权重 [2.0, 0.5]
    w = torch.tensor([2.0, 0.5], device="cuda", dtype=torch.float32)
    torch.manual_seed(42)
    loss_w, _, info_w = navit_packed_forward_and_loss(
        model, latents_list, t, cross_packed, text_lens, loss_fn,
        per_image_weights=w,
    )
    # per-image loss 被权重缩放
    torch.testing.assert_close(
        info_w["per_image_loss"].to(torch.float32), base * w, rtol=2e-3, atol=2e-3,
    )
    # 总 loss = mean(per_image × weights)
    torch.testing.assert_close(
        loss_w.detach().to(torch.float32), (base * w).mean(), rtol=2e-3, atol=2e-3,
    )
