"""NaViT / Patch-n-Pack — Stage 2: end-to-end model equivalence.

End-to-end model equivalence: ``MiniTrainDIT.forward_packed_navit`` on G
heterogeneous images packed into one sequence must equal running each image
independently through the dense ``forward`` (its own timestep) and, after
re-``patchify``-ing each dense latent output back into the packed
``(c pt ph pw)`` channel order, concatenating the per-image token outputs.

This pins down everything the packed path adds at once: block-diagonal self
attention, block-diagonal cross attention to per-image captions, per-image
RoPE grids, and per-token AdaLN with one timestep per image.

Needs CUDA + xformers (varlen kernels are GPU-only); skipped otherwise.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xformers")

from modeling.anima_modeling import Anima
from modeling.cosmos_predict2_modeling import set_xformers_enabled

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
        model_channels=128,   # /num_heads=2 -> head_dim 64 (xformers varlen needs 64/128/...)
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2.0,
        crossattn_emb_channels=128,
        pos_emb_cls="rope3d",
    )
    return m.to(device="cuda", dtype=dtype).eval()


@requires_cuda
def test_navit_equals_per_image_dense_forward():
    """``forward_packed_navit`` ≡ 各图独立走 dense ``forward``（patchify 对齐输出）。

    per-image 参照改用 dense ``forward``（原先用已删的 ``forward_packed_tokens``）：
    每张图单独走 dense forward 得预测 latent，用 ``patchify_latents_to_tokens`` 转到
    与 packed 输出相同的 ``(c pt ph pw)`` 通道序后拼接，对比 ``forward_packed_navit``。
    """
    set_xformers_enabled(True)
    dtype = torch.float16
    model = _model(dtype)
    D = 128

    # Heterogeneous latents → token counts N = (h/2)*(w/2) = 4, 12, 6.
    latent_shapes = [(4, 4), (6, 8), (4, 6)]
    text_lens = [5, 9, 3]
    timesteps = [0.2, 0.6, 0.9]

    tokens_list, grid_list, vseq = [], [], []
    cross_list, tseq, refs = [], [], []
    with torch.no_grad():
        for (h, w), L, t_val in zip(latent_shapes, text_lens, timesteps):
            lat = torch.randn(1, 16, 1, h, w, device="cuda", dtype=dtype)
            cross = torch.randn(1, L, D, device="cuda", dtype=dtype)
            tok, grid, _mask, _size = model.patchify_latents_to_tokens(lat)
            tokens_list.append(tok)
            grid_list.append(grid)
            vseq.append(tok.shape[1])
            cross_list.append(cross)
            tseq.append(L)

            # Per-image reference: dense forward → predicted latent → re-patchify
            # into the same (c pt ph pw) token order the packed path emits.
            dense_out = model.forward(
                lat, torch.tensor([[t_val]], device="cuda", dtype=dtype), cross
            )
            ref_tok, _, _, _ = model.patchify_latents_to_tokens(dense_out)
            refs.append(ref_tok)
        ref = torch.cat(refs, dim=1)

        # Packed NaViT forward.
        tokens_packed = torch.cat(tokens_list, dim=1)
        grid_packed = torch.cat(grid_list, dim=2)
        cross_packed = torch.cat(cross_list, dim=1)
        ts = torch.tensor(timesteps, device="cuda", dtype=dtype)
        packed = model.forward_packed_navit(
            tokens_packed, ts, cross_packed, grid_packed, vseq, tseq
        )

    assert tuple(packed.shape) == tuple(ref.shape)
    torch.testing.assert_close(packed, ref, rtol=5e-2, atol=5e-2)


@requires_cuda
def test_navit_checkpoint_matches_non_checkpoint():
    """Per-block gradient checkpointing must be numerically equivalent to the plain
    forward (same packed output), and must still backprop."""
    set_xformers_enabled(True)
    dtype = torch.float16
    model = _model(dtype)
    D = 128
    latent_shapes = [(4, 4), (6, 8), (4, 6)]
    text_lens = [5, 9, 3]
    timesteps = [0.2, 0.6, 0.9]

    toks, grids, vseq, cross_list, tseq = [], [], [], [], []
    for (h, w), L in zip(latent_shapes, text_lens):
        lat = torch.randn(1, 16, 1, h, w, device="cuda", dtype=dtype)
        tok, grid, _m, _s = model.patchify_latents_to_tokens(lat)
        toks.append(tok); grids.append(grid); vseq.append(tok.shape[1])
        cross_list.append(torch.randn(1, L, D, device="cuda", dtype=dtype)); tseq.append(L)
    tokens = torch.cat(toks, dim=1)
    grid = torch.cat(grids, dim=2)
    cross = torch.cat(cross_list, dim=1)
    ts = torch.tensor(timesteps, device="cuda", dtype=dtype)

    with torch.no_grad():
        plain = model.forward_packed_navit(tokens, ts, cross, grid, vseq, tseq,
                                            use_checkpoint=False)
        ckpt = model.forward_packed_navit(tokens, ts, cross, grid, vseq, tseq,
                                          use_checkpoint=True)
    torch.testing.assert_close(ckpt, plain, rtol=2e-3, atol=2e-3)

    # checkpoint path still backprops
    tokens_g = tokens.clone().requires_grad_(True)
    out = model.forward_packed_navit(tokens_g, ts, cross, grid, vseq, tseq,
                                     use_checkpoint=True)
    out.float().pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


@requires_cuda
def test_navit_rejects_seqlen_mismatch():
    """``visual_seqlens`` sum != packed token count must raise ValueError."""
    set_xformers_enabled(True)
    model = _model(torch.float16)
    tok = torch.randn(1, 10, model.x_embedder.proj[1].in_features,
                      device="cuda", dtype=torch.float16)
    grid = torch.zeros(1, 2, 10, device="cuda", dtype=torch.float16)
    cross = torch.randn(1, 4, 128, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        model.forward_packed_navit(
            tok, torch.tensor([0.5], device="cuda"), cross, grid,
            visual_seqlens=[4, 4],  # sums to 8, not 10
            text_seqlens=[4],
        )
