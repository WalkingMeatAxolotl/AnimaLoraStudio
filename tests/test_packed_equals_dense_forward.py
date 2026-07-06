"""NaViT / Patch-n-Pack — dense vs packed forward equivalence.

Equivalence probe: ``forward_packed_tokens`` vs dense ``forward``.

The NaViT training path runs ``forward_packed_navit`` (block-diagonal pack) whose
per-image reference is ``forward_packed_tokens``. Eval / ARB training run the dense
``forward``. The existing suite only proved ``forward_packed_navit`` ≡
``forward_packed_tokens`` (test_packed_navit_forward) — the dense↔packed equivalence
was never asserted. On top of that, those tests construct the model with
``concat_padding_mask=False``, whereas the real Anima is built with
``concat_padding_mask=True``, so the mask-channel branch in
``prepare_embedded_sequence`` — which dense runs but packed skips — was unexercised.

This test builds the model the way the trainer does (``concat_padding_mask=True``)
and checks, for a single image with an all-zero padding mask (exactly what both the
ARB training loop and eval feed), that:

    unpatchify_tokens(forward_packed_tokens(...))  ==  forward(...)

Uses ``assert_close`` (not ``torch.equal``) because fp16 inputs traverse different
rearrange sequences (PatchEmbed vs patchify_latents_to_tokens) and different
attention dispatch paths; while mathematically identical, fp16 reduction-order
differences preclude a byte-equality claim.

Needs CUDA + xformers; skipped otherwise.
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


def _model(dtype, concat_padding_mask=True):
    torch.manual_seed(0)
    m = Anima(
        max_img_h=64,
        max_img_w=64,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=concat_padding_mask,
        model_channels=128,   # /num_heads=2 -> head_dim 64
        num_blocks=2,
        num_heads=2,
        mlp_ratio=2.0,
        crossattn_emb_channels=128,
        pos_emb_cls="rope3d",
    )
    return m.to(device="cuda", dtype=dtype).eval()


def _run_one(model, lat, t_val, cross, dtype):
    """Return (dense_out, packed_out) both as [1,C,T,h,w] grids for one image."""
    h, w = lat.shape[-2], lat.shape[-1]
    # The all-zero padding mask is exactly what ARB training and eval feed into
    # dense forward.
    pad_mask = torch.zeros(1, 1, h, w, device="cuda", dtype=dtype)
    t = torch.tensor([t_val], device="cuda", dtype=dtype)

    with torch.no_grad():
        # Dense path: returns unpatchified [1,C,T,H,W].
        dense_out = model.forward(lat, t.view(-1, 1), cross, padding_mask=pad_mask)

        # Packed path: returns patch tokens [1,N,M]; unpatchify back to grid.
        tok, grid, _mask, size = model.patchify_latents_to_tokens(lat)
        mask_ones = torch.ones(1, tok.shape[1], device="cuda", dtype=dtype)
        packed_tok = model.forward_packed_tokens(
            tok, t.view(-1, 1), cross, grid, mask_ones, size,
        )
        packed_out = model.unpatchify_tokens(packed_tok, size)
    return dense_out, packed_out


@requires_cuda
def test_packed_equals_dense_concat_mask_true():
    """Real Anima config: concat_padding_mask=True, all-zero pad mask."""
    set_xformers_enabled(True)
    dtype = torch.float16
    model = _model(dtype, concat_padding_mask=True)
    D = 128
    # A few shapes / timesteps / caption lengths to stress the path.
    cases = [
        ((8, 8), 0.2, 7),
        ((6, 10), 0.6, 11),
        ((10, 6), 0.9, 3),
    ]
    max_abs_err = 0.0
    max_rel_err = 0.0
    for (shape, t_val, L) in cases:
        lat = torch.randn(1, 16, 1, shape[0], shape[1], device="cuda", dtype=dtype)
        cross = torch.randn(1, L, D, device="cuda", dtype=dtype)
        dense_out, packed_out = _run_one(model, lat, t_val, cross, dtype)
        assert tuple(dense_out.shape) == tuple(packed_out.shape), (
            f"shape mismatch for case {shape}"
        )
        # Report the worst-case error so a near-miss is visible, not just pass/fail.
        diff = (dense_out.float() - packed_out.float()).abs()
        max_abs_err = max(max_abs_err, float(diff.max().item()))
        denom = dense_out.float().abs().clamp(min=1e-3)
        max_rel_err = max(max_rel_err, float((diff / denom).max().item()))
    print(f"\n[concat_padding_mask=True] max_abs_err={max_abs_err:.6e} "
          f"max_rel_err={max_rel_err:.6e}")
    # fp16 + xformers varlen vs SDPA: allow a generous tolerance; if it passes
    # tight, the paths are equivalent.
    torch.testing.assert_close(dense_out, packed_out, rtol=5e-2, atol=5e-2)


@requires_cuda
def test_packed_equals_dense_concat_mask_false():
    """Sanity: with concat_padding_mask=False the mask-channel branch is gone,
    so dense and packed should be equivalent. Confirms the test harness is sound."""
    set_xformers_enabled(True)
    dtype = torch.float16
    model = _model(dtype, concat_padding_mask=False)
    D = 128
    lat = torch.randn(1, 16, 1, 8, 8, device="cuda", dtype=dtype)
    cross = torch.randn(1, 7, D, device="cuda", dtype=dtype)
    dense_out, packed_out = _run_one(model, lat, 0.5, cross, dtype)
    diff = (dense_out.float() - packed_out.float()).abs()
    print(f"\n[concat_padding_mask=False] max_abs_err={float(diff.max().item()):.6e}")
    torch.testing.assert_close(dense_out, packed_out, rtol=5e-2, atol=5e-2)
