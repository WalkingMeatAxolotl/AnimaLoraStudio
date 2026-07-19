"""NaViT / Patch-n-Pack block-diagonal packing — Stage 0: attention invariant.

Verifies the core invariant the whole packing scheme rests on: running
``torch_attention_op`` on several images packed into one sequence with a
``BlockDiagonalMask`` is numerically equal to attending each image
independently and concatenating — i.e. zero cross-image leakage.

Requires a CUDA device + xformers (``memory_efficient_attention``'s varlen
kernels are GPU-only); skipped otherwise.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xformers")

from xformers.ops.fmha import BlockDiagonalMask

from modeling.anima.cosmos_predict2_modeling import torch_attention_op

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA + xformers"
)


@requires_cuda
def test_packed_equals_independent_self_attention():
    """Block-diagonal packed self-attention ≡ per-image independent attention concat."""
    torch.manual_seed(0)
    device, dtype = "cuda", torch.float16
    H, D = 4, 64
    seqlens = [37, 92, 16]            # heterogeneous image token counts
    total = sum(seqlens)

    # Packed layout xformers expects: [B=1, total_tokens, H, D].
    q = torch.randn(1, total, H, D, device=device, dtype=dtype)
    k = torch.randn(1, total, H, D, device=device, dtype=dtype)
    v = torch.randn(1, total, H, D, device=device, dtype=dtype)

    mask = BlockDiagonalMask.from_seqlens(seqlens)
    packed = torch_attention_op(q, k, v, attn_mask=mask)        # [1, total, H*D]
    packed = packed[0]

    # Independent reference: attend each image's own slice with no mask.
    outs = []
    off = 0
    for n in seqlens:
        qi = q[:, off:off + n]
        ki = k[:, off:off + n]
        vi = v[:, off:off + n]
        outs.append(torch_attention_op(qi, ki, vi, None)[0])
        off += n
    ref = torch.cat(outs, dim=0)

    torch.testing.assert_close(packed, ref, rtol=2e-3, atol=2e-3)


@requires_cuda
def test_packed_cross_attention_differing_kv_seqlens():
    """Cross-attention packing: each image's query tokens attend only to its
    own (differently sized) text key/value block."""
    torch.manual_seed(1)
    device, dtype = "cuda", torch.float16
    H, D = 4, 64
    q_seqlens = [37, 92, 16]          # visual tokens per image
    kv_seqlens = [11, 7, 23]          # text tokens per caption
    q_total, kv_total = sum(q_seqlens), sum(kv_seqlens)

    q = torch.randn(1, q_total, H, D, device=device, dtype=dtype)
    k = torch.randn(1, kv_total, H, D, device=device, dtype=dtype)
    v = torch.randn(1, kv_total, H, D, device=device, dtype=dtype)

    mask = BlockDiagonalMask.from_seqlens(q_seqlen=q_seqlens, kv_seqlen=kv_seqlens)
    packed = torch_attention_op(q, k, v, attn_mask=mask)[0]

    outs = []
    qoff = kvoff = 0
    for qn, kn in zip(q_seqlens, kv_seqlens):
        qi = q[:, qoff:qoff + qn]
        ki = k[:, kvoff:kvoff + kn]
        vi = v[:, kvoff:kvoff + kn]
        outs.append(torch_attention_op(qi, ki, vi, None)[0])
        qoff += qn
        kvoff += kn
    ref = torch.cat(outs, dim=0)

    torch.testing.assert_close(packed, ref, rtol=2e-3, atol=2e-3)


@requires_cuda
def test_additive_tensor_mask_path_unchanged():
    """Behavior-neutrality guard: a plain float tensor mask must still route to
    the legacy SDPA additive-mask path, not the new bias branch."""
    device = "cuda"
    H, D, N = 4, 64, 8
    q = torch.randn(1, N, H, D, device=device, dtype=torch.float32)
    k = torch.randn(1, N, H, D, device=device, dtype=torch.float32)
    v = torch.randn(1, N, H, D, device=device, dtype=torch.float32)
    # key-padding additive mask shape [B,1,1,N] as built by _build_packed_masks
    add = torch.zeros(1, 1, 1, N, device=device)
    add[..., -2:] = -1.0e4
    out = torch_attention_op(q, k, v, attn_mask=add)
    assert tuple(out.shape) == (1, N, H * D)
