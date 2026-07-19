"""NaViT per-image AdaLN modulation path (``mod_index``) ≡ legacy per-token layout.

The NaViT packed forward can either expand per-image t_emb/adaln_lora via
``repeat_interleave`` to [1, ΣN, *] (legacy) or keep them at [1, G, *] and
gather per-token via ``index_select(mod_index)`` (new path). Same value within
a row → mathematically equivalent; differences can only come from GEMM row-count
differences in floating-point reduction order.

This test asserts Block (bf16 — varlen attention has no fp32 kernel on newer
archs like RTX 5090 sm_120) / FinalLayer (fp32, no attention) equivalence for
both layouts (AdaLN-LoRA on/off), and asserts ``index_select`` output contiguity.

Needs CUDA + xformers (block-diagonal varlen attention); skipped otherwise.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("xformers")

from xformers.ops.fmha import BlockDiagonalMask

from modeling.anima.cosmos_predict2_modeling import Block, FinalLayer

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA + xformers"
)

D = 128          # /num_heads=2 -> head_dim 64 (xformers varlen supports 64/128)
CTX = 96
COUNTS = [4, 12, 6]          # G=3 heterogeneous image token counts
TEXT = [5, 9, 3]


def _mk_inputs(dtype=torch.float32):
    torch.manual_seed(0)
    G = len(COUNTS)
    SN = sum(COUNTS)
    x = torch.randn(1, SN, D, device="cuda", dtype=dtype)
    cross = torch.randn(1, sum(TEXT), CTX, device="cuda", dtype=dtype)
    emb_G = torch.randn(1, G, D, device="cuda", dtype=dtype)
    lora_G = torch.randn(1, G, 3 * D, device="cuda", dtype=dtype)
    counts = torch.tensor(COUNTS, device="cuda")
    mod_index = torch.repeat_interleave(
        torch.arange(G, device="cuda"), counts
    )
    # legacy per-token layout: same row repeat_interleave-expanded
    emb_tok = emb_G[0].repeat_interleave(counts, dim=0).unsqueeze(0)
    lora_tok = lora_G[0].repeat_interleave(counts, dim=0).unsqueeze(0)
    self_bias = BlockDiagonalMask.from_seqlens(COUNTS)
    cross_bias = BlockDiagonalMask.from_seqlens(
        q_seqlen=COUNTS, kv_seqlen=TEXT
    )
    return (x, cross, emb_G, lora_G, emb_tok, lora_tok,
            mod_index, self_bias, cross_bias)


def _block(use_adaln_lora, dtype=torch.float32):
    torch.manual_seed(1)
    blk = Block(D, CTX, num_heads=2, mlp_ratio=2.0,
                use_adaln_lora=use_adaln_lora, adaln_lora_dim=16,
                self_attention_backend="torch", cross_attention_backend="torch")
    return blk.to(device="cuda", dtype=dtype).eval()


def _assert_block_equiv(use_adaln_lora):
    # block-diagonal varlen attention 只有 fp16/bf16 kernel（新架构如 RTX 5090
    # sm_120 无 fp32 varlen），且 navit 实际以 bf16/fp16 跑——用 bf16 反映真实运行。
    dtype = torch.bfloat16
    blk = _block(use_adaln_lora, dtype)
    (x, cross, emb_G, lora_G, emb_tok, lora_tok,
     mod_index, self_bias, cross_bias) = _mk_inputs(dtype)
    lora_G_arg = lora_G if use_adaln_lora else None
    lora_tok_arg = lora_tok if use_adaln_lora else None
    with torch.no_grad():
        legacy = blk.forward_tokens(
            x, emb_tok, cross, rope_emb_L_1_1_D=None,
            attn_mask=self_bias,
            adaln_lora_B_T_3D=lora_tok_arg, cross_attn_mask=cross_bias,
            token_wise_mod=True,
        )
        new = blk.forward_tokens(
            x, emb_G, cross, rope_emb_L_1_1_D=None,
            attn_mask=self_bias,
            adaln_lora_B_T_3D=lora_G_arg, cross_attn_mask=cross_bias,
            token_wise_mod=True, mod_index=mod_index,
        )
    # bf16 varlen: same-value rows via different M GEMMs; tolerance widened to
    # bf16 reduction level (fp32 varlen has no kernel on sm_120 / RTX 5090).
    torch.testing.assert_close(new, legacy, rtol=1.6e-2, atol=1e-2)


@requires_cuda
def test_block_equiv_adaln_lora():
    _assert_block_equiv(use_adaln_lora=True)


@requires_cuda
def test_block_equiv_plain_adaln():
    _assert_block_equiv(use_adaln_lora=False)


@requires_cuda
def test_final_layer_equiv():
    torch.manual_seed(2)
    fl = FinalLayer(D, spatial_patch_size=2, temporal_patch_size=1,
                    out_channels=16, use_adaln_lora=True, adaln_lora_dim=16)
    # init_weights zeros the modulation output → shift/scale always 0; re-randomize
    for lin in (fl.adaln_modulation[1], fl.adaln_modulation[2]):
        torch.nn.init.normal_(lin.weight, std=0.05)
    fl = fl.to(device="cuda", dtype=torch.float32).eval()
    (x, _cross, emb_G, lora_G, emb_tok, lora_tok,
     mod_index, _sb, _cb) = _mk_inputs()
    with torch.no_grad():
        legacy = fl.forward_tokens(
            x, emb_tok, adaln_lora_B_T_3D=lora_tok, token_wise_mod=True,
        )
        new = fl.forward_tokens(
            x, emb_G, adaln_lora_B_T_3D=lora_G, token_wise_mod=True,
            mod_index=mod_index,
        )
    torch.testing.assert_close(new, legacy, rtol=1e-5, atol=1e-5)


@requires_cuda
def test_gathered_mod_is_contiguous():
    """Performance invariant: ``index_select`` output must be contiguous
    (non-contiguous strided views are one reason the legacy layout is slower)."""
    counts = torch.tensor(COUNTS, device="cuda")
    mod_index = torch.repeat_interleave(torch.arange(3, device="cuda"), counts)
    chunk = torch.randn(1, 3, 3 * D, device="cuda").chunk(3, dim=-1)[0]
    assert not chunk.is_contiguous()          # chunk view itself is non-contiguous
    gathered = chunk.index_select(1, mod_index)
    assert gathered.is_contiguous()            # gather makes it contiguous
    assert tuple(gathered.shape) == (1, sum(COUNTS), D)
