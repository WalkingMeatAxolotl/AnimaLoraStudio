"""MOSS NaViT 打包训练（navit 能力位）。

- ``MossImageTransformer.forward_packed`` 与逐样本 ``forward`` 数值 / 梯度一致，
  保留各输入形状，每层只调用一次（block swap 每层每步只搬一次权重）
- 块对角注意力：逐段 SDPA 与 xformers ``BlockDiagonalMask`` 一致
- 训练步：逐图加噪、loss 图内均值再图间平均（两族同语义）
- 族钩子：打包族都提供 navit_encode_text / navit_forward_and_loss；MOSS 的文本条件
  去 padding，文本 token 计入打包预算（Anima 文本走 cross-attn，不计）
- 数据层：按索引取 caption、打包器叠加文本 token
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from modeling.attention import (
    SDPA,
    XFORMERS,
    attention_block_diagonal_bshd,
    attention_bshd,
    backend_available,
)
from modeling.moss.moss_modeling import MossConfig, MossImageTransformer
from studio.domain.common import FAMILY_CAPABILITIES
from training.dataset import (
    MergedDataset,
    NavitPackBatchSampler,
    RepeatDataset,
    dataset_captions,
)
from training.families import get_family
from training.families.moss import MossFamily
from training.families.moss.forward import MossTextCondition
from training.losses.mse import MseLoss
from training.phases.dataset import _navit_text_token_counts
import training.families.moss.navit as moss_navit

_CFG = MossConfig(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=32, freq_dim=16)


def _model() -> MossImageTransformer:
    torch.manual_seed(0)
    return MossImageTransformer(_CFG).double()


def _pack(seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    grids = [(8, 12), (6, 6), (4, 10)]
    text_lens = [5, 9, 2]
    latents = [torch.randn(1, 32, 1, h, w, generator=g, dtype=torch.float64) for h, w in grids]
    contexts = [torch.randn(n, _CFG.text_dim, generator=g, dtype=torch.float64) for n in text_lens]
    t = torch.tensor([0.2, 0.7, 0.95], dtype=torch.float64)
    return latents, contexts, t


def _per_sample(model, latents, contexts, t):
    return [model(x, t[i:i + 1], c[None]) for i, (x, c) in enumerate(zip(latents, contexts))]


# ── forward_packed ──────────────────────────────────────────────────────


def test_forward_packed_matches_per_sample_forward_and_grads() -> None:
    model = _model().train()
    latents, contexts, t = _pack()

    packed = model.forward_packed(latents, t, contexts, use_checkpoint=True)
    sum(p.square().mean() for p in packed).backward()
    packed_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()

    ref = _per_sample(model, latents, contexts, t)
    sum(r.square().mean() for r in ref).backward()

    for out, expected in zip(packed, ref):
        assert out.shape == expected.shape
        torch.testing.assert_close(out, expected, rtol=0, atol=1e-12)
    assert packed_grads.keys() == {n for n, p in model.named_parameters() if p.grad is not None}
    for name, param in model.named_parameters():
        if param.grad is not None:
            torch.testing.assert_close(packed_grads[name], param.grad, rtol=0, atol=1e-12)


def test_forward_packed_keeps_each_input_shape() -> None:
    model = _model().eval()
    g = torch.Generator().manual_seed(2)
    latents = [
        torch.randn(32, 4, 6, generator=g, dtype=torch.float64),
        torch.randn(1, 32, 6, 4, generator=g, dtype=torch.float64),
        torch.randn(1, 32, 1, 2, 2, generator=g, dtype=torch.float64),
    ]
    contexts = [torch.randn(1, 3, _CFG.text_dim, generator=g, dtype=torch.float64) for _ in latents]
    with torch.no_grad():
        outs = model.forward_packed(latents, torch.tensor([0.1, 0.5, 0.9]), contexts)
    assert [o.shape for o in outs] == [x.shape for x in latents]


def test_forward_packed_calls_each_layer_once_per_pack() -> None:
    model = _model().eval()
    calls = [0] * len(model.layers)
    for i, layer in enumerate(model.layers):
        layer.register_forward_pre_hook(lambda *_a, i=i: calls.__setitem__(i, calls[i] + 1))
    latents, contexts, t = _pack()
    with torch.no_grad():
        model.forward_packed(latents, t, contexts)
    assert calls == [1] * len(model.layers)


def test_forward_packed_rejects_inconsistent_inputs() -> None:
    model = _model().eval()
    latents, contexts, t = _pack()
    with pytest.raises(ValueError, match="一一对应"):
        model.forward_packed(latents, t, contexts[:2])
    with pytest.raises(ValueError, match="timestep"):
        model.forward_packed(latents, t[:2], contexts)
    with pytest.raises(ValueError, match="有效文本"):
        model.forward_packed(latents, t, [contexts[0], contexts[1][:0], contexts[2]])


# ── 块对角注意力 ────────────────────────────────────────────────────────


def _qkv(seqlens, device="cpu", dtype=torch.float32):
    g = torch.Generator(device=device).manual_seed(3)
    shape = (1, sum(seqlens), 2, 16)
    return [torch.randn(shape, generator=g, device=device, dtype=dtype) for _ in range(3)]


def _segment_reference(q, k, v, seqlens):
    return torch.cat([
        attention_bshd(qs, ks, vs, SDPA)
        for qs, ks, vs in zip(q.split(seqlens, 1), k.split(seqlens, 1), v.split(seqlens, 1))
    ], 1)


def test_block_diagonal_attention_isolates_segments() -> None:
    seqlens = (5, 3, 7)
    q, k, v = _qkv(seqlens)
    out = attention_block_diagonal_bshd(q, k, v, seqlens, SDPA)
    torch.testing.assert_close(out, _segment_reference(q, k, v, seqlens))
    # 改动第 2 段的 k/v，其余段输出不变
    k2, v2 = k.clone(), v.clone()
    k2[:, 5:8] += 1.0
    v2[:, 5:8] -= 1.0
    out2 = attention_block_diagonal_bshd(q, k2, v2, seqlens, SDPA)
    torch.testing.assert_close(out2[:, :5], out[:, :5])
    torch.testing.assert_close(out2[:, 8:], out[:, 8:])
    with pytest.raises(ValueError, match="段长"):
        attention_block_diagonal_bshd(q, k, v, (5, 3), SDPA)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not backend_available(XFORMERS),
    reason="需要 CUDA + xformers",
)
def test_block_diagonal_xformers_matches_segment_sdpa() -> None:
    seqlens = (64, 40, 96)
    q, k, v = _qkv(seqlens, device="cuda", dtype=torch.bfloat16)
    out = attention_block_diagonal_bshd(q, k, v, seqlens, XFORMERS)
    torch.testing.assert_close(out, _segment_reference(q, k, v, seqlens), rtol=2e-2, atol=2e-2)


# ── 训练步 ──────────────────────────────────────────────────────────────


def test_navit_loss_is_mean_over_images_of_per_image_mean(monkeypatch) -> None:
    model = _model().eval()
    latents, contexts, t = _pack()
    g = torch.Generator().manual_seed(4)
    noises = [torch.randn(x.shape, generator=g, dtype=x.dtype) for x in latents]
    queue = list(noises)
    monkeypatch.setattr(moss_navit, "make_noise", lambda x, **_kw: queue.pop(0))
    weights = torch.tensor([1.0, 0.5, 2.0])

    with torch.no_grad():
        # collate_fn_navit_pack 产出的逐图 latent 是 [C, T, h, w]（无 batch 维）
        loss, preds, info = moss_navit.navit_packed_forward_and_loss(
            model, [x[0] for x in latents], t, contexts, MseLoss(), per_image_weights=weights,
        )
        expected = []
        for i, (x, n, c) in enumerate(zip(latents, noises, contexts)):
            noisy = (1 - t[i]) * x + t[i] * n
            pred = model(noisy, t[i:i + 1], c[None])
            expected.append((pred - (n - x)).square().mean())
        expected = torch.stack(expected) * weights.double()

    assert len(preds) == 3
    torch.testing.assert_close(info["per_image_loss"], expected.to(info["per_image_loss"].dtype))
    torch.testing.assert_close(loss, expected.mean().to(loss.dtype))
    assert info["visual_seqlens"] == [24, 9, 10]


# ── 族钩子 ──────────────────────────────────────────────────────────────


def test_navit_families_provide_hooks() -> None:
    families = sorted(f for f, caps in FAMILY_CAPABILITIES.items() if "navit" in caps)
    assert families == ["anima", "moss"]
    for fam in families:
        family = get_family(fam)
        assert callable(getattr(family, "navit_encode_text", None)), fam
        assert callable(getattr(family, "navit_forward_and_loss", None)), fam
    # 文本只在单流序列里占预算：Anima 走 cross-attn，不提供计数钩子
    assert not hasattr(get_family("anima"), "navit_text_token_count")
    assert callable(getattr(get_family("moss"), "navit_text_token_count", None))


def test_moss_navit_encode_text_strips_padding() -> None:
    context = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    calls = []

    class _Text:
        def encode_text_for_batch(self, captions, *, device, dtype):
            calls.append(list(captions))
            return MossTextCondition(context, mask)

    out = MossFamily().navit_encode_text(
        _Text(), None, ["a", "b"], "cpu", torch.float32, text_trim_padding=False,
    )
    assert calls == [["a", "b"]]
    torch.testing.assert_close(out[0], context[0, :3])
    torch.testing.assert_close(out[1], context[1, :2])


def test_moss_navit_text_token_count_uses_stack() -> None:
    text = SimpleNamespace(token_count=lambda caption: len(caption) + 10)
    assert MossFamily().navit_text_token_count(text, "abc") == 13


# ── 数据层 ──────────────────────────────────────────────────────────────


class _TokenDataset:
    def __init__(self, token_counts):
        self.token_count_for_index = list(token_counts)

    def __len__(self):
        return len(self.token_count_for_index)


class _CaptionDataset:
    def __init__(self, captions):
        self.captions = list(captions)

    def caption_for_index(self, idx):
        return self.captions[idx]

    def __len__(self):
        return len(self.captions)


def test_pack_sampler_adds_text_tokens_to_budget() -> None:
    image_only = NavitPackBatchSampler(_TokenDataset([10] * 6), token_budget=30, shuffle=False)
    with_text = NavitPackBatchSampler(
        _TokenDataset([10] * 6), token_budget=30, shuffle=False, text_token_counts=[5] * 6,
    )
    assert with_text.token_counts == [15] * 6
    assert [len(p) for p in image_only] == [3, 3]
    assert [len(p) for p in with_text] == [2, 2, 2]
    with pytest.raises(ValueError, match="text_token_counts"):
        NavitPackBatchSampler(_TokenDataset([10] * 6), token_budget=30, text_token_counts=[5])


def test_dataset_captions_follows_repeat_and_merge_order() -> None:
    main = RepeatDataset(_CaptionDataset(["a", "b"]), repeats=2)
    merged = MergedDataset(main, _CaptionDataset(["r"]))
    assert dataset_captions(merged) == ["a", "b", "a", "b", "r"]


def test_phase_text_token_counts_only_for_counting_families() -> None:
    dataset = RepeatDataset(_CaptionDataset(["aa", "bbbb"]), repeats=3)
    assert _navit_text_token_counts(SimpleNamespace(family=object(), dataset=dataset)) is None

    seen = []

    class _Family:
        def navit_text_token_count(self, text, caption):
            seen.append(caption)
            return len(caption)

    ctx = SimpleNamespace(family=_Family(), dataset=dataset, text_stack="stack")
    assert _navit_text_token_counts(ctx) == [2, 4, 2, 4, 2, 4]
    assert seen == ["aa", "bbbb"]  # 同一 caption 只 tokenize 一次


def test_native_kwargs_reserve_text_only_for_counting_families() -> None:
    from training.phases.dataset import _native_dataset_kwargs

    args = SimpleNamespace(navit_packing=True, navit_native_resolution=True,
                           navit_token_budget=4096, navit_native_over_budget="downscale")
    plain = _native_dataset_kwargs(args, SimpleNamespace(family=object(), model=None))
    assert "native_text_tokens" not in plain

    class _Family:
        def navit_text_token_count(self, text, caption):
            return len(caption) + (100 if text == "stack" else 0)

    ctx = SimpleNamespace(family=_Family(), model=None, text_stack="stack")
    kwargs = _native_dataset_kwargs(args, ctx)
    assert kwargs["native_token_budget"] == 4096
    assert kwargs["native_text_tokens"]("abc") == 103


# ── 标准路径 batch>1 也走打包前向 ─────────────────────────────────────


def test_forward_train_batch_uses_packed_forward_once_per_layer() -> None:
    """batch>1 的 forward_train 与逐样本 forward 数值一致，且每层只调一次
    （逐样本循环会让 block swap 的换出层每个样本搬一次）；batch=1 仍走原 forward。"""
    from training.families.moss.forward import forward_train

    model = _model().train()
    g = torch.Generator().manual_seed(5)
    x = torch.randn(3, 32, 1, 6, 8, generator=g, dtype=torch.float64)
    context = torch.randn(3, 7, _CFG.text_dim, generator=g, dtype=torch.float64)
    mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3, [1] * 2 + [0] * 5])
    t = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    cond = MossTextCondition(context, mask)

    calls = [0] * len(model.layers)
    for i, layer in enumerate(model.layers):
        layer.register_forward_pre_hook(lambda *_a, i=i: calls.__setitem__(i, calls[i] + 1))
    out = forward_train(model, x, t, cond, use_checkpoint=True)
    assert calls == [1] * len(model.layers)

    ref = torch.cat([
        model(x[i:i + 1], t[i:i + 1], context[i:i + 1], attention_mask=mask[i:i + 1])
        for i in range(3)
    ])
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-12)
    # 标量 t 广播到每个样本
    torch.testing.assert_close(
        forward_train(model, x, torch.tensor(0.5, dtype=torch.float64), cond),
        forward_train(model, x, torch.full((3,), 0.5, dtype=torch.float64), cond),
    )

    one = forward_train(model, x[:1], t[:1], MossTextCondition(context[:1], mask[:1]))
    torch.testing.assert_close(one, model(x[:1], t[:1], context[:1], attention_mask=mask[:1]), rtol=0, atol=0)
