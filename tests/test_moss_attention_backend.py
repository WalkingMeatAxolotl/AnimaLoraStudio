"""MOSS-Image 注意力后端（flash-attn 包 / xformers / SDPA）的分发与接线。

- modeling.attention：选中包走 fast path（[B,S,H,D] 进出）、失败退回 SDPA 且只警告一次、
  不可用 / 未知后端直接 SDPA
- MOSS 训练前向与 ComfyUI 口径出图前向都按模型实例的 attention_backend 分发，
  与 SDPA 结果一致（假 fast path 内部即 SDPA，验布局与调用次数）
- apply_attention_backend：flash 有 / 无、xformers 经 enable_xformers 钩子、none
- 有 CUDA 且装了包时：真 flash-attn / xformers 与 SDPA 数值对照
"""
from __future__ import annotations

import logging
import sys
import types

import pytest
import torch
from torch.nn import functional as F

from modeling import attention as attn
from tests.test_moss_dit import _CFG, _comfy_state, _inputs, _model


def _sdpa_bshd(q, k, v):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


@pytest.fixture(autouse=True)
def _reset_warned():
    attn._FALLBACK_WARNED.clear()
    yield
    attn._FALLBACK_WARNED.clear()


@pytest.fixture
def fake_flash(monkeypatch):
    """把 flash-attn 换成记录调用的 SDPA 等价实现。"""
    calls: list[tuple[int, ...]] = []

    def flash_attn_func(q, k, v):
        calls.append(tuple(q.shape))
        return _sdpa_bshd(q, k, v)

    monkeypatch.setattr(attn, "_flash_attn_func", lambda: flash_attn_func)
    return calls


# ── modeling.attention ─────────────────────────────────────────────────────

def test_attention_bshd_uses_selected_package(fake_flash) -> None:
    q, k, v = (torch.randn(2, 7, 3, 8) for _ in range(3))
    out = attn.attention_bshd(q, k, v, attn.FLASH_ATTN)
    assert fake_flash == [(2, 7, 3, 8)]
    torch.testing.assert_close(out, _sdpa_bshd(q, k, v))


def test_attention_bshd_falls_back_and_warns_once(monkeypatch, caplog) -> None:
    def boom(q, k, v):
        raise RuntimeError("FlashAttention only supports fp16 and bf16")

    monkeypatch.setattr(attn, "_flash_attn_func", lambda: boom)
    q, k, v = (torch.randn(1, 5, 2, 8) for _ in range(3))
    with caplog.at_level(logging.WARNING, logger="modeling.attention"):
        first = attn.attention_bshd(q, k, v, attn.FLASH_ATTN, stage="moss")
        attn.attention_bshd(q, k, v, attn.FLASH_ATTN, stage="moss")
    torch.testing.assert_close(first, _sdpa_bshd(q, k, v))
    warnings = [r for r in caplog.records if "flash_attn fallback at moss" in r.getMessage()]
    assert len(warnings) == 1


@pytest.mark.parametrize("backend", ["none", "xformers", "bogus"])
def test_attention_bshd_unavailable_runs_sdpa(monkeypatch, backend) -> None:
    monkeypatch.setattr(attn, "_xformers_attention", lambda: None)
    q, k, v = (torch.randn(1, 4, 2, 8) for _ in range(3))
    torch.testing.assert_close(attn.attention_bshd(q, k, v, backend), _sdpa_bshd(q, k, v))
    assert attn.backend_available("none")
    assert not attn.backend_available("xformers")


# ── MOSS 模型接线 ──────────────────────────────────────────────────────────

def test_set_attention_backend_reaches_every_block() -> None:
    model = _model()
    model.set_attention_backend(attn.FLASH_ATTN)
    blocks = [*model.layers, *model.text_embedder.blocks]
    assert model.attention_backend == attn.FLASH_ATTN
    assert all(b.attention_backend == attn.FLASH_ATTN for b in blocks)
    model.enable_xformers_memory_efficient_attention()
    assert all(b.attention_backend == attn.XFORMERS for b in blocks)
    with pytest.raises(ValueError):
        model.set_attention_backend("sage")


def test_training_forward_dispatches_per_block(fake_flash) -> None:
    model = _model()
    x, ctx = _inputs(batch=2, text=5)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 3:] = False  # 第 2 个样本只有 3 个有效 token：每样本单独前向
    with torch.no_grad():
        ref = model(x, torch.tensor([0.3, 0.7]), ctx, mask)
        model.set_attention_backend(attn.FLASH_ATTN)
        out = model(x, torch.tensor([0.3, 0.7]), ctx, mask)
    torch.testing.assert_close(out, ref)
    per_sample = len(model.layers) + len(model.text_embedder.blocks)
    assert len(fake_flash) == 2 * per_sample
    heads, head_dim = _CFG.num_heads, _CFG.dim // _CFG.num_heads
    image_tokens = (4 // 2) * (6 // 2)
    assert fake_flash[0] == (1, 5, heads, head_dim)  # 文本精炼块：只有文本 token
    assert fake_flash[per_sample - 1] == (1, image_tokens + 5, heads, head_dim)
    assert fake_flash[-1] == (1, image_tokens + 3, heads, head_dim)


def test_comfy_forward_dispatches_per_block(fake_flash) -> None:
    from modeling.moss.moss_modeling import patchify
    from training.families.moss.comfy_forward import comfy_forward

    model = _model(seed=2)
    x32, ctx = _inputs(h=4, w=6, text=5)
    x128 = patchify(x32.squeeze(2))
    sigma = torch.tensor([0.6])
    with torch.no_grad():
        ref = comfy_forward(model, x128, sigma, ctx)
        model.set_attention_backend(attn.FLASH_ATTN)
        out = comfy_forward(model, x128, sigma, ctx, skip_layers=())
    torch.testing.assert_close(out, ref)
    assert len(fake_flash) == len(model.layers) + len(model.text_embedder.blocks)


# ── apply_attention_backend ────────────────────────────────────────────────

def test_apply_attention_backend_flash(fake_flash) -> None:
    from training.model_loading import apply_attention_backend

    model = _model()
    assert apply_attention_backend(model, "flash_attn") == attn.FLASH_ATTN
    assert model.layers[-1].attention_backend == attn.FLASH_ATTN


def test_apply_attention_backend_flash_missing(monkeypatch) -> None:
    from training.model_loading import apply_attention_backend

    monkeypatch.setattr(attn, "_flash_attn_func", lambda: None)
    model = _model()
    model.set_attention_backend(attn.XFORMERS)
    assert apply_attention_backend(model, "flash_attn") == attn.SDPA
    assert model.layers[0].attention_backend == attn.SDPA


def test_apply_attention_backend_xformers_via_hook(monkeypatch) -> None:
    """xformers 走 enable_xformers（与 daemon 出图同一入口）→ 模型的 diffusers 同名钩子。"""
    from training.model_loading import apply_attention_backend

    ops = types.ModuleType("xformers.ops")
    ops.memory_efficient_attention = _sdpa_bshd
    pkg = types.ModuleType("xformers")
    pkg.ops = ops
    monkeypatch.setitem(sys.modules, "xformers", pkg)
    monkeypatch.setitem(sys.modules, "xformers.ops", ops)
    model = _model()
    assert apply_attention_backend(model, "xformers") == attn.XFORMERS
    assert model.text_embedder.blocks[0].attention_backend == attn.XFORMERS
    assert apply_attention_backend(model, "none") == attn.SDPA
    assert model.layers[0].attention_backend == attn.SDPA


def test_load_dit_applies_backend(tmp_path, fake_flash) -> None:
    from safetensors.torch import save_file

    from training.families import get_family

    path = tmp_path / "moss.safetensors"
    save_file(_comfy_state(_model(seed=4), fuse_qkv=True, int8=False), str(path))
    fam = get_family("moss")
    for purpose in ("train", "generate"):
        model = fam.load_dit(str(path), "cpu", torch.float32, attention_backend="flash_attn", purpose=purpose)
        assert model.layers[0].attention_backend == attn.FLASH_ATTN
        assert fam.load_dit(str(path), "cpu", torch.float32, attention_backend="none",
                            purpose=purpose).layers[0].attention_backend == attn.SDPA


# ── 真包数值对照（CUDA） ────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
@pytest.mark.parametrize("backend", ["flash_attn", "xformers"])
def test_real_backend_matches_sdpa_on_cuda(backend) -> None:
    if not attn.backend_available(backend):
        pytest.skip(f"未安装 {backend}")
    model = _model().to("cuda", torch.bfloat16)
    x, ctx = _inputs(text=5)
    x, ctx = x.to("cuda", torch.bfloat16), ctx.to("cuda", torch.bfloat16)
    sigma = torch.tensor([0.5], device="cuda")
    with torch.no_grad():
        ref = model(x, sigma, ctx).float()
        model.set_attention_backend(backend)
        out = model(x, sigma, ctx).float()
    assert ((out - ref).norm() / ref.norm()).item() < 2e-2
