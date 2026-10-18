"""MOSS-Image block swap 与显存编排接入（PR-5）。

- 族接口 ``dit_blocks``：三族各自返回层容器（MOSS 叫 layers）
- loader：换出层直落 CPU pinned、不经显存；int8 的 weight_scale 留计算设备；
  pinned 预算按精确字节先行检查
- swap 后前向 / 反向与全驻显存逐元素一致
- ``swapped_param_ratio`` 按参数量（不按字节）
- daemon 的 TE 先行编排按文本策略判定；MOSS 逐图兜底路径 performance 档 TE 常驻
"""
from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from tests.test_moss_dit import _CFG, _comfy_state, _inputs, _model
from training.families import get_family
from training.families.moss.loader import (
    _swapped_pinned_bytes,
    load_moss_model,
    swapped_param_ratio,
)

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="block swap 是 CUDA-only 机制")


def _int8_checkpoint(tmp_path, *, seed: int = 4):
    path = tmp_path / "moss-int8.safetensors"
    save_file(_comfy_state(_model(seed=seed), fuse_qkv=True, int8=True), str(path))
    return path


# ── 族接口 ──────────────────────────────────────────────────────────────────

def test_dit_blocks_returns_each_family_layer_container() -> None:
    moss = _model()
    assert get_family("moss").dit_blocks(moss) is moss.layers

    class _Blocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])

    other = _Blocks()
    for family_id in ("anima", "krea2"):
        assert get_family(family_id).dit_blocks(other) is other.blocks


# ── 预算 / 比例（header，无 CUDA 依赖） ─────────────────────────────────────

def test_swapped_param_ratio_counts_params_not_bytes(tmp_path) -> None:
    path = _int8_checkpoint(tmp_path)
    model = _model(seed=4)
    total = sum(p.numel() for p in model.parameters())
    per_layer = sum(p.numel() for p in model.layers[0].parameters())
    assert swapped_param_ratio(path, 1) == pytest.approx(per_layer / total)
    assert swapped_param_ratio(path, 99) == pytest.approx(per_layer * _CFG.num_layers / total)
    assert swapped_param_ratio(path, 0) == 0.0
    fam = get_family("moss")
    assert fam.swapped_param_ratio(1, checkpoint_path=str(path)) == pytest.approx(per_layer / total)
    assert fam.swapped_param_ratio(1) == 0.0  # 无路径 → 保守（按完整模型预算）


def test_swapped_pinned_bytes_int8_one_byte_floats_cast(tmp_path) -> None:
    path = _int8_checkpoint(tmp_path)
    model = _model(seed=4)
    layer = model.layers[_CFG.num_layers - 1]
    int8_numel = sum(p.numel() for n, p in layer.named_parameters() if n.startswith("proj.") and n.endswith(".weight"))
    float_numel = sum(p.numel() for p in layer.parameters()) - int8_numel
    expected = int8_numel * 1 + float_numel * 2
    assert _swapped_pinned_bytes(path, (f"layers.{_CFG.num_layers - 1}.",), torch.bfloat16) == expected


def test_loader_checks_pinned_budget_before_reading(tmp_path, monkeypatch) -> None:
    import training.sysmem as sysmem

    path = _int8_checkpoint(tmp_path)

    def _deny(need, *, blocks):
        raise RuntimeError(f"budget need={need} blocks={blocks}")

    monkeypatch.setattr(sysmem, "check_pinned_budget", _deny)
    with pytest.raises(RuntimeError, match="blocks=2"):
        load_moss_model(path, "cpu", torch.float32, blocks_to_swap=2)


# ── CUDA：放置与数值一致 ─────────────────────────────────────────────────────

@_cuda
def test_loader_places_swapped_layers_in_pinned_memory(tmp_path) -> None:
    path = _int8_checkpoint(tmp_path)
    model = load_moss_model(path, "cuda", torch.float32, blocks_to_swap=2)
    first_swapped = _CFG.num_layers - 2
    for i, layer in enumerate(model.layers):
        for name, param in layer.named_parameters():
            if i >= first_swapped:
                assert param.device.type == "cpu" and param.is_pinned(), (i, name)
            else:
                assert param.device.type == "cuda", (i, name)
        # weight_scale 是 buffer、不归 swap 管：恒在计算设备
        assert layer.proj["k"].weight_scale.device.type == "cuda"
    assert model.text_embedder.blocks[0].qkv.weight.device.type == "cuda"


@_cuda
def test_block_swap_forward_backward_match_resident(tmp_path) -> None:
    from training.block_swap import PinnedBlockSwap

    path = _int8_checkpoint(tmp_path)
    resident = load_moss_model(path, "cuda", torch.float32)
    swapped = load_moss_model(path, "cuda", torch.float32, blocks_to_swap=2)
    blocks = get_family("moss").dit_blocks(swapped)
    swap = PinnedBlockSwap(blocks, 2, torch.device("cuda"))
    swap.attach()
    try:
        x, ctx = _inputs()
        outs = []
        for model in (resident, swapped):
            xi = x.cuda().requires_grad_(True)
            out = model(xi, torch.tensor([0.5], device="cuda"), ctx.cuda())
            out.square().mean().backward()
            outs.append((out.detach(), xi.grad.detach()))
        torch.testing.assert_close(outs[1][0], outs[0][0], rtol=0, atol=0)
        torch.testing.assert_close(outs[1][1], outs[0][1], rtol=0, atol=0)
    finally:
        swap.close()


# ── 显存编排 ────────────────────────────────────────────────────────────────

def test_daemon_te_first_follows_text_strategy() -> None:
    import anima_daemon

    assert anima_daemon._uses_te_first(get_family("moss"))
    assert anima_daemon._uses_te_first(get_family("krea2"))
    assert not anima_daemon._uses_te_first(get_family("anima"))


@pytest.mark.parametrize("policy,offloaded", [
    ("performance", False), ("auto", True), ("save_vram", True), (None, True),
])
def test_moss_sample_image_offloads_te_unless_performance(monkeypatch, policy, offloaded) -> None:
    from training.families import moss as moss_family

    calls: list[str] = []

    class _Text:
        def offload_model(self):
            calls.append("offload")

    monkeypatch.setattr(moss_family, "prepare_sampling_condition", lambda *a, **k: object())
    monkeypatch.setattr(moss_family, "sample_latents", lambda *a, **k: "latents")
    monkeypatch.setattr(moss_family, "decode_to_pil", lambda vae, latents: latents)
    moss_family.MossFamily().sample_image(
        None, None, _Text(), "p", steps=2, device="cpu", vram_policy=policy,
    )
    assert calls == (["offload"] if offloaded else [])


def test_text_stacks_declare_te_load_estimate(tmp_path) -> None:
    from training.families.moss.text_encoding import MossTextStack

    te = tmp_path / "te"
    te.mkdir()
    for flag, gib in ((True, 9), (False, 18)):
        stack = MossTextStack(te, device="cpu", tokenizer=object(), is_int8=flag)
        assert stack.te_load_bytes_estimate == gib * 1024 ** 3
