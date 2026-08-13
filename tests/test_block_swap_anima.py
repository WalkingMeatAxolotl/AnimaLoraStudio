"""Anima 族 block swap 接线（docs/design/block-swap.md B3：机制 family 无关，
接线 = loader 落位 + 能力位 + 版本感知折扣 + 采样 offload 互斥）。

机制核心（四钩子 / 双缓冲 / pinned 生命周期）已由 test_block_swap*.py 钉死，
这里只测 anima 侧新增的接线面。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training.families.anima.loader import (  # noqa: E402
    place_model_for_block_swap,
    swapped_param_ratio_from_header,
)

_CUDA = torch.cuda.is_available()


# ── swapped_param_ratio：safetensors header 数 numel（无 CUDA 依赖） ────────

def _write_fake_checkpoint(path: Path, *, num_blocks: int, prefix: str = "") -> dict:
    """写一个键形态贴近真实 checkpoint 的小 safetensors。

    每层一个 64 参数张量 + 两个非 block 张量（x_embedder 32 / final_layer 16）。
    prefix 模拟 model./module. 这类会被 _load_weights_best_effort 剥掉的前缀
    —— ratio 必须前缀无关。
    """
    from safetensors.torch import save_file

    tensors = {
        f"{prefix}x_embedder.proj.1.weight": torch.zeros(4, 8),
        f"{prefix}final_layer.linear.weight": torch.zeros(4, 4),
    }
    for i in range(num_blocks):
        tensors[f"{prefix}blocks.{i}.attn.qkv.weight"] = torch.zeros(8, 8)
    save_file(tensors, str(path))
    return tensors


def test_ratio_counts_trailing_blocks_from_header(tmp_path):
    ckpt = tmp_path / "anima.safetensors"
    _write_fake_checkpoint(ckpt, num_blocks=4)
    # 总量 = 32 + 16 + 4×64 = 304；换出末尾 2 层 = 128
    assert swapped_param_ratio_from_header(ckpt, 2) == pytest.approx(128 / 304)
    # 全换出
    assert swapped_param_ratio_from_header(ckpt, 4) == pytest.approx(256 / 304)
    # 超界 clamp 到总层数（跨版本共享的设置值，36 喂给 28 层版）
    assert swapped_param_ratio_from_header(ckpt, 99) == pytest.approx(256 / 304)
    assert swapped_param_ratio_from_header(ckpt, 0) == 0.0


def test_ratio_is_prefix_agnostic(tmp_path):
    """checkpoint 键带 model. 前缀（loader 加载时才剥）→ ratio 不受影响。"""
    ckpt = tmp_path / "prefixed.safetensors"
    _write_fake_checkpoint(ckpt, num_blocks=4, prefix="model.")
    assert swapped_param_ratio_from_header(ckpt, 2) == pytest.approx(128 / 304)


def test_family_ratio_degrades_to_conservative_zero(tmp_path):
    """无路径 / 文件坏 → 0（护栏按完整模型预算，不误放行）。"""
    from training.families.anima.family import AnimaFamily

    fam = AnimaFamily()
    assert fam.swapped_param_ratio(14) == 0.0  # 未给 checkpoint_path
    assert fam.swapped_param_ratio(14, checkpoint_path="") == 0.0
    bad = tmp_path / "not_a_checkpoint.safetensors"
    bad.write_bytes(b"garbage")
    assert fam.swapped_param_ratio(14, checkpoint_path=str(bad)) == 0.0
    ckpt = tmp_path / "ok.safetensors"
    _write_fake_checkpoint(ckpt, num_blocks=4)
    assert fam.swapped_param_ratio(2, checkpoint_path=str(ckpt)) == pytest.approx(128 / 304)


# ── loader 落位：换出层不上卡（§9.4 纪律） ─────────────────────────────────

class _TinyDiT(torch.nn.Module):
    """结构上贴近 Anima 的最小替身：.blocks ModuleList + 非 block 参数与 buffer。"""

    def __init__(self, num_blocks: int = 4) -> None:
        super().__init__()
        self.x_embedder = torch.nn.Linear(8, 8)
        self.blocks = torch.nn.ModuleList(
            [torch.nn.Linear(8, 8) for _ in range(num_blocks)]
        )
        self.register_buffer("pos_freq", torch.arange(4, dtype=torch.float32))


def test_pinned_budget_guard_fails_fast_before_any_placement(monkeypatch):
    """预算护栏在任何搬运 / 分配之前 raise（B6），无 CUDA 也必须成立。"""
    from training import sysmem

    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 1024)  # 1KB：必拒
    model = _TinyDiT()
    with pytest.raises(RuntimeError, match="内存不足以换出"):
        place_model_for_block_swap(model, "cuda", torch.float32, 2)
    # fail-fast 语义：模型分毫未动（全部仍在 CPU、无标记）
    assert all(p.device.type == "cpu" for p in model.parameters())
    assert getattr(model, "blocks_to_swap", 0) == 0


@pytest.mark.skipif(not _CUDA, reason="需要 CUDA")
def test_placement_keeps_swapped_blocks_on_cpu():
    """非换出部分上卡、末尾 N 层留 CPU、dtype 已 cast、标记落位、超界 clamp。"""
    model = _TinyDiT(num_blocks=4)
    num = place_model_for_block_swap(model, "cuda", torch.bfloat16, 99)
    assert num == 4  # clamp 到总层数

    model2 = _TinyDiT(num_blocks=4)
    num2 = place_model_for_block_swap(model2, "cuda", torch.bfloat16, 2)
    assert num2 == 2
    assert model2.blocks_to_swap == 2
    for i, block in enumerate(model2.blocks):
        expect = "cpu" if i >= 2 else "cuda"
        for p in block.parameters():
            assert p.device.type == expect, f"blocks.{i} 应在 {expect}"
            assert p.dtype == torch.bfloat16
    # 非 block 参数与 buffer 全部上卡
    assert model2.x_embedder.weight.device.type == "cuda"
    assert model2.pos_freq.device.type == "cuda"


@pytest.mark.skipif(not _CUDA, reason="需要 CUDA")
def test_anima_forward_backward_matches_without_swap():
    """真 Anima 小号模型：loader 落位 + 四钩子挂载后，前向输出与输入梯度
    同无 swap 对照一致（fp32 消除非确定性；机制竞态由 grad_fidelity 真尺寸
    测试把关，这里钉的是 anima 结构兼容性——rope/adaln_lora/cross-attn 一串
    预备张量都经手工展开循环传入 block）。
    """
    import copy

    from modeling.anima import anima_modeling
    from training.block_swap import PinnedBlockSwap
    from training.families.anima.forward import forward_with_optional_checkpoint

    config = dict(
        max_img_h=64, max_img_w=64, max_frames=4,
        in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        model_channels=64, num_blocks=4, num_heads=2,
        crossattn_emb_channels=1024,
        pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True, adaln_lora_dim=16,
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=1.0,
    )
    torch.manual_seed(0)
    ref = anima_modeling.Anima(**config)
    swapped = copy.deepcopy(ref)

    ref = ref.to("cuda", torch.float32)
    ref.requires_grad_(False)
    place_model_for_block_swap(swapped, "cuda", torch.float32, 2)
    swapped.requires_grad_(False)
    swap = PinnedBlockSwap(swapped.blocks, 2, "cuda")
    swap.attach()

    def _inputs():
        g = torch.Generator(device="cpu").manual_seed(7)
        latents = torch.randn(1, 16, 1, 8, 8, generator=g).to("cuda")
        t = torch.full((1, 1), 0.5, device="cuda")
        cross = torch.randn(1, 32, 1024, generator=g).to("cuda")
        pad = torch.zeros(1, 1, 8, 8, device="cuda")
        return latents.requires_grad_(True), t, cross, pad

    outs, grads = [], []
    for model in (ref, swapped):
        latents, t, cross, pad = _inputs()
        out = forward_with_optional_checkpoint(
            model, latents, t, cross, pad, use_checkpoint=True,
        )
        out.square().mean().backward()
        outs.append(out.detach())
        grads.append(latents.grad.detach().clone())
    torch.cuda.synchronize()

    torch.testing.assert_close(outs[0], outs[1], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-5)

    swap.close()


# ── 采样期 VAE decode offload 与 swap 互斥 ─────────────────────────────────

def test_decode_offload_skips_swapped_dit():
    from training.families.anima.sampling import _decode_offload_targets

    model = _TinyDiT()
    qwen = object()
    # 无 swap：DiT + Qwen 都可 offload
    assert _decode_offload_targets(model, qwen) == (model, qwen)
    # swap 生效（loader 落的标记）：只 offload Qwen —— 恢复期的一刀切 .to()
    # 会把 CPU pinned 主副本搬上卡，DiT 必须跳过
    model.blocks_to_swap = 2
    assert _decode_offload_targets(model, qwen) == (qwen,)
