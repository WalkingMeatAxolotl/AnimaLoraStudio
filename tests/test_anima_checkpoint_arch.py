"""Anima 底模架构探测（studio.services.inference.checkpoint_arch）+ loader 接线。

背景：第三方插层扩展版（Anima-2.9B = 40 层）进不来的两处根因 ——
① loader 按 model_channels 查表把层数写死 28（多出的层进 unexpected 只 log，静默丢层）；
② 文件里带 RoPE 派生缓冲 pos_embedder.seq [256]，与本地 [512] 形状不同，
   load_state_dict(strict=False) 对 shape 不匹配照样 raise。
修法：层数从 header 数；加载前剥派生缓冲；unexpected 关键层硬报错。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch

from studio.services.inference.checkpoint_arch import (
    AnimaCheckpointArch,
    CheckpointInspectError,
    arch_from_header,
    inspect_anima_checkpoint,
    read_safetensors_header,
)


# ── 造 header / 造文件 ────────────────────────────────────────────────────

def _anima_like_header(*, num_blocks: int, model_channels: int = 2048,
                       prefix: str = "net.", adapter_blocks: int = 6,
                       with_pos_embedder: bool = False) -> dict:
    """键形态贴近真实 checkpoint 的 header（形状缩小，只关心 numel 相对关系）。

    每个 DiT block 一个 [8, 8] 张量（64 参数）；llm_adapter 6 个 block 各一个 [4, 4]
    （16 参数）——它们也叫 blocks.N，探测器必须排除。
    """
    h: dict = {
        f"{prefix}x_embedder.proj.1.weight": {"dtype": "BF16", "shape": [model_channels, 68], "data_offsets": [0, 0]},
        f"{prefix}final_layer.linear.weight": {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 0]},
    }
    for i in range(num_blocks):
        h[f"{prefix}blocks.{i}.self_attn.q_proj.weight"] = {"dtype": "BF16", "shape": [8, 8], "data_offsets": [0, 0]}
    for i in range(adapter_blocks):
        h[f"{prefix}llm_adapter.blocks.{i}.self_attn.q_proj.weight"] = {"dtype": "BF16", "shape": [4, 4], "data_offsets": [0, 0]}
    if with_pos_embedder:
        h[f"{prefix}pos_embedder.seq"] = {"dtype": "BF16", "shape": [256], "data_offsets": [0, 0]}
        h[f"{prefix}pos_embedder.dim_spatial_range"] = {"dtype": "BF16", "shape": [21], "data_offsets": [0, 0]}
    return h


def _write_safetensors_with_header(path: Path, header: dict) -> None:
    """只写 header（payload 空）——探测器不读 payload，够用。"""
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


# ── arch_from_header：纯函数 ──────────────────────────────────────────────

def test_num_blocks_counted_from_header_not_table():
    """40 层第三方版：层数 = blocks.N 最大下标 + 1，与 model_channels 无关。"""
    arch = arch_from_header(_anima_like_header(num_blocks=40))
    assert arch.num_blocks == 40
    assert arch.model_channels == 2048
    assert arch.num_heads == 16
    assert arch.in_channels == 16
    assert arch.has_llm_adapter is True
    # 官方 28 层同一条路
    assert arch_from_header(_anima_like_header(num_blocks=28)).num_blocks == 28
    # 14B：通道 5120 → 40 头，层数仍数出来
    big = arch_from_header(_anima_like_header(num_blocks=36, model_channels=5120))
    assert (big.num_blocks, big.num_heads) == (36, 40)


def test_llm_adapter_blocks_excluded_from_dit_block_counts():
    """llm_adapter.blocks.0-5 不能算进 DiT block 0-5（旧 _BLOCK_KEY_RE 的 bug）。"""
    arch = arch_from_header(_anima_like_header(num_blocks=4, adapter_blocks=6))
    assert arch.num_blocks == 4  # 不会被 adapter 的 6 层污染成 6
    assert arch.block_param_counts == (64, 64, 64, 64)  # 每层只有自己的 64，不含 adapter 的 16
    # 但 adapter 参数计入总量：x_embedder 2048*68 + final 16 + 4*64 + 6*16
    assert arch.param_count == 2048 * 68 + 16 + 4 * 64 + 6 * 16


def test_swapped_param_ratio_from_arch():
    arch = arch_from_header(_anima_like_header(num_blocks=4, adapter_blocks=0, model_channels=2048))
    total = 2048 * 68 + 16 + 4 * 64
    assert arch.swapped_param_ratio(2) == pytest.approx(128 / total)
    assert arch.swapped_param_ratio(4) == pytest.approx(256 / total)
    assert arch.swapped_param_ratio(99) == pytest.approx(256 / total)  # 超界 clamp
    assert arch.swapped_param_ratio(0) == 0.0


def test_unknown_channels_keeps_heads_none_and_rejects_non_anima():
    """未知通道数：探测不报错（block swap 比例仍可算），建模时才 raise。"""
    arch = arch_from_header(_anima_like_header(num_blocks=4, model_channels=4))
    assert arch.num_heads is None
    from training.families.anima.loader import arch_to_config

    with pytest.raises(RuntimeError, match="未知的 model_channels"):
        arch_to_config(arch)
    # 缺 x_embedder → 不是 Anima 权重
    with pytest.raises(CheckpointInspectError, match="x_embedder"):
        arch_from_header({"blocks.0.w": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 0]}})
    # block 下标断档 → 报错而不是静默建一个层数错的模型
    h = _anima_like_header(num_blocks=4)
    del h["net.blocks.2.self_attn.q_proj.weight"]
    with pytest.raises(CheckpointInspectError, match="不连续"):
        arch_from_header(h)


def test_arch_to_config_feeds_num_blocks_through():
    from training.families.anima.loader import arch_to_config

    cfg = arch_to_config(arch_from_header(_anima_like_header(num_blocks=40)))
    assert cfg["num_blocks"] == 40
    assert cfg["num_heads"] == 16
    assert cfg["model_channels"] == 2048
    assert cfg["in_channels"] == 16
    assert cfg["rope_h_extrapolation_ratio"] == 4.0


# ── 文件级：header 直读、前缀无关、__metadata__ 忽略 ───────────────────────

def test_inspect_reads_header_only_and_is_prefix_agnostic(tmp_path):
    h = _anima_like_header(num_blocks=40, prefix="model.", with_pos_embedder=True)
    h["__metadata__"] = {"format": "pt"}
    p = tmp_path / "third_party.safetensors"
    _write_safetensors_with_header(p, h)
    arch = inspect_anima_checkpoint(p)
    assert isinstance(arch, AnimaCheckpointArch)
    assert arch.num_blocks == 40
    assert arch.file_bytes == p.stat().st_size
    assert read_safetensors_header(p)["__metadata__"] == {"format": "pt"}


def test_inspect_rejects_garbage(tmp_path):
    bad = tmp_path / "garbage.safetensors"
    bad.write_bytes(b"garbage")
    with pytest.raises(CheckpointInspectError):
        inspect_anima_checkpoint(bad)


def test_real_safetensors_file_roundtrip(tmp_path):
    """用 safetensors 真写一个文件（有 payload），探测结果与张量一致。"""
    from safetensors.torch import save_file

    tensors = {"net.x_embedder.proj.1.weight": torch.zeros(2048, 68)}
    for i in range(3):
        tensors[f"net.blocks.{i}.mlp.layer1.weight"] = torch.zeros(8, 8)
    p = tmp_path / "real.safetensors"
    save_file(tensors, str(p))
    arch = inspect_anima_checkpoint(p)
    assert arch.num_blocks == 3
    assert arch.has_llm_adapter is False
    assert arch.param_count == 2048 * 68 + 3 * 64


# ── loader 接线：派生缓冲剥离 + unexpected 关键层硬报错 ───────────────────

def test_drop_derived_buffers_strips_pos_embedder_only():
    from training.families.anima.loader import drop_derived_buffers

    sd = {
        "net.pos_embedder.seq": torch.zeros(256),
        "net.pos_embedder.dim_spatial_range": torch.zeros(21),
        "net.blocks.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        "net.x_embedder.proj.1.weight": torch.zeros(2, 2),
    }
    out = drop_derived_buffers(sd)
    assert set(out) == {"net.blocks.0.self_attn.q_proj.weight", "net.x_embedder.proj.1.weight"}
    # 没有派生缓冲时原样返回（不复制多 GB 的 dict）
    assert drop_derived_buffers(out) is out


class _TinyDiT(torch.nn.Module):
    def __init__(self, num_blocks: int) -> None:
        super().__init__()
        self.x_embedder = torch.nn.Linear(4, 4)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(num_blocks)])
        self.final_layer = torch.nn.Linear(4, 4)


def test_load_weights_rejects_checkpoint_with_more_blocks_than_model():
    """40 层权重灌 28 层模型：此前 coverage=100%、missing=0 → 只 log 一行放行
    （静默丢层）。现在关键前缀的 unexpected 必须硬报错。"""
    from training.model_loading import _load_weights_best_effort

    model = _TinyDiT(num_blocks=2)
    sd = _TinyDiT(num_blocks=3).state_dict()  # 多一层 blocks.2.*
    with pytest.raises(RuntimeError, match="不存在的关键参数"):
        _load_weights_best_effort(model, sd, label="Transformer")
    # 层数一致 → 正常
    info = _load_weights_best_effort(model, _TinyDiT(num_blocks=2).state_dict(), label="Transformer")
    assert info["coverage"] == pytest.approx(1.0)
    # 非关键的多余键（如剥漏的杂项）仍只记录不报错
    sd_ok = _TinyDiT(num_blocks=2).state_dict()
    sd_ok["some_extra.weight"] = torch.zeros(1)
    info = _load_weights_best_effort(model, sd_ok, label="Transformer")
    assert info["unexpected"] == ["some_extra.weight"]
