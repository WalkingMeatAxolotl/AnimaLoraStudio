"""tools/lora_merge.py（多 LoRA 精确 merge）+ _apply_reg_dims_ LoCon 子模块分支。

覆盖两块：
1. _apply_reg_dims_ 对本 lycoris 版本 LoConModule（lora_up/lora_down 是 nn.Linear
   子模块，非 lora_A/lora_B 参数）的分层 rank re-init —— 修复前该分支静默 skip，
   per-layer rank 的 plain LoRA 文件加载时 shape mismatch 被 strict=False 吞掉。
2. merge 的数学等价性：LoKr（kron 恒等式展开）+ plain LoRA 加权和 == 逐层 dense
   参考值；秩不齐时输出 lora_reg_dims 元数据 + per-layer alpha=rank。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from utils.lycoris_adapter import _apply_reg_dims_

REPO_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location("lora_merge", REPO_ROOT / "tools" / "lora_merge.py")
lora_merge = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("lora_merge", lora_merge)
_spec.loader.exec_module(lora_merge)


class _FakeNet:
    def __init__(self, loras: list) -> None:
        self.loras = loras


def _locon(name: str, org: nn.Module, dim: int) -> object:
    from lycoris.modules.locon import LoConModule

    return LoConModule(name, org, 1.0, lora_dim=dim, alpha=dim)


# ---------------------------------------------------------------------------
# _apply_reg_dims_ — LoCon lora_up/lora_down 子模块分支
# ---------------------------------------------------------------------------


def test_reg_dims_reinits_locon_linear_submodules() -> None:
    """fullmatch 命中的 LoCon(Linear) 层：down/up 重建为新 rank，up 归零保 ΔW=0。"""
    mod = _locon("layer_x", nn.Linear(16, 12), dim=8)
    _apply_reg_dims_(_FakeNet([mod]), {"layer_x": 4})
    assert mod.lora_dim == 4
    assert mod.lora_down.weight.shape == (4, 16)
    assert mod.lora_up.weight.shape == (12, 4)
    assert torch.all(mod.lora_up.weight == 0)


def test_reg_dims_locon_no_match_untouched() -> None:
    """pattern 不 fullmatch 时不动（re.search 语义会误伤，回归保护）。"""
    mod = _locon("layer_x_extra", nn.Linear(16, 12), dim=8)
    _apply_reg_dims_(_FakeNet([mod]), {"layer_x": 4})
    assert mod.lora_dim == 8
    assert mod.lora_down.weight.shape == (8, 16)


def test_reg_dims_locon_conv_skipped() -> None:
    """LoCon(Conv2d) 的 lora_down 不是 nn.Linear → 分层覆盖不适用，跳过不爆。"""
    mod = _locon("conv_x", nn.Conv2d(4, 4, 3, padding=1), dim=8)
    before = tuple(mod.lora_down.weight.shape)
    _apply_reg_dims_(_FakeNet([mod]), {"conv_x": 4})
    assert mod.lora_dim == 8
    assert tuple(mod.lora_down.weight.shape) == before


# ---------------------------------------------------------------------------
# merge — 数学等价 + 元数据
# ---------------------------------------------------------------------------

L1, L2 = "lora_unet_blocks_0_attn_q", "lora_unet_blocks_1_mlp"


def _write_lokr(path: Path) -> dict[str, torch.Tensor]:
    """两层 LoKr：w1 全矩阵 + w2_a/w2_b 分解；L2 dim 更小（模拟 lora_reg_dims）。"""
    torch.manual_seed(0)
    sd = {}
    for layer, dim in ((L1, 3), (L2, 2)):
        sd[f"{layer}.lokr_w1"] = torch.randn(2, 2)
        sd[f"{layer}.lokr_w2_a"] = torch.randn(4, dim)
        sd[f"{layer}.lokr_w2_b"] = torch.randn(dim, 6)
        sd[f"{layer}.alpha"] = torch.tensor(dim * 0.5)  # scale = alpha/dim = 0.5
    from safetensors.torch import save_file

    save_file(sd, str(path), metadata={
        "ss_network_dim": "3",
        "ss_network_alpha": "1.5",
        "ss_network_module": "lycoris.kohya",
        "ss_network_args": json.dumps({"algo": "lokr", "factor": 2}),
    })
    return sd


def _write_plain(path: Path, extra: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """两层 plain LoRA rank 2（8×12 的 Linear，与 LoKr 展开后同形）。"""
    torch.manual_seed(1)
    sd = {}
    for layer in (L1, L2):
        sd[f"{layer}.lora_down.weight"] = torch.randn(2, 12)
        sd[f"{layer}.lora_up.weight"] = torch.randn(8, 2)
        sd[f"{layer}.alpha"] = torch.tensor(1.0)  # scale = 0.5
    sd.update(extra or {})
    from safetensors.torch import save_file

    save_file(sd, str(path), metadata={
        "ss_network_dim": "2",
        "ss_network_alpha": "1.0",
        "ss_network_module": "lycoris.kohya",
        "ss_network_args": json.dumps({"algo": "lora"}),
    })
    return sd


def _dense_ref(lokr: dict, plain: dict, w_lokr: float, w_plain: float, layer: str) -> torch.Tensor:
    dim = lokr[f"{layer}.lokr_w2_a"].shape[1]
    kron = torch.kron(lokr[f"{layer}.lokr_w1"], lokr[f"{layer}.lokr_w2_a"] @ lokr[f"{layer}.lokr_w2_b"])
    ref = w_lokr * (float(lokr[f"{layer}.alpha"]) / dim) * kron
    rank = plain[f"{layer}.lora_down.weight"].shape[0]
    ref += w_plain * (float(plain[f"{layer}.alpha"]) / rank) * (
        plain[f"{layer}.lora_up.weight"] @ plain[f"{layer}.lora_down.weight"]
    )
    return ref


def test_merge_matches_dense_reference(tmp_path: Path) -> None:
    """merged 文件逐层 (alpha/rank)·up@down == LoKr 展开 + 加权 plain 的 dense 和。"""
    lokr_sd = _write_lokr(tmp_path / "style.safetensors")
    plain_sd = _write_plain(tmp_path / "slider.safetensors")
    out = tmp_path / "merged.safetensors"
    lora_merge.merge(
        [(tmp_path / "style.safetensors", 1.0), (tmp_path / "slider.safetensors", -5.0)],
        out, torch.float32, trim_energy=None, rank_cap=None,
    )
    got = lora_merge._load_layers(out)
    for layer in (L1, L2):
        t = got[layer]
        rank = t["lora_down.weight"].shape[0]
        assert float(t["alpha"]) == pytest.approx(float(rank))  # per-layer scale=1
        dw = t["lora_up.weight"] @ t["lora_down.weight"]
        ref = _dense_ref(lokr_sd, plain_sd, 1.0, -5.0, layer)
        assert torch.allclose(dw, ref, atol=1e-5), f"{layer} ΔW 偏离参考值"
    # 秩：L1 = 2·3+2 = 8（全局 max），L2 = 2·2+2 = 6 → 进 reg_dims
    assert got[L1]["lora_down.weight"].shape[0] == 8
    assert got[L2]["lora_down.weight"].shape[0] == 6


def test_merge_metadata_reg_dims_and_dim(tmp_path: Path) -> None:
    """元数据：ss_network_dim=max rank、algo=lora、秩≠max 的层落 lora_reg_dims。"""
    _write_lokr(tmp_path / "style.safetensors")
    _write_plain(tmp_path / "slider.safetensors")
    out = tmp_path / "merged.safetensors"
    lora_merge.merge(
        [(tmp_path / "style.safetensors", 1.0), (tmp_path / "slider.safetensors", -5.0)],
        out, torch.float32, trim_energy=None, rank_cap=None,
    )
    from safetensors import safe_open

    with safe_open(str(out), framework="pt", device="cpu") as f:
        meta = f.metadata()
    assert meta["ss_network_dim"] == "8"
    args = json.loads(meta["ss_network_args"])
    assert args["algo"] == "lora"
    assert args["lora_reg_dims"] == {L2: 6}
    sources = json.loads(meta["anima_merge_sources"])
    assert [s["weight"] for s in sources] == [1.0, -5.0]

    # read_lora_meta（推理加载入口）能按同一约定读回
    from studio.services.inference.core import read_lora_meta

    m = read_lora_meta(str(out))
    assert (m.rank, m.algo, m.lora_reg_dims) == (8, "lora", {L2: 6})


def test_merge_rank_cap_truncates(tmp_path: Path) -> None:
    """--rank-cap：SVD 截断到上限秩，结果是该秩下的最优近似（误差有限非零）。"""
    lokr_sd = _write_lokr(tmp_path / "style.safetensors")
    plain_sd = _write_plain(tmp_path / "slider.safetensors")
    out = tmp_path / "merged.safetensors"
    lora_merge.merge(
        [(tmp_path / "style.safetensors", 1.0), (tmp_path / "slider.safetensors", -5.0)],
        out, torch.float32, trim_energy=None, rank_cap=4,
    )
    got = lora_merge._load_layers(out)
    for layer in (L1, L2):
        assert got[layer]["lora_down.weight"].shape[0] <= 4
        dw = got[layer]["lora_up.weight"] @ got[layer]["lora_down.weight"]
        ref = _dense_ref(lokr_sd, plain_sd, 1.0, -5.0, layer)
        # 截秩后不再精确，但残差不应超过被丢弃的奇异值能量（宽松上界：范数量级）
        assert (dw - ref).norm() < ref.norm()


def test_merge_rejects_dora(tmp_path: Path) -> None:
    """含 dora_scale（DoRA 非线性）的源直接拒绝，不产出错误文件。"""
    _write_plain(tmp_path / "dora.safetensors", extra={f"{L1}.dora_scale": torch.ones(8, 1)})
    with pytest.raises(SystemExit, match="dora"):
        lora_merge.merge(
            [(tmp_path / "dora.safetensors", 1.0)],
            tmp_path / "out.safetensors", torch.float32, trim_energy=None, rank_cap=None,
        )
