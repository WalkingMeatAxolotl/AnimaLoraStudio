"""MuonWithAuxAdam 测试 — 与 torch 官方实现对拍 + LoRA 场景 + registry 接入。

- Newton-Schulz 把奇异值推到 ~1（含 batch 维、竖/横矩阵）
- Muon 分支与 torch.optim.Muon 逐步对拍（两种 adjust_lr_fn、带 weight decay、
  同形状矩阵走 batch 路径）
- 辅助 AdamW 分支与 torch.optim.AdamW 对拍
- param group 按维度拆分且保留组选项；LambdaLR 在拆分后的组上正常工作
- state_dict 往返后续训逐位一致
- LoRA 零初始化（up=0 → down 梯度为 0）不出 NaN，能拟合低秩目标
- create_optimizer / build_optimizer 派发到 muon

测试卫生：纯 CPU、无机器状态依赖、固定随机种子。
"""
from __future__ import annotations

import copy
import types

import pytest
import torch
from torch import nn

from utils.muon_optimizer import MuonWithAuxAdam, zeropower_via_newtonschulz5


@pytest.mark.parametrize("shape", [(16, 64), (64, 16), (3, 32, 8)])
def test_newton_schulz_pushes_singular_values_near_one(shape) -> None:
    torch.manual_seed(0)
    g = torch.randn(*shape)
    o = zeropower_via_newtonschulz5(g).float()
    sv = torch.linalg.svdvals(o)
    # quintic 系数刻意不精确收敛（Keller：约 Uniform(0.5, 1.5)）
    assert sv.min() > 0.5 and sv.max() < 1.5, sv


def test_newton_schulz_preserves_singular_vectors() -> None:
    torch.manual_seed(0)
    g = torch.randn(32, 8)
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    o = zeropower_via_newtonschulz5(g).float()
    # o ≈ U S' Vᵀ：投影到 U/V 上应近似对角
    core = u.T @ o @ vh.T
    off = core - torch.diag(torch.diagonal(core))
    assert off.abs().max() < 0.1 * torch.diagonal(core).abs().min()


def test_newton_schulz_zero_input_stays_zero() -> None:
    o = zeropower_via_newtonschulz5(torch.zeros(4, 16))
    assert torch.count_nonzero(o) == 0


def _quad_loss(params, targets):
    return sum(((p - t) ** 2).sum() for p, t in zip(params, targets))


@pytest.mark.parametrize("adjust_lr_fn", ["original", "match_rms_adamw"])
def test_muon_branch_matches_torch_muon(adjust_lr_fn) -> None:
    torch.manual_seed(0)
    # 两个同形状（走 batch NS）+ 一个异形状
    init = [torch.randn(8, 32), torch.randn(8, 32), torch.randn(48, 8)]
    targets = [torch.randn_like(t) for t in init]
    ours = [nn.Parameter(t.clone()) for t in init]
    ref = [nn.Parameter(t.clone()) for t in init]
    opt = MuonWithAuxAdam(ours, lr=0.02, weight_decay=0.1, adjust_lr_fn=adjust_lr_fn)
    ref_opt = torch.optim.Muon(ref, lr=0.02, weight_decay=0.1, adjust_lr_fn=adjust_lr_fn)
    for _ in range(5):
        for o, ps in ((opt, ours), (ref_opt, ref)):
            o.zero_grad()
            _quad_loss(ps, targets).backward()
            o.step()
    for a, b in zip(ours, ref):
        # bf16 NS + bmm vs addmm 路径差异，容差按 bf16 量级
        torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-3)


def test_aux_adam_branch_matches_torch_adamw() -> None:
    torch.manual_seed(0)
    init = [torch.randn(16), torch.randn(5)]
    targets = [torch.randn_like(t) for t in init]
    ours = [nn.Parameter(t.clone()) for t in init]
    ref = [nn.Parameter(t.clone()) for t in init]
    opt = MuonWithAuxAdam(ours, lr=1e-2, weight_decay=0.05)
    ref_opt = torch.optim.AdamW(ref, lr=1e-2, weight_decay=0.05, betas=(0.9, 0.999), eps=1e-8)
    assert [g["use_muon"] for g in opt.param_groups] == [False]
    for _ in range(10):
        for o, ps in ((opt, ours), (ref_opt, ref)):
            o.zero_grad()
            _quad_loss(ps, targets).backward()
            o.step()
    for a, b in zip(ours, ref):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_groups_split_by_ndim_and_keep_options() -> None:
    w = nn.Parameter(torch.randn(4, 8))
    b = nn.Parameter(torch.randn(4))
    w2 = nn.Parameter(torch.randn(8, 4))
    opt = MuonWithAuxAdam(
        [
            {"params": [w, b], "weight_decay": 0.1},
            {"params": [w2], "weight_decay": 0.0, "lr": 5e-3},
        ],
        lr=1e-2,
    )
    summary = [(g["use_muon"], len(g["params"]), g["weight_decay"], g["lr"]) for g in opt.param_groups]
    assert summary == [(True, 1, 0.1, 1e-2), (False, 1, 0.1, 1e-2), (True, 1, 0.0, 5e-3)]


@pytest.mark.parametrize("shape,expected", [
    ((4, 8), True), ((3, 4, 5), True),          # 线性层 / 卷积核
    ((8,), False), ((1, 8), False), ((8, 1), False), ((1, 3, 3), False),  # 向量形状
])
def test_is_muon_param_treats_vector_shapes_as_adam(shape, expected) -> None:
    from utils.muon_optimizer import is_muon_param

    assert is_muon_param(torch.empty(shape)) is expected


def test_lambda_lr_scales_every_split_group() -> None:
    w = nn.Parameter(torch.randn(4, 8))
    b = nn.Parameter(torch.randn(4))
    opt = MuonWithAuxAdam([w, b], lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 0.5)
    assert [g["lr"] for g in opt.param_groups] == [5e-3, 5e-3]
    assert sched.get_last_lr() == [5e-3, 5e-3]


def test_state_dict_roundtrip_resumes_identically() -> None:
    torch.manual_seed(0)
    init = [torch.randn(8, 16), torch.randn(8)]
    targets = [torch.randn_like(t) for t in init]

    def run(params, opt, n):
        for _ in range(n):
            opt.zero_grad()
            _quad_loss(params, targets).backward()
            opt.step()

    a = [nn.Parameter(t.clone()) for t in init]
    opt_a = MuonWithAuxAdam(a, lr=1e-2)
    run(a, opt_a, 3)

    b = [nn.Parameter(p.detach().clone()) for p in a]
    opt_b = MuonWithAuxAdam(b, lr=1e-2)
    opt_b.load_state_dict(copy.deepcopy(opt_a.state_dict()))

    run(a, opt_a, 3)
    run(b, opt_b, 3)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_lora_zero_init_fits_low_rank_target() -> None:
    """标准 LoRA 初始化（up=0）：首步 down 梯度为 0，NS 不能炸出 NaN，且能拟合。"""
    torch.manual_seed(0)
    d_in, d_out, r = 32, 24, 4
    target = torch.randn(d_out, r) @ torch.randn(r, d_in) * 0.3
    down = nn.Parameter(torch.randn(r, d_in) / d_in ** 0.5)
    up = nn.Parameter(torch.zeros(d_out, r))
    x = torch.randn(256, d_in)
    y = x @ target.T
    opt = MuonWithAuxAdam([down, up], lr=2e-2)
    losses = []
    for _ in range(300):
        opt.zero_grad()
        loss = nn.functional.mse_loss(x @ (up @ down).T, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        assert torch.isfinite(down).all() and torch.isfinite(up).all()
    assert losses[-1] < losses[0] * 0.05, f"{losses[0]} -> {losses[-1]}"


def test_create_and_build_dispatch_to_muon() -> None:
    from training.optimizers import build_optimizer
    from utils.optimizer_utils import create_optimizer

    w = nn.Parameter(torch.randn(4, 8))
    opt = create_optimizer("muon", [w], learning_rate=1e-3, weight_decay=0.0)
    assert isinstance(opt, MuonWithAuxAdam)

    args = types.SimpleNamespace(
        optimizer_type="muon", muon_momentum=0.9, muon_ns_steps=3, muon_adjust_lr="original",
    )
    opt = build_optimizer(args, [w], lr=1e-3, weight_decay=0.0)
    group = opt.param_groups[0]
    assert (group["momentum"], group["ns_steps"], group["adjust_lr_fn"]) == (0.9, 3, "original")


def test_rejects_unknown_adjust_lr_fn() -> None:
    with pytest.raises(ValueError, match="adjust_lr_fn"):
        MuonWithAuxAdam([nn.Parameter(torch.randn(2, 2))], adjust_lr_fn="bogus")
