"""Riemannion + RiemannLoRA adapter 测试 — 稠密参考实现对拍 + LoRA 导出 + registry。

- adapter 一次反向得到 ∇Z1 = G·B_R、∇B = Gᵀ·A_L（gradient trick），且 Z1 分支不改输出
- 低秩 ProjectLR 等于稠密切空间投影
- 两步 optimizer（含动量搬运）与稠密参考（显式投影 + 稠密 Newton-Schulz + 截断 SVD 回缩）一致
- CholeskyQR2 与 Householder QR 等价；秩亏时回退
- 训练中 A_L 保持列正交、ΔW 秩 ≤ r，能拟合低秩目标
- 导出标准 LoRA 后 up·down·(alpha/r) 精确等于 ΔW；从标准 LoRA 读回精确还原
- adapter + optimizer state_dict 往返后续训逐位一致
- registry：riemann ↔ riemannion 互相绑定；schema 同步

测试卫生：纯 CPU、float64 参考、固定随机种子。
"""
from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from utils.riemann_lora_adapter import RiemannLoRALinear
from utils.riemannion_optimizer import RiemannionWithAuxAdam, project_lr

M, N, R = 12, 10, 3


def _layer(seed: int = 0, m: int = M, n: int = N, r: int = R) -> RiemannLoRALinear:
    base = nn.Linear(n, m, bias=False)
    nn.init.zeros_(base.weight)
    return RiemannLoRALinear("lora_unet_x", base, rank=r, generator=torch.Generator().manual_seed(seed))


def _set_grads_from_dense(layer: RiemannLoRALinear, g: torch.Tensor) -> None:
    """等价于 autograd 结果（见 test_gradient_trick_matches_dense_gradient）。"""
    layer.lora_z1.grad = (g @ layer.b_r).float()
    layer.lora_b.grad = (g.mT @ layer.a_l).float()


def _dense_proj(a_l, b_r, z):
    pa = a_l @ a_l.mT
    return pa @ z + (torch.eye(a_l.shape[0], dtype=z.dtype) - pa) @ z @ b_r @ b_r.mT


def _dense_ortho(z):
    """稠密 Newton-Schulz（与实现同系数）：奇异值上的同一个奇多项式。"""
    from utils.riemannion_optimizer import NS_COEFFICIENTS, NS_STEPS

    a, b, c = NS_COEFFICIENTS
    x = z / z.norm()
    for _ in range(NS_STEPS):
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x


def _trunc(y, r):
    u, s, vh = torch.linalg.svd(y, full_matrices=False)
    return (u[:, :r] * s[:r]) @ vh[:r]


def test_gradient_trick_matches_dense_gradient() -> None:
    torch.manual_seed(0)
    layer = _layer()
    with torch.no_grad():
        layer.lora_b.copy_(torch.randn(N, R))
    x = torch.randn(7, N)
    c = torch.randn(7, M)
    y = layer(x)
    torch.testing.assert_close(y, x @ layer.delta_weight().mT)  # Z1 分支不改输出
    (c * y).sum().backward()
    g = c.mT @ x  # ∂L/∂ΔW
    torch.testing.assert_close(layer.lora_z1.grad, g @ layer.b_r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(layer.lora_b.grad, g.mT @ layer.a_l, rtol=1e-5, atol=1e-5)


def test_project_lr_matches_dense_projection() -> None:
    torch.manual_seed(0)
    a_l, _ = torch.linalg.qr(torch.randn(M, R, dtype=torch.float64))
    b_r, _ = torch.linalg.qr(torch.randn(N, R, dtype=torch.float64))
    a, b = torch.randn(M, 5, dtype=torch.float64), torch.randn(N, 5, dtype=torch.float64)
    a_dot, b_dot = project_lr(a, b, a_l, b_r)
    torch.testing.assert_close(a_dot @ b_r.mT + a_l @ b_dot.mT, _dense_proj(a_l, b_r, a @ b.mT))


@pytest.mark.parametrize("start_from_zero", [True, False])
def test_two_steps_match_dense_reference(start_from_zero) -> None:
    torch.manual_seed(1)
    layer = _layer()
    if not start_from_zero:
        with torch.no_grad():
            layer.lora_b.copy_(torch.randn(N, R))
            q, _ = torch.linalg.qr(layer.lora_b)
            layer.b_r.copy_(q)
    lr, wd, beta = 0.05, 0.1, 0.9
    opt = RiemannionWithAuxAdam([layer.lora_b, layer.lora_z1], lr=lr, weight_decay=wd, momentum=beta)

    x_ref = layer.delta_weight().double()
    a_l, b_r = layer.a_l.double(), layer.b_r.double()
    mom = None
    for _ in range(2):
        g = torch.randn(M, N, dtype=torch.float64)
        # 稠密参考
        xi = _dense_proj(a_l, b_r, g)
        if mom is not None:
            xi = beta * _dense_proj(a_l, b_r, mom) + xi
        mom = xi
        step_dir = _dense_proj(a_l, b_r, _dense_ortho(xi))
        x_ref = _trunc(x_ref * (1 - lr * wd) - lr * step_dir, R)
        # 实现
        _set_grads_from_dense(layer, g.float())
        opt.step()
        a_l, b_r = layer.a_l.double(), layer.b_r.double()
        torch.testing.assert_close(layer.delta_weight().double(), x_ref, rtol=1e-4, atol=1e-5)


def test_fits_low_rank_target_and_keeps_invariants() -> None:
    torch.manual_seed(0)
    m, n, r = 24, 32, 4
    layer = _layer(m=m, n=n, r=r)
    target = torch.randn(m, r) @ torch.randn(r, n) * 0.3
    x = torch.randn(256, n)
    y = x @ target.T
    opt = RiemannionWithAuxAdam([layer.lora_b, layer.lora_z1], lr=0.05)
    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = nn.functional.mse_loss(layer(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        assert torch.all(layer.lora_z1 == 0)
    assert losses[-1] < losses[0] * 0.05, f"{losses[0]} -> {losses[-1]}"
    eye = torch.eye(r)
    torch.testing.assert_close(layer.a_l.mT @ layer.a_l, eye, atol=1e-4, rtol=0)
    torch.testing.assert_close(layer.b_r.mT @ layer.b_r, eye, atol=1e-4, rtol=0)
    assert torch.linalg.matrix_rank(layer.delta_weight()) <= r


def test_export_plain_lora_is_exact_and_loads_back() -> None:
    torch.manual_seed(0)
    layer = _layer()
    with torch.no_grad():
        layer.lora_b.copy_(torch.randn(N, R))
    sd = layer.distilled_lora_state(dtype=torch.float32)
    up, down = sd["lora_unet_x.lora_up.weight"], sd["lora_unet_x.lora_down.weight"]
    scale = float(sd["lora_unet_x.alpha"]) / R
    torch.testing.assert_close(scale * up @ down, layer.delta_weight(), rtol=1e-5, atol=1e-5)

    other = _layer(seed=5)
    other.load_plain_lora_(down, up, scale)
    torch.testing.assert_close(other.delta_weight(), layer.delta_weight(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(other.a_l.mT @ other.a_l, torch.eye(R), atol=1e-5, rtol=0)


def test_state_dict_roundtrip_resumes_identically() -> None:
    from utils.riemann_lora_adapter import RiemannLoRAAdapter

    torch.manual_seed(0)
    x, c = torch.randn(16, N), torch.randn(16, M)

    def make():
        model = nn.Sequential(nn.Linear(N, M, bias=False))
        model[0].weight.data.zero_()
        adapter = RiemannLoRAAdapter(rank=R, preset={"target_name": ["0"]}, seed=0)
        adapter.inject(model)
        opt = RiemannionWithAuxAdam(adapter.get_param_groups(0.0), lr=0.05)
        return model, adapter, opt

    def run(model, opt, n):
        for _ in range(n):
            opt.zero_grad()
            (c * model(x)).sum().backward()
            opt.step()

    model_a, adapter_a, opt_a = make()
    run(model_a, opt_a, 3)
    model_b, adapter_b, opt_b = make()
    adapter_b.load_state_dict(adapter_a.state_dict())
    opt_b.load_state_dict(opt_a.state_dict())
    run(model_a, opt_a, 3)
    run(model_b, opt_b, 3)
    assert torch.equal(adapter_a.loras[0].delta_weight(), adapter_b.loras[0].delta_weight())


def test_unmarked_params_go_to_aux_adam() -> None:
    layer = _layer()
    bias = nn.Parameter(torch.zeros(4))
    opt = RiemannionWithAuxAdam([layer.lora_b, layer.lora_z1, bias], lr=1e-2)
    assert [(g["use_riemann"], len(g["params"])) for g in opt.param_groups] == [(True, 2), (False, 1)]


def test_riemann_and_riemannion_require_each_other() -> None:
    from training.adapters import build_adapter
    from training.optimizers import validate_optimizer

    with pytest.raises(ValueError, match="riemannion"):
        build_adapter(types.SimpleNamespace(lora_type="riemann", optimizer_type="adamw", lora_rank=4), preset={})
    with pytest.raises(ValueError, match="riemann"):
        validate_optimizer(types.SimpleNamespace(optimizer_type="riemannion", lora_type="lora"))
    validate_optimizer(types.SimpleNamespace(optimizer_type="riemannion", lora_type="riemann", blocks_to_swap=4))


def test_cholesky_qr2_matches_householder_and_falls_back() -> None:
    from utils.riemannion_optimizer import _qr

    torch.manual_seed(0)
    x = torch.randn(3, 200, 16)
    q, r = _qr(x)
    torch.testing.assert_close(q @ r, x, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(q.mT @ q, torch.eye(16).expand(3, 16, 16), atol=1e-5, rtol=0)
    x[:, :, 3] = 0  # 秩亏 → Cholesky 失败 → Householder 回退，Q 仍正交
    q, r = _qr(x)
    torch.testing.assert_close(q @ r, x, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(q.mT @ q, torch.eye(16).expand(3, 16, 16), atol=1e-5, rtol=0)


def test_zero_gradient_layer_is_skipped() -> None:
    layer = _layer()
    opt = RiemannionWithAuxAdam([layer.lora_b, layer.lora_z1], lr=0.1)
    before = (layer.a_l.clone(), layer.lora_b.clone())
    _set_grads_from_dense(layer, torch.zeros(M, N))
    opt.step()
    assert torch.equal(layer.a_l, before[0]) and torch.equal(layer.lora_b, before[1])


def test_wraps_int8_base_without_reading_weights() -> None:
    """int8 底模（MOSS W8A16）：包装层只读权重形状，原 forward（量化路径）照常调用。"""
    base = nn.Linear(N, M, bias=False)
    base.weight = nn.Parameter(torch.randint(-127, 127, (M, N), dtype=torch.int8), requires_grad=False)
    calls = []

    def fake_int8_forward(x):
        calls.append(x.shape)
        return torch.zeros(*x.shape[:-1], M, dtype=x.dtype)

    base.forward = fake_int8_forward
    layer = RiemannLoRALinear("lora_unet_x", base, rank=R, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        layer.lora_b.copy_(torch.randn(N, R))
    x = torch.randn(5, N)
    torch.testing.assert_close(layer(x), x @ layer.delta_weight().mT)
    assert calls == [torch.Size([5, N])]


# ---------------------------------------------------------------------------
# 配置联动：riemann ↔ riemannion 互绑 + 对 riemann 不生效的字段隐藏
# ---------------------------------------------------------------------------


def test_config_riemann_pins_riemannion_and_forbids_elsewhere() -> None:
    from pydantic import ValidationError

    from studio.domain.config_rules import apply_disable_rule_fixes
    from studio.schema import TrainingConfig

    # 缺省跟随：只选 riemann 时优化器自动落 riemannion
    assert TrainingConfig(lora_type="riemann").optimizer_type == "riemannion"
    TrainingConfig(lora_type="riemann", optimizer_type="riemannion")
    with pytest.raises(ValidationError, match="optimizer_type"):
        TrainingConfig(lora_type="riemann", optimizer_type="adamw")
    with pytest.raises(ValidationError, match="optimizer_type"):
        TrainingConfig(lora_type="lora", optimizer_type="riemannion")
    # tolerant 修复：离开 riemann 后残留的 riemannion 回到 schema 默认
    data = TrainingConfig().model_dump(mode="python")
    data.update(lora_type="lora", optimizer_type="riemannion")
    fixed, fields = apply_disable_rule_fixes(data, TrainingConfig)
    assert fixed["optimizer_type"] == "adamw" and fields == ["optimizer_type"]


def test_config_riemann_hides_fields_it_ignores() -> None:
    from studio.domain.config_prune import prune_inactive_fields
    from studio.schema import TrainingConfig

    ignored = ("lora_alpha", "lora_dora", "lora_rs", "lora_dropout",
               "lora_rank_dropout", "lora_module_dropout", "lora_reg_dims")
    pruned = prune_inactive_fields(TrainingConfig(lora_type="riemann").model_dump(mode="python"))
    assert not set(ignored) & set(pruned)
    kept = prune_inactive_fields(TrainingConfig(lora_type="lora").model_dump(mode="python"))
    assert "lora_alpha" in kept and "lora_dora" in kept


def test_resume_lora_from_plain_lora_and_rejects_incompatible_file(tmp_path) -> None:
    from safetensors.torch import save_file

    from utils.riemann_lora_adapter import RiemannLoRAAdapter

    def make():
        model = nn.Sequential(nn.Linear(N, M, bias=False))
        adapter = RiemannLoRAAdapter(rank=R, preset={"target_name": ["0"]}, seed=0)
        adapter.inject(model)
        return adapter

    src = make()
    with torch.no_grad():
        src.loras[0].lora_b.copy_(torch.randn(N, R))
    plain = tmp_path / "plain.safetensors"
    src.save(plain)  # 导出的就是标准 LoRA
    dst = make()
    dst.load(plain)
    # 导出存 bf16，按 bf16 精度比较
    torch.testing.assert_close(dst.loras[0].delta_weight(), src.loras[0].delta_weight(), rtol=2e-2, atol=5e-3)

    lokr = tmp_path / "lokr.safetensors"
    save_file({"lora_unet_0.lokr_w1": torch.zeros(2, 2)}, str(lokr))
    with pytest.raises(RuntimeError, match="resume_lora"):
        make().load(lokr)


def test_adapter_tensors_follow_training_device_not_offloaded_weight() -> None:
    """block swap：被换出块的底模权重在别的设备上，adapter 参数仍建在训练设备。"""
    base = nn.Linear(N, M, bias=False, device="meta")
    layer = RiemannLoRALinear(
        "lora_unet_x", base, rank=R, generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    for t in (layer.a_l, layer.b_r, layer.lora_b, layer.lora_z1):
        assert t.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gesvda 只在 CUDA 上")
def test_svd_falls_back_when_gesvda_fails_to_converge() -> None:
    """大量重复零奇异值（从低有效秩的普通 LoRA 热启动）会让 gesvda 抛错。"""
    from utils.riemannion_optimizer import _svd

    torch.manual_seed(0)
    u, _ = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(64, 64, dtype=torch.float64))
    s = torch.cat([torch.rand(10, dtype=torch.float64) * 0.1, torch.zeros(54, dtype=torch.float64)])
    core = ((u * s) @ v.T).float().cuda().expand(4, 64, 64).contiguous()
    with pytest.raises(torch.linalg.LinAlgError):
        torch.linalg.svd(core, driver="gesvda")
    uu, ss, vh = _svd(core)
    assert uu.is_cuda
    torch.testing.assert_close((uu * ss.unsqueeze(-2)) @ vh, core, atol=1e-5, rtol=1e-4)
