"""LeapAlign / FlowBP 四变体轨迹自蒸馏的回归测试。

核心 sanity check（commit d4ee36f 声称做过、这里沉淀为回归）：
- 恒等速度场（v_θ ≡ noise - x0）下，四变体都应精确重构 x0 ⇒ loss ≈ 0（~1e-12）。
  这是直线 rectified flow 的解析性质：任何沿真实速度的积分（Euler 望远镜、
  straight-through connector、Simpson 三点）都精确回到 x0。
- sample_activation_timesteps 的覆盖性 / 降序 / 开区间约束。
- _finalize_loss 的 traj-sim 开关分支。
- TrainingConfig 互斥校验对四变体一致触发。

约定见 runtime/training/leap.py 头部 docstring：t=0 数据端，t=1 噪声端，
x_t=(1-t)x0+t·noise，真实速度 v=noise-x0（常向量场）。
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.families.anima.leap import (  # noqa: E402
    _finalize_loss,
    bridge_training_step,
    lagrange_training_step,
    leap_training_step,
    sample_activation_timesteps,
    sample_two_timesteps,
    sparse_training_step,
)


class _IdentityVelocityModel:
    """Mock 模型：恒返回真实速度场 v = noise - x0（直线流的常向量场）。

    rectified flow 的真实速度在整条直线上处处相等 = noise - x0，与输入点 / t 无关，
    所以 mock 直接返回闭包捕获的常向量即可，忽略 latents / timesteps。
    记录调用次数以便断言各变体的前向次数。
    """

    def __init__(self, x0: torch.Tensor, noise: torch.Tensor):
        self._v = noise - x0
        self.n_calls = 0

    def __call__(self, latents, timesteps, cross, padding_mask=None):  # noqa: ARG002
        self.n_calls += 1
        return self._v.clone()


def _make_batch(bs: int = 4, c: int = 3, h: int = 8, w: int = 8):
    torch.manual_seed(0)
    x0 = torch.randn(bs, c, h, w)
    noise = torch.randn(bs, c, h, w)
    cross = torch.zeros(bs, 1, 16)  # mock 忽略，仅占位
    pad_mask = torch.zeros(bs, 1, h, w)
    return x0, noise, cross, pad_mask


# ---------------------------------------------------------------------------
# 恒等速度下四变体精确重构 x0 ⇒ loss ≈ 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ngc", [0.0, 0.3, 1.0])
def test_original_exact_reconstruction(ngc: float) -> None:
    x0, noise, cross, pad_mask = _make_batch()
    model = _IdentityVelocityModel(x0, noise)
    t_k, t_j = sample_two_timesteps(x0.shape[0], x0.device, min_gap=0.1)
    loss = leap_training_step(
        model, x0, noise, cross, pad_mask, t_k, t_j, nested_grad_coe=ngc,
    )
    assert torch.all(loss < 1e-10), f"original loss not ~0: {loss}"


@pytest.mark.parametrize("ngc", [0.0, 0.3, 1.0])
def test_bridge_exact_reconstruction(ngc: float) -> None:
    x0, noise, cross, pad_mask = _make_batch()
    model = _IdentityVelocityModel(x0, noise)
    t_k, t_j = sample_two_timesteps(x0.shape[0], x0.device, min_gap=0.1)
    loss = bridge_training_step(
        model, x0, noise, cross, pad_mask, t_k, t_j, nested_grad_coe=ngc,
    )
    assert torch.all(loss < 1e-10), f"bridge loss not ~0: {loss}"


@pytest.mark.parametrize("ngc", [0.0, 0.3, 1.0])
def test_lagrange_exact_reconstruction(ngc: float) -> None:
    x0, noise, cross, pad_mask = _make_batch()
    model = _IdentityVelocityModel(x0, noise)
    t_k, t_j = sample_two_timesteps(x0.shape[0], x0.device, min_gap=0.1)
    loss = lagrange_training_step(
        model, x0, noise, cross, pad_mask, t_k, t_j, nested_grad_coe=ngc,
    )
    assert torch.all(loss < 1e-10), f"lagrange loss not ~0: {loss}"


@pytest.mark.parametrize("k", [2, 3, 5, 8])
def test_sparse_exact_reconstruction(k: int) -> None:
    x0, noise, cross, pad_mask = _make_batch()
    model = _IdentityVelocityModel(x0, noise)
    t_steps = sample_activation_timesteps(x0.shape[0], x0.device, k=k)
    loss = sparse_training_step(
        model, x0, noise, cross, pad_mask, t_steps,
    )
    assert torch.all(loss < 1e-10), f"sparse(k={k}) loss not ~0: {loss}"


# ---------------------------------------------------------------------------
# Lagrange：变速度场下锁死 Simpson 权重(1:4:1) + 段终点取解析真值(非 Euler 预测)
#
# 恒等速度场的精确重构测试抓不到这两件事：常数场下任意归一化权重(1:4:1 / 旧 1:1
# 梯形 / ...)都精确重构、且段终点输入是真值还是 Euler 预测都一样(v 处处相等)。
# 这里用 *依赖位置* 的速度场 v_θ(x)=x，让两者都可观测：
#   - 第二段(j→0)的 x̂0 只由该段三点速度决定(与第一段解耦)，可解析算出期望值；
#   - 期望值按"段终点用 x0"算；若实现退回 Euler 预测点，结果会偏，断言失败。
# ---------------------------------------------------------------------------


class _PositionVelocityModel:
    """Mock：v_θ(x, t) = x（依赖输入位置，与 t / cross 无关）。

    用于区分 Simpson 三点权重与段终点取值——常数场下不可观测，位置相关场下可观测。
    """

    def __init__(self) -> None:
        self.n_calls = 0

    def __call__(self, latents, timesteps, cross, padding_mask=None):  # noqa: ARG002
        self.n_calls += 1
        return latents.clone()


def test_lagrange_simpson_weights_and_endpoint() -> None:
    """v_θ(x)=x 下，第二段 x̂0 应等于按 x0 端点 + 1:4:1 权重算出的解析值。"""
    x0, noise, cross, pad_mask = _make_batch()
    # 固定 (k, j) 避免随机性，便于解析对照
    bs = x0.shape[0]
    t_k = torch.full((bs,), 0.8)
    t_j = torch.full((bs,), 0.4)

    # ngc=1.0：connector 前向值 = x_j_real（straight-through 数值），第二段起点确定
    loss = lagrange_training_step(
        _PositionVelocityModel(), x0, noise, cross, pad_mask, t_k, t_j,
        nested_grad_coe=1.0,
    )

    # ── 解析复算第二段（前向数值，梯度无关）──
    view = (-1, *([1] * (x0.ndim - 1)))
    k = t_k.view(*view)
    j = t_j.view(*view)
    # 第一段 Simpson(x_k, x_m1, x_j_real)，v=x ⇒ 端点速度=端点本身
    x_k = (1.0 - k) * x0 + k * noise
    x_j_real = (1.0 - j) * x0 + j * noise
    m1 = (k + j) * 0.5
    x_m1 = (1.0 - m1) * x0 + m1 * noise
    v_seg1 = (x_k + 4.0 * x_m1 + x_j_real) / 6.0
    x_hat_j = x_k - (k - j) * v_seg1
    # connector straight-through：前向数值 = x_j_real，第二段起点 = x_j_real
    x_j_in = x_j_real
    # 第二段 Simpson(x_j_in, x_m2, x0)，段终点用 x0（非 Euler 预测）
    m2 = j * 0.5
    x_m2 = (1.0 - m2) * x0 + m2 * noise
    v_seg2 = (x_j_in + 4.0 * x_m2 + x0) / 6.0
    x_hat_0_expected = x_j_in - j * v_seg2
    expected_loss = (x_hat_0_expected.float() - x0.float()).pow(2).mean(dim=(1, 2, 3))

    assert torch.allclose(loss, expected_loss, atol=1e-6), (
        f"lagrange x̂0 偏离 Simpson(1:4:1)+x0 端点解析值: {loss} vs {expected_loss}"
    )

    # ── 反向断言：若段终点退回 Euler 预测点，结果应 *不同*（守住回归）──
    v_j_start = x_j_in  # v=x
    x_hat_0_euler = x_j_in - j * v_j_start  # Euler 预测端点
    v_seg2_euler = (x_j_in + 4.0 * x_m2 + x_hat_0_euler) / 6.0
    x_hat_0_euler_full = x_j_in - j * v_seg2_euler
    euler_loss = (x_hat_0_euler_full.float() - x0.float()).pow(2).mean(dim=(1, 2, 3))
    assert not torch.allclose(loss, euler_loss, atol=1e-6), (
        "段终点取真值 vs Euler 预测应当可区分；若相等说明位置场测试退化"
    )


# ---------------------------------------------------------------------------
# 前向次数（显存/算力估算的依据，diff 里写错过一次，锁死）
# ---------------------------------------------------------------------------


def test_forward_counts() -> None:
    x0, noise, cross, pad_mask = _make_batch()
    t_k, t_j = sample_two_timesteps(x0.shape[0], x0.device, min_gap=0.1)

    m = _IdentityVelocityModel(x0, noise)
    leap_training_step(m, x0, noise, cross, pad_mask, t_k, t_j)
    assert m.n_calls == 2, "original 应 2× 前向"

    m = _IdentityVelocityModel(x0, noise)
    bridge_training_step(m, x0, noise, cross, pad_mask, t_k, t_j)
    assert m.n_calls == 2, "bridge 应 2× 前向"

    m = _IdentityVelocityModel(x0, noise)
    lagrange_training_step(m, x0, noise, cross, pad_mask, t_k, t_j)
    assert m.n_calls == 6, "lagrange 应 6× 前向（两段各三点）"

    for k in (2, 3, 5):
        m = _IdentityVelocityModel(x0, noise)
        t_steps = sample_activation_timesteps(x0.shape[0], x0.device, k=k)
        sparse_training_step(m, x0, noise, cross, pad_mask, t_steps)
        assert m.n_calls == k, f"sparse(k={k}) 应 {k}× 前向"


# ---------------------------------------------------------------------------
# sample_activation_timesteps：覆盖性 / 降序 / 开区间
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 5, 8])
def test_activation_timesteps_descending_and_open_interval(k: int) -> None:
    t = sample_activation_timesteps(64, torch.device("cpu"), k=k)
    assert t.shape == (64, k)
    # 严格降序（每行）
    diffs = t[:, :-1] - t[:, 1:]
    assert torch.all(diffs > 0), "每行应严格降序"
    # 开区间 (0,1)
    assert torch.all(t > 0.0) and torch.all(t < 1.0), "应落在开区间 (0,1)"


def test_activation_timesteps_spans_full_range() -> None:
    """分层抖动应铺满 (0,1)，而非塌缩到一端（旧贪心下压的退化场景）。"""
    t = sample_activation_timesteps(256, torch.device("cpu"), k=4)
    # 噪声端（t_1，最大）批均值应明显高于 0.5，数据端（t_K，最小）应明显低于 0.5
    assert t[:, 0].mean() > 0.6, "噪声端支撑点未铺到高 t"
    assert t[:, -1].mean() < 0.4, "数据端支撑点未铺到低 t"


def test_activation_timesteps_rejects_small_k() -> None:
    with pytest.raises(ValueError, match="k>=2"):
        sample_activation_timesteps(4, torch.device("cpu"), k=1)


# ---------------------------------------------------------------------------
# _finalize_loss：traj-sim 开关分支
# ---------------------------------------------------------------------------


def test_finalize_loss_no_weighting() -> None:
    x0 = torch.randn(4, 3, 8, 8)
    x_hat_0 = x0 + 0.1  # 固定残差
    loss = _finalize_loss(x_hat_0, x0, traj_sim_weighting=False)
    expected = (x_hat_0.float() - x0.float()).pow(2).mean(dim=(1, 2, 3))
    assert torch.allclose(loss, expected)


def test_finalize_loss_traj_sim_two_endpoints() -> None:
    """有中间端点时 w_sim = 1/(d_inter + d_0)。"""
    x0 = torch.randn(4, 3, 8, 8)
    x_hat_0 = x0 + 0.1
    x_inter_real = torch.randn(4, 3, 8, 8)
    x_hat_inter = x_inter_real + 0.2
    loss = _finalize_loss(
        x_hat_0, x0, x_hat_inter=x_hat_inter, x_inter_real=x_inter_real,
        traj_sim_weighting=True, traj_sim_min=0.01,
    )
    base = (x_hat_0.float() - x0.float()).pow(2).mean(dim=(1, 2, 3))
    d0 = (x0 - x_hat_0).abs().mean(dim=(1, 2, 3)).clamp(min=0.01)
    di = (x_inter_real - x_hat_inter).abs().mean(dim=(1, 2, 3)).clamp(min=0.01)
    assert torch.allclose(loss, base / (di + d0))


def test_finalize_loss_traj_sim_sparse_endpoint_only() -> None:
    """sparse 无中间端点时退化为 w_sim = 1/d_0。"""
    x0 = torch.randn(4, 3, 8, 8)
    x_hat_0 = x0 + 0.1
    loss = _finalize_loss(
        x_hat_0, x0, traj_sim_weighting=True, traj_sim_min=0.01,
    )
    base = (x_hat_0.float() - x0.float()).pow(2).mean(dim=(1, 2, 3))
    d0 = (x0 - x_hat_0).abs().mean(dim=(1, 2, 3)).clamp(min=0.01)
    assert torch.allclose(loss, base / d0)


# ---------------------------------------------------------------------------
# TrainingConfig 互斥校验对四变体一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["original", "sparse", "bridge", "lagrange"])
def test_config_leap_exclusive_infonoise(variant: str) -> None:
    from pydantic import ValidationError

    from studio.schema import TrainingConfig

    with pytest.raises(ValidationError, match="infonoise"):
        TrainingConfig(
            transformer_path="x", data_dir="x", output_dir="x",
            leap_enabled=True, leap_variant=variant, infonoise_enabled=True,
        )


@pytest.mark.parametrize("variant", ["original", "sparse", "bridge", "lagrange"])
def test_config_leap_exclusive_huber(variant: str) -> None:
    from pydantic import ValidationError

    from studio.schema import TrainingConfig

    with pytest.raises(ValidationError, match="huber"):
        TrainingConfig(
            transformer_path="x", data_dir="x", output_dir="x",
            leap_enabled=True, leap_variant=variant, loss_type="huber",
        )


@pytest.mark.parametrize("variant", ["original", "sparse", "bridge", "lagrange"])
def test_config_leap_variant_accepts_all(variant: str) -> None:
    from studio.schema import TrainingConfig

    cfg = TrainingConfig(
        transformer_path="x", data_dir="x", output_dir="x",
        leap_enabled=True, leap_variant=variant,
    )
    assert cfg.leap_variant == variant
