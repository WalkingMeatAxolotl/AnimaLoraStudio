"""非有限 loss micro-batch 的累积组语义（runtime/training/loop.py）。

回归背景（task 1848/1853 事故）：旧实现对 loss=NaN 的 micro-batch 做
`zero_grad() + continue`——
1. 把同一累积组内其它正常 micro-batch 已积累的梯度一并清掉；
2. 跳过组尾结算，组尾撞上 NaN 时 global_step 冻结，权重坏死后训练
   以「空转」跑完所有 epoch 并以 exit 0 伪装成功。

新语义：非有限 loss 只跳过本 micro-batch 的 backward，组内其它梯度保留、
组尾照常结算；组内全部被跳过（无任何梯度）时不 step、不推进 global_step。

harness：为 loop.run() 搭最小 fake（CPU、标准 rectified-flow 路径、真 SGD），
NaN 注入走真实事故同路径——latents 含 NaN → forward → loss NaN。
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from runtime.training import loop as loop_mod  # noqa: E402
from runtime.training.context import TrainingContext  # noqa: E402


class _ScalarGainModel(nn.Module):
    """pred = 输入 × 标量参数：梯度直达唯一参数，更新与否一眼可判。"""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * self.w


class _FakeFamily:
    spec = types.SimpleNamespace(family_id="fake")

    def encode_text_for_batch(self, text_stack, model, captions, device, dtype, **kw):
        return torch.zeros(len(captions), 1, 8)

    def forward_train(self, model, noisy, t, cross, use_checkpoint=False):
        return model(noisy)


class _FakeTimestepSampler:
    def sample(self, bs, device):
        return torch.full((bs,), 0.5)

    def record(self, t, mse):
        pass

    def maybe_refresh(self, step):
        pass

    def status(self):
        return {"kind": "fake"}


class _FakeInjector:
    def on_step_begin(self, step_ctx):
        pass

    def regularization_loss(self, step_ctx):
        return None

    def save(self, path):
        pass


class _FakeLoss:
    def compute(self, pred, target, t):
        return (pred - target) ** 2


class _FakeWandb:
    def log(self, *a, **k):
        pass

    def upload_model(self, *a, **k):
        pass

    def upload_state_auto(self, *a, **k):
        pass

    def upload_state_manual(self, *a, **k):
        pass

    def finish(self):
        pass


def _make_args(**over):
    d = dict(
        epochs=1,
        grad_accum=2,
        grad_checkpoint=False,
        max_steps=0,
        sample_steps=0,
        sample_every=0,
        save_every_epochs=0,
        save_every_steps=0,
        save_state_every_epochs=0,
        save_state_every_steps=0,
        log_every=10,  # 0 会在既有 infonoise 可观测性分支里除零（真实默认 10）
        loss_curve_steps=100,
        output_name="t",
        navit_packing=False,
        leap_enabled=False,
        masked_loss=False,
        loss_weighting="none",
        noise_enhancement_type="none",
        timestep_shift_resolution_aware=False,
        caption_comfy_encoding=True,
        kv_trim=False,
    )
    d.update(over)
    return types.SimpleNamespace(**d)


def _batch(nan: bool = False):
    lat = torch.randn(1, 4, 1, 8, 8)
    if nan:
        lat[...] = float("nan")
    return {"captions": ["c"], "latents": lat}


def _make_ctx(tmp_path, batches, monkeypatch, **args_over):
    # epoch 末的周期 IO（auto_epoch_state 写盘 + event）与本测试无关，打掉
    monkeypatch.setattr(loop_mod, "save_training_state", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "write_config_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "emit_event", lambda *a, **k: None)

    ctx = TrainingContext(args=_make_args(**args_over))
    model = _ScalarGainModel()
    ctx.family = _FakeFamily()
    ctx.device = "cpu"
    ctx.dtype = torch.float32
    ctx.use_cached = True
    ctx.dataloader = batches
    ctx.model = model
    ctx.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ctx.optimizer_type = "sgd"
    ctx.trainable_params = list(model.parameters())
    ctx.timestep_sampler = _FakeTimestepSampler()
    ctx.injector = _FakeInjector()
    ctx.loss_fn = _FakeLoss()
    ctx.wandb_monitor = _FakeWandb()
    ctx.output_dir = tmp_path / "out"
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    ctx.sample_dir = tmp_path / "samples"
    ctx.grad_clip = 0.0
    ctx.total_steps = 10
    return ctx


def test_baseline_two_finite_microbatches_step_once(tmp_path, monkeypatch):
    # harness 自检：正常两 micro-batch（ga=2）→ 恰好 1 次 step，参数被更新
    ctx = _make_ctx(tmp_path, [_batch(), _batch()], monkeypatch)
    loop_mod.run(ctx)
    assert ctx.global_step == 1
    assert float(ctx.model.w.detach()) != 1.0
    assert len(ctx.loss_history) == 1
    assert all(v == v for v in ctx.loss_history)  # 无 NaN


def test_nan_tail_microbatch_still_steps_with_kept_grads(tmp_path, monkeypatch):
    # [好, NaN]（ga=2）：旧实现组尾 NaN → zero_grad + continue → 好梯度作废、
    # global_step 冻结。新语义：好 micro-batch 的梯度保留，组尾照常 step。
    ctx = _make_ctx(tmp_path, [_batch(), _batch(nan=True)], monkeypatch)
    loop_mod.run(ctx)
    assert ctx.global_step == 1
    assert float(ctx.model.w.detach()) != 1.0
    # 组尾 micro-batch 非有限 → 本 step 跳过 loss 记录，不得混入 NaN
    assert all(v == v for v in ctx.loss_history)


def test_nan_head_microbatch_group_still_steps(tmp_path, monkeypatch):
    # [NaN, 好]：NaN 在组头不得影响随后的正常 micro-batch 结算
    ctx = _make_ctx(tmp_path, [_batch(nan=True), _batch()], monkeypatch)
    loop_mod.run(ctx)
    assert ctx.global_step == 1
    assert float(ctx.model.w.detach()) != 1.0
    assert len(ctx.loss_history) == 1
    assert all(v == v for v in ctx.loss_history)


def test_all_nan_group_does_not_step(tmp_path, monkeypatch):
    # 组内全 NaN → 无梯度可结算：不 step、global_step 不推进、参数不动
    # （空梯度 step 会污染 Prodigy 的 k / scheduler 进度，fp16 下 GradScaler 直接崩）
    ctx = _make_ctx(tmp_path, [_batch(nan=True), _batch(nan=True)], monkeypatch)
    loop_mod.run(ctx)
    assert ctx.global_step == 0
    assert float(ctx.model.w.detach()) == 1.0
    assert ctx.loss_history == []


# ── 连续非有限 loss 的 fail-fast（NonFiniteLossStreak）──


def test_streak_raises_at_threshold():
    s = loop_mod.NonFiniteLossStreak(abort_streak=3)
    s.record(False, global_step=7)
    s.record(False, global_step=7)
    with pytest.raises(RuntimeError, match="非有限"):
        s.record(False, global_step=7)


def test_streak_resets_on_finite_loss():
    s = loop_mod.NonFiniteLossStreak(abort_streak=3)
    s.record(False, global_step=0)
    s.record(False, global_step=0)
    s.record(True, global_step=0)  # 有限 loss 复位计数
    s.record(False, global_step=0)
    s.record(False, global_step=0)
    assert s.streak == 2  # 未达阈值，没 raise


def test_all_nan_training_fails_fast(tmp_path, monkeypatch):
    # 权重坏死场景（所有 micro-batch 永远 NaN）：达到阈值 raise 终止训练，
    # 而不是空转跑完全部 epoch 后以 exit 0 伪装完成（task 1848/1853 事故）
    monkeypatch.setattr(loop_mod, "NONFINITE_LOSS_ABORT_STREAK", 3)
    ctx = _make_ctx(tmp_path, [_batch(nan=True)] * 6, monkeypatch, epochs=2)
    with pytest.raises(RuntimeError, match="非有限"):
        loop_mod.run(ctx)
