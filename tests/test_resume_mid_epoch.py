"""epoch 中途 step 恢复点续训的逐步一致性（runtime/training/loop.py + state.py）。

回归背景：`save_state_every_steps` 存的是「当前 epoch 序号」，旧 resume 走
`for epoch in range(start_epoch, ...)` 从该 epoch 第 0 个 batch 重新开始——
恢复点之前已训过的 batch 被再训一遍，恢复后第 N+1 步吃的是原轨迹第 1 步的数据，
逐步 loss 与不中断训练对不上（真机 AdamW 最大差 ≈0.049）。另外 DataLoader 每次
iter() 从全局 torch CPU RNG 抽 worker seed，恢复后重建迭代器多抽一次，CPU 侧
随机流（InfoNoise 采样等）整体错一位。

修复：恢复点记 `epoch_batch_offset`，sampler 在索引层跳过已训 batch；DataLoader
用独立 generator。本文件用真 DataLoader / BucketBatchSampler / save+load_training_state
跑「不中断」与「中途恢复」两条轨迹，断言恢复后逐步 loss 与终态权重逐位一致。
"""
from __future__ import annotations

import random
import types

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

from runtime.training import loop as loop_mod  # noqa: E402
from runtime.training.dataset import BucketBatchSampler, NavitPackBatchSampler  # noqa: E402
from runtime.training.phases import resume as resume_mod  # noqa: E402
from runtime.training.phases.dataset import build_dataloader  # noqa: E402
from tests.test_loop_nonfinite_loss import _make_ctx  # noqa: E402

N_ITEMS = 6


class _IdxDataset(Dataset):
    """每个样本的 latent 由下标决定；__getitem__ 掷一次 Python random 模拟 flip 增广，
    让「被跳过的 batch 是否经过 __getitem__」反映到随机流上。"""

    def __len__(self):
        return N_ITEMS

    def __getitem__(self, idx):
        sign = -1.0 if random.random() < 0.5 else 1.0
        g = torch.Generator().manual_seed(1000 + idx)
        return {"latent": sign * torch.randn(4, 1, 8, 8, generator=g), "caption": str(idx)}


def _collate(items):
    return {
        "latents": torch.stack([it["latent"] for it in items]),
        "captions": [it["caption"] for it in items],
    }


class _CpuRandTimestepSampler:
    """CPU 全局 RNG 采样（同 InfoNoise）：DataLoader 偷抽全局 RNG 会直接改变 t。"""

    def sample(self, bs, device):
        return torch.rand(bs).clamp(0.05, 0.95)

    def record(self, t, mse):
        pass

    def maybe_refresh(self, step):
        pass

    def status(self):
        return {"kind": "fake"}


class _StateInjector:
    """有 state_dict / load_state_dict 的 injector 替身，供真 save/load_training_state 用。"""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, sd, strict=False):
        return self.model.load_state_dict(sd, strict=strict)

    def on_step_begin(self, step_ctx):
        pass

    def regularization_loss(self, step_ctx):
        return None

    def save(self, path):
        pass


def _ctx(tmp_path, monkeypatch, *, grad_accum, save_every, resume_state=""):
    ctx = _make_ctx(
        tmp_path, None, monkeypatch,
        epochs=2, grad_accum=grad_accum, save_state_every_steps=save_every,
        resume_state=resume_state, seed=7, batch_size=1,
        no_progress=True, no_live_curve=True,
        sample_prompts=[], sample_prompt="",
    )
    # _make_ctx 打掉了 save_training_state；本测试要真存真读
    monkeypatch.undo()
    monkeypatch.setattr(loop_mod, "write_config_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "emit_event", lambda *a, **k: None)
    monkeypatch.setattr(resume_mod, "emit_event", lambda *a, **k: None)
    monkeypatch.setattr(resume_mod.signal, "signal", lambda *a, **k: None)

    ds = _IdxDataset()
    sampler = BucketBatchSampler(ds, batch_size=1, drop_last=False, shuffle=True, seed=7)
    ctx.dataloader = build_dataloader(ds, sampler, _collate, num_workers=0, seed=7)
    ctx.timestep_sampler = _CpuRandTimestepSampler()
    ctx.injector = _StateInjector(ctx.model)
    ctx.optimizer = torch.optim.AdamW(ctx.model.parameters(), lr=0.05)
    ctx.monitor_server = None
    ctx.family.spec = types.SimpleNamespace(family_id="anima")
    return ctx


def _run(ctx, seed=7):
    torch.manual_seed(seed)
    random.seed(seed)
    resume_mod.run(ctx)
    loop_mod.run(ctx)
    return list(ctx.loss_history), float(ctx.model.w.detach())


@pytest.mark.parametrize(
    "grad_accum,save_every,resume_step",
    [
        (1, 2, 2),   # epoch 0 中途
        (1, 2, 8),   # epoch 1 中途
        (1, 3, 6),   # 恰好落在 epoch 0 最后一个 batch（offset == dl_len）
        (2, 1, 2),   # 梯度累积：offset = 4 个 micro-batch
    ],
)
def test_mid_epoch_resume_matches_uninterrupted(tmp_path, monkeypatch, grad_accum, save_every, resume_step):
    ref_ctx = _ctx(tmp_path / "ref", monkeypatch, grad_accum=grad_accum, save_every=save_every)
    ref_losses, ref_w = _run(ref_ctx)
    state = ref_ctx.state_dir() / f"training_state_step{resume_step}.pt"
    assert state.exists()

    res_ctx = _ctx(
        tmp_path / "res", monkeypatch,
        grad_accum=grad_accum, save_every=0, resume_state=str(state),
    )
    # 恢复前先把全局 RNG 搅乱：恢复后的随机流只能来自恢复点
    res_losses, res_w = _run(res_ctx, seed=12345)

    assert res_losses == ref_losses
    assert res_w == ref_w
    assert res_ctx.global_step == ref_ctx.global_step


def test_bucket_sampler_skip_batches_is_one_shot():
    ds = _IdxDataset()
    sampler = BucketBatchSampler(ds, batch_size=1, drop_last=False, shuffle=True, seed=3)
    sampler.set_epoch(1)
    full = list(sampler)
    sampler.skip_batches(4)
    assert list(sampler) == full[4:]
    assert len(sampler) == len(full)  # 长度仍是整 epoch（batch_idx 从 offset 起编号）
    assert list(sampler) == full      # 只生效一次


def test_navit_sampler_skip_batches_is_one_shot():
    ds = _IdxDataset()
    ds.token_count_for_index = [16 + 8 * i for i in range(N_ITEMS)]
    sampler = NavitPackBatchSampler(ds, token_budget=48, shuffle=True, seed=3)
    sampler.set_epoch(2)
    full = list(sampler)
    assert len(full) > 2
    sampler.set_epoch(2)
    sampler.skip_batches(2)
    assert list(sampler) == full[2:]
    assert list(sampler) == full


def test_dataloader_iter_does_not_consume_global_rng():
    ds = _IdxDataset()
    sampler = BucketBatchSampler(ds, batch_size=1, drop_last=False, shuffle=False)
    dl = build_dataloader(ds, sampler, _collate, num_workers=0, seed=0)
    torch.manual_seed(0)
    expected = torch.rand(1)
    torch.manual_seed(0)
    iter(dl)
    assert torch.equal(torch.rand(1), expected)
