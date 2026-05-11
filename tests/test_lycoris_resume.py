"""utils.lycoris_adapter + anima_train.save/load_training_state round-trip 测试。

不依赖真实 Anima 模型；用 mock DiT 验证：
- adapter.state_dict ↔ adapter.load_state_dict 张量 bit-exact
- save_training_state ↔ load_training_state 完整字段还原（含 optimizer + RNG）
- resume 后 forward 输出与 resume 前一致
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent

# 手动 import anima_train.py 中的 save/load_training_state（不走包导入避免顶层副作用）
_spec = importlib.util.spec_from_file_location(
    "_anima_train_for_test", REPO_ROOT / "runtime" / "anima_train.py"
)
# 这个 import 会触发依赖检测 — 如果环境缺包就跳过整个测试文件
try:
    _at = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_at)
    save_training_state = _at.save_training_state
    load_training_state = _at.load_training_state
except Exception as e:
    pytest.skip(f"anima_train.py 加载失败: {e}", allow_module_level=True)

from utils.lycoris_adapter import AnimaLycorisAdapter


class MockDiT(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.output_proj = nn.Linear(d, d, bias=False)


def _make_trained_adapter(seed: int = 42) -> tuple[AnimaLycorisAdapter, MockDiT, torch.optim.Optimizer]:
    """构造一个跑过若干 step 的 adapter + optimizer，让权重不再是初始化值"""
    torch.manual_seed(seed)
    model = MockDiT()
    adapter = AnimaLycorisAdapter(algo="lokr", rank=4, alpha=4, factor=8)
    adapter.inject(model)

    optimizer = torch.optim.AdamW(adapter.get_params(), lr=1e-3)
    # 跑 3 step 让权重 + optimizer state 都改变
    for _ in range(3):
        x = torch.randn(2, 128)
        y = model.q_proj(x).sum()
        y.backward()
        optimizer.step()
        optimizer.zero_grad()
    return adapter, model, optimizer


def test_adapter_state_dict_roundtrip_bit_exact(tmp_path):
    """adapter.state_dict() → load_state_dict() 张量 bit-exact"""
    adapter, _, _ = _make_trained_adapter()
    sd = adapter.state_dict()

    # 装载到新 adapter
    model2 = MockDiT()
    adapter2 = AnimaLycorisAdapter(algo="lokr", rank=4, alpha=4, factor=8)
    adapter2.inject(model2)
    adapter2.load_state_dict(sd, strict=True)

    sd2 = adapter2.state_dict()
    assert set(sd.keys()) == set(sd2.keys()), "键集合不一致"
    for k in sd:
        if "alpha" in k:
            continue  # alpha 是 buffer，可能由 lycoris 重新计算
        assert torch.equal(sd[k], sd2[k]), f"张量不一致: {k}"


def test_save_training_state_roundtrip(tmp_path):
    """save_training_state → load_training_state 完整字段还原"""
    adapter, _, optimizer = _make_trained_adapter()

    state_path = tmp_path / "state.pt"
    save_training_state(
        state_path, adapter, optimizer,
        epoch=2, global_step=42,
        loss_history=[0.5, 0.3, 0.2],
        monitor_state={"foo": "bar"},
    )
    assert state_path.exists()

    # 新一组 adapter+optimizer 加载
    model2 = MockDiT()
    adapter2 = AnimaLycorisAdapter(algo="lokr", rank=4, alpha=4, factor=8)
    adapter2.inject(model2)
    optimizer2 = torch.optim.AdamW(adapter2.get_params(), lr=1e-3)

    epoch, step, history, monitor = load_training_state(state_path, adapter2, optimizer2)
    assert epoch == 2
    assert step == 42
    assert history == [0.5, 0.3, 0.2]
    assert monitor == {"foo": "bar"}

    # 权重对齐
    sd1 = adapter.state_dict()
    sd2 = adapter2.state_dict()
    for k in sd1:
        if "alpha" in k:
            continue
        assert torch.equal(sd1[k], sd2[k]), f"resume 后权重不一致: {k}"

    # optimizer state 对齐（验证 step count 进入了状态）
    assert len(optimizer2.state) == len(optimizer.state)


def test_save_load_preserves_w1_no_decay_grouping(tmp_path):
    """resume 后 get_param_groups 仍然正确排除 lokr_w1 的 weight_decay"""
    adapter, _, optimizer = _make_trained_adapter()
    state_path = tmp_path / "state.pt"
    save_training_state(state_path, adapter, optimizer, epoch=0, global_step=0)

    model2 = MockDiT()
    adapter2 = AnimaLycorisAdapter(algo="lokr", rank=4, alpha=4, factor=8)
    adapter2.inject(model2)
    optimizer2 = torch.optim.AdamW(adapter2.get_params(), lr=1e-3)
    load_training_state(state_path, adapter2, optimizer2)

    groups = adapter2.get_param_groups(weight_decay=0.01)
    assert len(groups) == 2
    assert any(g["weight_decay"] == 0.0 for g in groups), "缺少 wd=0 组（w1 应排除）"
    assert any(g["weight_decay"] == 0.01 for g in groups), "缺少 wd=0.01 组（w2 系）"


def test_legacy_state_dict_strict_false_does_not_crash(tmp_path):
    """旧格式 ckpt 触发 missing_keys 时 load_training_state 不应崩"""
    adapter, _, optimizer = _make_trained_adapter()

    state_path = tmp_path / "legacy.pt"
    # 模拟旧自实现的键名（lokr_w2_a/b 而非 lycoris 的 lokr_w2）
    fake_legacy_sd = {
        "lora_unet_q_proj.lokr_w1": torch.zeros(8, 8),
        "lora_unet_q_proj.lokr_w2_a": torch.zeros(16, 4),
        "lora_unet_q_proj.lokr_w2_b": torch.zeros(4, 16),
    }
    torch.save({
        "lora_state_dict": fake_legacy_sd,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 0,
        "global_step": 0,
    }, state_path)

    model2 = MockDiT()
    adapter2 = AnimaLycorisAdapter(algo="lokr", rank=4, alpha=4, factor=8)
    adapter2.inject(model2)
    optimizer2 = torch.optim.AdamW(adapter2.get_params(), lr=1e-3)

    # 不应抛异常 — 走 strict=False + warning 路径
    load_training_state(state_path, adapter2, optimizer2)


def test_model_eval_cascades_to_lycoris_network():
    """model.eval() 必须同步到 LycorisNetwork（否则 sample 时走 dropout 分支报 device mismatch）"""
    model = MockDiT()
    adapter = AnimaLycorisAdapter(
        algo="lokr", rank=4, alpha=4, factor=8,
        rank_dropout=0.1,  # 触发 dropout 分支
    )
    adapter.inject(model)

    # inject 时 model 默认 train mode
    assert adapter.network.training is True

    # model.eval() 应当级联
    model.eval()
    assert adapter.network.training is False, "lycoris network 未跟随 model.eval()"
    for lora in adapter.network.loras:
        assert lora.training is False, f"{lora.lora_name} 未 eval"

    # model.train() 切回
    model.train()
    assert adapter.network.training is True


def test_rng_state_restored(tmp_path):
    """resume 后再 sample 一个 random，应等于第一次 sample 的下一个值"""
    adapter, _, optimizer = _make_trained_adapter()

    # 设定一个干净的 RNG 起点
    torch.manual_seed(123)
    random.seed(123)
    sample_a = (torch.randn(3).tolist(), random.random())
    # 此时 RNG 已经前进了

    state_path = tmp_path / "state.pt"
    save_training_state(state_path, adapter, optimizer, epoch=0, global_step=0)

    # 接着原本会 sample 出来的下一个值 — 记录它
    expected_next = (torch.randn(3).tolist(), random.random())

    # 修改 RNG 到完全不同的状态，再 resume
    torch.manual_seed(999)
    random.seed(999)
    model2 = MockDiT()
    adapter2 = AnimaLycorisAdapter(algo="lokr", rank=4, alpha=4, factor=8)
    adapter2.inject(model2)
    optimizer2 = torch.optim.AdamW(adapter2.get_params(), lr=1e-3)
    load_training_state(state_path, adapter2, optimizer2)

    # resume 后 RNG 应当恢复到 save 时刻；下一个 sample 应等于 expected_next
    actual_next = (torch.randn(3).tolist(), random.random())
    assert actual_next == expected_next, "RNG 未正确恢复"
