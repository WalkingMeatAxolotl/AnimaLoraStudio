"""ADR 0003 PR-C：plugin registry + AdapterProtocol 单元测试。

覆盖：
- 4 个 plugin 子包的 BUILDERS / build_X / validate_schema_consistency 三件套
- AdapterProtocol runtime_checkable 对 AnimaLycorisAdapter 返回 True
- 动态/per-step / loss 加项 hook 在 mock adapter 上能被正确调用
- train_loop.py / phases/optimizer.py 已不含 if optimizer_type == / if lora_type ==
  风格的 dispatch（防回归）
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = REPO_ROOT / "runtime"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RUNTIME_DIR))


@pytest.fixture(scope="module")
def AnimaLycorisAdapter():
    """AnimaLycorisAdapter 类（lycoris-lora 后端可用时跑，否则 skip）。"""
    pytest.importorskip("lycoris")
    from utils.lycoris_adapter import AnimaLycorisAdapter as cls
    return cls


# ---------------------------------------------------------------------------
# Registry 三件套
# ---------------------------------------------------------------------------


def test_adapter_builders_dict_has_lokr_loha_lora() -> None:
    from training.adapters import BUILDERS
    assert set(BUILDERS) == {"lokr", "loha", "lora"}


def test_optimizer_builders_dict_has_5_variants() -> None:
    from training.optimizers import BUILDERS, VALIDATORS
    assert set(BUILDERS) == {"adamw", "automagic", "lion", "prodigy", "prodigy_plus_schedulefree"}
    # Automagic / PPSF 有专属 validator，adamw / lion / prodigy 没有
    assert set(VALIDATORS) == {"automagic", "prodigy_plus_schedulefree"}


def test_scheduler_builders_dict_excludes_none() -> None:
    from training.schedulers import BUILDERS, SCHEMA_ONLY_OPTIONS
    assert set(BUILDERS) == {"cosine", "cosine_with_restart"}
    assert SCHEMA_ONLY_OPTIONS == {"none"}


def test_inference_sampler_builders_has_er_sde() -> None:
    from training.inference_samplers import BUILDERS
    assert "er_sde" in BUILDERS


def test_loss_builders_dict_has_mse_huber() -> None:
    from training.losses import BUILDERS
    assert set(BUILDERS) == {"mse", "huber"}


def test_build_adapter_raises_on_unknown_lora_type() -> None:
    from training.adapters import build_adapter
    args = argparse.Namespace(lora_type="bogus_xyz")
    with pytest.raises(ValueError, match="未知 lora_type"):
        build_adapter(args)


def test_build_scheduler_returns_none_when_lr_scheduler_is_none() -> None:
    from training.schedulers import build_scheduler
    args = argparse.Namespace(lr_scheduler="none")
    assert build_scheduler(args, optimizer=None, total_steps=None) is None


def test_ppsf_zero_prodigy_steps_disables_freeze(monkeypatch) -> None:
    """ppsf_prodigy_steps=0 keeps the upstream PPSF meaning: never freeze d."""
    from training.optimizers import prodigy_plus_schedulefree as ppsf

    captured = {}

    def fake_create_optimizer(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "utils.optimizer_utils",
        types.SimpleNamespace(create_optimizer=fake_create_optimizer),
    )
    args = argparse.Namespace(
        ppsf_beta1=0.9,
        ppsf_beta2=0.99,
        ppsf_d_coef=1.0,
        ppsf_prodigy_steps=0,
        ppsf_split_groups=True,
        ppsf_split_groups_mean=False,
        ppsf_use_speed=False,
        ppsf_fused_back_pass=False,
        ppsf_use_stableadamw=True,
    )

    ppsf.build(args, params=[], lr=1.0, weight_decay=0.0)

    assert captured["prodigy_steps"] == 0


def test_ppsf_explicit_prodigy_steps_is_preserved(monkeypatch) -> None:
    from training.optimizers import prodigy_plus_schedulefree as ppsf

    captured = {}

    def fake_create_optimizer(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "utils.optimizer_utils",
        types.SimpleNamespace(create_optimizer=fake_create_optimizer),
    )
    args = argparse.Namespace(
        ppsf_beta1=0.9,
        ppsf_beta2=0.99,
        ppsf_d_coef=1.0,
        ppsf_prodigy_steps=750,
        ppsf_split_groups=True,
        ppsf_split_groups_mean=False,
        ppsf_use_speed=False,
        ppsf_fused_back_pass=False,
        ppsf_use_stableadamw=True,
    )

    ppsf.build(args, params=[], lr=1.0, weight_decay=0.0)

    assert captured["prodigy_steps"] == 750


# ---------------------------------------------------------------------------
# Schema↔registry 一致性自动校验（PR-C R4 缓解）
# ---------------------------------------------------------------------------


def test_adapter_schema_consistency_passes_on_clean_dev() -> None:
    from training.adapters import validate_schema_consistency
    # dev 上 schema.lora_type == {"lora","lokr","loha"} == BUILDERS keys
    validate_schema_consistency()  # 不抛即 pass


def test_optimizer_schema_consistency_passes_on_clean_dev() -> None:
    from training.optimizers import validate_schema_consistency
    validate_schema_consistency()


def test_scheduler_schema_consistency_passes_on_clean_dev() -> None:
    from training.schedulers import validate_schema_consistency
    validate_schema_consistency()


def test_loss_schema_consistency_passes_on_clean_dev() -> None:
    from training.losses import validate_schema_consistency
    validate_schema_consistency()


def test_schema_consistency_raises_when_builder_missing(monkeypatch) -> None:
    """模拟漏注册：schema 加了 lora_type=tlora 但 BUILDERS 没注册时，校验
    必须 raise，而不是放行让训练跑半天才暴露。"""
    from training import adapters
    monkeypatch.setitem(adapters.BUILDERS.copy(), "tlora", lambda args: None)
    # 临时改 schema 的 Literal 表演成 "schema 有 tlora 但 registry 没有"
    from studio.schema import TrainingConfig
    field = TrainingConfig.model_fields["lora_type"]
    original = field.annotation
    try:
        # 用 typing.Literal 重建一个含 "tlora" 的 annotation
        from typing import Literal
        field.annotation = Literal["lora", "lokr", "loha", "tlora"]  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="不同步"):
            adapters.validate_schema_consistency()
    finally:
        field.annotation = original  # 恢复


# ---------------------------------------------------------------------------
# AdapterProtocol runtime_checkable
# ---------------------------------------------------------------------------


def test_animalycoris_satisfies_adapter_protocol(AnimaLycorisAdapter) -> None:
    """AnimaLycorisAdapter 实现了全部 4 必需 + 3 可选 hook，
    isinstance(_, AdapterProtocol) 必须 True。"""
    from training.adapters.protocol import AdapterProtocol
    adapter = AnimaLycorisAdapter(algo="lokr")
    assert isinstance(adapter, AdapterProtocol)


def test_animalycoris_hooks_are_noop(AnimaLycorisAdapter) -> None:
    """LyCORIS 路径下 3 个 hook 必须 default no-op；
    on_step_begin 返回 None；regularization_loss 返回 None。"""
    from training.adapters.protocol import StepContext
    adapter = AnimaLycorisAdapter(algo="lokr")

    # 用极简 StepContext —— sigma_t 在 lyc 路径下不会被读
    import torch
    step_ctx = StepContext(
        global_step=0,
        total_steps=100,
        epoch=0,
        sigma_t=torch.zeros(1),
        args=argparse.Namespace(),
    )

    assert adapter.on_step_begin(step_ctx) is None
    assert adapter.regularization_loss(step_ctx) is None


def test_animalycoris_lokr_excludes_weight_decay_for_w1(AnimaLycorisAdapter) -> None:
    """LoKr 模式下 'lokr_w1' 子串参数排除 weight_decay。"""
    adapter = AnimaLycorisAdapter(algo="lokr")
    assert adapter.excludes_weight_decay("lora_unet_xxx.lokr_w1") is True
    assert adapter.excludes_weight_decay("lora_unet_xxx.lokr_w2_a") is False


def test_animalycoris_non_lokr_does_not_exclude_weight_decay(AnimaLycorisAdapter) -> None:
    """非 LoKr（lora / loha）模式：excludes_weight_decay 永远 False。"""
    adapter = AnimaLycorisAdapter(algo="lora")
    assert adapter.excludes_weight_decay("lora_unet_xxx.lokr_w1") is False
    adapter = AnimaLycorisAdapter(algo="loha")
    assert adapter.excludes_weight_decay("lora_unet_xxx.lokr_w1") is False


# ---------------------------------------------------------------------------
# Mock 论文级变体：验证 hook 接口能 cover T-LoRA / OFT 类需求
# ---------------------------------------------------------------------------


class _MockTLoRAAdapter:
    """模拟 T-LoRA：on_step_begin 按 sigma_t 调内部 mask（这里仅记录调用次数）。"""

    def __init__(self) -> None:
        self.step_begin_calls = 0
        self.last_sigma_t = None

    # 必需 4 个：no-op stubs（满足 Protocol 形状即可）
    def inject(self, model) -> None: pass
    def get_param_groups(self, weight_decay): return []
    def save(self, path) -> None: pass
    def load(self, path) -> None: pass

    def on_step_begin(self, ctx) -> None:
        self.step_begin_calls += 1
        self.last_sigma_t = ctx.sigma_t

    def regularization_loss(self, ctx): return None
    def excludes_weight_decay(self, name): return False


class _MockOFTAdapter:
    """模拟 OFT：regularization_loss 返回 orthogonality penalty。"""

    def __init__(self) -> None:
        self.reg_calls = 0

    def inject(self, model) -> None: pass
    def get_param_groups(self, weight_decay): return []
    def save(self, path) -> None: pass
    def load(self, path) -> None: pass
    def on_step_begin(self, ctx) -> None: pass

    def regularization_loss(self, ctx):
        import torch
        self.reg_calls += 1
        return torch.tensor(0.42)

    def excludes_weight_decay(self, name): return False


def test_mock_tlora_implements_protocol() -> None:
    from training.adapters.protocol import AdapterProtocol
    assert isinstance(_MockTLoRAAdapter(), AdapterProtocol)


def test_mock_oft_regularization_returns_tensor() -> None:
    import torch
    from training.adapters.protocol import StepContext
    adapter = _MockOFTAdapter()
    ctx = StepContext(global_step=5, total_steps=100, epoch=0,
                      sigma_t=torch.zeros(1), args=argparse.Namespace())
    loss = adapter.regularization_loss(ctx)
    assert isinstance(loss, torch.Tensor)
    assert float(loss) == pytest.approx(0.42)
    assert adapter.reg_calls == 1


def test_mock_tlora_on_step_begin_receives_sigma() -> None:
    import torch
    from training.adapters.protocol import StepContext
    adapter = _MockTLoRAAdapter()
    sigma = torch.tensor([0.3, 0.7])
    ctx = StepContext(global_step=10, total_steps=100, epoch=0,
                      sigma_t=sigma, args=argparse.Namespace())
    adapter.on_step_begin(ctx)
    assert adapter.step_begin_calls == 1
    assert torch.equal(adapter.last_sigma_t, sigma)


# ---------------------------------------------------------------------------
# 防回归：phases / loop 不再有 if optimizer_type == / if lora_type == 风格 dispatch
# ---------------------------------------------------------------------------


def test_no_optimizer_type_dispatch_in_phases_optimizer() -> None:
    text = (RUNTIME_DIR / "training" / "phases" / "optimizer.py").read_text(encoding="utf-8")
    # PR-C 后这些 if-elif 应该都被 build_optimizer 替代
    assert 'if ctx.optimizer_type == "prodigy"' not in text
    assert 'if optimizer_type ==' not in text
    assert 'optimizer_type == "prodigy_plus_schedulefree"' not in text


def test_no_lora_type_dispatch_in_phases_models() -> None:
    text = (RUNTIME_DIR / "training" / "phases" / "models.py").read_text(encoding="utf-8")
    # 应该看不到 AnimaLycorisAdapter 直接实例化（被 build_adapter 替代）
    assert "AnimaLycorisAdapter(" not in text


def test_no_lr_scheduler_dispatch_in_phases_optimizer() -> None:
    text = (RUNTIME_DIR / "training" / "phases" / "optimizer.py").read_text(encoding="utf-8")
    assert 'if lr_sched == "cosine"' not in text
    assert 'CosineAnnealingLR' not in text
    assert 'CosineAnnealingWarmRestarts' not in text


def test_no_er_sde_inline_dispatch_in_sampling() -> None:
    text = (RUNTIME_DIR / "training" / "sampling.py").read_text(encoding="utf-8")
    # sample_image 应该通过 build_inference_sampler 派发
    assert 'if sampler_name_l == "er_sde"' not in text
    assert "build_inference_sampler" in text
