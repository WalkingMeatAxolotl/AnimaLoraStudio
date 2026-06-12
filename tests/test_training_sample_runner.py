from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training import sample_runner


class FakeImage:
    def __init__(self) -> None:
        self.saved_to: Path | None = None

    def save(self, path: Path) -> None:
        self.saved_to = path


class FakeWandb:
    log_samples = False

    def log_image(self, *_args, **_kwargs) -> None:
        raise AssertionError("log_image should not be called when log_samples=False")


class FakeInjector:
    def __init__(self) -> None:
        self.cleared = 0

    def clear_timestep_mask(self) -> None:
        self.cleared += 1


def _ctx(*, sample_negative_prompt: str = "") -> SimpleNamespace:
    args = SimpleNamespace(
        resolution=1024,
        sample_width=512,
        sample_height=512,
        sample_cfg_scale=4.0,
        sample_negative_prompt=sample_negative_prompt,
        sample_seed=123,
        sample_infer_steps=25,
        sample_sampler_name="er_sde",
        sample_scheduler="simple",
    )
    return SimpleNamespace(
        args=args,
        model=torch.nn.Linear(1, 1),
        vae=object(),
        qwen_model=object(),
        qwen_tok=object(),
        t5_tok=object(),
        optimizer=object(),
        injector=FakeInjector(),
        device="cpu",
        dtype=torch.float32,
        wandb_monitor=FakeWandb(),
        monitor_server=False,
        emit=lambda *_args, **_kwargs: None,
    )


def test_run_sample_preserves_empty_negative_and_passes_seed(monkeypatch, tmp_path) -> None:
    records: list[dict] = []

    def fake_sample_image(*_args, **kwargs):
        records.append(kwargs)
        return FakeImage()

    monkeypatch.setattr(sample_runner, "sample_image", fake_sample_image)

    ctx = _ctx(sample_negative_prompt="")
    sample_runner.run_sample(ctx, prompt="1girl", sample_path=tmp_path / "sample.png")

    # 对齐 ComfyUI：空负面就是空，不注入隐式默认串
    assert records[-1]["negative_prompt"] == ""
    assert records[-1]["seed"] == 123
    assert "comfy_parity" not in records[-1]


def test_run_sample_clears_tlora_timestep_mask_before_sampling(monkeypatch, tmp_path) -> None:
    """#215 T-LoRA：sample 前必须清训练态 mask（merge 冲突解法的回归锚）。"""
    monkeypatch.setattr(sample_runner, "sample_image", lambda *a, **k: FakeImage())

    ctx = _ctx()
    sample_runner.run_sample(ctx, prompt="1girl", sample_path=tmp_path / "sample.png")

    assert ctx.injector.cleared == 1


def test_run_sample_logs_failure_and_restores_train_mode(monkeypatch, tmp_path) -> None:
    def fail_sample_image(*_args, **_kwargs):
        raise RuntimeError("sample boom")

    monkeypatch.setattr(sample_runner, "sample_image", fail_sample_image)

    ctx = _ctx()
    ctx.model.train()

    sample_runner.run_sample(ctx, prompt="1girl", sample_path=tmp_path / "sample.png")

    assert ctx.model.training is True
