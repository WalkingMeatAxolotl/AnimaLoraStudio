from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest


class _FakeModel:
    def train(self):
        return None


def _ctx(tmp_path: Path, *, sample_every: int, sample_steps: int = 0):
    from training.context import TrainingContext

    args = Namespace(
        no_progress=True,
        loss_curve_steps=100,
        no_live_curve=True,
        resume_state="",
        sample_prompts=["p1", "p2"],
        sample_prompt="fallback",
        sample_steps=sample_steps,
        sample_every=sample_every,
        optimizer_type="adamw",
    )
    ctx = TrainingContext(args=args)
    ctx.total_steps = 10
    ctx.sample_dir = tmp_path
    ctx.optimizer = None
    ctx.model = _FakeModel()
    ctx.monitor_server = False
    ctx.wandb_monitor = Namespace(log_samples=False)
    return ctx


def test_resume_phase_runs_startup_baseline_at_step0(tmp_path, monkeypatch):
    """周期采样开启时，新训练（global_step == 0）跑 step 0 baseline 采样。"""
    pytest.importorskip("torch")
    from training.phases import resume

    calls = []
    monkeypatch.setattr(resume, "run_sample", lambda *args, **kwargs: calls.append(kwargs))

    ctx = _ctx(tmp_path, sample_every=2)
    resume.run(ctx)

    assert [call["prompt"] for call in calls] == ["p1", "p2"]
    assert [call["sample_path"].name for call in calls] == [
        "step_0_baseline_0.png",
        "step_0_baseline_1.png",
    ]


def test_resume_phase_skips_baseline_when_sampling_disabled(tmp_path, monkeypatch):
    """sample_every / sample_steps 都为 0（周期采样禁用）时不跑 baseline。"""
    pytest.importorskip("torch")
    from training.phases import resume

    calls = []
    monkeypatch.setattr(resume, "run_sample", lambda *args, **kwargs: calls.append(kwargs))

    ctx = _ctx(tmp_path, sample_every=0, sample_steps=0)
    resume.run(ctx)

    assert calls == []
