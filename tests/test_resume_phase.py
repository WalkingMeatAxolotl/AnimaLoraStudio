from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest


class _FakeModel:
    def train(self):
        return None


def _ctx(tmp_path: Path, *, sample_on_start: bool):
    from training.context import TrainingContext

    args = Namespace(
        no_progress=True,
        loss_curve_steps=100,
        no_live_curve=True,
        resume_state="",
        sample_prompts=["p1", "p2"],
        sample_prompt="fallback",
        sample_steps=0,
        sample_every=2,
        sample_on_start=sample_on_start,
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


def test_resume_phase_skips_startup_baseline_by_default(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from training.phases import resume

    calls = []
    monkeypatch.setattr(resume, "run_sample", lambda *args, **kwargs: calls.append(kwargs))

    ctx = _ctx(tmp_path, sample_on_start=False)
    resume.run(ctx)

    assert calls == []


def test_resume_phase_runs_startup_baseline_when_enabled(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from training.phases import resume

    calls = []
    monkeypatch.setattr(resume, "run_sample", lambda *args, **kwargs: calls.append(kwargs))

    ctx = _ctx(tmp_path, sample_on_start=True)
    resume.run(ctx)

    assert [call["prompt"] for call in calls] == ["p1", "p2"]
    assert [call["sample_path"].name for call in calls] == [
        "step_0_baseline_0.png",
        "step_0_baseline_1.png",
    ]
