"""Production observation seam: same pipeline, no default instrumentation."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.test_loop_nonfinite_loss import _batch, _make_ctx, loop_mod


class Recorder:
    def __init__(self):
        self.events = []

    def loop_started(self, ctx):
        self.events.append("start")

    def loop_finished(self, ctx):
        self.events.append("finish")

    def forward_started(self, ctx, *, batch_size):
        self.events.append(("forward", batch_size))

    def backward_finished(self, ctx, *, loss_is_finite):
        self.events.append(("backward", loss_is_finite))

    def optimizer_step_finished(self, ctx, *, reason):
        self.events.append(("update", reason, ctx.global_step))


def test_observer_none_does_not_change_rng_updates_or_new_timers(tmp_path, monkeypatch):
    counts = []
    original_clock = loop_mod.time.perf_counter
    for observed in (False, True):
        torch.manual_seed(567)
        ctx = _make_ctx(tmp_path, [_batch() for _ in range(5)], monkeypatch)
        calls = []
        monkeypatch.setattr(loop_mod.time, "perf_counter", lambda: calls.append(1) or 1.0)
        observer = Recorder() if observed else None
        loop_mod.run(ctx, observer=observer)
        counts.append((ctx.model.w.detach().clone(), torch.get_rng_state(), len(calls)))
    monkeypatch.setattr(loop_mod.time, "perf_counter", original_clock)
    assert torch.equal(counts[0][0], counts[1][0])
    assert torch.equal(counts[0][1], counts[1][1])
    assert counts[0][2] == counts[1][2]
    assert observer.events[0] == "start" and observer.events[-1] == "finish"
    assert [e for e in observer.events if e[0] == "update"] == [
        ("update", "updated", 1), ("update", "updated", 2), ("update", "updated", 3)]


@pytest.mark.parametrize("nan_flags, expected", [
    ([False, True], ["updated"]), ([True, False], ["updated"]),
    ([True, True], ["no_gradients"]), ([False, False, False], ["updated", "updated"]),
])
def test_accumulation_tail_and_nan_boundaries(tmp_path, monkeypatch, nan_flags, expected):
    ctx = _make_ctx(tmp_path, [_batch(nan=v) for v in nan_flags], monkeypatch)
    observer = Recorder()
    loop_mod.run(ctx, observer=observer)
    assert [e[1] for e in observer.events if e[0] == "update"] == expected
    assert sum(e == ("backward", False) for e in observer.events) == sum(nan_flags)


def test_nonfinite_gradient_notification(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path, [_batch(), _batch()], monkeypatch)
    ctx.model.w.register_hook(lambda g: g * float("nan"))
    observer = Recorder()
    loop_mod.run(ctx, observer=observer)
    assert ("update", "nonfinite_gradients", 0) in observer.events
    assert ctx.global_step == 0


def test_max_steps_still_respected(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path, [_batch() for _ in range(8)], monkeypatch, max_steps=1)
    observer = Recorder()
    loop_mod.run(ctx, observer=observer)
    assert ctx.global_step == 1
    assert sum(e[0] == "forward" for e in observer.events) == 2


def test_exception_has_no_fake_loop_completion(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path, [_batch()], monkeypatch)
    observer = Recorder()
    monkeypatch.setattr(ctx.family, "forward_train", lambda *a, **k: (_ for _ in ()).throw(ValueError("fixture")))
    with pytest.raises(ValueError):
        loop_mod.run(ctx, observer=observer)
    assert "finish" not in observer.events


def test_single_pipeline_order_and_main_signature(monkeypatch):
    import anima_train
    from training import phases, loop

    events = []
    names = ["bootstrap", "models", "dataset", "text_cache", "models_finish", "optimizer", "resume", "loop", "finalize"]
    def operation(name):
        def call(ctx, **kwargs):
            events.append((name, kwargs))
        return call
    for name in names:
        if name == "models_finish":
            monkeypatch.setattr(phases.models, "finish", operation(name))
        elif name == "loop":
            monkeypatch.setattr(loop, "run", operation(name))
        else:
            monkeypatch.setattr(getattr(phases, name), "run", operation(name))
    monkeypatch.setattr(anima_train, "parse_args", lambda: SimpleNamespace())
    anima_train.main()
    assert events == [(name, {}) for name in names]
    events.clear()
    observer = SimpleNamespace(
        phase_started=lambda name, ctx: events.append(("start", name)),
        phase_finished=lambda name, ctx: events.append(("finish", name)))
    anima_train.run_training(SimpleNamespace(), observer=observer)
    assert [e for e in events if e[0] == "start"] == [("start", name) for name in names]
    assert ("loop", {"observer": observer}) in events
    assert [e for e in events if e[0] == "finish"] == [("finish", name) for name in names]


def test_every_observer_call_is_guarded_and_runtime_has_no_tools_import():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "runtime/training/loop.py", root / "runtime/anima_train.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("tools")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "observer":
                current = node
                while current in parent and not isinstance(current, ast.If):
                    current = parent[current]
                assert isinstance(current, ast.If)
                assert ast.unparse(current.test) == "observer is not None"
