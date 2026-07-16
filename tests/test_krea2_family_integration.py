"""Krea2 family registration and cache-aware model loading lifecycle."""

from __future__ import annotations

from types import SimpleNamespace

from training.context import TrainingContext
from training.families.krea2 import KREA2_SPEC
from training.phases import models


def _ctx(*, cache_enabled: bool) -> TrainingContext:
    args = SimpleNamespace(text_encoder_cache=cache_enabled)
    family = SimpleNamespace(spec=KREA2_SPEC)
    return TrainingContext(args=args, family=family)


def _patch_loaders(monkeypatch, events: list[str]) -> None:
    monkeypatch.setattr(models, "_resolve_paths", lambda ctx: events.append("paths"))

    def load_dit(ctx):
        events.append("dit")
        ctx.model = object()

    def load_vae(ctx):
        events.append("vae")
        ctx.vae = object()

    def load_text(ctx):
        events.append("text")
        ctx.text_stack = object()

    def inject(ctx):
        events.append("inject")
        ctx.injector = object()

    monkeypatch.setattr(models, "_load_dit", load_dit)
    monkeypatch.setattr(models, "_load_vae", load_vae)
    monkeypatch.setattr(models, "_load_text", load_text)
    monkeypatch.setattr(models, "_inject_adapter", inject)


def test_cached_krea2_defers_dit_until_after_text_cache(monkeypatch):
    events: list[str] = []
    _patch_loaders(monkeypatch, events)
    ctx = _ctx(cache_enabled=True)

    models.run(ctx)
    assert events == ["paths", "vae", "text"]
    assert ctx.model is None

    events.append("text-cache-release")
    models.finish(ctx)
    assert events == [
        "paths", "vae", "text", "text-cache-release", "dit", "inject",
    ]


def test_storage_free_krea2_keeps_dit_and_te_resident(monkeypatch):
    events: list[str] = []
    _patch_loaders(monkeypatch, events)
    ctx = _ctx(cache_enabled=False)

    models.run(ctx)
    models.finish(ctx)

    assert events == ["paths", "dit", "vae", "text", "inject"]
