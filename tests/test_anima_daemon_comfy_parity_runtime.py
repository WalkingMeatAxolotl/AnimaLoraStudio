from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image


_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "runtime"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)


def test_daemon_model_cache_preserves_user_backend_and_sets_comfy_style_dtype(monkeypatch) -> None:
    mod = importlib.import_module("anima_daemon")
    cache = mod.ModelCache()

    monkeypatch.setattr(mod._T, "find_diffusion_pipe_root", lambda: _REPO / "modeling")
    monkeypatch.setattr(
        mod._T,
        "resolve_path_best_effort",
        lambda value, bases: str(value),
    )

    loads: list[dict] = []

    def fake_load(self, **kwargs):
        loads.append(kwargs)
        self.model = object()
        self.transformer_path = kwargs["transformer_path"]
        self.vae_path = kwargs["vae_path"]
        self.text_encoder_path = kwargs["text_encoder_path"]
        self.t5_tokenizer_path = kwargs["t5_tokenizer_path"]
        self.attention_backend = kwargs["backend"]
        self.mixed_precision = kwargs["precision"]

    monkeypatch.setattr(mod.ModelCache, "_load", fake_load)
    monkeypatch.setattr(mod, "_emit_evt", lambda *args, **kwargs: None)

    cache.ensure_loaded({
        "transformer_path": "transformer.safetensors",
        "vae_path": "vae.safetensors",
        "text_encoder_path": "text_encoder",
        "attention_backend": "flash_attn",
        "mixed_precision": "fp32",
    })

    assert loads[0]["backend"] == "flash_attn"
    assert loads[0]["precision"] == "bf16"
    assert loads[0]["vae_precision"] == "bf16"
    assert loads[0]["text_encoder_backend"] == "comfy_qwen3"
    assert loads[0]["t5_tokenizer_backend"] == "fast"


def test_exact_ksampler_parity_backend_semantics() -> None:
    from studio.domain.comfy_parity import is_exact_ksampler_parity_backend

    assert is_exact_ksampler_parity_backend("xformers") is True
    assert is_exact_ksampler_parity_backend("flash_attn") is False
    assert is_exact_ksampler_parity_backend("none") is False


def test_daemon_exact_ksampler_parity_fails_when_xformers_unavailable(monkeypatch) -> None:
    mod = importlib.import_module("anima_daemon")
    cache = mod.ModelCache()
    reached: list[str] = []

    monkeypatch.setattr(mod._T, "find_diffusion_pipe_root", lambda: _REPO / "modeling")
    # daemon 加载已经 family 派发（多模型 PR-2b D8'）：mock 面升级为 family 缝
    class _FakeFamily:
        def load_dit(self, *args, **kwargs):
            return object()

        def load_vae(self, *args, **kwargs):
            reached.append("vae")

        def load_text(self, *args, **kwargs):
            reached.append("text")
            return (object(), object(), object())

    monkeypatch.setattr(mod._T, "get_family", lambda _fid: _FakeFamily())
    monkeypatch.setattr(mod._T, "enable_xformers", lambda _model: False)

    with pytest.raises(RuntimeError, match="xformers"):
        cache._load(
            family_id="anima",
            transformer_path="transformer.safetensors",
            vae_path="vae.safetensors",
            text_encoder_path="text_encoder",
            t5_tokenizer_path="",
            backend="xformers",
            precision="bf16",
            vae_precision="fp32",
            text_encoder_backend="comfy_qwen3",
            t5_tokenizer_backend="fast",
        )

    assert reached == []


def test_daemon_worker_reports_error_when_all_images_fail(monkeypatch, tmp_path) -> None:
    mod = importlib.import_module("anima_daemon")

    class _BoomFamily:
        def sample_image(self, *args, **kwargs):
            raise RuntimeError("boom")

    class FakeCache:
        model = object()
        vae = object()
        text_stack = (object(), object(), object())
        family = _BoomFamily()
        device = "cpu"
        dtype = None

        def ensure_loaded(self, _cfg):
            return None

        def apply_loras(self, _lora_configs):
            return []

    events: list[dict] = []

    monkeypatch.setattr(mod, "CACHE", FakeCache())
    monkeypatch.setattr(mod, "_emit_for", lambda req_id, kind, **extra: events.append({"id": req_id, "kind": kind, **extra}))

    mod._run_generate_worker(
        "req-1",
        123,
        {
            "prompts": ["a"],
            "count": 1,
            "steps": 1,
            "width": 64,
            "height": 64,
            "seed": 1,
        },
        tmp_path,
        threading.Event(),
    )

    kinds = [event["kind"] for event in events]
    assert "image_error" in kinds
    assert "error" in kinds
    assert "done" not in kinds


def test_daemon_restores_runtime_to_device_after_successful_generate(monkeypatch, tmp_path) -> None:
    mod = importlib.import_module("anima_daemon")

    events: list[str] = []

    class _RecordingFamily:
        def sample_image(self, *_args, **_kwargs):
            events.append("sample_image")
            return Image.new("RGB", (1, 1))

    class FakeCache:
        model = object()
        vae = object()
        text_stack = (object(), object(), object())
        family = _RecordingFamily()
        device = "cuda"
        dtype = None

        def ensure_loaded(self, _cfg):
            events.append("ensure_loaded")

        def apply_loras(self, _lora_configs):
            events.append("apply_loras")
            return []

        def _move_runtime_to_device(self):
            events.append("restore_runtime")

    monkeypatch.setattr(mod, "CACHE", FakeCache())
    monkeypatch.setattr(
        mod,
        "_emit_for",
        lambda _req_id, kind, **_extra: events.append(f"emit:{kind}"),
    )

    mod._run_generate(
        "req-1",
        123,
        {
            "prompts": ["a"],
            "count": 1,
            "steps": 1,
            "width": 64,
            "height": 64,
            "seed": 1,
        },
        tmp_path,
        threading.Event(),
    )

    assert events.index("sample_image") < events.index("restore_runtime")
    assert events.index("restore_runtime") < events.index("emit:image_done")
