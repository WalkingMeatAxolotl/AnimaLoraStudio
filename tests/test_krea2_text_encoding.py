"""Krea2 Qwen3-VL varlen encoding, cache lifecycle, and online fallback."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from training.families.krea2.text_encoding import (
    Krea2TextStack,
    gather_valid_text,
    pad_text_conditions,
)
from training.text_cache import TextCacheEntry


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        texts = [text] if isinstance(text, str) else list(text)
        if kwargs.get("return_tensors") != "pt":
            if "assistant" in texts[0]:
                return {"input_ids": list(range(5))}
            return {"input_ids": list(range(34))}

        max_length = kwargs.get("max_length")
        if max_length is None:  # fixed five-token suffix
            return {
                "input_ids": torch.arange(5).repeat(len(texts), 1),
                "attention_mask": torch.ones(len(texts), 5, dtype=torch.long),
            }

        ids = torch.zeros(len(texts), max_length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, value in enumerate(texts):
            caption = value.rsplit("user\n", 1)[-1]
            valid = min(max_length, 34 + len(caption))
            ids[index] = torch.arange(max_length)
            mask[index, :valid] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _FakeBackbone:
    def __init__(self, width=4, layers=4):
        self.width = width
        self.layers = layers
        self.calls = 0

    def __call__(self, *, input_ids, attention_mask, **kwargs):
        self.calls += 1
        batch, seq = input_ids.shape
        positions = torch.arange(seq, device=input_ids.device, dtype=torch.float32)
        positions = positions.view(1, seq, 1).expand(batch, seq, self.width)
        hidden_states = tuple(positions + layer * 100 for layer in range(self.layers))
        return SimpleNamespace(hidden_states=hidden_states)


class _FakeModel:
    def __init__(self):
        self.model = _FakeBackbone()


class _Loader:
    def __init__(self):
        self.calls = 0
        self.models = []

    def __call__(self, model_path, device, dtype):
        self.calls += 1
        model = _FakeModel()
        self.models.append(model)
        return model


def _stack(tmp_path, loader, *, cache_enabled):
    return Krea2TextStack(
        tmp_path / "qwen",
        device="cpu",
        dtype=torch.float32,
        cache_enabled=cache_enabled,
        tokenizer=_FakeTokenizer(),
        model_loader=loader,
        max_length=8,
        selected_layers=(1, 3),
        hidden_width=4,
        text_fingerprint="krea2-test-v1",
        cache_batch_size=2,
    )


def test_gather_valid_text_preserves_suffix_after_interior_padding():
    hidden = torch.arange(7, dtype=torch.float32).view(1, 7, 1)
    mask = torch.tensor([[True, True, False, False, False, True, True]])

    gathered = gather_valid_text(hidden, mask)

    assert gathered[0].flatten().tolist() == [0.0, 1.0, 5.0, 6.0]


def test_pad_text_conditions_builds_bool_mask_and_requested_dtype():
    condition = pad_text_conditions(
        [torch.ones(2, 2, 4), torch.full((4, 2, 4), 2.0)],
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert condition.context.shape == (2, 4, 2, 4)
    assert condition.context.dtype == torch.bfloat16
    assert condition.attention_mask.dtype == torch.bool
    assert condition.attention_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert torch.count_nonzero(condition.context[0, 2:]) == 0


def test_online_mode_never_writes_cache_and_keeps_model_loaded(tmp_path):
    loader = _Loader()
    stack = _stack(tmp_path, loader, cache_enabled=False)
    entry = TextCacheEntry.for_image(tmp_path / "image.png", "ab")

    stack.prepare_text_cache(
        ["ab"], [""], cache_entries=[entry], cache_root=tmp_path,
    )
    condition = stack.encode_text_for_batch(
        ["ab", "c"], device="cpu", dtype=torch.float32,
    )

    assert loader.calls == 1
    assert stack.is_model_loaded
    assert not entry.cache_path.exists()
    assert not (tmp_path / ".text-cache").exists()
    assert condition.context.shape == (2, 7, 2, 4)
    assert condition.attention_mask.sum(dim=1).tolist() == [7, 6]
    # For "ab": positions 34/35, interior pad 36 skipped, suffix 37..41 retained.
    assert condition.context[0, :, 0, 0].tolist() == [
        134.0, 135.0, 137.0, 138.0, 139.0, 140.0, 141.0,
    ]


def test_cached_mode_precaches_reuses_and_releases_model(tmp_path):
    first_loader = _Loader()
    first = _stack(tmp_path, first_loader, cache_enabled=True)
    entries = [
        TextCacheEntry.for_image(tmp_path / "one.png", "ab"),
        TextCacheEntry.for_image(tmp_path / "two.png", "ab"),
        TextCacheEntry.for_image(tmp_path / "three.png", "c"),
    ]

    first.prepare_text_cache(
        ["ab", "ab", "c"], ["sample", ""],
        cache_entries=entries,
        cache_root=tmp_path,
    )

    assert first_loader.calls == 1
    assert not first.is_model_loaded
    assert all(entry.cache_path.is_file() for entry in entries)
    assert first_loader.models[0].model.calls == 2

    def fail_loader(*args):
        raise AssertionError("完整缓存命中不应加载 Qwen3-VL")

    second = _stack(tmp_path, fail_loader, cache_enabled=True)
    second.prepare_text_cache(
        ["ab", "ab", "c"], ["sample", ""],
        cache_entries=entries,
        cache_root=tmp_path,
    )
    condition = second.encode_text_for_batch(
        ["c", "ab", "sample"], device="cpu", dtype=torch.float32,
    )

    assert not second.is_model_loaded
    assert condition.attention_mask.sum(dim=1).tolist() == [6, 7, 8]


def test_invalid_sidecar_is_reencoded_and_released(tmp_path):
    loader = _Loader()
    stack = _stack(tmp_path, loader, cache_enabled=True)
    entry = TextCacheEntry.for_image(tmp_path / "image.png", "ab")
    stack.store.write_caption(entry, {"context": torch.ones(2, 99, 4)})

    stack.prepare_text_cache(
        ["ab"], [], cache_entries=[entry], cache_root=tmp_path,
    )
    cached = stack.store.read_caption(entry)

    assert loader.calls == 1
    assert not stack.is_model_loaded
    assert cached["context"].shape == (7, 2, 4)


def test_cached_batch_miss_repairs_sidecar_after_prepare(tmp_path):
    loader = _Loader()
    stack = _stack(tmp_path, loader, cache_enabled=True)
    entry = TextCacheEntry.for_image(tmp_path / "image.png", "ab")
    stack.prepare_text_cache(
        ["ab"], [], cache_entries=[entry], cache_root=tmp_path,
    )
    entry.cache_path.unlink()

    condition = stack.encode_text_for_batch(
        ["ab"], device="cpu", dtype=torch.float32,
    )

    assert loader.calls == 2
    assert entry.cache_path.is_file()
    assert not stack.is_model_loaded
    assert condition.attention_mask.sum().item() == 7


def test_offload_model_round_trip():
    """offload_model 把 TE 挪 CPU（DiT 独占显存）；ensure_model 搬回（Comfy
    free_memory 语义）。未加载 / 重复调用安全 no-op。"""
    import torch

    from training.families.krea2.text_encoding import Krea2TextStack

    class _FakeModel:
        def __init__(self):
            self.device_history = []

        def to(self, device):
            self.device_history.append(str(device))
            return self

    fake = _FakeModel()
    stack = Krea2TextStack(
        "unused-dir", device="cpu", cache_enabled=False,
        tokenizer=object(), model_loader=lambda *a: fake,
    )
    stack.offload_model()          # 未加载 → no-op
    assert fake.device_history == []
    assert stack.ensure_model() is fake
    stack.offload_model()
    assert fake.device_history == ["cpu"]
    stack.offload_model()          # 重复 → no-op
    assert fake.device_history == ["cpu"]
    assert stack.ensure_model() is fake   # 搬回 self.device
    assert fake.device_history == ["cpu", "cpu"]
