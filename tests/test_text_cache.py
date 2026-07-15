"""Phase 2 文本缓存协议：key、sidecar、varlen safetensors 与失效重算。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from training.text_cache import (
    TEXT_CACHE_FORMAT_VERSION,
    TextCacheEntry,
    TextCacheStore,
    caption_sidecar_path,
    prompt_cache_path,
    text_cache_key,
)


def test_key_covers_caption_fingerprint_and_format_version():
    base = text_cache_key("a cat", "qwen-v1")
    assert base != text_cache_key("a dog", "qwen-v1")
    assert base != text_cache_key("a cat", "qwen-v2")
    assert base != text_cache_key(
        "a cat", "qwen-v1", format_version=TEXT_CACHE_FORMAT_VERSION + 1,
    )


def test_sidecar_keeps_full_image_name_to_avoid_extension_collision(tmp_path):
    jpg = tmp_path / "same.jpg"
    png = tmp_path / "same.png"
    assert caption_sidecar_path(jpg).name == "same.jpg.text.safetensors"
    assert caption_sidecar_path(png).name == "same.png.text.safetensors"
    assert caption_sidecar_path(jpg) != caption_sidecar_path(png)


def test_caption_roundtrip_preserves_varlen_bfloat16_and_is_atomic(tmp_path):
    entry = TextCacheEntry.for_image(tmp_path / "image.webp", "final caption")
    store = TextCacheStore("qwen3-vl-test")
    encoded = {
        "hidden_states": torch.randn(7, 12, 8, dtype=torch.bfloat16),
        "attention_mask": torch.ones(7, dtype=torch.int64),
    }

    store.write_caption(entry, encoded)
    loaded = store.read_caption(entry)

    assert loaded is not None
    assert loaded["hidden_states"].shape == (7, 12, 8)  # varlen，未 pad 512
    assert loaded["hidden_states"].dtype == torch.bfloat16
    assert torch.equal(loaded["attention_mask"], encoded["attention_mask"])
    assert not list(tmp_path.glob(entry.cache_path.name + ".*.tmp"))


def test_caption_mismatch_or_corruption_is_cache_miss(tmp_path):
    entry = TextCacheEntry.for_image(tmp_path / "image.png", "old caption")
    store = TextCacheStore("te-v1")
    store.write_caption(entry, {"hidden": torch.ones(3, 2)})

    changed = TextCacheEntry.for_image(entry.image_path, "new caption")
    assert store.read_caption(changed) is None
    assert TextCacheStore("te-v2").read_caption(entry) is None

    entry.cache_path.write_bytes(b"not safetensors")
    assert store.read_caption(entry) is None


def test_get_or_encode_reuses_hit_and_reencodes_stale_caption(tmp_path):
    entry = TextCacheEntry.for_image(tmp_path / "image.png", "caption v1")
    store = TextCacheStore("te-v1")
    calls = []

    def encode(caption):
        calls.append(caption)
        return {"hidden": torch.full((len(caption), 2), float(len(calls)))}

    first, first_hit = store.get_or_encode_caption(entry, encode)
    second, second_hit = store.get_or_encode_caption(entry, encode)
    changed = TextCacheEntry.for_image(entry.image_path, "caption v2")
    third, third_hit = store.get_or_encode_caption(changed, encode)

    assert first_hit is False
    assert second_hit is True
    assert third_hit is False
    assert calls == ["caption v1", "caption v2"]
    assert torch.equal(first["hidden"], second["hidden"])
    assert not torch.equal(second["hidden"], third["hidden"])


def test_prompt_bundle_supports_multiple_variable_lengths(tmp_path):
    store = TextCacheStore("te-v1")
    path = store.write_prompt_bundle(tmp_path, {
        "short": {"hidden": torch.randn(2, 4)},
        "a much longer prompt": {"hidden": torch.randn(11, 4)},
        "": {"hidden": torch.randn(1, 4)},
    })

    assert path == prompt_cache_path(tmp_path, "te-v1")
    assert path.parent == tmp_path / ".text-cache"
    assert path.name.startswith("prompts.")
    assert store.read_prompt(tmp_path, "short")["hidden"].shape == (2, 4)
    assert store.read_prompt(tmp_path, "a much longer prompt")["hidden"].shape == (11, 4)
    assert store.read_prompt(tmp_path, "")["hidden"].shape == (1, 4)
    assert store.read_prompt(tmp_path, "missing") is None


def test_get_or_encode_prompts_only_encodes_bundle_misses(tmp_path):
    store = TextCacheStore("te-v1")
    calls = []

    def encode(caption):
        calls.append(caption)
        return {"hidden": torch.ones(max(1, len(caption)), 2)}

    first, first_hits = store.get_or_encode_prompts(
        tmp_path, ["positive", "", "positive"], encode,
    )
    second, second_hits = store.get_or_encode_prompts(
        tmp_path, ["positive", ""], encode,
    )

    assert list(first) == ["positive", ""]
    assert first_hits == 0
    assert second_hits == 2
    assert calls == ["positive", ""]
    assert torch.equal(first["positive"]["hidden"], second["positive"]["hidden"])


def test_tensor_names_reject_bundle_separator(tmp_path):
    entry = TextCacheEntry.for_image(tmp_path / "image.png", "caption")
    store = TextCacheStore("te-v1")
    with pytest.raises(ValueError, match="tensor"):
        store.write_caption(entry, {"bad::name": torch.ones(1)})
