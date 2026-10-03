"""ADR 0021: caption presence does not select training images."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from runtime.training.dataset import BucketManager, CachedLatentDataset, ImageDataset
from runtime.training.phases.text_cache import _collect_entries
from studio.services.projects.versions import compute_bucket_histogram, compute_navit_pack_estimate


@pytest.mark.parametrize("prefer_json", [True, False])
def test_missing_and_empty_captions_train_with_and_without_latent_cache(
    tmp_path: Path, prefer_json: bool,
) -> None:
    folder = tmp_path / "2_data" / "nested"
    folder.mkdir(parents=True)
    for name in ("missing", "empty", "whitespace", "tagged"):
        image = folder / f"{name}.png"
        Image.new("RGB", (64, 64)).save(image)
        np.savez(image.with_suffix(".npz"), latent=np.zeros((16, 1, 4, 4)))
    (folder / "empty.txt").write_text("", encoding="utf-8")
    (folder / "whitespace.txt").write_text(" \n\t ", encoding="utf-8")
    (folder / "tagged.txt").write_text("known, tag", encoding="utf-8")

    dataset = ImageDataset(tmp_path, 256, BucketManager(256), prefer_json=prefer_json)
    assert len(dataset) == 8  # repeat=2 includes all four images
    assert len(dataset.bucket_for_index) == 8
    expected = {"missing": "", "empty": "", "whitespace": "", "tagged": "known, tag"}
    for index, sample in enumerate(dataset.samples):
        assert dataset[index]["caption"] == expected[sample["image"].stem]

    cached = object.__new__(CachedLatentDataset)
    cached.base_dataset = dataset
    cached.samples = dataset.samples
    cached.np = np
    cached.flip_augment = False
    cached.load_masks = False
    cached._multi_reso = set()
    for index, sample in enumerate(cached.samples):
        assert cached[index]["caption"] == expected[sample["image"].stem]
    # Both caption-less and empty-caption images are included in text-cache entries,
    # deduplicated by image rather than repeat-expanded training sample.
    entries = _collect_entries(SimpleNamespace(base_dataset=cached, reg_dataset=None))
    assert {entry.image_path.stem: entry.caption for entry in entries} == expected
    # Reading never creates missing sidecars just to represent an empty caption.
    assert not (folder / "missing.txt").exists()
    assert not (folder / "missing.json").exists()


@pytest.mark.parametrize("caption_override", ["class prompt", ""])
def test_caption_override_includes_uncaptioned_images(tmp_path: Path, caption_override: str) -> None:
    Image.new("RGB", (32, 32)).save(tmp_path / "missing.png")
    dataset = ImageDataset(tmp_path, 256, caption_override=caption_override)
    assert len(dataset) == 1
    assert dataset[0]["caption"] == caption_override


def test_empty_json_is_valid_but_broken_preferred_json_is_not(tmp_path: Path) -> None:
    image = tmp_path / "sample.png"
    Image.new("RGB", (32, 32)).save(image)
    image.with_suffix(".txt").write_text("stale fallback", encoding="utf-8")
    sidecar = image.with_suffix(".json")
    sidecar.write_text(json.dumps({"tags": []}), encoding="utf-8")
    dataset = ImageDataset(tmp_path, 256)
    assert len(dataset) == 1
    assert dataset[0]["caption"] == ""
    sidecar.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON caption"):
        ImageDataset(tmp_path, 256)


def test_source_empty_summary_does_not_consume_augmentation_rng(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    folder = tmp_path / "2_data"
    folder.mkdir()
    for name in ("missing", "empty", "tagged"):
        Image.new("RGB", (32, 32)).save(folder / f"{name}.png")
    (folder / "empty.json").write_text('{"tags": []}', encoding="utf-8")
    (folder / "tagged.txt").write_text("a, b", encoding="utf-8")
    before = random.getstate()
    with caplog.at_level(logging.INFO, logger="runtime.training.dataset"):
        dataset = ImageDataset(tmp_path, 256, shuffle_caption=True, tag_dropout=1.0)
    assert random.getstate() == before
    assert len(dataset) == 6
    summary = next(r.message for r in caplog.records if "空文本训练" in r.message or "Empty-text training" in r.message)
    assert "2" in summary and "4" in summary  # two images, four repeat-expanded samples


@pytest.mark.parametrize("native", [False, True])
def test_previews_include_all_images_and_match_runtime_scan(tmp_path: Path, native: bool) -> None:
    for folder, name in ((tmp_path, "root"), (tmp_path / "3_data" / "nested", "nested")):
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64)).save(folder / f"{name}.png")
    resolutions = [256, 512]
    dataset = ImageDataset(
        tmp_path, 256, BucketManager(256), resolutions=resolutions,
        native_resolution=native, native_token_budget=1024,
    )
    estimate = compute_navit_pack_estimate(
        [tmp_path], resolutions, native_resolution=native, token_budget=1024,
    )
    assert estimate["samples"] == len(dataset)
    if not native:
        histogram = compute_bucket_histogram(tmp_path, resolutions)
        assert sum(b["count"] for group in histogram for b in group["buckets"]) == len(dataset)
