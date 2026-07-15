"""dataset → text_cache → optimizer 生命周期接线。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from training.context import TrainingContext
from training.phases import text_cache


class _Dataset:
    def __init__(self, samples, captions):
        self.samples = samples
        self._captions = captions

    def caption_for_sample(self, sample):
        return self._captions[str(sample["image"])]


class _Family:
    def __init__(self, strategy):
        self.spec = SimpleNamespace(text=SimpleNamespace(strategy=strategy))
        self.call = None

    def prepare_text_cache(self, captions, extra_prompts, **kwargs):
        self.call = (list(captions), list(extra_prompts), kwargs)


def _args(data_dir):
    return SimpleNamespace(
        data_dir=str(data_dir),
        sample_prompts=["trigger, portrait"],
        sample_prompt="",
        sample_negative_prompt="blurry",
        sample_steps=20,
        sample_every=0,
    )


def test_cached_varlen_phase_collects_main_reg_and_deduplicates_fanout(tmp_path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "reg" / "b.png"
    main = _Dataset(
        [
            {"image": image_a, "target_reso": 1024},
            {"image": image_a, "target_reso": 768},
        ],
        {str(image_a): "caption a"},
    )
    reg = _Dataset(
        [{"image": image_b, "target_reso": 1024}],
        {str(image_b): "class caption"},
    )
    family = _Family("cached_varlen")
    ctx = TrainingContext(args=_args(tmp_path), family=family)
    ctx.base_dataset = main
    ctx.reg_dataset = reg
    ctx.text_stack = object()
    ctx.device = "cuda"

    text_cache.run(ctx)

    captions, extras, kwargs = family.call
    assert captions == ["caption a", "class caption"]
    assert extras == ["trigger, portrait", "blurry"]
    assert [e.image_path for e in kwargs["cache_entries"]] == [image_a, image_b]
    assert kwargs["cache_entries"][0].cache_path.parent == image_a.parent
    assert kwargs["cache_root"] == tmp_path
    assert kwargs["text"] is ctx.text_stack
    assert kwargs["device"] == "cuda"


def test_online_strategy_is_noop_and_does_not_scan_dataset(tmp_path):
    class _ExplodingDataset:
        @property
        def samples(self):  # pragma: no cover - access is the failure
            raise AssertionError("online family 不应扫描 caption")

    family = _Family("online")
    ctx = TrainingContext(args=_args(tmp_path), family=family)
    ctx.base_dataset = _ExplodingDataset()

    text_cache.run(ctx)

    assert family.call[0] == []
    assert family.call[1] == []
    assert family.call[2] == {}


def test_sampling_default_and_empty_negative_are_cached(tmp_path):
    args = _args(tmp_path)
    args.sample_prompts = []
    args.sample_negative_prompt = ""
    family = _Family("cached_varlen")
    ctx = TrainingContext(args=args, family=family)
    ctx.base_dataset = _Dataset([], {})

    text_cache.run(ctx)

    assert family.call[1] == ["1girl, masterpiece", ""]
