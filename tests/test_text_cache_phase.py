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


def _ctx(args, family, task_archive_dir=None, output_dir=None):
    ctx = TrainingContext(args=args, family=family)
    ctx.task_archive_dir = task_archive_dir
    ctx.output_dir = output_dir
    return ctx


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
    task_dir = tmp_path / "tasks" / "7"
    ctx = _ctx(_args(tmp_path), family, task_archive_dir=task_dir)
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
    # prompt 聚合缓存根 = task 档案，不是数据集 train/（否则被数据集扫描误触）
    assert kwargs["cache_root"] == task_dir
    assert kwargs["text"] is ctx.text_stack
    assert kwargs["device"] == "cuda"


def test_prompt_cache_root_falls_back_to_output_dir_without_task_archive(tmp_path):
    family = _Family("cached_varlen")
    out_dir = tmp_path / "output"
    ctx = _ctx(_args(tmp_path), family, output_dir=out_dir)
    ctx.base_dataset = _Dataset([], {})

    text_cache.run(ctx)

    assert family.call[2]["cache_root"] == out_dir


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
    ctx = _ctx(args, family, output_dir=tmp_path / "output")
    ctx.base_dataset = _Dataset([], {})

    text_cache.run(ctx)

    assert family.call[1] == ["1girl, masterpiece", ""]


def test_cached_strategy_disabled_keeps_te_online_without_scanning(tmp_path):
    class _ExplodingDataset:
        @property
        def samples(self):  # pragma: no cover - access is the failure
            raise AssertionError("关闭缓存后不应扫描 caption")

    args = _args(tmp_path)
    args.text_encoder_cache = False
    family = _Family("cached_varlen")
    ctx = TrainingContext(args=args, family=family)
    ctx.base_dataset = _ExplodingDataset()
    ctx.text_stack = object()
    ctx.device = "cuda"

    text_cache.run(ctx)

    assert family.call[0] == []
    assert family.call[1] == []
    assert family.call[2]["text"] is ctx.text_stack
