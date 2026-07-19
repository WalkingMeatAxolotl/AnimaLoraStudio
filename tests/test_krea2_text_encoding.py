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
        # 后缀调用：无 padding 无 max_length（在线主调用带 padding="longest"）
        if max_length is None and kwargs.get("padding") is None:
            return {
                "input_ids": torch.arange(5).repeat(len(texts), 1),
                "attention_mask": torch.ones(len(texts), 5, dtype=torch.long),
            }

        def _valid(value: str) -> int:
            caption = value.rsplit("user\n", 1)[-1]
            n = 34 + len(caption)
            return min(max_length, n) if max_length else n

        width = max_length or max(_valid(v) for v in texts)
        ids = torch.zeros(len(texts), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, value in enumerate(texts):
            ids[index] = torch.arange(width)
            mask[index, :_valid(value)] = 1
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
    # 在线模式不截断 + longest pad：最长条 "ab" 无 interior pad，token 连续
    # （positions 34/35 + suffix 36..40）；旧定长口径的 pad 位移不复存在。
    assert condition.context[0, :, 0, 0].tolist() == [
        134.0, 135.0, 136.0, 137.0, 138.0, 139.0, 140.0,
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


def test_manual_cast_patch_fp16_storage_fp32_compute():
    """manual_cast 等价 patch（ComfyUI sd.py:258 口径）：权重 fp16 常驻，
    Embedding 输出与 Linear 计算都进 fp32 域；fp16→fp32 cast 精确，数值与
    整模 upcast 逐位一致。"""

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(8, 4)
            self.proj = torch.nn.Linear(4, 4)

    model = _Tiny().to(torch.float16)
    stack = Krea2TextStack(
        "unused-dir", device="cpu", dtype=torch.float16,
        compute_dtype=torch.float32, cache_enabled=False,
        tokenizer=object(), model_loader=lambda *a: model,
    )

    patched = stack.ensure_model()

    embedded = patched.embed(torch.tensor([1, 3]))
    out = patched.proj(embedded)
    assert embedded.dtype == torch.float32
    assert out.dtype == torch.float32
    assert patched.embed.weight.dtype == torch.float16    # 存储不动
    assert patched.proj.weight.dtype == torch.float16
    expected = torch.nn.functional.linear(
        embedded, model.proj.weight.float(), model.proj.bias.float(),
    )
    assert torch.equal(out, expected)


def test_compute_dtype_none_leaves_model_unpatched():
    """compute_dtype 缺省（训练路径现状）不 patch 任何模块。"""
    model = torch.nn.Linear(2, 2)
    stack = Krea2TextStack(
        "unused-dir", device="cpu", dtype=torch.float32, cache_enabled=False,
        tokenizer=object(), model_loader=lambda *a: model,
    )
    stack.ensure_model()
    assert "forward" not in vars(model)


def test_online_lru_skips_backbone_on_repeat_and_matches_first_encode(tmp_path):
    """在线模式 prompt LRU：同 caption 二次编码零 backbone 调用，condition
    与首次逐位一致（Comfy conditioning 节点缓存同款语义）。"""
    loader = _Loader()
    stack = _stack(tmp_path, loader, cache_enabled=False)

    first = stack.encode_text_for_batch(["ab"], device="cpu", dtype=torch.float32)
    calls_after_first = loader.models[0].model.calls
    assert stack.online_conditions_cached(["ab"])
    assert not stack.online_conditions_cached(["ab", "new"])

    second = stack.encode_text_for_batch(["ab"], device="cpu", dtype=torch.float32)
    assert loader.models[0].model.calls == calls_after_first
    assert torch.equal(first.context, second.context)
    assert torch.equal(first.attention_mask, second.attention_mask)


def test_online_lru_evicts_beyond_capacity(tmp_path):
    loader = _Loader()
    stack = _stack(tmp_path, loader, cache_enabled=False)
    stack._online_lru_capacity = 2

    stack.encode_text_for_batch(["a"], device="cpu", dtype=torch.float32)
    stack.encode_text_for_batch(["b"], device="cpu", dtype=torch.float32)
    stack.encode_text_for_batch(["c"], device="cpu", dtype=torch.float32)

    assert not stack.online_conditions_cached(["a"])   # 最旧被逐出
    assert stack.online_conditions_cached(["b", "c"])


def test_online_lru_dedupes_batch_and_not_used_in_cached_mode(tmp_path):
    loader = _Loader()
    stack = _stack(tmp_path, loader, cache_enabled=False)

    condition = stack.encode_text_for_batch(
        ["ab", "ab", "c"], device="cpu", dtype=torch.float32,
    )
    assert condition.context.shape[0] == 3      # 重复 caption 仍按位置返回

    cached_stack = _stack(tmp_path, _Loader(), cache_enabled=True)
    assert not cached_stack.online_conditions_cached(["ab"])  # cached 模式恒 False


def test_precache_online_prompts_fills_lru_and_skips_hits(tmp_path):
    """任务级预编码：miss 编码进 LRU（去重），已命中不再触发 backbone。"""
    loader = _Loader()
    stack = _stack(tmp_path, loader, cache_enabled=False)

    encoded = stack.precache_online_prompts(["a", "b", "a"])
    assert encoded == 2
    assert stack.online_conditions_cached(["a", "b"])
    calls = loader.models[0].model.calls

    assert stack.precache_online_prompts(["a", "b"]) == 0
    assert loader.models[0].model.calls == calls  # 全命中零编码

    # 预编码后 encode_text_for_batch 命中 LRU，backbone 不再前向
    stack.encode_text_for_batch(["a", "b"], device="cpu", dtype=torch.float32)
    assert loader.models[0].model.calls == calls


def test_precache_online_prompts_noop_in_cached_mode(tmp_path):
    loader = _Loader()
    stack = _stack(tmp_path, loader, cache_enabled=True)
    assert stack.precache_online_prompts(["a", "b"]) == 0
    assert loader.calls == 0  # cached 模式不加载模型


def test_precache_online_prompts_raises_capacity_for_large_tasks(tmp_path):
    """多 prompt 任务（>16 条默认容量）：容量抬到本批需求，预编码不自逐出。"""
    loader = _Loader()
    stack = _stack(tmp_path, loader, cache_enabled=False)
    prompts = [f"p{i}" for i in range(20)]

    encoded = stack.precache_online_prompts(prompts)

    assert encoded == 20
    assert stack._online_lru_capacity >= 20
    assert stack.online_conditions_cached(prompts)
