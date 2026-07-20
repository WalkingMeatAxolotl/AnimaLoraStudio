"""daemon TE 先行编排：_precache_prompts_and_release + ensure_text_ready。

XY / 多 prompt generate 的 prompt 集合任务开始前封闭：krea2 在 DiT 加载
前先就绪 TE 栈 → 全部编进在线 LRU → 彻底释放 TE（release 非 offload，
不留 CPU 副本）——任一时刻 GPU 只有一个大模型。anima 文本栈无此 API
安全跳过；performance 档不释放；预编码异常由逐格惰性路径兜底。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "runtime"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

import anima_daemon  # noqa: E402


class _Stack:
    def __init__(
        self, fail: bool = False, *, cached: bool = False,
        events: list[str] | None = None,
    ):
        self.fail = fail
        self.cached = cached
        self.events = events
        self.precached: list[list[str]] = []
        self.released = 0
        self.is_model_loaded = False
        self.is_model_on_device = False
        self.is_fp8_storage = True

    def precache_online_prompts(self, prompts):
        if self.fail:
            raise RuntimeError("boom")
        if self.events is not None:
            self.events.append("precache")
        self.precached.append(list(prompts))
        return len(prompts)

    def release_model(self):
        if self.events is not None:
            self.events.append("release_te")
        self.released += 1

    def offload_model(self):
        if self.events is not None:
            self.events.append("offload_te")

    def online_conditions_cached(self, _prompts):
        return self.cached


def _with_stack(monkeypatch, stack):
    monkeypatch.setattr(anima_daemon.CACHE, "text_stack", stack, raising=False)
    return stack


def test_precache_encodes_and_releases(monkeypatch):
    stack = _with_stack(monkeypatch, _Stack())
    phases: list[str] = []

    anima_daemon._precache_prompts_and_release(
        ["p", "neg"], "auto", phases.append,
    )

    assert stack.precached == [["p", "neg"]]
    assert stack.released == 1  # 彻底释放，不留 CPU 副本
    assert phases == ["clip"]


def test_precache_performance_policy_keeps_te_resident(monkeypatch):
    stack = _with_stack(monkeypatch, _Stack())

    anima_daemon._precache_prompts_and_release(["p"], "performance")

    assert stack.precached == [["p"]]
    assert stack.released == 0


def test_precache_failure_falls_back_without_release(monkeypatch):
    stack = _with_stack(monkeypatch, _Stack(fail=True))

    anima_daemon._precache_prompts_and_release(["p"], "auto")  # 不抛

    assert stack.released == 0


def test_precache_skips_stacks_without_api(monkeypatch):
    """anima 文本栈是 (qwen, tok, t5_tok) tuple——无 API 直接跳过。"""
    monkeypatch.setattr(
        anima_daemon.CACHE, "text_stack",
        (SimpleNamespace(), SimpleNamespace(), SimpleNamespace()),
        raising=False,
    )
    anima_daemon._precache_prompts_and_release(["p"], "auto")  # 不抛


def test_task_level_te_budget_policies(monkeypatch):
    gib = 1024 ** 3
    monkeypatch.setattr(
        anima_daemon, "_cuda_mem_info", lambda _device: (6 * gib, 24 * gib),
    )
    assert anima_daemon._should_yield_dit_for_te(
        "performance", "cuda", 7 * gib,
    )[0] is False
    assert anima_daemon._should_yield_dit_for_te(
        "save_vram", "cuda", 1,
    )[0] is True
    # auto: 7 GiB TE + 9.6 GiB WDDM residency reserve > 6 GiB free
    should_yield, budget = anima_daemon._should_yield_dit_for_te(
        "auto", "cuda", 7 * gib,
    )
    assert should_yield is True
    assert budget is not None and budget["margin"] == int(24 * gib * 0.40)


def _set_second_task_runtime(
    monkeypatch, *, free_gib: int, calibrated_gib: int, total_gib: int = 24,
):
    events: list[str] = []

    class _Model:
        def to(self, device):
            events.append(f"dit:{device}")
            return self

    stack = _Stack(events=events)
    key = ("krea2", "te")
    monkeypatch.setattr(anima_daemon.CACHE, "text_stack", stack)
    monkeypatch.setattr(anima_daemon.CACHE, "model", _Model())
    monkeypatch.setattr(anima_daemon.CACHE, "adapters", [])
    monkeypatch.setattr(anima_daemon.CACHE, "device", "cuda")
    monkeypatch.setattr(anima_daemon.CACHE, "text_encoder_path", "te")
    monkeypatch.setattr(anima_daemon.CACHE, "_text_ready_key", key)
    monkeypatch.setattr(
        anima_daemon.CACHE, "_te_peak_calibration",
        {key: (calibrated_gib * 1024 ** 3, 512 * 1024 ** 2)},
    )
    monkeypatch.setattr(anima_daemon.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        anima_daemon, "_cuda_mem_info",
        lambda _device: (free_gib * 1024 ** 3, total_gib * 1024 ** 3),
    )
    return events


def test_second_task_save_vram_yields_dit_around_te(monkeypatch):
    events = _set_second_task_runtime(
        monkeypatch, free_gib=20, calibrated_gib=7,
    )
    anima_daemon._precache_prompts_and_release(["new"], "save_vram")
    assert events == ["dit:cpu", "precache", "release_te", "dit:cuda"]


def test_second_task_auto_uses_calibrated_peak(monkeypatch):
    events = _set_second_task_runtime(
        monkeypatch, free_gib=6, calibrated_gib=7,
    )
    anima_daemon._precache_prompts_and_release(["new"], "auto")
    assert events == ["dit:cpu", "precache", "release_te", "dit:cuda"]


def test_second_task_auto_keeps_dit_when_calibrated_peak_fits(monkeypatch):
    events = _set_second_task_runtime(
        monkeypatch, free_gib=30, calibrated_gib=7, total_gib=48,
    )
    anima_daemon._precache_prompts_and_release(["new"], "auto")
    assert events == ["precache", "release_te"]


def test_second_task_lru_hit_skips_dit_yield(monkeypatch):
    events = _set_second_task_runtime(
        monkeypatch, free_gib=1, calibrated_gib=7,
    )
    anima_daemon.CACHE.text_stack.cached = True
    anima_daemon._precache_prompts_and_release(["cached"], "save_vram")
    assert events == ["precache", "release_te"]


def test_dit_restore_waits_for_cuda_copies_before_sampling(monkeypatch):
    events: list[str] = []

    class _Model:
        def to(self, device):
            events.append(f"dit:{device}")
            return self

    cache = anima_daemon.ModelCache()
    cache.model = _Model()
    cache.adapters = []
    monkeypatch.setattr(anima_daemon.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        anima_daemon.torch.cuda, "synchronize",
        lambda device: events.append(f"sync:{device}"),
    )
    monkeypatch.setattr(
        anima_daemon.torch.cuda, "empty_cache",
        lambda: events.append("empty_cache"),
    )

    cache._move_dit_and_adapters("cuda")

    assert events == ["dit:cuda", "sync:cuda", "empty_cache"]


def test_cuda_oom_retries_once_with_same_seed(monkeypatch):
    events: list[str] = []
    attempts = 0

    def sample_once():
        nonlocal attempts
        attempts += 1
        events.append(f"sample:{attempts}")
        if attempts == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 100 MiB")
        return "image"

    monkeypatch.setattr(
        anima_daemon, "_recover_cuda_allocator",
        lambda device: events.append(f"recover:{device}"),
    )
    monkeypatch.setattr(
        anima_daemon.torch, "manual_seed",
        lambda seed: events.append(f"torch_seed:{seed}"),
    )
    monkeypatch.setattr(
        anima_daemon.random, "seed",
        lambda seed: events.append(f"random_seed:{seed}"),
    )

    result = anima_daemon._sample_with_cuda_oom_retry(
        sample_once,
        seed=123,
        device="cuda",
        before_retry=lambda: events.append("release_vae"),
    )

    assert result == "image"
    assert events == [
        "sample:1", "release_vae", "recover:cuda",
        "torch_seed:123", "random_seed:123", "sample:2",
    ]


def test_cuda_oom_retry_is_bounded_and_does_not_mask_other_errors(monkeypatch):
    recoveries: list[str] = []
    monkeypatch.setattr(
        anima_daemon, "_recover_cuda_allocator", recoveries.append,
    )

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        anima_daemon._sample_with_cuda_oom_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("CUDA out of memory")),
            seed=1,
            device="cuda",
        )
    assert recoveries == ["cuda"]

    with pytest.raises(RuntimeError, match="bad sampler"):
        anima_daemon._sample_with_cuda_oom_retry(
            lambda: (_ for _ in ()).throw(RuntimeError("bad sampler")),
            seed=1,
            device="cuda",
        )
    assert recoveries == ["cuda"]


def test_initial_te_peak_calibration_spans_load_and_encode(monkeypatch):
    gib = 1024 ** 3
    key = ("krea2", "te")
    monkeypatch.setattr(anima_daemon.CACHE, "model", None)
    monkeypatch.setattr(anima_daemon.CACHE, "_text_ready_key", None)
    monkeypatch.setattr(anima_daemon.CACHE, "_te_peak_calibration", {})
    monkeypatch.setattr(anima_daemon.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(anima_daemon.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(anima_daemon.torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(
        anima_daemon.torch.cuda, "reset_peak_memory_stats", lambda _device: None,
    )
    monkeypatch.setattr(
        anima_daemon.torch.cuda, "memory_allocated", lambda _device: 2 * gib,
    )
    monkeypatch.setattr(
        anima_daemon.torch.cuda, "max_memory_allocated", lambda _device: 9 * gib,
    )

    start = anima_daemon._begin_initial_te_peak_calibration(
        {"model_family": "krea2"},
    )
    monkeypatch.setattr(anima_daemon.CACHE, "_text_ready_key", key)
    anima_daemon._finish_initial_te_peak_calibration(start)

    assert start == 2 * gib
    assert anima_daemon.CACHE._te_peak_calibration[key] == (7 * gib, 0)


# ---------------------------------------------------------------------------
# ensure_text_ready（TE 先行第一段）
# ---------------------------------------------------------------------------


def _te_cfg(path="G:/models/te-dir"):
    return {
        "model_family": "krea2",
        "text_encoder_path": path,
        "transformer_path": "G:/models/dit.safetensors",
        "vae_path": "G:/models/vae.safetensors",
        "mixed_precision": "bf16",
    }


def test_ensure_text_ready_builds_and_reuses(monkeypatch):
    cache = anima_daemon.ModelCache()
    built = []

    class _Family:
        def load_text(self, path, device, dtype, **kw):
            built.append(str(path))
            return _Stack()

    monkeypatch.setattr(anima_daemon._T, "get_family", lambda fid: _Family())
    monkeypatch.setattr(
        anima_daemon._T, "resolve_path_best_effort", lambda p, bases: str(p),
    )

    cache.ensure_text_ready(_te_cfg())
    assert len(built) == 1
    stack_first = cache.text_stack

    cache.ensure_text_ready(_te_cfg())  # 同参数复用（LRU 保留）
    assert len(built) == 1
    assert cache.text_stack is stack_first

    cache.ensure_text_ready(_te_cfg(path="G:/models/te-fp8"))  # 换 TE 重建
    assert len(built) == 2
    assert cache.text_stack is not stack_first


def test_ensure_text_ready_noop_for_anima(monkeypatch):
    cache = anima_daemon.ModelCache()
    cfg = _te_cfg()
    cfg["model_family"] = "anima"
    cache.ensure_text_ready(cfg)
    assert cache.text_stack is None


def test_unload_keep_text_preserves_stack(monkeypatch):
    """ensure_loaded 重载路径用 keep_text——刚编码完的 LRU 不被清。"""
    cache = anima_daemon.ModelCache()
    stack = _Stack()
    cache.text_stack = stack
    cache._text_ready_key = ("krea2", "p")
    cache.model = object()
    cache.vae = object()

    cache.unload(keep_text=True)
    assert cache.text_stack is stack
    assert cache._text_ready_key == ("krea2", "p")
    assert cache.model is None

    cache.unload()  # 完整卸载清 text 栈
    assert cache.text_stack is None
    assert cache._text_ready_key is None
