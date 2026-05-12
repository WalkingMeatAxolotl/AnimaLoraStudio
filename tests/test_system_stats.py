"""services/system_stats.py — 采集 + NVML 优雅降级 + SSE sampler 线程。"""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from studio.services import system_stats


def test_collect_stats_returns_sane_basic():
    """实环境采集：CPU/RAM 范围合理，结构完整。"""
    stats = system_stats.collect_stats()
    assert 0.0 <= stats.cpu_pct <= 100.0
    assert stats.ram_used_gb >= 0.0
    assert stats.ram_total_gb > 0.0
    assert stats.ram_used_gb <= stats.ram_total_gb
    # gpu 字段在 CI 环境通常是 None；本地有卡时是 list[GpuStats]


def test_stats_to_json_no_gpu():
    s = system_stats.SystemStats(
        cpu_pct=12.5, ram_used_gb=8.0, ram_total_gb=32.0, gpu=None,
    )
    j = system_stats.stats_to_json(s)
    assert set(j.keys()) == {"cpu_pct", "ram_used_gb", "ram_total_gb", "gpu"}
    assert j["gpu"] is None


def test_stats_to_json_with_gpu():
    g = system_stats.GpuStats(
        index=0, name="Test GPU", util_pct=42,
        vram_used_gb=4.0, vram_total_gb=24.0, temp_c=55,
    )
    s = system_stats.SystemStats(
        cpu_pct=1.0, ram_used_gb=8.0, ram_total_gb=32.0, gpu=[g],
    )
    j = system_stats.stats_to_json(s)
    assert isinstance(j["gpu"], list) and len(j["gpu"]) == 1
    g0 = j["gpu"][0]
    assert g0["index"] == 0
    assert g0["name"] == "Test GPU"
    assert g0["util_pct"] == 42
    assert g0["temp_c"] == 55


def test_nvml_init_failure_returns_none(monkeypatch: pytest.MonkeyPatch):
    """模拟 nvmlInit 抛错：collect_gpu 永久返回 None。"""
    monkeypatch.setattr(
        system_stats, "_nvml_state", {"inited": False, "ok": False},
    )
    fake = types.ModuleType("pynvml")

    def boom() -> None:
        raise RuntimeError("simulated init failure")

    fake.nvmlInit = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    assert system_stats._collect_gpu() is None
    # 第二次调用走缓存，仍是 None；不应再次抛错
    assert system_stats._collect_gpu() is None


def test_nvml_zero_devices_returns_empty_list(monkeypatch: pytest.MonkeyPatch):
    """NVML 可用但没卡：返回 [] (前端跟 None 一样隐藏 GPU pill)。"""
    monkeypatch.setattr(
        system_stats, "_nvml_state", {"inited": True, "ok": True},
    )
    fake = types.ModuleType("pynvml")
    fake.nvmlDeviceGetCount = lambda: 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    assert system_stats._collect_gpu() == []


def test_nvml_one_fake_gpu(monkeypatch: pytest.MonkeyPatch):
    """单张 mock 卡：字段按预期映射。"""
    monkeypatch.setattr(
        system_stats, "_nvml_state", {"inited": True, "ok": True},
    )

    class FakeMem:
        used = 4 * 1024 ** 3
        total = 24 * 1024 ** 3

    class FakeUtil:
        gpu = 67

    fake = types.ModuleType("pynvml")
    fake.NVML_TEMPERATURE_GPU = 0  # type: ignore[attr-defined]
    fake.nvmlDeviceGetCount = lambda: 1  # type: ignore[attr-defined]
    fake.nvmlDeviceGetHandleByIndex = lambda i: f"h{i}"  # type: ignore[attr-defined]
    fake.nvmlDeviceGetName = lambda h: "Mock GPU"  # type: ignore[attr-defined]
    fake.nvmlDeviceGetMemoryInfo = lambda h: FakeMem()  # type: ignore[attr-defined]
    fake.nvmlDeviceGetUtilizationRates = lambda h: FakeUtil()  # type: ignore[attr-defined]
    fake.nvmlDeviceGetTemperature = lambda h, t: 50  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    result = system_stats._collect_gpu()
    assert result is not None and len(result) == 1
    g = result[0]
    assert g.index == 0
    assert g.name == "Mock GPU"
    assert g.util_pct == 67
    assert g.vram_used_gb == 4.0
    assert g.vram_total_gb == 24.0
    assert g.temp_c == 50


def test_sampler_emits_payloads(monkeypatch: pytest.MonkeyPatch):
    """SystemStatsSampler 启动后会定期 callback；stop() 干净退出。"""
    samples: list[dict] = []
    event = threading.Event()

    def on_sample(payload: dict) -> None:
        samples.append(payload)
        if len(samples) >= 2:
            event.set()

    sampler = system_stats.SystemStatsSampler(on_sample, interval=0.05)
    sampler.start()
    try:
        assert event.wait(timeout=2.0), f"only got {len(samples)} samples"
    finally:
        sampler.stop()

    assert len(samples) >= 2
    for p in samples:
        assert set(p.keys()) == {"cpu_pct", "ram_used_gb", "ram_total_gb", "gpu"}


def test_sampler_swallows_collection_errors(monkeypatch: pytest.MonkeyPatch):
    """采集抛错时 sampler 不应崩溃，继续下一轮。"""
    fail_count = [0]
    samples: list[dict] = []

    real_collect = system_stats.collect_stats

    def flaky_collect() -> system_stats.SystemStats:
        fail_count[0] += 1
        if fail_count[0] == 1:
            raise RuntimeError("simulated transient failure")
        return real_collect()

    monkeypatch.setattr(system_stats, "collect_stats", flaky_collect)

    sampler = system_stats.SystemStatsSampler(samples.append, interval=0.05)
    sampler.start()
    try:
        deadline = time.time() + 2.0
        while len(samples) < 1 and time.time() < deadline:
            time.sleep(0.05)
    finally:
        sampler.stop()

    # 第一次 collect 抛错被吞，第二次成功 → samples >= 1
    assert fail_count[0] >= 2
    assert len(samples) >= 1


def test_nvml_temp_failure_keeps_other_fields(monkeypatch: pytest.MonkeyPatch):
    """部分指标 (温度) 抛错时不应让整张卡丢失。"""
    monkeypatch.setattr(
        system_stats, "_nvml_state", {"inited": True, "ok": True},
    )

    class FakeMem:
        used = 1 * 1024 ** 3
        total = 8 * 1024 ** 3

    class FakeUtil:
        gpu = 10

    def temp_boom(*a, **k):
        raise RuntimeError("temp sensor unavailable")

    fake = types.ModuleType("pynvml")
    fake.NVML_TEMPERATURE_GPU = 0  # type: ignore[attr-defined]
    fake.nvmlDeviceGetCount = lambda: 1  # type: ignore[attr-defined]
    fake.nvmlDeviceGetHandleByIndex = lambda i: f"h{i}"  # type: ignore[attr-defined]
    fake.nvmlDeviceGetName = lambda h: "Old GPU"  # type: ignore[attr-defined]
    fake.nvmlDeviceGetMemoryInfo = lambda h: FakeMem()  # type: ignore[attr-defined]
    fake.nvmlDeviceGetUtilizationRates = lambda h: FakeUtil()  # type: ignore[attr-defined]
    fake.nvmlDeviceGetTemperature = temp_boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)

    result = system_stats._collect_gpu()
    assert result is not None and len(result) == 1
    assert result[0].temp_c is None
    assert result[0].util_pct == 10
