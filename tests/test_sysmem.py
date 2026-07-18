"""系统内存工具：working set trim + RAM 水位护栏。

真机案例：mmap 读 13GB DiT + 5GB TE 后文件缓存页驻留 working set，物理
内存紧张的机器换页风暴整机卡死（显存全程健康）。trim 归还缓存页；护栏
在加载前 fail-fast（可配置，Settings → 显存策略）。
"""
from __future__ import annotations

import pytest

from training import sysmem


def test_trim_working_set_returns_bool():
    assert sysmem.trim_working_set() in (True, False)


def test_available_ram_bytes_positive():
    avail = sysmem.available_ram_bytes()
    assert avail is None or avail > 0


def test_ram_guard_disabled_skips(monkeypatch):
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 0)
    sysmem.check_ram_guard(False, stage="测试")  # 关闭时不抛


def test_ram_guard_raises_below_threshold(monkeypatch):
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 1 * 1024**3)
    with pytest.raises(RuntimeError, match="可用内存不足"):
        sysmem.check_ram_guard(True, stage="模型加载")


def test_ram_guard_passes_with_headroom(monkeypatch):
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 32 * 1024**3)
    sysmem.check_ram_guard(True, stage="模型加载")


def test_ram_guard_query_failure_is_permissive(monkeypatch):
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: None)
    sysmem.check_ram_guard(True, stage="模型加载")  # 查询失败放行
