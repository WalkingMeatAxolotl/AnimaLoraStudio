"""系统内存工具：working set trim + 双预算水位护栏。

真机案例：mmap 读 13GB DiT + 5GB TE 后文件缓存页瞬时冲高（≈文件大小），
物理内存紧张的机器换页风暴整机卡死（显存全程健康）；多进程叠加（server
daemon 驻留模型 + 手动 daemon）会打穿显存。预算制护栏按即将读取的文件
实际大小同时查 RAM 与 GPU free（可配置，Settings → 显存策略）。
"""
from __future__ import annotations

import pytest

from training import sysmem


def test_trim_working_set_returns_bool():
    assert sysmem.trim_working_set() in (True, False)


def test_available_ram_bytes_positive():
    avail = sysmem.available_ram_bytes()
    assert avail is None or avail > 0


def _write_weight(tmp_path, size=1024):
    p = tmp_path / "w.safetensors"
    p.write_bytes(b"\0" * size)
    return p


def test_budget_disabled_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 0)
    sysmem.check_load_budget(
        False, weight_paths=[_write_weight(tmp_path)], stage="测试",
    )  # 关闭时不抛


def test_budget_raises_when_ram_below_file_size(tmp_path, monkeypatch):
    """预算制：avail < 文件大小 + 基底 → 拦（回答「做完之后还好吗」）。

    文件用 1KB 真身 + mock 大小（不写 GB 级临时文件——磁盘卫生）。
    """
    weight = _write_weight(tmp_path)
    monkeypatch.setattr(sysmem, "_file_bytes", lambda paths: 13 * 1024**3)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 10 * 1024**3)
    with pytest.raises(RuntimeError, match="可用内存不足"):
        sysmem.check_load_budget(True, weight_paths=[weight], stage="模型加载")


def test_budget_raises_when_gpu_free_below_need(tmp_path, monkeypatch):
    """VRAM 预算（NVML 全卡视角）：free < 文件 + 基底 → 拦（跨进程场景）。"""
    weight = _write_weight(tmp_path)
    monkeypatch.setattr(sysmem, "_file_bytes", lambda paths: 13 * 1024**3)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: 7 * 1024**3)
    with pytest.raises(RuntimeError, match="显存不足"):
        sysmem.check_load_budget(True, weight_paths=[weight], stage="模型加载")


def test_budget_passes_with_headroom(tmp_path, monkeypatch):
    weight = _write_weight(tmp_path)
    monkeypatch.setattr(sysmem, "_file_bytes", lambda paths: 13 * 1024**3)
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 64 * 1024**3)
    monkeypatch.setattr(sysmem, "gpu_free_bytes_global", lambda: 30 * 1024**3)
    sysmem.check_load_budget(True, weight_paths=[weight], stage="模型加载")


def test_budget_empty_paths_and_missing_files_are_noop(tmp_path, monkeypatch):
    """零预算（TE 已在卡上）与不存在的路径直通——结构校验由 loader 兜底。"""
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 0)
    sysmem.check_load_budget(True, weight_paths=[], stage="测试")
    sysmem.check_load_budget(
        True, weight_paths=[tmp_path / "missing.safetensors"], stage="测试",
    )


def test_budget_query_failure_is_permissive(tmp_path, monkeypatch):
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: None)
    sysmem.check_load_budget(
        True, weight_paths=[_write_weight(tmp_path)], stage="模型加载",
    )  # RAM 查询失败放行（GPU free 检查仍在，CPU 测试环境跳过）


def test_budget_sums_directory_contents(tmp_path):
    """目录型资产（HF TE）按目录内文件求和。"""
    d = tmp_path / "te"
    d.mkdir()
    (d / "a.safetensors").write_bytes(b"\0" * 100)
    (d / "b.safetensors").write_bytes(b"\0" * 200)
    assert sysmem._file_bytes([d]) == 300
