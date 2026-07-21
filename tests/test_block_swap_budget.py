"""block swap 的预算护栏（docs/design/block-swap.md §3.2 ① / 刀 3）。

两条独立的预算，语义不同、不能互相复用：
- ``check_load_budget`` 的显存侧：换出层**永不上卡**，必须折扣，否则小显存卡
  开满 swap 会被按「完整模型装不下」误拒。
- ``check_pinned_budget``：换出层锁定在内存里、``trim_working_set`` 对它无效，
  按可用物理内存的安全比例把关。

不需要 CUDA（纯预算算术 + 猴补查询函数）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "runtime"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training import sysmem  # noqa: E402

_GIB = 1024 ** 3


@pytest.fixture
def fake_env(monkeypatch):
    """把 RAM / VRAM 查询与文件大小都换成可控值。"""

    def _apply(*, ram_gb: float, vram_gb: float, file_gb: float):
        monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: int(ram_gb * _GIB))
        monkeypatch.setattr(
            sysmem, "gpu_free_bytes_global", lambda: int(vram_gb * _GIB)
        )
        monkeypatch.setattr(sysmem, "_file_bytes", lambda _paths: int(file_gb * _GIB))

    return _apply


def test_vram_discount_lets_small_card_pass(fake_env):
    """16GB 卡 + 25.8GB 模型 + 换出 22.6GB → 应放行（这正是 B12 的场景）。

    不折扣的话按完整模型算需 25.8+3=28.8GB，会被误拒。
    """
    fake_env(ram_gb=64, vram_gb=15.0, file_gb=25.8)

    # 不折扣：拒绝
    with pytest.raises(RuntimeError, match="GPU 空闲显存不足"):
        sysmem.check_load_budget(True, weight_paths=["x"], stage="测试")

    # 折扣掉换出的 22.64GB：常驻仅 3.2GB + 基底 3GB < 15GB，放行
    sysmem.check_load_budget(
        True, weight_paths=["x"], stage="测试",
        vram_discount_bytes=int(22.64 * _GIB),
    )


def test_vram_discount_still_rejects_when_genuinely_short(fake_env):
    """折扣不是免死金牌：常驻部分仍装不下时照样拒。"""
    fake_env(ram_gb=64, vram_gb=4.0, file_gb=25.8)
    with pytest.raises(RuntimeError, match="GPU 空闲显存不足"):
        sysmem.check_load_budget(
            True, weight_paths=["x"], stage="测试",
            vram_discount_bytes=int(14 * _GIB),
        )


def test_vram_discount_does_not_relax_ram_side(fake_env):
    """折扣只作用于显存侧 —— 换出层仍要占内存，RAM 预算照算。"""
    fake_env(ram_gb=8, vram_gb=80, file_gb=25.8)
    with pytest.raises(RuntimeError, match="系统可用内存不足"):
        sysmem.check_load_budget(
            True, weight_paths=["x"], stage="测试",
            vram_discount_bytes=int(22.64 * _GIB),
        )


def test_pinned_budget_rejects_over_safe_fraction(monkeypatch):
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 16 * _GIB)
    # 安全上限 = 16 × 0.6 = 9.6GB
    sysmem.check_pinned_budget(int(9 * _GIB), blocks=14)
    with pytest.raises(RuntimeError, match="内存不足以换出"):
        sysmem.check_pinned_budget(int(12 * _GIB), blocks=28)


def test_pinned_budget_message_is_actionable(monkeypatch):
    """B6：报错不静默降级，且文案要能指导操作。"""
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: 8 * _GIB)
    with pytest.raises(RuntimeError) as exc:
        sysmem.check_pinned_budget(int(20 * _GIB), blocks=28)
    msg = str(exc.value)
    assert "28" in msg
    assert "blocks_to_swap" in msg
    assert "锁定" in msg


def test_pinned_budget_silent_when_query_fails(monkeypatch):
    """查询失败静默放行（与既有护栏口径一致，不因探测不到就挡住训练）。"""
    monkeypatch.setattr(sysmem, "available_ram_bytes", lambda: None)
    sysmem.check_pinned_budget(int(999 * _GIB), blocks=28)


def test_pinned_budget_noop_for_zero():
    sysmem.check_pinned_budget(0, blocks=0)
