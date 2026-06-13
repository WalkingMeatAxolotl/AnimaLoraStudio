"""PR-7a — 模型层 flash_attn fast path + warn-once + set_flash_attn_enabled。

测 cosmos_predict2_modeling 的状态机和 fast path：
- set_flash_attn_enabled 在 _FLASH_ATTN_AVAILABLE=False 时永远返回 False
- enable + 提供 _flash_attn_func mock 时 torch_attention_op 走 fast path
- fast path 异常 → fallback 到 SDPA + warn-once（同 shape 不重复警告）

注：训练里 cosmos_predict2_modeling 通过 importlib.spec_from_file_location 加载
（standalone 副本），但 tests 直接走包导入也能命中同 module instance；副作用是
test 之间需要 reset 状态，每个测试用 fixture 拿干净状态。
"""
from __future__ import annotations

import logging

import pytest
import torch


@pytest.fixture
def cosmos_module():
    """每个测试拿一份重置过的 cosmos_predict2_modeling 状态。"""
    from models import cosmos_predict2_modeling as m
    orig_use = m._USE_FLASH_ATTN
    orig_avail = m._FLASH_ATTN_AVAILABLE
    orig_func = m._flash_attn_func
    orig_warned = set(m._FLASH_FALLBACK_WARNED)
    orig_x_use = m._USE_XFORMERS
    orig_x_avail = m._XFORMERS_AVAILABLE
    orig_xops = m._xops
    orig_x_warned = set(m._XFORMERS_FALLBACK_WARNED)
    yield m
    m._USE_FLASH_ATTN = orig_use
    m._FLASH_ATTN_AVAILABLE = orig_avail
    m._flash_attn_func = orig_func
    m._FLASH_FALLBACK_WARNED.clear()
    m._FLASH_FALLBACK_WARNED.update(orig_warned)
    m._USE_XFORMERS = orig_x_use
    m._XFORMERS_AVAILABLE = orig_x_avail
    m._xops = orig_xops
    m._XFORMERS_FALLBACK_WARNED.clear()
    m._XFORMERS_FALLBACK_WARNED.update(orig_x_warned)


# ---------------------------------------------------------------------------
# set_flash_attn_enabled
# ---------------------------------------------------------------------------


def test_set_flash_attn_enabled_returns_false_when_unavailable(cosmos_module) -> None:
    """flash_attn 没装（_FLASH_ATTN_AVAILABLE=False）→ enable 永远返回 False。"""
    cosmos_module._FLASH_ATTN_AVAILABLE = False
    assert cosmos_module.set_flash_attn_enabled(True) is False
    assert cosmos_module._USE_FLASH_ATTN is False


def test_set_flash_attn_enabled_returns_true_when_available(cosmos_module) -> None:
    cosmos_module._FLASH_ATTN_AVAILABLE = True
    cosmos_module._flash_attn_func = lambda q, k, v: q  # 不会真调，但要非 None
    assert cosmos_module.set_flash_attn_enabled(True) is True
    assert cosmos_module._USE_FLASH_ATTN is True
    assert cosmos_module._USE_XFORMERS is False


def test_set_flash_attn_enabled_disable(cosmos_module) -> None:
    cosmos_module._FLASH_ATTN_AVAILABLE = True
    cosmos_module._flash_attn_func = lambda q, k, v: q
    cosmos_module.set_flash_attn_enabled(True)
    assert cosmos_module.set_flash_attn_enabled(False) is False
    assert cosmos_module._USE_FLASH_ATTN is False


def test_set_flash_attn_disabled_clears_existing_xformers(cosmos_module) -> None:
    cosmos_module._XFORMERS_AVAILABLE = True
    cosmos_module.set_xformers_enabled(True)

    assert cosmos_module.set_flash_attn_enabled(False) is False

    assert cosmos_module._USE_FLASH_ATTN is False
    assert cosmos_module._USE_XFORMERS is False


# ---------------------------------------------------------------------------
# set_xformers_enabled
# ---------------------------------------------------------------------------


def test_set_xformers_enabled_returns_false_when_unavailable(cosmos_module) -> None:
    """xformers 没装（_XFORMERS_AVAILABLE=False）→ enable 永远返回 False。"""
    cosmos_module._XFORMERS_AVAILABLE = False
    assert cosmos_module.set_xformers_enabled(True) is False
    assert cosmos_module._USE_XFORMERS is False


def test_set_xformers_enabled_returns_true_when_available(cosmos_module) -> None:
    cosmos_module._XFORMERS_AVAILABLE = True
    cosmos_module._USE_FLASH_ATTN = True
    assert cosmos_module.set_xformers_enabled(True) is True
    assert cosmos_module._USE_XFORMERS is True
    assert cosmos_module._USE_FLASH_ATTN is False


def test_set_attention_backend_none_clears_all_fast_paths(cosmos_module) -> None:
    cosmos_module._XFORMERS_AVAILABLE = True
    cosmos_module.set_xformers_enabled(True)

    assert cosmos_module.set_attention_backend("none") == "none"

    assert cosmos_module._USE_FLASH_ATTN is False
    assert cosmos_module._USE_XFORMERS is False


# ---------------------------------------------------------------------------
# torch_attention_op fast path
# ---------------------------------------------------------------------------


def test_torch_attention_op_uses_flash_when_enabled(cosmos_module) -> None:
    """启用 flash_attn 后 torch_attention_op 应直接调 _flash_attn_func（不走 SDPA）。"""
    captured: list = []
    expected_out = torch.zeros(2, 4, 3, 8)  # b=2 s=4 h=3 d=8
    def fake_flash(q, k, v):
        captured.append((q.shape, k.shape, v.shape))
        return expected_out

    cosmos_module._FLASH_ATTN_AVAILABLE = True
    cosmos_module._flash_attn_func = fake_flash
    cosmos_module._USE_FLASH_ATTN = True

    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 3, 8)
    v = torch.randn(2, 4, 3, 8)
    out = cosmos_module.torch_attention_op(q, k, v)
    assert len(captured) == 1
    # rearrange "b s h d -> b s (h d)" → 形状 (2, 4, 24)
    assert out.shape == (2, 4, 24)


def test_torch_attention_op_skips_flash_when_disabled(cosmos_module) -> None:
    """_USE_FLASH_ATTN=False → 即便 _flash_attn_func 非 None 也不调它。"""
    called = []
    cosmos_module._FLASH_ATTN_AVAILABLE = True
    cosmos_module._flash_attn_func = lambda *a, **k: called.append(1)  # 调了就记录
    cosmos_module._USE_FLASH_ATTN = False

    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)
    cosmos_module.torch_attention_op(q, k, v)  # 应走 SDPA fallback
    assert called == []


def test_torch_attention_op_uses_xformers_when_enabled(cosmos_module) -> None:
    """启用 xformers 后 torch_attention_op 应直接调 memory_efficient_attention。"""
    captured: list = []
    expected_out = torch.zeros(2, 4, 3, 8)  # b=2 s=4 h=3 d=8

    class FakeXops:
        @staticmethod
        def memory_efficient_attention(q, k, v):
            captured.append((q.shape, k.shape, v.shape))
            return expected_out

    cosmos_module._FLASH_ATTN_AVAILABLE = False
    cosmos_module._USE_FLASH_ATTN = False
    cosmos_module._XFORMERS_AVAILABLE = True
    cosmos_module._xops = FakeXops
    cosmos_module._USE_XFORMERS = True

    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 4, 3, 8)
    v = torch.randn(2, 4, 3, 8)
    out = cosmos_module.torch_attention_op(q, k, v)
    assert captured == [(q.shape, k.shape, v.shape)]
    assert out.shape == (2, 4, 24)


def test_llm_adapter_attention_does_not_use_xformers(cosmos_module) -> None:
    """ComfyUI Anima keeps LLMAdapterAttention on SDPA even when main attention uses xformers."""
    from models.anima_modeling import LLMAdapterAttention

    class BoomXops:
        @staticmethod
        def memory_efficient_attention(*_args, **_kwargs):
            raise AssertionError("LLMAdapterAttention should not call xformers")

    cosmos_module._FLASH_ATTN_AVAILABLE = False
    cosmos_module._USE_FLASH_ATTN = False
    cosmos_module._XFORMERS_AVAILABLE = True
    cosmos_module._xops = BoomXops
    cosmos_module._USE_XFORMERS = True

    attn = LLMAdapterAttention(query_dim=8, context_dim=8, n_heads=2, head_dim=4)
    x = torch.randn(1, 3, 8)
    context = torch.randn(1, 4, 8)

    out = attn(x, context=context)
    assert out.shape == x.shape


def test_torch_attention_op_fallback_on_flash_error(
    cosmos_module, caplog: pytest.LogCaptureFixture
) -> None:
    """flash_attn 抛异常 → fallback SDPA + warn-once（同 shape 第二次不再 warn）。"""
    cosmos_module._FLASH_ATTN_AVAILABLE = True

    def boom(*_a, **_k):
        raise RuntimeError("seq_len must be multiple of 8")

    cosmos_module._flash_attn_func = boom
    cosmos_module._USE_FLASH_ATTN = True

    q = torch.randn(1, 4, 2, 8)
    k = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 4, 2, 8)

    with caplog.at_level(logging.WARNING, logger="models.cosmos_predict2_modeling"):
        out1 = cosmos_module.torch_attention_op(q, k, v)
        out2 = cosmos_module.torch_attention_op(q, k, v)
        out3 = cosmos_module.torch_attention_op(q, k, v)

    # 三次调用都 fallback 到 SDPA，shape 一致
    assert out1.shape == out2.shape == out3.shape == (1, 4, 16)
    # 但只 warn 了一次（同 shape 去重）
    flash_warnings = [r for r in caplog.records if "flash_attn fallback" in r.message]
    assert len(flash_warnings) == 1
    assert "seq_len must be multiple of 8" in flash_warnings[0].message
    assert "torch_attention_op" in flash_warnings[0].message


def test_warn_flash_fallback_keys_by_shape(
    cosmos_module, caplog: pytest.LogCaptureFixture
) -> None:
    """不同 shape 各 warn 一次。"""
    cosmos_module._FLASH_FALLBACK_WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="models.cosmos_predict2_modeling"):
        cosmos_module.warn_flash_fallback("op_a", (1, 4, 2, 8), "reason")
        cosmos_module.warn_flash_fallback("op_a", (1, 4, 2, 8), "reason")  # 同 shape 同 stage
        cosmos_module.warn_flash_fallback("op_a", (2, 4, 2, 8), "reason")  # 不同 shape
        cosmos_module.warn_flash_fallback("op_b", (1, 4, 2, 8), "reason")  # 不同 stage
    assert len([r for r in caplog.records if "flash_attn fallback" in r.message]) == 3


def test_try_flash_attn_returns_used_false_when_disabled(cosmos_module) -> None:
    """helper 在 disabled 时返回 (None, False)，调用方走 fallback。"""
    cosmos_module._FLASH_ATTN_AVAILABLE = True
    cosmos_module._flash_attn_func = lambda q, k, v: q
    cosmos_module._USE_FLASH_ATTN = False
    q = torch.randn(1, 4, 2, 8)
    out, used = cosmos_module.try_flash_attn(q, q, q, "test")
    assert out is None
    assert used is False


def test_try_flash_attn_returns_used_true_when_enabled(cosmos_module) -> None:
    """helper 启用且 callable 成功时返回 (out, True)。"""
    expected = torch.zeros(1, 4, 2, 8)
    cosmos_module._FLASH_ATTN_AVAILABLE = True
    cosmos_module._flash_attn_func = lambda q, k, v: expected
    cosmos_module._USE_FLASH_ATTN = True
    q = torch.randn(1, 4, 2, 8)
    out, used = cosmos_module.try_flash_attn(q, q, q, "test")
    assert out is expected
    assert used is True


def test_try_flash_attn_returns_used_false_on_exception(
    cosmos_module, caplog: pytest.LogCaptureFixture
) -> None:
    """helper 在 fast path 抛异常时 warn-once 并返回 (None, False)。"""
    cosmos_module._FLASH_FALLBACK_WARNED.clear()
    cosmos_module._FLASH_ATTN_AVAILABLE = True

    def boom(*_a):
        raise RuntimeError("fake")
    cosmos_module._flash_attn_func = boom
    cosmos_module._USE_FLASH_ATTN = True

    q = torch.randn(1, 4, 2, 8)
    with caplog.at_level(logging.WARNING, logger="models.cosmos_predict2_modeling"):
        out, used = cosmos_module.try_flash_attn(q, q, q, "test_stage")
    assert out is None
    assert used is False
    assert any("test_stage" in r.message and "fake" in r.message for r in caplog.records)
