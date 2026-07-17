"""krea2 TE fp8（load-time rescale）：quantize_linears_to_fp8 + 文本栈接线。

官方 qwen3vl fp8 文件是 comfy 单文件布局、接不进 transformers HF 目录
loader——改为加载后逐 Linear 现场量化（per-tensor scaled，RTN），dequant
前向与 DiT fp8_scaled 同款。生成侧专属（te_precision=fp8）；训练侧不引入。
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from training.families.krea2.quant_fp8 import (
    model_has_fp8_layers,
    quantize_linears_to_fp8,
)


class _TinyTe(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 4, bias=True)
        self.out = nn.Linear(4, 2, bias=False)


def test_quantize_linears_converts_weights_and_registers_scale():
    torch.manual_seed(0)
    model = _TinyTe().to(torch.float16)
    reference = {
        name: module.weight.detach().float().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }

    count = quantize_linears_to_fp8(model)

    assert count == 2
    assert model_has_fp8_layers(model)
    assert model.embed.weight.dtype == torch.float16  # Embedding 不量化
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        assert module.weight.dtype == torch.float8_e4m3fn
        assert module.weight_scale.dtype == torch.float32
        # dequant 近似原权重（e4m3 相对误差 ≤ 2^-3 + scale 域余量）
        back = module.weight.float() * module.weight_scale
        ref = reference[name]
        tol = ref.abs().amax() * 2**-3 + 1e-6
        assert torch.all((back - ref).abs() <= tol)


def test_quantized_forward_dequants_to_input_dtype_and_casts_bias():
    torch.manual_seed(1)
    model = _TinyTe().to(torch.float16)
    quantize_linears_to_fp8(model)

    x = torch.randn(3, 4, dtype=torch.float32)  # TE compute=fp32（manual_cast）
    got = model.proj(x)

    expected_weight = (
        model.proj.weight.to(torch.float32) * model.proj.weight_scale.to(torch.float32)
    )
    expected = torch.nn.functional.linear(
        x, expected_weight, model.proj.bias.to(torch.float32),
    )
    assert got.dtype == torch.float32
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_quantize_is_idempotent_and_defensive():
    model = _TinyTe().to(torch.float16)
    assert quantize_linears_to_fp8(model) == 2
    assert quantize_linears_to_fp8(model) == 0  # 已量化幂等
    assert quantize_linears_to_fp8(object()) == 0  # 非 Module 防御
    assert quantize_linears_to_fp8(None) == 0


def test_quantize_zero_weight_layer_scale_falls_back_to_one():
    model = _TinyTe().to(torch.float16)
    with torch.no_grad():
        model.out.weight.zero_()
    quantize_linears_to_fp8(model)
    assert model.out.weight_scale.item() == pytest.approx(1.0)
    assert torch.all(model.out.weight.float() == 0)


def test_text_stack_te_quantize_wiring(tmp_path):
    """te_quantize=True：ensure_model 后 Linear 量化；fake 非 Module 安全跳过。"""
    from tests.test_krea2_text_encoding import _FakeTokenizer

    from training.families.krea2.text_encoding import Krea2TextStack

    loads = []

    def _loader(path, device, dtype):
        model = _TinyTe().to(torch.float16)
        loads.append(model)
        return model

    stack = Krea2TextStack(
        tmp_path / "qwen",
        device="cpu",
        dtype=torch.float16,
        compute_dtype=torch.float32,
        cache_enabled=False,
        tokenizer=_FakeTokenizer(),
        model_loader=_loader,
        max_length=8,
        selected_layers=(1, 3),
        hidden_width=4,
        text_fingerprint="krea2-test-v1",
        te_quantize=True,
    )
    model = stack.ensure_model()
    assert model_has_fp8_layers(model)
    assert stack.te_quantize is True

    # 默认（te_quantize=False）不量化
    stack_plain = Krea2TextStack(
        tmp_path / "qwen",
        device="cpu",
        dtype=torch.float16,
        cache_enabled=False,
        tokenizer=_FakeTokenizer(),
        model_loader=_loader,
        max_length=8,
        selected_layers=(1, 3),
        hidden_width=4,
        text_fingerprint="krea2-test-v1",
    )
    assert not model_has_fp8_layers(stack_plain.ensure_model())
