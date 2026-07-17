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


# ---------------------------------------------------------------------------
# comfy 单文件形态（官方 qwen3vl_4b_fp8_scaled）加载
# ---------------------------------------------------------------------------


def _tiny_qwen3vl_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            vocab_size=128, tie_word_embeddings=True,
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, depth=2, num_heads=4,
            out_hidden_size=32, patch_size=4, temporal_patch_size=1,
            spatial_merge_size=1,
        ),
    )


def _write_comfy_te_file(dir_path, config):
    """从 tiny HF 模型反向构造 comfy 单文件：language_model 前缀剥掉、text
    侧 Linear 量化 fp8 + weight_scale + comfy_quant 假 blob，visual 原样。"""
    from safetensors.torch import save_file
    from transformers import Qwen3VLForConditionalGeneration

    torch.manual_seed(7)
    ref = Qwen3VLForConditionalGeneration(config).to(torch.bfloat16)
    out: dict[str, torch.Tensor] = {}
    fp8_layers: list[str] = []
    for key, value in ref.state_dict().items():
        if key == "lm_head.weight":
            continue  # tied，comfy 文件不含
        comfy_key = key
        if key.startswith("model.language_model."):
            comfy_key = "model." + key[len("model.language_model."):]
        is_text_linear = (
            key.startswith("model.language_model.layers.")
            and key.endswith(".weight") and value.ndim == 2
        )
        if is_text_linear:
            scale = value.float().abs().amax() / torch.finfo(torch.float8_e4m3fn).max
            scale = scale if scale > 0 else torch.tensor(1.0)
            out[comfy_key] = (value.float() / scale).to(torch.float8_e4m3fn)
            layer = comfy_key[: -len(".weight")]
            out[f"{layer}.weight_scale"] = scale.to(torch.float32)
            out[f"{layer}.comfy_quant"] = torch.zeros(64, dtype=torch.uint8)
            fp8_layers.append(key[: -len(".weight")])
        else:
            out[comfy_key] = value.detach().clone().contiguous()
    dir_path.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dir_path / "qwen3vl_fp8_scaled.safetensors"))
    return ref, fp8_layers


def test_comfy_single_file_te_loads_maps_keys_and_patches(tmp_path, monkeypatch):
    import training.families.krea2.text_encoding as te_mod

    config = _tiny_qwen3vl_config()
    te_dir = tmp_path / "qwen3vl-fp8"
    ref, fp8_layers = _write_comfy_te_file(te_dir, config)
    monkeypatch.setattr(
        te_mod, "AutoConfig", None, raising=False,
    )
    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained",
        classmethod(lambda cls, *a, **k: config),
    )

    model = te_mod._default_model_loader(
        te_dir, torch.device("cpu"), torch.float16,
    )

    assert fp8_layers  # fixture 必须真的量化了 text Linear
    quantized = {
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and module.weight.dtype == torch.float8_e4m3fn
    }
    assert quantized == set(fp8_layers)
    # visual 原样 fp16（cast 到存储 dtype）；lm_head tied 回 embed
    assert model.model.visual.blocks[0].attn.proj.weight.dtype == torch.float16
    assert model.lm_head.weight.data_ptr() == \
        model.model.language_model.embed_tokens.weight.data_ptr()
    assert all(not p.requires_grad for p in model.parameters())
    # dequant 数值近似原 bf16 权重
    name = fp8_layers[0]
    module = dict(model.named_modules())[name]
    back = module.weight.float() * module.weight_scale
    orig = dict(ref.named_modules())[name].weight.float()
    tol = orig.abs().amax() * 2**-3 + 1e-6
    assert torch.all((back - orig).abs() <= tol)


def test_comfy_single_file_detection(tmp_path):
    from training.families.krea2.text_encoding import _comfy_te_single_file

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "model.safetensors.index.json").write_text("{}")
    (hf_dir / "model-00001-of-00002.safetensors").write_bytes(b"")
    assert _comfy_te_single_file(hf_dir) is None

    single_dir = tmp_path / "single"
    single_dir.mkdir()
    (single_dir / "qwen3vl_fp8.safetensors").write_bytes(b"")
    assert _comfy_te_single_file(single_dir) is not None


def test_manual_cast_skips_fp8_linears():
    """patch_manual_cast 不覆盖 fp8 层的 dequant 前向（覆盖会丢 scale）。"""
    from training.families.krea2.text_encoding import patch_manual_cast

    model = _TinyTe().to(torch.float16)
    quantize_linears_to_fp8(model)
    fp8_forward = model.proj.forward

    patch_manual_cast(model, torch.float32)

    assert model.proj.forward is fp8_forward  # 未被覆盖
    x = torch.randn(2, 4, dtype=torch.float32)
    expected_weight = model.proj.weight.to(torch.float32) * model.proj.weight_scale
    expected = torch.nn.functional.linear(
        x, expected_weight, model.proj.bias.to(torch.float32),
    )
    torch.testing.assert_close(model.proj(x), expected, rtol=0, atol=0)
