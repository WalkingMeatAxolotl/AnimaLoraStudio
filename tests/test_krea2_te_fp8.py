"""krea2 TE fp8：官方 comfy 单文件形态加载 + dequant 前向。

官方 qwen3vl_4b_fp8_scaled 是 comfy 单文件布局：权重键是 HF 命名（text
侧差一个 ``language_model.`` 前缀）、text Linear 为 F8_E4M3 + F32 标量
weight_scale、visual/embed/norm 保持 bf16。loader 做前缀映射 + scale 收集
+ patch_fp8_linears（DiT fp8_scaled 完全同款）。TE 精度由目录形态决定
（selected_te variant 选择），训练文本缓存指纹按形态区分（-tefp8）。
"""
from __future__ import annotations

import torch
from torch import nn

from training.families.krea2.quant_fp8 import (
    _fp8_linear_forward,
    model_has_fp8_layers,
    patch_fp8_linears,
)


class _TinyTe(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 4, bias=True)
        self.out = nn.Linear(4, 2, bias=False)


def _quantize_tiny(model: _TinyTe) -> None:
    """手工构造 fp8_scaled 形态（per-tensor amax/448 + patch）。"""
    scales = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        with torch.no_grad():
            wf = module.weight.detach().float()
            scale = wf.abs().amax() / torch.finfo(torch.float8_e4m3fn).max
            module.weight = nn.Parameter(
                (wf / scale).to(torch.float8_e4m3fn), requires_grad=False,
            )
        scales[name] = scale.to(torch.float32)
    patch_fp8_linears(model, scales)


def test_fp8_forward_dequants_to_input_dtype_and_casts_bias():
    """dequant 前向：weight/scale/bias 全 cast 到 input.dtype（TE fp32
    compute 场景 fp16 bias 必须 cast；DiT 场景 no-op 等价）。"""
    torch.manual_seed(1)
    model = _TinyTe().to(torch.float16)
    _quantize_tiny(model)
    assert model_has_fp8_layers(model)

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


def test_fp8_forward_bias_cast_is_noop_when_dtypes_match():
    """DiT 场景：bias 与 input 同 dtype 时 cast 为恒等（parity 不变）。"""
    torch.manual_seed(2)
    module = nn.Linear(4, 4, bias=True).to(torch.bfloat16)
    with torch.no_grad():
        module.weight = nn.Parameter(
            module.weight.to(torch.float8_e4m3fn), requires_grad=False,
        )
    from types import MethodType

    module.forward = MethodType(_fp8_linear_forward, module)
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    expected = torch.nn.functional.linear(
        x, module.weight.to(torch.bfloat16), module.bias,
    )
    torch.testing.assert_close(module(x), expected, rtol=0, atol=0)


def test_manual_cast_skips_fp8_linears():
    """patch_manual_cast 不覆盖 fp8 层的 dequant 前向（覆盖会丢 scale）。"""
    from training.families.krea2.text_encoding import patch_manual_cast

    model = _TinyTe().to(torch.float16)
    _quantize_tiny(model)
    fp8_forward = model.proj.forward

    patch_manual_cast(model, torch.float32)

    assert model.proj.forward is fp8_forward  # 未被覆盖
    x = torch.randn(2, 4, dtype=torch.float32)
    expected_weight = model.proj.weight.to(torch.float32) * model.proj.weight_scale
    expected = torch.nn.functional.linear(
        x, expected_weight, model.proj.bias.to(torch.float32),
    )
    torch.testing.assert_close(model.proj(x), expected, rtol=0, atol=0)


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
    # buffers（rotary inv_freq 等）必须物化——meta 残留曾致 offload 的
    # .to("cpu") 崩（真机案例：编码侥幸能跑，全模型遍历即崩）
    assert all(b.device.type != "meta" for _, b in model.named_buffers())
    model.to("cpu")  # offload_model 同款全模型遍历，回归不崩
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


def test_text_stack_fp8_storage_changes_fingerprint(tmp_path):
    """fp8 单文件目录 → is_fp8_storage=True + 缓存指纹加 -tefp8 后缀
    （fp8/bf16 编码的嵌入不混源）；HF 目录 → 原指纹。"""
    from tests.test_krea2_text_encoding import _FakeTokenizer

    from training.families.krea2.text_encoding import Krea2TextStack

    fp8_dir = tmp_path / "fp8"
    fp8_dir.mkdir()
    (fp8_dir / "w.safetensors").write_bytes(b"")
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "model.safetensors.index.json").write_text("{}")

    common = dict(
        device="cpu", dtype=torch.float16, cache_enabled=True,
        tokenizer=_FakeTokenizer(), model_loader=lambda *a: None,
        max_length=8, selected_layers=(1, 3), hidden_width=4,
        text_fingerprint="krea2-test-v1",
    )
    fp8_stack = Krea2TextStack(fp8_dir, **common)
    hf_stack = Krea2TextStack(hf_dir, **common)

    assert fp8_stack.is_fp8_storage is True
    assert fp8_stack.store.text_fingerprint == "krea2-test-v1-tefp8"
    assert hf_stack.is_fp8_storage is False
    assert hf_stack.store.text_fingerprint == "krea2-test-v1"
