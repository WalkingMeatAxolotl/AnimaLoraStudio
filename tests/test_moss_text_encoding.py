"""MOSS-Image 文本编码器单测（MOSS 设计稿 §5.5）。

小配置的真实 ``Qwen3_5TextModel``（GDN 线性注意力 + 全注意力混合，CPU / fp32）：
- HF 分片布局（``language_model.*``）与 ComfyUI int8 单文件布局（``model.language_model.*``，
  Linear + embed_tokens 均为 int8 convrot）都能 strict 加载，视觉塔 / MTP / lm_head 被跳过
- int8 TE 与「反量化参照模型」输出一致；int8 embedding 按行反量化正确
- 模板与 strip、逐条不 padding 编码、4096 截断
- 缓存：sidecar 写入后第二次全命中不再加载 TE；int8 TE 指纹不同
- transformers 缺 qwen3_5 时给明确提示
"""
from __future__ import annotations

import builtins
import json

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("transformers.models.qwen3_5")

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config  # noqa: E402
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel  # noqa: E402

import training.families.moss.text_encoding as te  # noqa: E402
from training.families.moss.quant_int8 import dequantize, hadamard, patch_int8_embedding  # noqa: E402
from training.text_cache import TextCacheEntry  # noqa: E402

_WIDTH = 64
_GROUP = 16
_TEXT_CONFIG = dict(
    vocab_size=256, hidden_size=_WIDTH, intermediate_size=96, num_hidden_layers=4,
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    num_attention_heads=2, num_key_value_heads=1, head_dim=32,
    linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
    tie_word_embeddings=False,
)


@pytest.fixture(autouse=True)
def _small_width(monkeypatch):
    monkeypatch.setattr(te, "MOSS_TEXT_WIDTH", _WIDTH)


def _config_json(path) -> None:
    (path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": _TEXT_CONFIG}), encoding="utf-8",
    )


def _source_model(seed: int = 0) -> Qwen3_5TextModel:
    torch.manual_seed(seed)
    cfg = Qwen3_5Config(text_config=_TEXT_CONFIG).text_config
    return Qwen3_5TextModel(cfg).eval()


def _quantize(w: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """comfy-kitchen 约定：W@Hᵀ（按输入维分组）后逐行 int8。"""
    out_f, in_f = w.shape
    h = hadamard(group, dtype=torch.float32)
    rot = (w.float().reshape(out_f, in_f // group, group) @ h.T).reshape(out_f, in_f)
    scale = (rot.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)
    return torch.round(rot / scale).clamp(-127, 127).to(torch.int8), scale


def _blob(group: int) -> torch.Tensor:
    conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group}
    return torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)


def _quantizable(name: str, tensor: torch.Tensor) -> bool:
    # Comfy 的 int8 TE：embed_tokens 与各 Linear 量化；GDN 的 in_proj_a/b、norm、conv 不量化
    if name == "embed_tokens.weight":
        return True
    return (
        tensor.ndim == 2 and name.endswith(".weight") and "norm" not in name
        and "in_proj_a" not in name and "in_proj_b" not in name and tensor.shape[1] % _GROUP == 0
    )


def _hf_dir(tmp_path, model) -> "Path":  # noqa: F821
    d = tmp_path / "te_hf"
    d.mkdir()
    _config_json(d)
    sd = {f"language_model.{k}": v.contiguous() for k, v in model.state_dict().items()}
    sd["visual.patch_embed.proj.weight"] = torch.zeros(2, 2)  # 视觉塔：应被跳过
    keys = sorted(sd)
    half = len(keys) // 2
    shards = {"model-00001-of-00002.safetensors": keys[:half], "model-00002-of-00002.safetensors": keys[half:]}
    weight_map = {}
    for name, ks in shards.items():
        save_file({k: sd[k] for k in ks}, str(d / name))
        weight_map.update({k: name for k in ks})
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    return d


def _comfy_int8_dir(tmp_path, model) -> "Path":  # noqa: F821
    d = tmp_path / "te_comfy"
    d.mkdir()
    _config_json(d)
    out = {}
    for k, v in model.state_dict().items():
        key = f"model.language_model.{k}"
        if _quantizable(k, v):
            q, s = _quantize(v, _GROUP)
            out[key] = q
            out[f"{key[: -len('.weight')]}.weight_scale"] = s
            out[f"{key[: -len('.weight')]}.comfy_quant"] = _blob(_GROUP)
        else:
            out[key] = v.contiguous()
    out["lm_head.weight"] = torch.zeros(4, 4, dtype=torch.int8)  # 应被跳过
    out["mtp.norm.weight"] = torch.zeros(4)
    out["model.visual.merger.norm.weight"] = torch.zeros(4)
    save_file(out, str(d / "qwen35-int8.safetensors"))
    return d


class _FakeTokenizer:
    """按字节映射 id（< vocab），记录收到的文本。"""

    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, text, *, return_tensors, truncation, max_length):
        self.seen.append(text)
        ids = [b % _TEXT_CONFIG["vocab_size"] for b in text.encode("utf-8")]
        if truncation:
            ids = ids[:max_length]
        t = torch.tensor([ids])
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


class _CountingLoader:
    def __init__(self):
        self.calls = 0

    def __call__(self, path, device, dtype):
        self.calls += 1
        return te.load_moss_text_model(path, device, dtype)


def _run(model, ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state


# ── 加载 ────────────────────────────────────────────────────────────────


def test_load_hf_shards_matches_source(tmp_path) -> None:
    src = _source_model()
    d = _hf_dir(tmp_path, src)
    assert not te.is_int8_text_encoder(d)
    loaded = te.load_moss_text_model(d, torch.device("cpu"), torch.float32)
    for k, v in src.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v)
    ids = torch.tensor([[1, 5, 9, 30, 2, 7]])
    torch.testing.assert_close(_run(loaded, ids), _run(src, ids), rtol=1e-4, atol=1e-4)


def test_load_comfy_int8_matches_dequantized_reference(tmp_path) -> None:
    src = _source_model(seed=1)
    d = _comfy_int8_dir(tmp_path, src)
    assert te.is_int8_text_encoder(d)
    loaded = te.load_moss_text_model(d, torch.device("cpu"), torch.float32)
    assert loaded.embed_tokens.weight.dtype == torch.int8
    assert loaded.layers[0].mlp.down_proj.weight.dtype == torch.int8
    assert loaded.layers[0].linear_attn.in_proj_a.weight.dtype == torch.float32

    ref = _source_model(seed=1)
    modules = dict(loaded.named_modules())
    with torch.no_grad():
        for name, mod in ref.named_modules():
            target = modules[name]
            if isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)) and target.weight.dtype == torch.int8:
                mod.weight.copy_(dequantize(target.weight, target.weight_scale, _GROUP))
    ids = torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]])
    torch.testing.assert_close(_run(loaded, ids), _run(ref, ids), rtol=1e-4, atol=1e-4)


def test_int8_embedding_rows_dequantize_and_unrotate() -> None:
    torch.manual_seed(0)
    w = torch.randn(10, 32)
    q, s = _quantize(w, _GROUP)
    emb = torch.nn.Embedding(10, 32)
    emb.weight = torch.nn.Parameter(q, requires_grad=False)
    patch_int8_embedding(emb, s, _GROUP, torch.float32)
    ids = torch.tensor([[0, 7, 7, 3]])
    torch.testing.assert_close(emb(ids), dequantize(q, s, _GROUP)[ids])


def test_load_rejects_unpaired_int8_and_missing_config(tmp_path) -> None:
    src = _source_model()
    d = _comfy_int8_dir(tmp_path, src)
    from safetensors.torch import load_file

    path = d / "qwen35-int8.safetensors"
    sd = {k: v.clone() for k, v in load_file(str(path)).items()}  # 先脱离 mmap（Windows 不能覆盖映射中的文件）
    sd.pop("model.language_model.layers.0.mlp.down_proj.weight_scale")
    path.unlink()
    save_file(sd, str(path))
    with pytest.raises(ValueError, match="一一对应"):
        te.load_moss_text_model(d, torch.device("cpu"), torch.float32)

    empty = tmp_path / "empty"
    empty.mkdir()
    save_file({"x": torch.zeros(1)}, str(empty / "a.safetensors"))
    with pytest.raises(ValueError, match="config.json"):
        te.load_moss_text_model(empty, torch.device("cpu"), torch.float32)


# ── 编码与缓存 ──────────────────────────────────────────────────────────


def test_prompt_template_strips_caption() -> None:
    assert te.moss_prompt("  a cat \n") == "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n"


def test_encode_per_caption_unpadded_and_truncated(tmp_path) -> None:
    src = _source_model()
    d = _hf_dir(tmp_path, src)
    tok = _FakeTokenizer()
    stack = te.MossTextStack(d, device="cpu", dtype=torch.float32, cache_enabled=False,
                             tokenizer=tok, max_tokens=100)
    short, long = "1girl", "x" * 200
    cond = stack.encode_text_for_batch([short, long], device="cpu", dtype=torch.float32)
    short_len = len(te.moss_prompt(short).encode())
    assert short_len < 100
    assert cond.context.shape == (2, 100, _WIDTH)
    assert cond.attention_mask.sum(1).tolist() == [short_len, 100]
    # 逐条编码：短 caption 的结果与单独跑完全一致（没有被 batch 内 padding 影响）
    ids = torch.tensor([[b % 256 for b in te.moss_prompt(short).encode()]])
    torch.testing.assert_close(cond.context[0, :short_len], _run(src, ids)[0], rtol=1e-4, atol=1e-4)
    assert tok.seen[0] == te.moss_prompt(short)


def test_cached_mode_writes_sidecar_and_hits_without_loading(tmp_path) -> None:
    d = _hf_dir(tmp_path, _source_model())
    loader = _CountingLoader()
    entry = TextCacheEntry.for_image(tmp_path / "img.png", "1girl, solo")
    first = te.MossTextStack(d, device="cpu", dtype=torch.float32, tokenizer=_FakeTokenizer(),
                             model_loader=loader)
    first.prepare_text_cache(["1girl, solo"], ["sample prompt"], cache_entries=[entry], cache_root=tmp_path)
    assert loader.calls == 1 and not first.is_model_loaded
    cached = first.store.read_caption(entry)["context"]
    assert cached.ndim == 2 and cached.shape[1] == _WIDTH

    second = te.MossTextStack(d, device="cpu", dtype=torch.float32, tokenizer=_FakeTokenizer(),
                              model_loader=loader)
    second.prepare_text_cache(["1girl, solo"], ["sample prompt"], cache_entries=[entry], cache_root=tmp_path)
    cond = second.encode_text_for_batch(["1girl, solo", "sample prompt"], device="cpu", dtype=torch.float32)
    assert loader.calls == 1  # 全命中：TE 没再加载
    torch.testing.assert_close(cond.context[0, : cached.shape[0]], cached)


def test_int8_text_encoder_has_distinct_fingerprint(tmp_path) -> None:
    src = _source_model()
    hf = te.MossTextStack(_hf_dir(tmp_path, src), device="cpu", tokenizer=_FakeTokenizer())
    q8 = te.MossTextStack(_comfy_int8_dir(tmp_path, src), device="cpu", tokenizer=_FakeTokenizer())
    assert hf.store.text_fingerprint == te.MOSS_TEXT_FINGERPRINT
    assert q8.store.text_fingerprint == te.MOSS_TEXT_FINGERPRINT + "-int8convrot-dq"
    assert q8.is_int8_storage and not hf.is_int8_storage


def test_tokenizer_falls_back_to_sibling_dir(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    (root / "text_encoder").mkdir(parents=True)
    (root / "tokenizer").mkdir()
    (root / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    seen = {}

    class _Auto:
        @staticmethod
        def from_pretrained(path, local_files_only):
            seen["path"] = path
            return "tok"

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", _Auto)
    assert te._load_tokenizer(root / "text_encoder") == "tok"
    assert seen["path"].endswith("tokenizer")
    with pytest.raises(ValueError, match="找不到 MOSS tokenizer"):
        te._load_tokenizer(tmp_path / "nowhere")


def test_old_transformers_gives_clear_message(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("transformers.models.qwen3_5"):
            raise ImportError("no qwen3_5")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("ANIMA_UI_LANG", "zh")
    with pytest.raises(RuntimeError, match=r"transformers 5\.3"):
        te.check_transformers_supports_qwen35()
