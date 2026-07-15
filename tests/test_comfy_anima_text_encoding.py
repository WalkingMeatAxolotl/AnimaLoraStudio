from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from training import inference_samplers
from training.families.anima import sampling
from training.families.anima.sampling import sample_image
from training.families.anima.text_encoding import build_comfy_anima_conditioning_inputs, encode_qwen


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False, **_kwargs):
        value = str(text)
        ids = [self.token_id(ch) for ch in value]
        ids.append(self.eos_token_id)
        return {"input_ids": ids}

    @staticmethod
    def token_id(ch: str) -> int:
        return ord(ch) % 251 + 2


class FakeBatch(dict):
    def to(self, device):
        return FakeBatch({k: v.to(device) for k, v in self.items()})


class RecordingQwenTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts, **_kwargs):
        values = [str(text) for text in texts]
        self.calls.append(values)
        max_len = max(1, *(len(value) for value in values))
        input_ids = torch.full((len(values), max_len), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(values), max_len), dtype=torch.long)
        for row, value in enumerate(values):
            ids = [FakeTokenizer.token_id(ch) for ch in value] or [self.eos_token_id]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention[row, : len(ids)] = 1
        return FakeBatch({"input_ids": input_ids, "attention_mask": attention})


class EmptyQwenTokenizer:
    pad_token_id = 151643
    eos_token_id = 151645

    def __call__(self, texts, **_kwargs):
        return FakeBatch(
            {
                "input_ids": torch.empty((len(texts), 0), dtype=torch.long),
                "attention_mask": torch.empty((len(texts), 0), dtype=torch.long),
            }
        )


class FakeQwenModel:
    def __call__(self, input_ids, attention_mask, **_kwargs):
        hidden = torch.ones((*input_ids.shape, 4), dtype=torch.float32)
        return SimpleNamespace(hidden_states=[hidden])


class FakeComfyQwenModel:
    uses_comfy_clip_masking = True

    def __call__(self, input_ids, attention_mask, **_kwargs):
        hidden = torch.full((*input_ids.shape, 4), 3.0, dtype=torch.float32)
        return SimpleNamespace(hidden_states=[hidden])


class RecordingAnimaModel:
    def __init__(self) -> None:
        self.preprocess_calls: list[dict[str, torch.Tensor | None]] = []

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass

    def preprocess_text_embeds(self, qwen_embeds, t5_ids, t5xxl_weights=None):
        self.preprocess_calls.append(
            {
                "qwen_embeds": qwen_embeds,
                "t5_ids": t5_ids,
                "t5xxl_weights": t5xxl_weights,
            }
        )
        return torch.ones((t5_ids.shape[0], t5_ids.shape[1], 4), dtype=torch.float32)

    def __call__(self, *args, **kwargs):
        raise AssertionError("sampler test should not run denoise")


class RecordingBatchedDenoiseModel(RecordingAnimaModel):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls: list[dict[str, torch.Tensor]] = []

    def preprocess_text_embeds(self, qwen_embeds, t5_ids, t5xxl_weights=None):
        self.preprocess_calls.append(
            {
                "qwen_embeds": qwen_embeds,
                "t5_ids": t5_ids,
                "t5xxl_weights": t5xxl_weights,
            }
        )
        value = float(len(self.preprocess_calls))
        return torch.full((t5_ids.shape[0], t5_ids.shape[1], 4), value, dtype=torch.float32)

    def __call__(self, x, timesteps, cross, **kwargs):
        self.forward_calls.append(
            {
                "x": x.detach().clone(),
                "timesteps": timesteps.detach().clone(),
                "cross": cross.detach().clone(),
                "padding_mask": kwargs["padding_mask"].detach().clone(),
            }
        )
        return torch.zeros_like(x)


class XformersNaNThenFiniteModel(RecordingBatchedDenoiseModel):
    def __call__(self, x, timesteps, cross, **kwargs):
        from modeling.anima import cosmos_predict2_modeling as cosmos

        self.forward_calls.append(
            {
                "x": x.detach().clone(),
                "timesteps": timesteps.detach().clone(),
                "cross": cross.detach().clone(),
                "padding_mask": kwargs["padding_mask"].detach().clone(),
            }
        )
        if cosmos._USE_XFORMERS:
            return torch.full_like(x, float("nan"))
        return torch.zeros_like(x)


class FakeVAEModel:
    def decode(self, latents, scale):
        batch, _channels, frames, height, width = latents.shape
        return torch.zeros((batch, 3, frames, height, width), dtype=torch.float32)


class FakeVAE:
    scale = 1.0
    model = FakeVAEModel()


class TinyTrackedAnimaModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.preprocess_devices: list[str] = []

    def preprocess_text_embeds(self, qwen_embeds, t5_ids, t5xxl_weights=None):
        self.preprocess_devices.append(self.marker.device.type)
        return torch.ones((t5_ids.shape[0], t5_ids.shape[1], 4), device=t5_ids.device, dtype=torch.float32)

    def forward(self, *args, **kwargs):
        raise AssertionError("sampler test should not run denoise")


class TinyTrackedQwenModel(torch.nn.Module):
    uses_comfy_clip_masking = True

    def __init__(self) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.forward_devices: list[str] = []

    def forward(self, input_ids, attention_mask, **_kwargs):
        self.forward_devices.append(self.marker.device.type)
        hidden = torch.ones((*input_ids.shape, 4), device=input_ids.device, dtype=torch.float32)
        return SimpleNamespace(hidden_states=[hidden])


class AssertingVAEModel:
    def __init__(self, anima_model: TinyTrackedAnimaModel, qwen_model: TinyTrackedQwenModel) -> None:
        self.anima_model = anima_model
        self.qwen_model = qwen_model
        self.decode_calls = 0

    def parameters(self):
        yield torch.zeros((), dtype=torch.float32)

    def decode(self, latents, scale):
        # 包它的 TrackedVAE.should_offload_for_whole_decode=True → decode 时模型应已 offload 到 CPU
        self.decode_calls += 1
        assert next(self.anima_model.parameters()).device.type == "cpu"
        assert next(self.qwen_model.parameters()).device.type == "cpu"
        batch, _channels, frames, height, width = latents.shape
        return torch.zeros((batch, 3, frames, height, width), device=latents.device, dtype=torch.float32)


class TrackedVAE:
    scale = 1.0

    def __init__(self, anima_model: TinyTrackedAnimaModel, qwen_model: TinyTrackedQwenModel) -> None:
        self.model = AssertingVAEModel(anima_model, qwen_model)

    def should_offload_for_whole_decode(self, z):
        return True


class FailingAssertingVAEModel(AssertingVAEModel):
    def decode(self, latents, scale):
        super().decode(latents, scale)
        raise RuntimeError("decode boom")


class FailingTrackedVAE:
    scale = 1.0

    def __init__(self, anima_model: TinyTrackedAnimaModel, qwen_model: TinyTrackedQwenModel) -> None:
        self.model = FailingAssertingVAEModel(anima_model, qwen_model)

    def should_offload_for_whole_decode(self, z):
        return True


def _tokenized_non_eos(prompt: str) -> tuple[list[int], list[float]]:
    _qwen_text, t5_ids, t5_attn, t5_weights = build_comfy_anima_conditioning_inputs(FakeTokenizer(), prompt)
    valid = t5_attn[0].bool()
    ids = t5_ids[0][valid].tolist()
    weights = t5_weights[0][valid].tolist()
    assert ids[-1] == FakeTokenizer.eos_token_id
    return ids[:-1], weights[:-1]


def _weights_for_char(prompt: str, ch: str) -> list[float]:
    ids, weights = _tokenized_non_eos(prompt)
    token_id = FakeTokenizer.token_id(ch)
    return [weights[idx] for idx, value in enumerate(ids) if value == token_id]


def _assert_all_weights_are_one(weights: list[float]) -> None:
    assert weights
    assert all(abs(weight - 1.0) < 1e-6 for weight in weights)


def _ids_for_text(text: str) -> list[int]:
    return [FakeTokenizer.token_id(ch) for ch in text]


def test_comfy_helper_keeps_raw_prompt_for_qwen_text() -> None:
    prompt = "1girl, (masterpiece:1.2)"

    qwen_text, _t5_ids, _t5_attn, _t5_weights = build_comfy_anima_conditioning_inputs(FakeTokenizer(), prompt)

    assert qwen_text == prompt


def test_comfy_helper_preserves_t5_commas_and_word_order() -> None:
    prompt = "1girl, (masterpiece:1.2)"

    _qwen_text, t5_ids, _t5_attn, _t5_weights = build_comfy_anima_conditioning_inputs(FakeTokenizer(), prompt)
    ids = t5_ids[0].tolist()

    comma_index = ids.index(FakeTokenizer.token_id(","))
    girl_index = ids.index(FakeTokenizer.token_id("g"))
    masterpiece_index = ids.index(FakeTokenizer.token_id("m"))

    assert girl_index < comma_index < masterpiece_index
    assert FakeTokenizer.token_id(",") in ids


def test_comfy_helper_applies_weight_syntax_to_t5_tokens_without_cleaning_qwen_text() -> None:
    prompt = "1girl, (masterpiece:1.2), [flat color]"

    qwen_text, t5_ids, _t5_attn, t5_weights = build_comfy_anima_conditioning_inputs(FakeTokenizer(), prompt)
    ids = t5_ids[0].tolist()
    weights = t5_weights[0].tolist()

    masterpiece_token = FakeTokenizer.token_id("m")
    weighted_positions = [
        idx
        for idx, token_id in enumerate(ids)
        if token_id == masterpiece_token and abs(weights[idx] - 1.2) < 1e-5
    ]

    assert qwen_text == prompt
    assert weighted_positions


def test_comfy_helper_empty_prompt_returns_one_valid_token_path() -> None:
    qwen_text, t5_ids, t5_attn, t5_weights = build_comfy_anima_conditioning_inputs(FakeTokenizer(), "")

    assert qwen_text == ""
    assert t5_ids.shape[1] >= 1
    assert t5_attn.shape == t5_ids.shape
    assert t5_weights.shape == t5_ids.shape
    assert t5_attn.sum().item() >= 1
    assert t5_weights[t5_attn.bool()].min().item() > 0


def test_encode_qwen_keeps_comfy_masked_hidden_for_empty_prompt() -> None:
    hidden, attention = encode_qwen(
        FakeComfyQwenModel(),
        EmptyQwenTokenizer(),
        [""],
        "cpu",
        preserve_empty_text=True,
    )

    assert attention.shape == (1, 1)
    assert attention.item() == 0
    assert hidden.abs().sum().item() > 0


@pytest.mark.parametrize(
    ("prompt", "expected_weight"),
    [
        ("(masterpiece:.5)", 0.5),
        ("(masterpiece:1.)", 1.0),
        ("((masterpiece:1.2))", 1.2),
        ("((masterpiece))", 1.21),
    ],
)
def test_comfy_helper_parentheses_weights_match_sdtokenizer_edges(prompt: str, expected_weight: float) -> None:
    weights = _weights_for_char(prompt, "m")

    assert weights
    assert all(abs(weight - expected_weight) < 1e-5 for weight in weights)


@pytest.mark.parametrize("prompt", ["(masterpiece:.5)", "(masterpiece:1.)"])
def test_comfy_helper_explicit_weight_suffix_is_not_tokenized(prompt: str) -> None:
    ids, _weights = _tokenized_non_eos(prompt)

    assert FakeTokenizer.token_id(":") not in ids
    assert FakeTokenizer.token_id(".") not in ids


@pytest.mark.parametrize(
    "prompt",
    [
        "[flat color]",
        "(masterpiece",
        r"\(masterpiece\)",
        r"\[flat\]",
    ],
)
def test_comfy_helper_literal_prompt_syntax_keeps_visible_text_at_weight_one(prompt: str) -> None:
    ids, weights = _tokenized_non_eos(prompt)

    _assert_all_weights_are_one(weights)
    if prompt == "[flat color]":
        assert FakeTokenizer.token_id("[") in ids
        assert FakeTokenizer.token_id("]") in ids
    elif prompt == "(masterpiece":
        assert FakeTokenizer.token_id("(") in ids
    elif prompt == r"\(masterpiece\)":
        assert FakeTokenizer.token_id("(") in ids
        assert FakeTokenizer.token_id(")") in ids
    elif prompt == r"\[flat\]":
        assert ids.count(FakeTokenizer.token_id("\\")) == 2


@pytest.mark.parametrize(
    ("prompt", "expected_text", "expected_weight"),
    [
        ("((masterpiece)", "(masterpiece", 1.1),
        ("(foo (bar)", "foo (bar", 1.1),
    ],
)
def test_comfy_helper_malformed_nested_parentheses_follow_recursive_grouping(
    prompt: str,
    expected_text: str,
    expected_weight: float,
) -> None:
    ids, weights = _tokenized_non_eos(prompt)

    assert ids == _ids_for_text(expected_text)
    assert all(abs(weight - expected_weight) < 1e-5 for weight in weights)


def test_sample_image_comfy_parity_uses_raw_qwen_text_and_model_weight_api(monkeypatch) -> None:
    prompt = "1girl, (masterpiece:1.2)"
    model = RecordingAnimaModel()
    qwen_tokenizer = RecordingQwenTokenizer()

    monkeypatch.setattr(
        inference_samplers,
        "build_inference_sampler",
        lambda _name: (lambda denoise_fn, x, sigmas, **_kwargs: x),
    )

    image = sample_image(
        model,
        FakeVAE(),
        FakeQwenModel(),
        qwen_tokenizer,
        FakeTokenizer(),
        prompt,
        height=16,
        width=16,
        steps=1,
        cfg_scale=1.0,
        negative_prompt="",
        sampler_name="er_sde",
        scheduler="simple",
        device="cpu",
        dtype=torch.float32,
        seed=123,
    )

    assert image.size == (2, 2)
    assert qwen_tokenizer.calls[0] == [prompt]
    assert qwen_tokenizer.calls[1] == [""]
    assert model.preprocess_calls[0]["t5xxl_weights"] is not None
    assert model.preprocess_calls[1]["t5xxl_weights"] is not None


def test_sample_image_comfy_parity_batches_cfg_like_comfyui(monkeypatch) -> None:
    prompt = "1girl, (masterpiece:1.2)"
    model = RecordingBatchedDenoiseModel()

    def one_step_sampler(denoise_fn, x, sigmas, **_kwargs):
        denoise_fn(x, sigmas[0])
        return x

    monkeypatch.setattr(inference_samplers, "build_inference_sampler", lambda _name: one_step_sampler)

    image = sample_image(
        model,
        FakeVAE(),
        FakeQwenModel(),
        RecordingQwenTokenizer(),
        FakeTokenizer(),
        prompt,
        height=16,
        width=16,
        steps=1,
        cfg_scale=4.0,
        negative_prompt="",
        sampler_name="er_sde",
        scheduler="simple",
        device="cpu",
        dtype=torch.float32,
        seed=123,
    )

    assert image.size == (2, 2)
    assert len(model.forward_calls) == 1
    call = model.forward_calls[0]
    assert call["x"].shape[0] == 2
    assert call["timesteps"].shape == (2,)
    assert call["timesteps"].dtype == torch.float32
    assert call["padding_mask"].shape[0] == 2

    # ComfyUI batches uncond then cond in the common txt2img path. The first
    # preprocess call is positive, second is negative, so the batched cross order
    # should be [negative, positive].
    cross_max = call["cross"].amax(dim=(1, 2)).tolist()
    assert cross_max == [2.0, 1.0]


def test_sample_image_retries_with_sdpa_when_xformers_outputs_nan(monkeypatch) -> None:
    from modeling.anima import cosmos_predict2_modeling as cosmos

    model = XformersNaNThenFiniteModel()

    def one_step_sampler(denoise_fn, x, sigmas, **_kwargs):
        denoise_fn(x, sigmas[0])
        return x

    monkeypatch.setattr(inference_samplers, "build_inference_sampler", lambda _name: one_step_sampler)
    monkeypatch.setattr(cosmos, "_XFORMERS_AVAILABLE", True)
    monkeypatch.setattr(cosmos, "_USE_XFORMERS", True)

    image = sample_image(
        model,
        FakeVAE(),
        FakeQwenModel(),
        RecordingQwenTokenizer(),
        FakeTokenizer(),
        "1girl",
        height=16,
        width=16,
        steps=1,
        cfg_scale=4.0,
        negative_prompt="",
        sampler_name="er_sde",
        scheduler="simple",
        device="cpu",
        dtype=torch.float32,
        seed=123,
    )

    assert image.size == (2, 2)
    assert len(model.forward_calls) == 2
    # 采样结束后开关复位：本张图用 SDPA 跑完，下一张重新尝试 xformers
    assert cosmos._USE_XFORMERS is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only VRAM offload behavior")
def test_vae_decode_offload_helper_round_trips_cuda_modules() -> None:
    module = torch.nn.Linear(1, 1).cuda()

    offloaded = sampling._offload_modules_for_vae_decode(module)
    try:
        assert offloaded
        assert next(module.parameters()).device.type == "cpu"
    finally:
        sampling._restore_offloaded_modules(offloaded)

    assert next(module.parameters()).device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only VRAM offload behavior")
def test_sample_image_offloads_inactive_modules_for_vae_decode_and_restores_after_success(monkeypatch) -> None:
    model = TinyTrackedAnimaModel().cuda()
    qwen_model = TinyTrackedQwenModel().cuda()
    vae = TrackedVAE(model, qwen_model)

    monkeypatch.setattr(
        inference_samplers,
        "build_inference_sampler",
        lambda _name: (lambda denoise_fn, x, sigmas, **_kwargs: x),
    )

    for _ in range(2):
        image = sample_image(
            model,
            vae,
            qwen_model,
            RecordingQwenTokenizer(),
            FakeTokenizer(),
            "1girl",
            height=16,
            width=16,
            steps=1,
            cfg_scale=1.0,
            negative_prompt="",
            sampler_name="er_sde",
            scheduler="simple",
            device="cuda",
            dtype=torch.float32,
            seed=123,
        )
        assert image.size == (2, 2)
        assert next(model.parameters()).device.type == "cuda"
        assert next(qwen_model.parameters()).device.type == "cuda"

    assert vae.model.decode_calls == 2
    assert model.preprocess_devices == ["cuda", "cuda", "cuda", "cuda"]
    assert qwen_model.forward_devices == ["cuda", "cuda", "cuda", "cuda"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only VRAM offload behavior")
def test_sample_image_restores_offloaded_modules_when_vae_decode_fails(monkeypatch) -> None:
    model = TinyTrackedAnimaModel().cuda()
    qwen_model = TinyTrackedQwenModel().cuda()
    vae = FailingTrackedVAE(model, qwen_model)

    monkeypatch.setattr(
        inference_samplers,
        "build_inference_sampler",
        lambda _name: (lambda denoise_fn, x, sigmas, **_kwargs: x),
    )

    with pytest.raises(RuntimeError, match="decode boom"):
        sample_image(
            model,
            vae,
            qwen_model,
            RecordingQwenTokenizer(),
            FakeTokenizer(),
            "1girl",
            height=16,
            width=16,
            steps=1,
            cfg_scale=1.0,
            negative_prompt="",
            sampler_name="er_sde",
            scheduler="simple",
            device="cuda",
            dtype=torch.float32,
            seed=123,
        )

    assert next(model.parameters()).device.type == "cuda"
    assert next(qwen_model.parameters()).device.type == "cuda"


class NoOffloadAssertingVAEModel:
    """should_offload_for_whole_decode=False：decode 时模型必须仍在 CUDA（不应触发 offload）。"""

    def __init__(self, anima_model: TinyTrackedAnimaModel, qwen_model: TinyTrackedQwenModel) -> None:
        self.anima_model = anima_model
        self.qwen_model = qwen_model
        self.decode_calls = 0

    def parameters(self):
        yield torch.zeros((), dtype=torch.bfloat16)

    def decode(self, latents, scale):
        self.decode_calls += 1
        assert next(self.anima_model.parameters()).device.type == "cuda"
        assert next(self.qwen_model.parameters()).device.type == "cuda"
        batch, _channels, frames, height, width = latents.shape
        return torch.zeros((batch, 3, frames, height, width), device=latents.device, dtype=torch.float32)


class NoOffloadTrackedVAE:
    scale = 1.0

    def __init__(self, anima_model: TinyTrackedAnimaModel, qwen_model: TinyTrackedQwenModel) -> None:
        self.model = NoOffloadAssertingVAEModel(anima_model, qwen_model)

    def should_offload_for_whole_decode(self, z):
        return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only VRAM offload behavior")
def test_sample_image_skips_offload_when_vae_declines(monkeypatch) -> None:
    model = TinyTrackedAnimaModel().cuda()
    qwen_model = TinyTrackedQwenModel().cuda()
    vae = NoOffloadTrackedVAE(model, qwen_model)

    monkeypatch.setattr(
        inference_samplers,
        "build_inference_sampler",
        lambda _name: (lambda denoise_fn, x, sigmas, **_kwargs: x),
    )

    image = sample_image(
        model,
        vae,
        qwen_model,
        RecordingQwenTokenizer(),
        FakeTokenizer(),
        "1girl",
        height=16,
        width=16,
        steps=1,
        cfg_scale=1.0,
        negative_prompt="",
        sampler_name="er_sde",
        scheduler="simple",
        device="cuda",
        dtype=torch.float32,
        seed=123,
    )
    assert image.size == (2, 2)
    assert vae.model.decode_calls == 1
    assert next(model.parameters()).device.type == "cuda"


# ---------------- tokenize_t5_comfy_literal（训练 caption 编码） ----------------

def test_tokenize_t5_comfy_literal_keeps_parentheses_literal_weight_one() -> None:
    from training.families.anima.text_encoding import tokenize_t5_comfy_literal

    caption = "ganyu (genshin impact), 1girl"
    ids, attn, w = tokenize_t5_comfy_literal(FakeTokenizer(), [caption], max_length=512)

    valid = attn[0].bool()
    valid_ids = ids[0][valid].tolist()
    # 字面 tokenize：括号字符保留在 token 序列里，不做权重语法解析
    assert FakeTokenizer.token_id("(") in valid_ids
    assert FakeTokenizer.token_id(")") in valid_ids
    assert valid_ids == _ids_for_text(caption) + [FakeTokenizer.eos_token_id]
    # 权重全 1.0
    assert torch.all(w[0][valid] == 1.0)


def test_tokenize_t5_comfy_literal_differs_from_prompt_weight_parsing() -> None:
    from training.families.anima.text_encoding import tokenize_t5_comfy_literal

    caption = "ganyu (genshin impact)"
    ids, attn, _w = tokenize_t5_comfy_literal(FakeTokenizer(), [caption], max_length=512)
    valid_ids = ids[0][attn[0].bool()].tolist()

    # prompt 权重解析会吃掉括号并给 1.1 倍权重——caption 路径必须不同
    _qwen, p_ids, p_attn, p_w = build_comfy_anima_conditioning_inputs(FakeTokenizer(), caption)
    parsed_ids = p_ids[0][p_attn[0].bool()].tolist()
    assert FakeTokenizer.token_id("(") not in parsed_ids
    assert valid_ids != parsed_ids


def test_tokenize_t5_comfy_literal_batch_padding_conventions() -> None:
    from training.families.anima.text_encoding import tokenize_t5_comfy_literal

    ids, attn, w = tokenize_t5_comfy_literal(FakeTokenizer(), ["1girl", "a"], max_length=512)

    assert ids.shape == attn.shape == w.shape
    assert ids.shape[0] == 2
    # 短样本 padding：pad_id=0 / attn=0 / weight=0.0（下游清零 padding cross）
    short_pad = ~attn[1].bool()
    assert short_pad.any()
    assert torch.all(ids[1][short_pad] == FakeTokenizer.pad_token_id)
    assert torch.all(w[1][short_pad] == 0.0)
    # 两行都以 EOS 结尾
    assert ids[0][attn[0].bool()][-1].item() == FakeTokenizer.eos_token_id
    assert ids[1][attn[1].bool()][-1].item() == FakeTokenizer.eos_token_id
