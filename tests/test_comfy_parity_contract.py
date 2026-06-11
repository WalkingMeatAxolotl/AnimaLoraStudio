import pytest

from studio.domain.comfy_parity import force_comfy_parity_runtime_config
from studio.domain.training import TrainingConfig
from training.sampling import (
    _normalize_negative_prompt,
    _resolve_parity_sampler_scheduler,
    sample_image,
)


class _MinimalModel:
    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass


@pytest.mark.parametrize("sampler", ["dpmpp_3m_sde", "er_sde"])
@pytest.mark.parametrize("scheduler", ["sgm_uniform", "simple"])
def test_resolve_parity_accepts_supported_matrix(sampler: str, scheduler: str) -> None:
    assert _resolve_parity_sampler_scheduler(sampler, scheduler) == (sampler, scheduler)


@pytest.mark.parametrize(
    ("sampler", "scheduler"),
    [
        ("euler", "simple"),
        ("dpmpp_3m_sde", "normal"),
        ("dpmpp_3m_sde", "sgm_uiform"),
    ],
)
def test_resolve_parity_rejects_unsupported_names(sampler: str, scheduler: str) -> None:
    with pytest.raises(ValueError, match="unsupported Comfy parity"):
        _resolve_parity_sampler_scheduler(sampler, scheduler)


def test_sample_image_comfy_parity_rejects_unsupported_scheduler_before_fallback() -> None:
    with pytest.raises(ValueError, match="unsupported Comfy parity"):
        sample_image(
            _MinimalModel(),
            object(),
            object(),
            object(),
            object(),
            "prompt",
            height=16,
            width=16,
            steps=1,
            cfg_scale=1.0,
            negative_prompt="",
            sampler_name="er_sde",
            scheduler="normal",
            device="cpu",
            dtype=None,
            comfy_parity=True,
        )


def test_sample_image_comfy_parity_rejects_unsupported_sampler_before_fallback() -> None:
    with pytest.raises(ValueError, match="unsupported Comfy parity"):
        sample_image(
            _MinimalModel(),
            object(),
            object(),
            object(),
            object(),
            "prompt",
            height=16,
            width=16,
            steps=1,
            cfg_scale=1.0,
            negative_prompt="",
            sampler_name="euler",
            scheduler="simple",
            device="cpu",
            dtype=None,
            comfy_parity=True,
        )


def test_negative_prompt_empty_string_stays_empty_for_comfy_parity() -> None:
    assert _normalize_negative_prompt("", comfy_parity=True) == ""


def test_negative_prompt_none_uses_legacy_default_outside_comfy_parity() -> None:
    value = _normalize_negative_prompt(None, comfy_parity=False)
    assert "worst quality" in value


def test_negative_prompt_none_stays_empty_for_comfy_parity() -> None:
    assert _normalize_negative_prompt(None, comfy_parity=True) == ""


def test_training_config_exposes_sample_comfy_parity_default_true() -> None:
    cfg = TrainingConfig()
    assert cfg.sample_comfy_parity is True
    schema = TrainingConfig.model_json_schema()
    field = schema["properties"]["sample_comfy_parity"]
    assert field["default"] is True


def test_comfy_parity_runtime_config_forces_comfy_aki_runtime() -> None:
    cfg = force_comfy_parity_runtime_config({
        "attention_backend": "flash_attn",
        "mixed_precision": "bf16",
        "xformers": True,
        "flash_attn": True,
        "sampler_name": "dpmpp_3m_sde",
        "scheduler": "sgm_uniform",
    })

    assert cfg["attention_backend"] == "xformers"
    assert cfg["mixed_precision"] == "bf16"
    assert cfg["vae_precision"] == "fp32"
    assert cfg["text_encoder_backend"] == "comfy_qwen3"
    assert cfg["t5_tokenizer_backend"] == "fast"
    assert "xformers" not in cfg
    assert "flash_attn" not in cfg
