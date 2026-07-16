import pytest

from studio.domain.comfy_parity import force_comfy_parity_runtime_config
from studio.domain.training import TrainingConfig
from training.families.anima.sampling import (
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
        )


def test_training_config_caption_comfy_encoding_default_true() -> None:
    cfg = TrainingConfig()
    assert cfg.caption_comfy_encoding is True
    schema = TrainingConfig.model_json_schema()
    field = schema["properties"]["caption_comfy_encoding"]
    assert field["default"] is True


def test_training_config_sampler_scheduler_are_enums() -> None:
    """收紧为 Literal 后 UI 自动渲染下拉，非法值在 config 层就挡掉。"""
    schema = TrainingConfig.model_json_schema()
    assert set(schema["properties"]["sample_sampler_name"]["enum"]) == {
        "er_sde", "dpmpp_3m_sde", "euler",
    }
    assert set(schema["properties"]["sample_scheduler"]["enum"]) == {
        "simple", "sgm_uniform", "krea2_shift",
    }


def test_training_config_coerces_legacy_sampler_values() -> None:
    """旧 preset 可能存了 euler（当年走 inline Euler 兜底）：归并默认而非炸 config。"""
    cfg = TrainingConfig(sample_sampler_name="euler", sample_scheduler="normal")
    assert cfg.sample_sampler_name == "er_sde"
    assert cfg.sample_scheduler == "simple"


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
    assert cfg["vae_precision"] == "bf16"
    assert cfg["text_encoder_backend"] == "comfy_qwen3"
    assert cfg["t5_tokenizer_backend"] == "fast"
    assert "xformers" not in cfg
    assert "flash_attn" not in cfg


def test_comfy_parity_runtime_config_preserves_explicit_vae_precision() -> None:
    """vae_precision 是用户选项（settings.generate.vae_precision），不被强制覆盖。"""
    cfg = force_comfy_parity_runtime_config({"vae_precision": "fp32"})
    assert cfg["vae_precision"] == "fp32"

    cfg = force_comfy_parity_runtime_config({})
    assert cfg["vae_precision"] == "bf16"
