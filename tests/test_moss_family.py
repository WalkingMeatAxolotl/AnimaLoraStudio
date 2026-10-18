"""MOSS-Image 族注册与训练闭环（MOSS 设计稿 §5.6–§5.8，PR-4a）。

- registry / 能力矩阵 / 族默认 / 采样白名单
- Comfy parity 采样：simple σ 表、percent_to_sigma、初始噪声、uncond 跳层生效区间、CFG=1
- 端到端出图（小 DiT + 假文本栈 + 假 VAE）
- LoRA preset 命中范围；int8 底模下 LyCORIS 强制 bypass，前向不变、梯度可达
- 训练前 int8 底模组合校验
- 资产清单默认路径与 TE 目录就绪判定
"""
from __future__ import annotations

import json
from fnmatch import fnmatch
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from modeling.moss.moss_modeling import MossConfig, MossImageTransformer
from training.families import get_family, get_spec
from training.families.latent_spaces import FLUX2_F8C32_BN
from training.families.moss import MOSS_SPEC, MossFamily
from training.families.moss.forward import MossTextCondition
from training.families.moss.preset import MOSS_PRESET
from training.families.moss.quant_int8 import hadamard, patch_int8_linear
from training.families.moss.sampling import (
    MossSamplingCondition,
    build_sigmas,
    comfy_initial_noise,
    percent_to_sigma,
    sample_latents,
)

_CFG = MossConfig(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=32, freq_dim=16)
_GROUP = 16


def _model(seed: int = 0) -> MossImageTransformer:
    torch.manual_seed(seed)
    return MossImageTransformer(_CFG).eval()


def _cond(length: int = 5, seed: int = 0) -> MossTextCondition:
    g = torch.Generator().manual_seed(seed)
    return MossTextCondition(torch.randn(1, length, _CFG.text_dim, generator=g),
                             torch.ones(1, length, dtype=torch.bool))


# ── 注册 ────────────────────────────────────────────────────────────────


def test_registered_spec_and_family() -> None:
    assert get_spec("moss") is MOSS_SPEC
    assert isinstance(get_family("moss"), MossFamily)
    assert MOSS_SPEC.latent is FLUX2_F8C32_BN
    assert MOSS_SPEC.text.strategy == "cached_varlen"
    assert {"masked_loss", "text_cache", "int8_base"} <= MOSS_SPEC.capabilities
    assert "block_swap" in MOSS_SPEC.capabilities  # PR-5 接入（层容器经 dit_blocks 取）
    s = MOSS_SPEC.sampling
    assert (s.default_sampler, s.default_scheduler, s.default_steps, s.default_cfg) == ("euler", "simple", 50, 3.0)
    assert s.shift_policy.shift == 5.0


def test_training_config_family_defaults_overlay() -> None:
    from studio.domain.training import TrainingConfig

    cfg = TrainingConfig(model_family="moss")
    assert cfg.timestep_sampling == "logit_normal" and cfg.timestep_shift == 1.0
    assert cfg.text_encoder_cache is True and cfg.shuffle_caption is False
    assert (cfg.sample_sampler_name, cfg.sample_scheduler) == ("euler", "simple")
    assert (cfg.sample_infer_steps, cfg.sample_cfg_scale) == (50, 3.0)
    # 显式值不被族默认改写
    assert TrainingConfig(model_family="moss", timestep_shift=3.0).timestep_shift == 3.0
    assert TrainingConfig(model_family="moss", sample_sampler_name="er_sde").sample_sampler_name == "er_sde"


# ── 采样（Comfy parity） ─────────────────────────────────────────────────


def test_simple_sigmas_match_comfy_formula() -> None:
    sig = build_sigmas(4)
    table = 5.0 * (torch.arange(1, 1001) / 1000.0) / (1 + 4.0 * (torch.arange(1, 1001) / 1000.0))
    expected = [float(table[-(1 + int(i * 250))]) for i in range(4)] + [0.0]
    torch.testing.assert_close(sig, torch.tensor(expected))
    assert float(sig[0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="simple"):
        build_sigmas(4, scheduler="sgm_uniform")


def test_percent_to_sigma_matches_comfy_discrete_flow() -> None:
    assert percent_to_sigma(0.0) == 1.0 and percent_to_sigma(1.0) == 0.0
    assert percent_to_sigma(0.5) == pytest.approx(5 * 0.5 / (1 + 4 * 0.5))


def test_initial_noise_is_comfy_layout_and_seeded() -> None:
    a = comfy_initial_noise(64, 32, seed=7)
    g = torch.Generator(device="cpu").manual_seed(7)
    assert a.shape == (1, 128, 4, 2)
    torch.testing.assert_close(a, torch.randn((1, 128, 4, 2), generator=g))


class _RecordingModel(torch.nn.Module):
    """记录每次前向的 σ 与 skip_layers；输出常数速度。"""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[float, tuple]] = []

    def forward(self, x, t, context, attention_mask=None, skip_layers=None):
        self.calls.append((float(t.reshape(-1)[0]), tuple(skip_layers or ())))
        return torch.zeros_like(x)


def test_uncond_skip_layers_only_within_percent_window() -> None:
    model = _RecordingModel()
    cond = MossSamplingCondition(_cond(), _cond(seed=1))
    sample_latents(model, cond, height=32, width=32, steps=6, cfg_scale=3.0, device="cpu",
                   dtype=torch.float32, uncond_skip_layers=(1,), uncond_skip_start_percent=0.0,
                   uncond_skip_end_percent=0.5)
    lo = percent_to_sigma(0.5)
    uncond_calls = model.calls[1::2]
    assert len(model.calls) == 12  # 每步 cond + uncond
    for sigma, skip in uncond_calls:
        assert skip == ((1,) if sigma >= lo else ())
    assert all(skip == () for _s, skip in model.calls[0::2])


def test_cfg_one_skips_uncond_forward() -> None:
    model = _RecordingModel()
    sample_latents(model, MossSamplingCondition(_cond(), None), height=32, width=32, steps=3,
                   cfg_scale=1.0, device="cpu", dtype=torch.float32)
    assert len(model.calls) == 3


@pytest.mark.parametrize("sampler", ["euler", "er_sde"])
def test_sample_latents_shape_with_real_dit(sampler) -> None:
    out = sample_latents(_model(), MossSamplingCondition(_cond(), _cond(seed=2)), height=32, width=48,
                         steps=3, sampler_name=sampler, device="cpu", dtype=torch.float32, seed=3,
                         uncond_skip_layers=(1,))
    assert out.shape == (1, 32, 1, 4, 6) and torch.isfinite(out).all()
    with pytest.raises(ValueError, match="sampler"):
        sample_latents(_model(), MossSamplingCondition(_cond(), None), height=32, width=32,
                       steps=2, sampler_name="dpmpp_3m_sde", device="cpu")


def test_euler_single_step_lands_on_denoised() -> None:
    from training.inference_samplers import build_inference_sampler

    x0 = torch.randn(1, 4, 2, 2)
    out = build_inference_sampler("euler")(lambda x, s: x0, torch.randn(1, 4, 2, 2), torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(out, x0)


def test_family_sample_image_end_to_end() -> None:
    class _Text:
        def __init__(self):
            self.offloaded = False

        def encode_text_for_batch(self, captions, *, device, dtype):
            return _cond(length=len(captions[0]) % 7 + 1)

        def offload_model(self):
            self.offloaded = True

    class _Vae:
        def decode(self, z):
            b, _c, t, h, w = z.shape
            return torch.zeros(b, 3, t, h * 8, w * 8)

    text = _Text()
    img = MossFamily().sample_image(_model(), _Vae(), text, "1girl", negative_prompt="bad",
                                    height=32, width=48, steps=2, device="cpu",
                                    dtype=torch.float32, seed=1,
                                    uncond_skip_layers=(1,))  # 小模型只有 3 层；族默认 22 对应 42 层真模型
    assert img.size == (48, 32) and text.offloaded


# ── LoRA ────────────────────────────────────────────────────────────────


def test_preset_targets_only_block_projections() -> None:
    names = [n for n, m in _model().named_modules() if isinstance(m, torch.nn.Linear)]
    hit = [n for n in names
           if any(fnmatch(n, p) for p in MOSS_PRESET["target_name"])
           and not any(fnmatch(n, p) for p in MOSS_PRESET["exclude_name"])]
    assert len(hit) == _CFG.num_layers * 6
    assert all(n.startswith("layers.") and ".proj." in n for n in hit)


def _int8_model() -> MossImageTransformer:
    model = _model(seed=5)
    model.requires_grad_(False)
    for block in model.layers:
        for lin in block.proj.values():
            w = lin.weight.detach().float()
            out_f, in_f = w.shape
            h = hadamard(_GROUP)
            rot = (w.reshape(out_f, in_f // _GROUP, _GROUP) @ h.T).reshape(out_f, in_f)
            s = (rot.abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-30)
            lin.weight = torch.nn.Parameter(torch.round(rot / s).clamp(-127, 127).to(torch.int8),
                                            requires_grad=False)
            patch_int8_linear(lin, s, _GROUP)
    return model


@pytest.mark.parametrize("algo", ["lora", "lokr", "loha"])
def test_lycoris_on_int8_base_is_bypass_identity_and_trainable(algo) -> None:
    pytest.importorskip("lycoris")
    from utils.lycoris_adapter import LycorisAdapter

    model = _int8_model()
    x = torch.randn(1, 32, 1, 4, 4)
    ctx = _cond()
    with torch.no_grad():
        base = model(x, torch.tensor([0.5]), ctx.context)
    adapter = LycorisAdapter(preset=MOSS_PRESET, algo=algo, rank=4, alpha=4.0, factor=4)
    adapter.inject(model)
    with torch.no_grad():
        injected = model(x, torch.tensor([0.5]), ctx.context)
    # 初始增量为 0：bypass 下输出与底模一致（rebuild 路径会把增量加到 int8 原码上直接崩或错）
    torch.testing.assert_close(injected, base, rtol=1e-4, atol=1e-4)
    out = model(x, torch.tensor([0.5]), ctx.context)
    out.square().mean().backward()
    grads = [p.grad for p in adapter.get_params() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


# ── 训练前 int8 底模校验 ─────────────────────────────────────────────────


def _int8_file(tmp_path):
    path = tmp_path / "moss-int8.safetensors"
    conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 16}
    save_file({
        "layers.0.proj.q.weight": torch.zeros(16, 16, dtype=torch.int8),
        "layers.0.proj.q.weight_scale": torch.ones(16, 1),
        "layers.0.proj.q.comfy_quant": torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8),
    }, str(path))
    return path


def _ctx(path, family="moss", **overrides):
    args = SimpleNamespace(transformer_path=str(path), grad_checkpoint=True, lora_dora=False,
                           lora_type="lokr")
    for k, v in overrides.items():
        setattr(args, k, v)
    return SimpleNamespace(args=args, family=get_family(family))


def test_validate_int8_base_accepts_and_rejects(tmp_path) -> None:
    from training.phases.models import _validate_int8_base

    path = _int8_file(tmp_path)
    assert _validate_int8_base(_ctx(path)) is True
    with pytest.raises(RuntimeError, match="grad_checkpoint"):
        _validate_int8_base(_ctx(path, grad_checkpoint=False))
    with pytest.raises(RuntimeError, match="lora_dora"):
        _validate_int8_base(_ctx(path, lora_dora=True))
    for bad in ("ortho", "tlora"):
        with pytest.raises(RuntimeError, match=f"lora_type={bad}"):
            _validate_int8_base(_ctx(path, lora_type=bad))
    # 未声明 int8_base 的族不在这里拦（交给 loader 报「不支持的量化格式」）
    assert _validate_int8_base(_ctx(path, family="krea2", grad_checkpoint=False)) is False
    bf16 = tmp_path / "bf16.safetensors"
    save_file({"layers.0.proj.q.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}, str(bf16))
    assert _validate_int8_base(_ctx(bf16, grad_checkpoint=False)) is False


# ── 资产清单 ────────────────────────────────────────────────────────────


def test_assets_default_paths_and_te_readiness(tmp_path, monkeypatch) -> None:
    from studio import secrets
    from studio.services.models.families import get_assets
    from studio.services.models.families import moss as moss_assets

    monkeypatch.setattr(secrets, "load", lambda: secrets.Secrets(models={"selected": {}}))
    monkeypatch.setattr(moss_assets, "models_root", lambda: tmp_path)
    paths = get_assets("moss").default_paths_for_new_version()
    assert paths["transformer_path"].endswith("moss-image-base-int8-convrot.safetensors")
    assert paths["vae_path"].endswith("flux2-vae.safetensors")
    assert paths["text_encoder_path"].endswith("moss-qwen3.5-9b-int8")
    assert paths["t5_tokenizer_path"] == ""
    assert get_assets("moss").transformer_path_for("sft_bf16").endswith("moss-image-sft-bf16.safetensors")

    cfg = secrets.Secrets(models={"selected": {}}).models
    assert moss_assets.path_choices(tmp_path, cfg)["text_encoder_path"] == []
    te_dir = moss_assets.moss_te_dir(tmp_path, "int8")
    te_dir.mkdir(parents=True)
    for item in moss_assets.te_download_files("int8"):
        (te_dir / item["filename"]).write_bytes(b"x")
    choices = moss_assets.path_choices(tmp_path, cfg)["text_encoder_path"]
    assert [c["note"] for c in choices] == ["int8"]
    sections = moss_assets.catalog_sections(tmp_path, cfg)
    assert {"moss_main", "moss_text_encoder_int8", "moss_text_encoder_bf16", "moss_vae"} <= set(sections)
