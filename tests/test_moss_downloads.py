"""MOSS-Image 下载中心接入（PR-4b 后端）。

- 主模型 / 两种 TE / FLUX.2 VAE 的 HF 与 ModelScope 渠道（官方仓库在 MS 上 id 不同）
- trigger 路由与 delete_asset
- catalog：族声明支持的量化格式（MOSS 的 int8 convrot）不再被标「不可用」，其余族照旧
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio.services import models as model_downloader
from studio.services.models import catalog as catalog_mod
from studio.services.models.families import moss as moss_assets


def _recorder(write: bool = True):
    calls: list[tuple[str, str, str]] = []

    def fake(repo_id, repo_subpath, target, *, on_log=print):
        calls.append((repo_id, repo_subpath, str(target)))
        if write:
            Path(target).parent.mkdir(parents=True, exist_ok=True)
            Path(target).write_bytes(b"x")
        return True

    return calls, fake


def test_download_moss_main_uses_comfy_org_on_modelscope(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_SOURCE", "modelscope")
    hf, fake_hf = _recorder()
    ms, fake_ms = _recorder()
    monkeypatch.setattr("studio.services.models.sources.download_flat", fake_hf)
    monkeypatch.setattr("studio.services.models.sources.download_flat_ms", fake_ms)
    assert model_downloader.download_moss_main(tmp_path, "latest", on_log=lambda _l: None)
    assert hf == []
    assert ms == [(
        "Comfy-Org/MOSS-Image",
        "diffusion_models/moss_image_base_int8_convrot.safetensors",
        str(tmp_path / "diffusion_models" / "moss-image-base-int8-convrot.safetensors"),
    )]
    assert not model_downloader.download_moss_main(tmp_path, "bogus", on_log=lambda _l: None)


def test_download_moss_te_int8_modelscope_maps_official_repo_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_SOURCE", "modelscope")
    ms, fake_ms = _recorder()
    monkeypatch.setattr("studio.services.models.sources.download_flat_ms", fake_ms)
    assert model_downloader.download_moss_te(tmp_path, "int8", on_log=lambda _l: None)
    repos = {repo for repo, _, _ in ms}
    assert repos == {"Comfy-Org/MOSS-Image", "openmoss/MOSS-Image-Base"}
    assert ms[0][1] == "text_encoders/qwen3.5_9b_int8_convrot.safetensors"
    # 落成训练侧认的布局：一个目录 = 权重 + config + tokenizer
    te_dir = moss_assets.moss_te_dir(tmp_path, "int8")
    for name in ("qwen3.5_9b_int8_convrot.safetensors", "config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja"):
        assert (te_dir / name).exists()


def test_download_moss_te_bf16_huggingface_shards(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_SOURCE", "huggingface")
    hf, fake_hf = _recorder()
    monkeypatch.setattr("studio.services.models.sources.download_flat", fake_hf)
    assert model_downloader.download_moss_te(tmp_path, "bf16", on_log=lambda _l: None)
    assert {repo for repo, _, _ in hf} == {"OpenMOSS-Team/MOSS-Image-Base"}
    subpaths = [sub for _, sub, _ in hf]
    assert "text_encoder/model.safetensors.index.json" in subpaths
    assert "tokenizer/tokenizer.json" in subpaths
    from studio import secrets

    choices = moss_assets.path_choices(tmp_path, secrets.Secrets(models={"selected": {}}).models)
    assert [c["note"] for c in choices["text_encoder_path"]] == ["bf16"]
    assert not model_downloader.download_moss_te(tmp_path, "fp4", on_log=lambda _l: None)


def test_download_moss_vae_shared_flux2_target(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_SOURCE", "huggingface")
    hf, fake_hf = _recorder()
    monkeypatch.setattr("studio.services.models.sources.download_flat", fake_hf)
    assert model_downloader.download_moss_vae(tmp_path, on_log=lambda _l: None)
    assert hf == [("Comfy-Org/MOSS-Image", "vae/flux2-vae.safetensors",
                   str(tmp_path / "vae" / "flux2-vae.safetensors"))]


def test_trigger_routes_moss_assets(tmp_path, monkeypatch) -> None:
    captured: list[str] = []
    calls: list[tuple[str, str]] = []
    dl = "studio.services.models.downloader"
    monkeypatch.setattr(f"{dl}.models_root", lambda: tmp_path)
    monkeypatch.setattr(f"{dl}.start_download_async", lambda key, fn: (captured.append(key), fn(print)))
    monkeypatch.setattr(f"{dl}.download_moss_main", lambda root, v, *, on_log=print: calls.append(("main", v)) or True)
    monkeypatch.setattr(f"{dl}.download_moss_te", lambda root, v, *, on_log=print: calls.append(("te", v)) or True)
    monkeypatch.setattr(f"{dl}.download_moss_vae", lambda root, *, on_log=print: calls.append(("vae", "")) or True)

    assert model_downloader.trigger("moss_main") == "moss_main:base_int8"
    assert model_downloader.trigger("moss_main", "sft_bf16") == "moss_main:sft_bf16"
    assert model_downloader.trigger("moss_text_encoder_int8") == "moss_text_encoder_int8"
    assert model_downloader.trigger("moss_text_encoder_bf16") == "moss_text_encoder_bf16"
    assert model_downloader.trigger("moss_vae") == "moss_vae"
    assert calls == [("main", "base_int8"), ("main", "sft_bf16"), ("te", "int8"), ("te", "bf16"), ("vae", "")]
    with pytest.raises(ValueError, match="MOSS"):
        model_downloader.trigger("moss_main", "bogus")
    with pytest.raises(ValueError, match="MOSS"):
        model_downloader.trigger("moss_text_encoder_fp4")


def test_delete_moss_assets(tmp_path, monkeypatch) -> None:
    from studio.services.models import downloader as dl

    monkeypatch.setattr(dl, "models_root", lambda: tmp_path)
    main = moss_assets.moss_main_target(tmp_path, "base_bf16")
    main.parent.mkdir(parents=True)
    main.write_bytes(b"w")
    model_downloader.delete_asset("moss_main", "base_bf16")
    assert not main.exists()

    te_dir = moss_assets.moss_te_dir(tmp_path, "bf16")
    te_dir.mkdir(parents=True)
    (te_dir / "config.json").write_text("{}", encoding="utf-8")
    model_downloader.delete_asset("moss_text_encoder_bf16")
    assert not te_dir.exists()

    vae = tmp_path / "vae" / "flux2-vae.safetensors"
    vae.parent.mkdir(parents=True)
    vae.write_bytes(b"v")
    model_downloader.delete_asset("moss_vae")
    assert not vae.exists()

    with pytest.raises(ValueError, match="variant"):
        model_downloader.delete_asset("moss_main", "bogus")


@pytest.mark.parametrize("family,flagged", [("moss", False), ("anima", True), ("krea2", True)])
def test_catalog_quant_flag_respects_family_support(tmp_path, monkeypatch, family, flagged) -> None:
    f = tmp_path / "m.safetensors"
    f.write_bytes(b"x")
    monkeypatch.setattr(catalog_mod, "arch_summary", lambda _p: {"unsupported_quant": "int8_tensorwise + ConvRot"})
    arch = catalog_mod._family_arch(family, f)
    assert bool(arch and arch.get("unsupported_quant")) is flagged
    # 族不支持的格式（非 convrot 的 int8）对 MOSS 仍然标不可用
    monkeypatch.setattr(catalog_mod, "arch_summary", lambda _p: {"unsupported_quant": "int8_rowwise"})
    assert catalog_mod._family_arch(family, f)["unsupported_quant"] == "int8_rowwise"


def test_catalog_sections_carry_frontend_fields(tmp_path) -> None:
    from studio import secrets

    sections = moss_assets.catalog_sections(tmp_path, secrets.Secrets(models={"selected": {}}).models)
    for key in ("moss_main", "moss_text_encoder_int8", "moss_text_encoder_bf16", "moss_vae"):
        assert sections[key]["description"] and sections[key]["repo"], key
    assert sections["moss_text_encoder_int8"]["selected"] == "int8"
