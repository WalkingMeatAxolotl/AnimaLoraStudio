"""自定义本地主模型（custom Anima）：路径解析 + catalog 暴露 + 增删端点。

覆盖 feat：设置页 PathPicker 注册本地 .safetensors 主模型，驱动训练新建默认
+ 测试出图（在微调权重上炼丹 / 验证）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import secrets
from studio.services import models as model_downloader


def _secrets(tmp_path: Path, *, selected: str = "1.0", custom: list[str] | None = None):
    """构造一份 root 指到 tmp_path 的 Secrets（models_root() → tmp_path）。"""
    return secrets.Secrets(models={
        "root": str(tmp_path),
        "selected_anima": selected,
        "custom_anima_paths": custom or [],
    })


# ---------------------------------------------------------------------------
# selected_anima_transformer_path 解析
# ---------------------------------------------------------------------------


def test_resolver_uses_custom_path_when_selected_and_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "my-finetune.safetensors"
    custom.write_bytes(b"weights")
    monkeypatch.setattr(
        secrets, "load",
        lambda: _secrets(tmp_path, selected=str(custom), custom=[str(custom)]),
    )
    assert model_downloader.selected_anima_transformer_path() == str(custom)


def test_resolver_falls_back_to_variant_when_custom_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """选中的 custom 路径文件不存在（被删/移走）→ 回退到当前 variant，不返回死路径。"""
    ghost = tmp_path / "gone.safetensors"  # 不创建
    monkeypatch.setattr(
        secrets, "load",
        lambda: _secrets(tmp_path, selected=str(ghost), custom=[str(ghost)]),
    )
    resolved = model_downloader.selected_anima_transformer_path()
    expected = str(model_downloader.anima_main_target(tmp_path, "1.0"))
    assert resolved == expected


def test_resolver_uses_variant_target_for_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(secrets, "load", lambda: _secrets(tmp_path, selected="1.0"))
    assert model_downloader.selected_anima_transformer_path() == str(
        model_downloader.anima_main_target(tmp_path, "1.0")
    )


def test_default_paths_for_new_version_follows_custom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "ft.safetensors"
    custom.write_bytes(b"x")
    monkeypatch.setattr(
        secrets, "load",
        lambda: _secrets(tmp_path, selected=str(custom), custom=[str(custom)]),
    )
    paths = model_downloader.default_paths_for_new_version()
    assert paths["transformer_path"] == str(custom)
    # 其余三件套仍走标准位置（微调复用同一套 VAE/TE/T5）
    assert paths["vae_path"] == str(model_downloader.qwen_image_vae_target(tmp_path))


def test_generate_resolver_follows_selected_custom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """测试/出图页 deps._resolve_model_paths 跟随 selected_anima（含 custom），
    不再写死 v1.0。"""
    from studio.api import deps

    custom = tmp_path / "ft.safetensors"
    custom.write_bytes(b"x")
    monkeypatch.setattr(
        secrets, "load",
        lambda: _secrets(tmp_path, selected=str(custom), custom=[str(custom)]),
    )
    assert deps._resolve_model_paths()["transformer_path"] == str(custom)


# ---------------------------------------------------------------------------
# base_model 本次请求覆盖（先验生成 / 测试出图页「底模」下拉）
# ---------------------------------------------------------------------------


def test_base_model_override_picks_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """override 指定非默认官方 variant → 只换 transformer，无视 selected。"""
    monkeypatch.setattr(secrets, "load", lambda: _secrets(tmp_path, selected="1.0"))
    paths = model_downloader.default_paths_for_new_version("preview3-base")
    assert paths["transformer_path"] == str(
        model_downloader.anima_main_target(tmp_path, "preview3-base")
    )
    # 其余三件套仍跟随全局，不受 override 影响
    assert paths["vae_path"] == str(model_downloader.qwen_image_vae_target(tmp_path))


def test_base_model_override_picks_custom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "ft.safetensors"
    custom.write_bytes(b"x")
    monkeypatch.setattr(secrets, "load", lambda: _secrets(tmp_path, selected="1.0"))
    paths = model_downloader.default_paths_for_new_version(str(custom))
    assert paths["transformer_path"] == str(custom)


def test_base_model_override_none_follows_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """None / 空 → 回退 selected_anima（保持原有「跟随设置」行为）。"""
    monkeypatch.setattr(
        secrets, "load", lambda: _secrets(tmp_path, selected="preview2")
    )
    expected = str(model_downloader.anima_main_target(tmp_path, "preview2"))
    assert model_downloader.anima_transformer_path_for(None) == expected
    assert model_downloader.anima_transformer_path_for("") == expected
    assert model_downloader.default_paths_for_new_version()["transformer_path"] == expected


def test_base_model_override_missing_custom_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """override 给了不存在的 custom 路径 → 回退 selected，不返回死路径。"""
    ghost = tmp_path / "gone.safetensors"  # 不创建
    monkeypatch.setattr(secrets, "load", lambda: _secrets(tmp_path, selected="1.0"))
    resolved = model_downloader.anima_transformer_path_for(str(ghost))
    assert resolved == str(model_downloader.anima_main_target(tmp_path, "1.0"))


def test_resolve_model_paths_threads_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """deps._resolve_model_paths(base_model) 透传到 transformer_path。"""
    from studio.api import deps

    monkeypatch.setattr(secrets, "load", lambda: _secrets(tmp_path, selected="1.0"))
    got = deps._resolve_model_paths("preview3-base")["transformer_path"]
    assert got == str(model_downloader.anima_main_target(tmp_path, "preview3-base"))


# ---------------------------------------------------------------------------
# catalog 暴露 custom 列表
# ---------------------------------------------------------------------------


def test_build_catalog_exposes_custom_anima(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = tmp_path / "my-finetune.safetensors"
    custom.write_bytes(b"x" * 2048)
    monkeypatch.setattr(
        secrets, "load",
        lambda: _secrets(tmp_path, selected=str(custom), custom=[str(custom)]),
    )
    cat = model_downloader.build_catalog(tmp_path)
    anima = cat["anima_main"]
    assert anima["selected"] == str(custom)
    entry = next(c for c in anima["custom"] if c["path"] == str(custom))
    assert entry["name"] == "my-finetune.safetensors"
    assert entry["exists"] is True
    assert entry["size"] == 2048


def test_build_catalog_custom_marks_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ghost = tmp_path / "gone.safetensors"
    monkeypatch.setattr(
        secrets, "load",
        lambda: _secrets(tmp_path, custom=[str(ghost)]),
    )
    cat = model_downloader.build_catalog(tmp_path)
    entry = next(c for c in cat["anima_main"]["custom"] if c["path"] == str(ghost))
    assert entry["exists"] is False
    assert entry["size"] == 0


# ---------------------------------------------------------------------------
# 增删端点
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """内存模拟 secrets 持久化：load 返回当前值，save 落回内存。"""
    state = {"s": _secrets(tmp_path)}
    monkeypatch.setattr(secrets, "load", lambda: state["s"])
    monkeypatch.setattr(secrets, "save", lambda s: state.update(s=s))
    return state


def test_add_custom_anima_registers_and_dedupes(
    tmp_path: Path, fake_store
) -> None:
    from studio.api.routers.models import add_model_source
    from studio.api.schemas.models import ModelSourceCandidateRequest

    f = tmp_path / "ft.safetensors"
    f.write_bytes(b"x")
    cat = add_model_source(
        "anima", ModelSourceCandidateRequest(kind="local", path=str(f)))
    assert fake_store["s"].models.custom_anima_paths == [str(f)]
    assert any(c["path"] == str(f) for c in cat["anima_main"]["custom"])

    # 重复添加不产生第二条
    add_model_source(
        "anima", ModelSourceCandidateRequest(kind="local", path=str(f)))
    assert fake_store["s"].models.custom_anima_paths == [str(f)]


def test_add_custom_anima_rejects_bad_ext(tmp_path: Path, fake_store) -> None:
    from studio.api.routers.models import add_model_source
    from studio.api.schemas.models import ModelSourceCandidateRequest
    from studio.domain.errors import ValidationError

    bad = tmp_path / "evil.txt"
    bad.write_bytes(b"x")
    with pytest.raises(ValidationError) as exc:
        add_model_source(
            "anima", ModelSourceCandidateRequest(kind="local", path=str(bad)))
    assert exc.value.code == "file.ext_invalid"


def test_add_custom_anima_rejects_missing_file(tmp_path: Path, fake_store) -> None:
    from studio.api.routers.models import add_model_source
    from studio.api.schemas.models import ModelSourceCandidateRequest
    from studio.domain.errors import ValidationError

    with pytest.raises(ValidationError) as exc:
        add_model_source("anima", ModelSourceCandidateRequest(
            kind="local", path=str(tmp_path / "ghost.safetensors")))
    assert exc.value.code == "model.not_found"


def test_remove_custom_anima_resets_selected_when_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.api.routers.models import remove_model_source
    from studio.api.schemas.models import ModelSourceCandidateRequest

    a = str(tmp_path / "a.safetensors")
    b = str(tmp_path / "b.safetensors")
    state = {"s": _secrets(tmp_path, selected=a, custom=[a, b])}
    monkeypatch.setattr(secrets, "load", lambda: state["s"])
    monkeypatch.setattr(secrets, "save", lambda s: state.update(s=s))

    remove_model_source(
        "anima", ModelSourceCandidateRequest(kind="local", path=a))
    assert state["s"].models.custom_anima_paths == [b]
    # 删的是当前默认 → 重置回最新官方 variant
    assert state["s"].models.selected_anima == model_downloader.LATEST_ANIMA


def test_remove_custom_anima_keeps_selected_when_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studio.api.routers.models import remove_model_source
    from studio.api.schemas.models import ModelSourceCandidateRequest

    a = str(tmp_path / "a.safetensors")
    b = str(tmp_path / "b.safetensors")
    state = {"s": _secrets(tmp_path, selected=a, custom=[a, b])}
    monkeypatch.setattr(secrets, "load", lambda: state["s"])
    monkeypatch.setattr(secrets, "save", lambda s: state.update(s=s))

    remove_model_source(
        "anima", ModelSourceCandidateRequest(kind="local", path=b))
    assert state["s"].models.custom_anima_paths == [a]
    assert state["s"].models.selected_anima == a  # 未动当前默认
