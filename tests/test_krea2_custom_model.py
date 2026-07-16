"""Krea2 本地主模型注册、选择与回退。"""
from __future__ import annotations

from pathlib import Path

import pytest

from studio import secrets
from studio.api.routers.models import add_custom_model, remove_custom_model
from studio.api.schemas.models import AnimaCustomModelRequest
from studio.services import models as model_downloader


def _settings(root: Path, *, selected: str = "raw", custom: list[str] | None = None):
    return secrets.Secrets(models={
        "root": str(root),
        "selected": {"anima": "1.0", "krea2": selected},
        "custom": {"krea2": custom or []},
    })


def test_selected_krea2_transformer_accepts_existing_custom_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom = tmp_path / "custom-krea2.safetensors"
    custom.write_bytes(b"weights")
    monkeypatch.setattr(
        secrets, "load",
        lambda: _settings(tmp_path, selected=str(custom), custom=[str(custom)]),
    )

    assert model_downloader.selected_krea2_transformer_path() == str(custom)
    assert model_downloader.krea2_transformer_path_for(None) == str(custom)


def test_krea2_custom_endpoint_registers_dedupes_and_resets_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom = tmp_path / "custom-krea2.safetensors"
    custom.write_bytes(b"weights")
    state = {"settings": _settings(tmp_path)}
    monkeypatch.setattr(secrets, "load", lambda: state["settings"])
    monkeypatch.setattr(
        secrets, "save", lambda value: state.update(settings=value),
    )
    request = AnimaCustomModelRequest(path=str(custom))

    catalog = add_custom_model("krea2", request)
    add_custom_model("krea2", request)
    assert state["settings"].models.custom["krea2"] == [str(custom)]
    assert catalog["krea2_main"]["custom"][0]["path"] == str(custom)

    selected = state["settings"].models.model_copy(update={
        "selected": {"anima": "1.0", "krea2": str(custom)},
    })
    state["settings"] = state["settings"].model_copy(update={"models": selected})
    remove_custom_model("krea2", request)

    assert state["settings"].models.custom["krea2"] == []
    assert state["settings"].models.selected["krea2"] == "raw"
