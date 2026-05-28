"""CLTagger tagger：mock onnx + mapping 解析、阈值过滤。"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from studio import secrets
from studio.services.tagging import cltagger as cltagger_tagger


@pytest.fixture
def isolated_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sf = tmp_path / "secrets.json"
    monkeypatch.setattr(secrets, "SECRETS_FILE", sf)
    return tmp_path


def _make_local_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.onnx").write_bytes(b"fake-onnx")
    (model_dir / "tag_mapping.json").write_text(
        json.dumps({
            "idx_to_tag": {
                "0": "general_tag",
                "1": "hero_name",
                "2": "explicit",
                "3": "model_tag",
            },
            "tag_to_category": {
                "general_tag": "General",
                "hero_name": "Character",
                "explicit": "Rating",
                "model_tag": "Model",
            },
        }),
        encoding="utf-8",
    )


def test_resolve_local_dir_flat_files(isolated_secrets: Path) -> None:
    model_dir = isolated_secrets / "cl"
    _make_local_model(model_dir)
    secrets.update({"cltagger": {"local_dir": str(model_dir)}})
    t = cltagger_tagger.CLTagger()
    model_path, mapping_path = t._resolve_model_files()
    assert model_path == model_dir / "model.onnx"
    assert mapping_path == model_dir / "tag_mapping.json"


def test_is_available_does_not_download_when_model_missing(
    isolated_secrets: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = {"download": False}

    def _boom(*args, **kwargs):
        called["download"] = True
        raise AssertionError("download should not run in is_available")

    monkeypatch.setattr(cltagger_tagger.model_downloader, "download_cltagger", _boom)
    t = cltagger_tagger.CLTagger()
    ok, msg = t.is_available()
    assert ok is False
    assert "需下载模型" in msg
    assert called["download"] is False


def test_postprocess_uses_character_threshold_and_optional_categories(
    isolated_secrets: Path,
) -> None:
    secrets.update({
        "cltagger": {
            "threshold_general": 0.5,
            "threshold_character": 0.7,
            "add_rating_tag": False,
            "add_model_tag": False,
        }
    })
    t = cltagger_tagger.CLTagger()
    t._labels = cltagger_tagger._LabelData(
        names=["general_tag", "hero_name", "explicit", "model_tag"],
        categories=["General", "Character", "Rating", "Model"],
    )
    logits = np.array([2.0, 0.0, 4.0, 4.0], dtype=np.float32)
    tags, raw = t._postprocess_one(logits)
    assert tags == ["general tag"]
    assert raw["general tag"] > 0.5

    secrets.update({"cltagger": {"add_rating_tag": True, "add_model_tag": True}})
    tags, _ = t._postprocess_one(logits)
    assert tags[:2] == ["explicit", "model tag"]


def test_tag_iterator_runs_onnx_batch(isolated_secrets: Path, tmp_path: Path) -> None:
    secrets.update({"cltagger": {"threshold_general": 0.1, "batch_size": 2}})
    t = cltagger_tagger.CLTagger()
    t._session = MagicMock()
    t._session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    t._session.run.return_value = (np.array([[2.0], [3.0]], dtype=np.float32),)
    t._input_name = "input"
    t._output_names = ["output"]
    t._input_size = 4
    t._labels = cltagger_tagger._LabelData(names=["x"], categories=["General"])

    paths = []
    for i in range(2):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8)).save(p)
        paths.append(p)

    results = list(t.tag(paths))
    assert [r["tags"] for r in results] == [["x"], ["x"]]
    fed = t._session.run.call_args[0][1]["input"]
    assert fed.shape == (2, 3, 4, 4)


def test_load_tag_mapping_supports_inline_object_schema(tmp_path: Path) -> None:
    """支持 {idx: {tag, category}} 这种内联 schema —— cella110n 老版本 mapping 格式。"""
    mapping_path = tmp_path / "tag_mapping.json"
    mapping_path.write_text(
        json.dumps({
            "0": {"tag": "general_tag", "category": "General"},
            "1": {"tag": "hero_name", "category": "Character"},
            "3": {"tag": "explicit"},  # category 缺省 → "General"
        }),
        encoding="utf-8",
    )
    labels = cltagger_tagger.CLTagger._load_tag_mapping(mapping_path)
    assert labels.names == ["general_tag", "hero_name", None, "explicit"]
    assert labels.categories == ["General", "Character", "General", "General"]


def test_preprocess_supports_nhwc_layout(isolated_secrets: Path) -> None:
    t = cltagger_tagger.CLTagger()
    t._input_size = 4
    t._input_layout = "nhwc"
    arr = t._preprocess(Image.new("RGB", (8, 8), (255, 0, 0)))
    assert arr.shape == (4, 4, 3)
    assert arr[0, 0, 0] == pytest.approx(-1.0, abs=1e-4)  # B
    assert arr[0, 0, 2] == pytest.approx(1.0, abs=1e-4)   # R
