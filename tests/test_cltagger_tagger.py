"""CLTagger tagger：mock onnx + mapping 解析、阈值过滤。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from studio import secrets
from studio.services import models as model_downloader
from studio.services.tagging import cltagger as cltagger_tagger


@pytest.fixture
def isolated_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    sf = tmp_path / "secrets.json"
    monkeypatch.setattr(secrets, "SECRETS_FILE", sf)
    # 隔离 models_root：默认回退到 REPO_ROOT/models，dev 机上可能真有
    # cl_tagger_1_02，导致 is_available 误判 True。
    monkeypatch.setattr(model_downloader, "models_root", lambda: tmp_path / "models")
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


def test_v2_legacy_root_paths_resolve_versioned_files(isolated_secrets: Path) -> None:
    base = model_downloader.cltagger_target_root(
        model_downloader.models_root(),
        "cella110n/cl_tagger_v2",
    )
    version_dir = base / "v2_01a"
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "model.onnx").write_bytes(b"fake-onnx")
    (version_dir / "model.onnx.data").write_bytes(b"fake-weights")
    (version_dir / "model_metadata.json").write_text("{}", encoding="utf-8")
    (version_dir / "model_vocabulary.json").write_text(
        json.dumps({"idx_to_tag": {"0": "1girl"}}),
        encoding="utf-8",
    )
    secrets.update({
        "cltagger": {
            "model_id": "cella110n/cl_tagger_v2",
            "model_path": "model.onnx",
            "tag_mapping_path": "model_vocabulary.json",
        }
    })

    t = cltagger_tagger.CLTagger()
    model_path, mapping_path, ok = t._local_model_files_status()

    assert ok
    assert model_path == version_dir / "model.onnx"
    assert mapping_path == version_dir / "model_vocabulary.json"


def test_v2_missing_external_data_reports_not_ready(isolated_secrets: Path) -> None:
    """v2 权重在 model.onnx.data sidecar 里；缺它必须报"未就绪"，

    否则 onnx 图就绪会让 is_available 误报"可用"，等到 prepare 加载 external
    data 时才在 onnxruntime 层黑盒炸。
    """
    base = model_downloader.cltagger_target_root(
        model_downloader.models_root(),
        "cella110n/cl_tagger_v2",
    )
    version_dir = base / "v2_01a"
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "model.onnx").write_bytes(b"fake-onnx")
    (version_dir / "model_vocabulary.json").write_text(
        json.dumps({"idx_to_tag": {"0": "1girl"}}),
        encoding="utf-8",
    )
    # 故意不写 model.onnx.data —— 模拟部分下载
    secrets.update({
        "cltagger": {
            "model_id": "cella110n/cl_tagger_v2",
            "model_path": "v2_01a/model.onnx",
            "tag_mapping_path": "v2_01a/model_vocabulary.json",
        }
    })

    t = cltagger_tagger.CLTagger()
    _, _, ok = t._local_model_files_status()
    available, _ = t.is_available()

    assert ok is False
    assert available is False


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


def test_postprocess_default_gates_copyright_on_meta_quality_off(
    isolated_secrets: Path,
) -> None:
    """默认勾 General / Character / Copyright 三类；Artist / Meta / Quality / Model /
    Rating 默认关。保证老 secrets 升级后用户能直接拿到"干净"的 caption。"""
    secrets.update({"cltagger": {"threshold_general": 0.1, "threshold_character": 0.1}})
    t = cltagger_tagger.CLTagger()
    t._labels = cltagger_tagger._LabelData(
        names=["1girl", "hero_name", "fate_series", "some_artist", "highres", "best_quality", "model_tag", "explicit"],
        categories=["General", "Character", "Copyright", "Artist", "Meta", "Quality", "Model", "Rating"],
    )
    logits = np.array([4.0] * 8, dtype=np.float32)  # sigmoid≈0.98，全部超阈值
    tags, _ = t._postprocess_one(logits)
    assert set(tags) == {"1girl", "hero name", "fate series"}


def test_postprocess_artist_gate_can_be_enabled(isolated_secrets: Path) -> None:
    secrets.update({
        "cltagger": {
            "threshold_general": 0.1,
            "threshold_character": 0.1,
            "add_artist_tag": True,
        }
    })
    t = cltagger_tagger.CLTagger()
    t._labels = cltagger_tagger._LabelData(
        names=["1girl", "some_artist"],
        categories=["General", "Artist"],
    )
    logits = np.array([4.0, 4.0], dtype=np.float32)
    tags, _ = t._postprocess_one(logits)
    assert set(tags) == {"1girl", "some artist"}


def test_postprocess_meta_and_quality_gates_can_be_enabled(
    isolated_secrets: Path,
) -> None:
    secrets.update({
        "cltagger": {
            "threshold_general": 0.1,
            "threshold_character": 0.1,
            "add_meta_tag": True,
            "add_quality_tag": True,
            "add_copyright_tag": False,
        }
    })
    t = cltagger_tagger.CLTagger()
    t._labels = cltagger_tagger._LabelData(
        names=["1girl", "fate_series", "highres", "best_quality"],
        categories=["General", "Copyright", "Meta", "Quality"],
    )
    logits = np.array([4.0] * 4, dtype=np.float32)
    tags, _ = t._postprocess_one(logits)
    assert set(tags) == {"1girl", "highres", "best quality"}


def test_postprocess_blacklist_underscore_and_case_insensitive(isolated_secrets: Path) -> None:
    """cltagger 的 tag 是下划线形式；blacklist 填空格 'cat girl' / 大写
    'BLUE_EYES' 也能屏蔽（_/空格、大小写不敏感，与 wd14 一致）。"""
    secrets.update({
        "cltagger": {
            "threshold_general": 0.1, "threshold_character": 0.1,
            "add_rating_tag": False, "add_model_tag": False,
            "blacklist_tags": ["cat girl", "BLUE_EYES"],
        }
    })
    t = cltagger_tagger.CLTagger()
    t._labels = cltagger_tagger._LabelData(
        names=["cat_girl", "blue_eyes", "1girl"],
        categories=["General", "General", "General"],
    )
    logits = np.array([4.0, 4.0, 4.0], dtype=np.float32)  # sigmoid≈0.98 > 0.1
    tags, _ = t._postprocess_one(logits)
    assert tags == ["1girl"]


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


def test_load_model_vocabulary_maps_numeric_category_aliases(tmp_path: Path) -> None:
    """CLTagger v2 model_vocabulary.json may use numeric category ids."""
    mapping_path = tmp_path / "model_vocabulary.json"
    mapping_path.write_text(
        json.dumps({
            "idx_to_tag": {
                "2": "touhou",
                "0": "1girl",
                "1": "hakurei_reimu",
            },
            "tag_to_category": {
                "1girl": "0",
                "hakurei_reimu": "4",
                "touhou": "Copyright",
            },
        }),
        encoding="utf-8",
    )

    labels = cltagger_tagger.CLTagger._load_tag_mapping(mapping_path)

    assert labels.names == ["1girl", "hakurei_reimu", "touhou"]
    assert labels.categories == ["General", "Character", "Copyright"]


def test_load_model_vocabulary_supports_tag_to_idx_fallback(tmp_path: Path) -> None:
    """Some v2 vocabularies expose tag_to_idx instead of idx_to_tag."""
    mapping_path = tmp_path / "model_vocabulary.json"
    mapping_path.write_text(
        json.dumps({
            "tag_to_idx": {
                "hakurei_reimu": 1,
                "1girl": 0,
            },
            "tag_to_category": {
                "1girl": "General",
                "hakurei_reimu": "Character",
            },
        }),
        encoding="utf-8",
    )

    labels = cltagger_tagger.CLTagger._load_tag_mapping(mapping_path)

    assert labels.names == ["1girl", "hakurei_reimu"]
    assert labels.categories == ["General", "Character"]


def test_preprocess_supports_nhwc_layout(isolated_secrets: Path) -> None:
    t = cltagger_tagger.CLTagger()
    t._input_size = 4
    t._input_layout = "nhwc"
    arr = t._preprocess(Image.new("RGB", (8, 8), (255, 0, 0)))
    assert arr.shape == (4, 4, 3)
    assert arr[0, 0, 0] == pytest.approx(-1.0, abs=1e-4)  # B
    assert arr[0, 0, 2] == pytest.approx(1.0, abs=1e-4)   # R


def test_v2_preprocess_uses_rgb_chw_minus_one_to_one(isolated_secrets: Path) -> None:
    secrets.update({
        "cltagger": {
            "model_path": "cl_tagger_v2/v2_01a/model.onnx",
            "tag_mapping_path": "cl_tagger_v2/v2_01a/model_vocabulary.json",
        }
    })
    t = cltagger_tagger.CLTagger()
    t._input_size = 4
    t._input_layout = "nchw"

    arr = t._preprocess(Image.new("RGB", (8, 8), (255, 0, 0)))

    assert arr.shape == (3, 4, 4)
    assert arr[0, 0, 0] == pytest.approx(1.0, abs=1e-4)    # R
    assert arr[1, 0, 0] == pytest.approx(-1.0, abs=1e-4)   # G
    assert arr[2, 0, 0] == pytest.approx(-1.0, abs=1e-4)   # B


def test_prepare_v2_prefers_pixel_values_input_and_logits_output(
    isolated_secrets: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = model_downloader.cltagger_target_root(
        model_downloader.models_root(),
        "cella110n/cl_tagger_v2",
    )
    version_dir = base / "v2_01a"
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "model.onnx").write_bytes(b"fake-onnx")
    (version_dir / "model.onnx.data").write_bytes(b"fake-weights")
    (version_dir / "model_metadata.json").write_text("{}", encoding="utf-8")
    (version_dir / "model_vocabulary.json").write_text(
        json.dumps({"idx_to_tag": {"0": "1girl"}}),
        encoding="utf-8",
    )
    secrets.update({
        "cltagger": {
            "model_id": "cella110n/cl_tagger_v2",
            "model_path": "model.onnx",
            "tag_mapping_path": "model_vocabulary.json",
        }
    })

    def _fake_create_session(self: cltagger_tagger.CLTagger, model_path: Path) -> None:
        self._model_path = model_path
        self._input_name = "image"
        self._output_names = ["probabilities"]
        self._session = SimpleNamespace(
            get_inputs=lambda: [
                SimpleNamespace(name="image", shape=[1, 3, 448, 448]),
                SimpleNamespace(name="pixel_values", shape=[1, 3, 384, 384]),
            ],
            get_outputs=lambda: [
                SimpleNamespace(name="probabilities"),
                SimpleNamespace(name="logits"),
            ],
        )

    monkeypatch.setattr(cltagger_tagger.CLTagger, "_create_session", _fake_create_session)
    t = cltagger_tagger.CLTagger()

    t.prepare()

    assert t._input_name == "pixel_values"
    assert t._output_names == ["logits"]
    assert t._input_size == 384
