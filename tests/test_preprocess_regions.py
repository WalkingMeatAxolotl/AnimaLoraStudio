from __future__ import annotations

import json

import pytest

from studio.domain.errors import ValidationError
from studio.services.preprocess import regions
from studio.services.data_io import train_io


def _doc() -> dict:
    return {
        "version": 1,
        "image_size": {"w": 100, "h": 80},
        "regions": [{
            "id": "primary",
            "label": "face",
            "class_word": "1girl",
            "caption": "blue eyes",
            "weight": 1.0,
            "box": {"x": 0.2, "y": 0.25, "w": 0.4, "h": 0.5},
        }],
    }


def test_region_sidecar_write_read_and_delete(tmp_path) -> None:
    result = regions.write_region(tmp_path, "1_data/a.png", _doc(), expected_size=(100, 80))
    assert result["regions"][0]["box"]["x"] == 0.2
    path = tmp_path / "1_data" / "a.regions.json"
    assert json.loads(path.read_text(encoding="utf-8"))["image_size"] == {"w": 100, "h": 80}
    assert regions.read_region(tmp_path, "1_data/a.png")["regions"][0]["class_word"] == "1girl"
    assert regions.delete_region(tmp_path, "1_data/a.png") is True
    assert regions.read_region(tmp_path, "1_data/a.png") is None


def test_region_validation_rejects_invalid_version_and_box(tmp_path) -> None:
    bad = _doc()
    bad["version"] = "invalid"
    with pytest.raises(ValidationError):
        regions.write_region(tmp_path, "a.png", bad, expected_size=(100, 80))
    bad = _doc()
    bad["regions"][0]["box"]["w"] = 0.9
    with pytest.raises(ValidationError):
        regions.write_region(tmp_path, "a.png", bad, expected_size=(100, 80))


def test_region_reader_accepts_windows_utf8_bom(tmp_path) -> None:
    path = regions.region_path_for(tmp_path, "a.png")
    path.write_text(json.dumps(_doc()), encoding="utf-8-sig")
    assert regions.read_region(tmp_path, "a.png") is not None


def test_crop_intersects_and_renormalizes_region(tmp_path) -> None:
    regions.write_region(tmp_path, "a.png", _doc(), expected_size=(100, 80))
    regions.crop_region_like(
        tmp_path,
        "a.png",
        boxes=[(20, 20, 60, 60)],
        out_rels=["crop.png"],
    )
    cropped = regions.read_region(tmp_path, "crop.png")
    assert cropped is not None
    box = cropped["regions"][0]["box"]
    assert box == pytest.approx({"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0})
    assert cropped["image_size"] == {"w": 40, "h": 40}
    assert regions.read_region(tmp_path, "a.png") is None


def test_bundle_collector_includes_regions_only_when_requested(tmp_path) -> None:
    folder = tmp_path / "1_data"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"image")
    (folder / "a.regions.json").write_text(json.dumps(_doc()), encoding="utf-8")

    payload, stats = train_io._collect_train(
        tmp_path, include_captions=True, include_regions=True,
    )
    assert stats["region_count"] == 1
    assert "train/1_data/a.regions.json" in {arc for _, arc in payload}

    payload_without, stats_without = train_io._collect_train(
        tmp_path, include_captions=True,
    )
    assert stats_without["region_count"] == 0
    assert not any(arc.endswith(".regions.json") for _, arc in payload_without)
