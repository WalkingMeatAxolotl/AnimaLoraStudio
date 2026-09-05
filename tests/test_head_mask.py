from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from studio.domain.errors import ConflictError
from studio.infrastructure import paths as studio_paths
from studio.services.preprocess import head_mask
from studio.services.preprocess import masks as train_masks


def _write_image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path, "PNG")


def _proposal(
    name: str,
    path: Path,
    regions: list[dict] | None = None,
) -> dict:
    size = Image.open(path).size
    return head_mask.make_image_proposal(
        name,
        path,
        size,
        regions or [{"score": 0.9, "box": [20, 10, 50, 40]}],
        padding_ratio=0.1,
        feather_ratio=0.03,
    )


def test_decode_letterbox_restores_source_coordinates() -> None:
    # Source 320x160 -> 640x320 plus 160px top padding. Raw xywh represents
    # source box [50,20,150,100] after 2x scale + letterbox.
    output = np.array([[[200.0], [280.0], [200.0], [160.0], [0.9]]])
    got = head_mask.decode_output(
        output,
        confidence=0.413,
        iou_threshold=0.7,
        scale=2.0,
        pad_left=0,
        pad_top=160,
        source_size=(320, 160),
    )
    assert got[0]["box"] == [50.0, 20.0, 150.0, 100.0]


def test_decode_nms_keeps_multiple_people_and_clips_bounds() -> None:
    # (1, 5, N): two overlapping boxes + one separate box.
    output = np.array([[
        [50, 52, 120], [50, 52, 120], [40, 40, 30], [40, 40, 30],
        [0.95, 0.90, 0.80],
    ]], dtype=np.float32)
    got = head_mask.decode_output(
        output,
        confidence=0.4,
        iou_threshold=0.5,
        scale=1.0,
        pad_left=0,
        pad_top=0,
        source_size=(128, 128),
    )
    assert len(got) == 2
    assert all(0 <= value <= 128 for row in got for value in row["box"])


def test_decode_zero_detections() -> None:
    output = np.array([[[10], [10], [4], [4], [0.1]]], dtype=np.float32)
    assert head_mask.decode_output(
        output,
        confidence=0.413,
        iou_threshold=0.7,
        scale=1.0,
        pad_left=0,
        pad_top=0,
        source_size=(100, 100),
    ) == []


def test_expand_and_feather_mask() -> None:
    region = head_mask.expand_detection(
        {"score": 0.8, "box": [20, 20, 40, 40]},
        (100, 100), padding_ratio=0.1, feather_ratio=0.1,
    )
    assert region["mask_region"] == {
        "x1": 18, "y1": 18, "x2": 42, "y2": 42,
        "feather_x": 2, "feather_y": 2,
    }
    mask = head_mask.render_auto_mask((100, 100), [region])
    assert mask[20, 20] == 0
    assert 0 < mask[17, 20] < 255
    assert mask[0, 0] == 255


def test_multiple_regions_merge_by_minimum() -> None:
    regions = [
        {"mask_region": {"x1": 5, "y1": 5, "x2": 15, "y2": 15,
                         "feather_x": 0, "feather_y": 0}},
        {"mask_region": {"x1": 20, "y1": 20, "x2": 30, "y2": 30,
                         "feather_x": 0, "feather_y": 0}},
    ]
    mask = head_mask.render_auto_mask((40, 40), regions)
    assert mask[10, 10] == 0
    assert mask[25, 25] == 0
    assert mask[18, 18] == 255


@pytest.fixture
def apply_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    train = tmp_path / "train"
    image = train / "1_data" / "A.png"
    _write_image(image)
    monkeypatch.setattr(studio_paths, "TASKS_DIR", tmp_path / "tasks")
    proposal = _proposal("1_data/A.png", image)
    result = head_mask.new_result(
        7,
        confidence=0.413,
        iou_threshold=0.7,
        padding_ratio=0.1,
        feather_ratio=0.03,
        provider="CPUExecutionProvider",
        images=[proposal],
    )
    head_mask.write_result(7, result)
    return {"train": train, "image": image, "proposal": proposal, "result": result}


def test_apply_unions_manual_mask_and_undo_restores(apply_env) -> None:
    train = apply_env["train"]
    mask_path = train_masks.mask_path_for(train, "1_data/A.png")
    manual = np.full((80, 100), 255, dtype=np.uint8)
    manual[60:70, 70:90] = 0
    Image.fromarray(manual, mode="L").save(mask_path, "PNG")
    before = mask_path.read_bytes()
    region_id = apply_env["proposal"]["regions"][0]["id"]

    applied = head_mask.apply_proposals(
        7, train, {"1_data/A.png": [region_id]},
    )
    assert applied["applied"] == 1
    with Image.open(mask_path) as raw:
        mask = np.asarray(raw)
    assert mask[20, 30] == 0  # auto head
    assert mask[65, 80] == 0  # existing manual area survived

    undone = head_mask.undo_apply(7, train)
    assert undone["undone"] == 1
    assert mask_path.read_bytes() == before


def test_apply_refuses_stale_proposal(apply_env) -> None:
    image = apply_env["image"]
    os.utime(image, ns=(image.stat().st_atime_ns, image.stat().st_mtime_ns + 1_000_000))
    region_id = apply_env["proposal"]["regions"][0]["id"]
    with pytest.raises(ConflictError) as exc:
        head_mask.apply_proposals(
            7, apply_env["train"], {"1_data/A.png": [region_id]},
        )
    assert exc.value.code == "preprocess.head_mask_proposals_stale"


def test_undo_refuses_subsequent_manual_edit(apply_env) -> None:
    train = apply_env["train"]
    region_id = apply_env["proposal"]["regions"][0]["id"]
    head_mask.apply_proposals(7, train, {"1_data/A.png": [region_id]})
    mask_path = train_masks.mask_path_for(train, "1_data/A.png")
    with Image.open(mask_path) as raw:
        changed = raw.convert("L")
    changed.putpixel((99, 79), 0)
    changed.save(mask_path, "PNG")
    with pytest.raises(ConflictError) as exc:
        head_mask.undo_apply(7, train)
    assert exc.value.code == "preprocess.head_mask_undo_modified"


def test_batch_replace_failure_rolls_back_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train = tmp_path / "train"
    monkeypatch.setattr(studio_paths, "TASKS_DIR", tmp_path / "tasks")
    images = []
    selections = {}
    before = {}
    for name, value in (("A.png", 180), ("B.png", 200)):
        image = train / "1_data" / name
        _write_image(image)
        proposal = _proposal(f"1_data/{name}", image)
        images.append(proposal)
        selections[f"1_data/{name}"] = [proposal["regions"][0]["id"]]
        mask_path = train_masks.mask_path_for(train, f"1_data/{name}")
        Image.new("L", (100, 80), value).save(mask_path, "PNG")
        before[name] = mask_path.read_bytes()
    head_mask.write_result(9, head_mask.new_result(
        9, confidence=0.413, iou_threshold=0.7,
        padding_ratio=0.1, feather_ratio=0.03,
        provider="CPUExecutionProvider", images=images,
    ))
    real_replace = head_mask.os.replace
    commits = 0

    def fail_second_mask_commit(src, dst):
        nonlocal commits
        if str(dst).endswith(".mask") and "staging" in str(src):
            commits += 1
            if commits == 2:
                raise OSError("injected failure")
        return real_replace(src, dst)

    monkeypatch.setattr(head_mask.os, "replace", fail_second_mask_commit)
    with pytest.raises(OSError, match="injected failure"):
        head_mask.apply_proposals(9, train, selections)
    assert (train / "1_data" / "A.mask").read_bytes() == before["A.png"]
    assert (train / "1_data" / "B.mask").read_bytes() == before["B.png"]


def test_result_reports_stale_without_mutation(apply_env) -> None:
    fresh = head_mask.result_with_staleness(apply_env["result"], apply_env["train"])
    assert fresh["stale_count"] == 0
    apply_env["image"].unlink()
    stale = head_mask.result_with_staleness(apply_env["result"], apply_env["train"])
    assert stale["images"][0]["stale_reason"] == "missing"


def test_apply_state_is_json_reviewable(apply_env) -> None:
    region_id = apply_env["proposal"]["regions"][0]["id"]
    head_mask.apply_proposals(
        7, apply_env["train"], {"1_data/A.png": [region_id]},
    )
    state = json.loads(head_mask.apply_state_path(7).read_text(encoding="utf-8"))
    assert state["records"][0]["selected_region_ids"] == [region_id]


def test_session_creation_falls_back_from_directml_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class Session:
        def __init__(self, _path: str, *, providers: list[str]) -> None:
            calls.append(providers)
            if providers[0] == "DmlExecutionProvider":
                raise RuntimeError("DML unavailable")
            self._providers = providers

        def get_inputs(self):
            return [types.SimpleNamespace(name="images")]

        def get_providers(self):
            return self._providers

    fake = types.SimpleNamespace(
        get_available_providers=lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=Session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    detector = head_mask.HeadDetector(tmp_path / "model.onnx")
    assert calls == [
        ["DmlExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    assert detector.provider == "CPUExecutionProvider"


def test_cuda_inference_failure_recreates_cpu_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class Session:
        def __init__(self, _path: str, *, providers: list[str]) -> None:
            calls.append(providers)
            self._providers = providers

        def get_inputs(self):
            return [types.SimpleNamespace(name="images")]

        def get_providers(self):
            return self._providers

        def run(self, _outputs, _inputs):
            if self._providers[0] == "CUDAExecutionProvider":
                raise RuntimeError("CUDA OOM")
            return [np.zeros((1, 5, 1), dtype=np.float32)]

    fake = types.SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        InferenceSession=Session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    detector = head_mask.HeadDetector(tmp_path / "model.onnx")
    output = detector.run(np.zeros((1, 3, 640, 640), dtype=np.float32))
    assert output.shape == (1, 5, 1)
    assert calls == [
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CPUExecutionProvider"],
    ]
    assert detector.provider == "CPUExecutionProvider"
