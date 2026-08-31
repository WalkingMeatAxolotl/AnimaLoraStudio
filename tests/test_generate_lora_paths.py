from pathlib import Path

import pytest
from fastapi import HTTPException

from studio.api.routers.generate import _validate_lora_files
from studio.api.schemas.generate import GenerateRequest


def _request(*, loras: list[str], xy_values: list[str] | None = None) -> GenerateRequest:
    return GenerateRequest(
        lora_configs=[{"path": path, "scale": 1.0} for path in loras],
        xy_matrix=(
            {
                "x": {"axis": "lora_ckpt", "values": xy_values, "lora_index": 0},
                "y": None,
            }
            if xy_values is not None else None
        ),
    )


def test_validate_lora_files_accepts_existing_static_and_xy_paths(tmp_path: Path) -> None:
    anchor = tmp_path / "anchor.safetensors"
    epoch = tmp_path / "epoch8.safetensors"
    anchor.write_bytes(b"anchor")
    epoch.write_bytes(b"epoch")

    _validate_lora_files(_request(loras=[str(anchor)], xy_values=[str(epoch)]))


def test_validate_lora_files_rejects_snapshot_basename_before_task_creation() -> None:
    request = _request(
        loras=["anchor.safetensors"],
        xy_values=["epoch8.safetensors", "epoch12.safetensors"],
    )

    with pytest.raises(HTTPException) as caught:
        _validate_lora_files(request)

    assert caught.value.status_code == 422
    assert "尚未解析为绝对路径" in str(caught.value.detail)
    assert "epoch8.safetensors" in str(caught.value.detail)


def test_validate_lora_files_rejects_missing_absolute_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.safetensors"

    with pytest.raises(HTTPException) as caught:
        _validate_lora_files(_request(loras=[str(missing)]))

    assert caught.value.status_code == 422
    assert "missing.safetensors" in str(caught.value.detail)
