"""Version eval manifest endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ...schemas.projects import EvalManifestPut
from ._shared import _version_dir_or_404
from ....services import eval_manifest

router = APIRouter()


def _payload(vdir, manifest: dict[str, Any] | None, default_manifest=None) -> dict[str, Any]:
    return {
        "has_manifest": manifest is not None,
        "manifest": manifest,
        "default_manifest": default_manifest,
        "path": eval_manifest.manifest_path(vdir).relative_to(vdir).as_posix(),
    }


@router.get("/api/projects/{pid}/versions/{vid}/eval/manifest")
def get_eval_manifest_endpoint(pid: int, vid: int) -> dict[str, Any]:
    p, v, vdir = _version_dir_or_404(pid, vid)
    try:
        manifest = eval_manifest.load_manifest(vdir)
        default_manifest = (
            None
            if manifest is not None
            else eval_manifest.create_default_manifest(p, v, vdir)
        )
    except eval_manifest.EvalManifestError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _payload(vdir, manifest, default_manifest)


@router.post("/api/projects/{pid}/versions/{vid}/eval/manifest/default")
def create_default_eval_manifest_endpoint(pid: int, vid: int) -> dict[str, Any]:
    p, v, vdir = _version_dir_or_404(pid, vid)
    try:
        manifest = eval_manifest.save_default_manifest(p, v, vdir)
    except eval_manifest.EvalManifestError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _payload(vdir, manifest)


@router.put("/api/projects/{pid}/versions/{vid}/eval/manifest")
def put_eval_manifest_endpoint(
    pid: int, vid: int, body: EvalManifestPut
) -> dict[str, Any]:
    p, v, vdir = _version_dir_or_404(pid, vid)
    try:
        manifest = eval_manifest.save_manifest(p, v, vdir, body.manifest)
    except eval_manifest.EvalManifestError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _payload(vdir, manifest)
