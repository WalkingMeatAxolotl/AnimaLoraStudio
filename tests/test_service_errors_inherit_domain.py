"""PR-2 C3 — 验证 11 个 service error 全部继承 DomainError。

后续 C4/C5 删 router try/except 之后，service raise 直接被 handler 翻 envelope
的前提就是 isinstance(exc, DomainError)。本测锁继承链不被未来 PR 破。
"""
from __future__ import annotations

import pytest

from studio.domain.errors import DomainError
from studio.services.data_io.train_io import TrainIOError
from studio.services.dataset.browse import BrowseError
from studio.services.dataset.curation import CurationError
from studio.services.preprocess.core import PreprocessError
from studio.services.preprocess.duplicates import DuplicateFinderError
from studio.services.presets.io import PresetError
from studio.services.projects.jobs import JobError
from studio.services.projects.projects import ProjectError
from studio.services.projects.versions import VersionError
from studio.services.tagging.caption_snapshot import SnapshotError
from studio.services.version_config import VersionConfigError


@pytest.mark.parametrize("cls,expected_code", [
    (PresetError, "preset.error"),
    (ProjectError, "project.error"),
    (VersionError, "version.error"),
    (JobError, "job.error"),
    (CurationError, "curation.error"),
    (TrainIOError, "train_io.error"),
    (VersionConfigError, "version_config.error"),
    (DuplicateFinderError, "duplicate.error"),
    (PreprocessError, "preprocess.error"),
    (BrowseError, "browse.error"),
    (SnapshotError, "snapshot.error"),
])
def test_service_error_inherits_domain_with_code(cls, expected_code) -> None:
    assert issubclass(cls, DomainError), (
        f"{cls.__name__} 必须继承 DomainError 让 exception handler 自动 catch"
    )
    e = cls("test message")
    assert isinstance(e, DomainError)
    assert e.code == expected_code
    assert e.http_status == 400  # default; router 或 raise 时按情况覆盖


def test_raise_service_error_caught_by_handler_via_isinstance(tmp_path) -> None:
    """端到端：raise PresetError 通过 webui app 被 DomainError handler 翻 envelope。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from studio.api.exception_handlers import register_exception_handlers
    from studio.api.trace_middleware import TraceIdMiddleware

    a = FastAPI()
    a.add_middleware(TraceIdMiddleware)
    register_exception_handlers(a)

    @a.get("/raise_preset_err")
    def _r():
        raise PresetError("preset 'x' missing")

    c = TestClient(a)
    resp = c.get("/raise_preset_err")
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"] == "preset 'x' missing"
    assert body["error"]["code"] == "preset.error"
    assert body["error"]["message"] == "preset 'x' missing"
    assert body["error"]["trace_id"] is not None


def test_raise_with_override_http_status() -> None:
    """service 仍可 raise PresetError("x", http_status=404) 让 handler 翻 404。

    C4/C5 router 迁移时常用这个 — 比如 read_preset 不存在时 raise
    PresetError("...", http_status=404, code="preset.not_found")。
    """
    e = PresetError("missing", http_status=404, code="preset.not_found")
    assert e.http_status == 404
    assert e.code == "preset.not_found"
    # 但 isinstance 仍是 PresetError + DomainError
    assert isinstance(e, PresetError)
    assert isinstance(e, DomainError)
