"""PR-2 C1 — DomainError 体系基础测试。

不依赖 fastapi / handler（C2 才有）；只验类层级 + 字段。
"""
from __future__ import annotations

import pytest

from studio.domain.errors import (
    AuthError,
    ConflictError,
    DomainError,
    ForbiddenError,
    InvalidPathError,
    NotFoundError,
    PresetConflictError,
    PresetNameInvalidError,
    PresetNotFoundError,
    ValidationError,
)


# ── 基类 ─────────────────────────────────────────────────────────────────


def test_domain_error_default_fields() -> None:
    e = DomainError("something broke")
    assert e.message == "something broke"
    assert e.code == "domain.error"
    assert e.details == {}
    assert e.http_status == 400
    assert str(e) == "something broke"


def test_domain_error_custom_code_and_details() -> None:
    e = DomainError(
        "x is bad", code="my.custom", details={"field": "x"}, http_status=418,
    )
    assert e.code == "my.custom"
    assert e.details == {"field": "x"}
    assert e.http_status == 418


def test_domain_error_inherits_from_exception() -> None:
    assert issubclass(DomainError, Exception)
    with pytest.raises(DomainError):
        raise DomainError("boom")


def test_domain_error_repr_contains_code() -> None:
    e = DomainError("x", code="my.code")
    assert "my.code" in repr(e)
    assert "DomainError" in repr(e)


# ── 5 核心子类 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("cls,expected_status,expected_code", [
    (NotFoundError, 404, "not_found"),
    (ValidationError, 422, "validation"),
    (ConflictError, 409, "conflict"),
    (AuthError, 401, "auth"),
    (ForbiddenError, 403, "forbidden"),
])
def test_core_subclass_status_and_code(cls, expected_status, expected_code) -> None:
    e = cls("msg")
    assert e.http_status == expected_status
    assert e.code == expected_code
    assert isinstance(e, DomainError)


# ── preset / path 别名 ───────────────────────────────────────────────────


def test_preset_not_found_is_404_with_preset_code() -> None:
    e = PresetNotFoundError("preset 'foo' missing")
    assert isinstance(e, NotFoundError)
    assert e.http_status == 404
    assert e.code == "preset.not_found"


def test_preset_name_invalid_is_400() -> None:
    """name_invalid 是 400 而非 422 — name 是 URL path 一部分，URL 校验是 400 惯例。"""
    e = PresetNameInvalidError("preset name contains '/'")
    assert e.http_status == 400
    assert e.code == "preset.name_invalid"


def test_preset_conflict_is_409() -> None:
    e = PresetConflictError("preset 'foo' already exists")
    assert isinstance(e, ConflictError)
    assert e.http_status == 409
    assert e.code == "preset.exists"


def test_invalid_path_is_400() -> None:
    e = InvalidPathError("path '../etc' escapes safe root")
    assert isinstance(e, ValidationError)
    assert e.http_status == 400
    assert e.code == "path.invalid"


# ── 不依赖 fastapi ──────────────────────────────────────────────────────


def test_domain_errors_module_does_not_import_fastapi() -> None:
    """domain/ 层禁止反向依赖 api（ADR-0008 / ADR-0009 §4）。"""
    import studio.domain.errors as _e
    import sys
    # 检查 module 不直接 import fastapi（不能 100% 阻止子模块传染，但能 catch 直接 import）
    src = open(_e.__file__, encoding="utf-8").read()
    for forbidden in ("import fastapi", "from fastapi"):
        assert forbidden not in src, (
            f"domain/errors.py 不应 import fastapi（services 通过 domain raise）；"
            f"找到: {forbidden!r}"
        )


def test_subclass_can_override_per_instance() -> None:
    """子类可以 per-instance 覆盖 http_status（罕见但合法）。"""
    e = NotFoundError("x", http_status=410)  # Gone 而非 404
    assert e.http_status == 410
    assert e.code == "not_found"  # code 不变
