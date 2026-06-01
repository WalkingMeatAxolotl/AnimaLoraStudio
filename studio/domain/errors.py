"""统一异常体系（ADR-0009 §4 / PR-2 C1）。

业务层 (`studio.services.*`) raise DomainError 子类；FastAPI 在 api 层装
exception_handler 翻成统一 JSON envelope 给前端。router 不再 try/except 翻
HTTPException（PR-2 C4/C5 渐进迁移现有 380 处）。

为什么放 `domain/` 而不是 `api/`：
  - services 反向依赖 api 是反模式（ADR-0008 4 层架构 services → domain，不
    应该 services → api）。把基类放 domain/errors.py 让 services 可直接
    raise 而不破层依赖。
  - api 层只装 exception_handler 注册，不持有错误类定义。
  - 不引 fastapi 依赖（纯 Python Exception 子类）；api/exception_handlers.py
    才 import FastAPI / Request / JSONResponse。

文案规约（ADR-0009 §4.1 + B audit 跨D）：
  - `message` 字段**英文** — 前端用 `code` 查 i18n 表渲染本地化字串；message
    兜底显示英文。这避开 ADR-0008 §跨D 中文字符串匹配陷阱借 DomainError 复活。
  - `code` 字段**领域.动作** 命名（`preset.not_found` / `curation.duplicate`）。
  - 短期（0.12.0）现有 PresetError 等 service 错误的中文 message 仍存在
    （C3 加 base 不改 message），但**新代码必须**英文 message + i18n code。

用法：
    from studio.domain.errors import NotFoundError, PresetNotFoundError

    if not preset_exists(name):
        raise PresetNotFoundError(f"preset {name!r} does not exist",
                                  details={"name": name})

handler 自动翻成（C2 后）：
    HTTP 404
    Headers: X-Trace-Id: <id>
    Body: {
      "detail": "preset 'foo' does not exist",   # legacy contract
      "error": {
        "code": "preset.not_found",
        "message": "preset 'foo' does not exist",
        "trace_id": "<id>",
        "details": {"name": "foo"}
      }
    }
"""
from typing import Any, ClassVar, Dict, Optional


class DomainError(Exception):
    """业务异常基类。子类化指定 `http_status` + `default_code`。"""

    http_status: ClassVar[int] = 400
    default_code: ClassVar[str] = "domain.error"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        http_status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.details = details or {}
        if http_status is not None:
            self.http_status = http_status

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.message!r}, code={self.code!r})"


# ── 5 个核心子类（ADR-0009 §C round2 §1.1 决策："5 核心而非 7 一次落"）────


class NotFoundError(DomainError):
    """资源不存在 — 404。"""
    http_status = 404
    default_code = "not_found"


class ValidationError(DomainError):
    """请求字段 / 业务规则校验失败 — 422。

    `details` 通常含 `{"field": "...", "reason": "..."}` 给前端字段级提示。
    跟 fastapi RequestValidationError 区分：RequestValidationError 是 pydantic
    解析 body 失败；本类是业务层判断（如 "epoch must be > 0"）。
    """
    http_status = 422
    default_code = "validation"


class ConflictError(DomainError):
    """资源冲突（重名、状态冲突） — 409。"""
    http_status = 409
    default_code = "conflict"


class AuthError(DomainError):
    """未认证 — 401。当前 webui 单用户无认证，此类预留给未来多用户场景。"""
    http_status = 401
    default_code = "auth"


class ForbiddenError(DomainError):
    """已认证但无权限 — 403。同 AuthError 预留。"""
    http_status = 403
    default_code = "forbidden"


# ── 子类常用别名（让 service raise 更可读） ──────────────────────────────


class InvalidPathError(ValidationError):
    """路径越界 / 非法分量 — 400（不是 422 因为 path 是 URL 一部分）。

    `_safe_join_or_400` 触发本类。
    """
    default_code = "path.invalid"
    http_status = 400


class PresetNotFoundError(NotFoundError):
    default_code = "preset.not_found"


class PresetNameInvalidError(ValidationError):
    default_code = "preset.name_invalid"
    http_status = 400


class PresetConflictError(ConflictError):
    default_code = "preset.exists"


__all__ = [
    # 基类
    "DomainError",
    # 5 核心子类
    "NotFoundError", "ValidationError", "ConflictError",
    "AuthError", "ForbiddenError",
    # preset / path 常用别名
    "InvalidPathError",
    "PresetNotFoundError", "PresetNameInvalidError", "PresetConflictError",
]
