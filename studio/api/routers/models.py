"""模型 catalog / 下载（PR-6 commit 2 从 server.py 抽出）。

路由（PP7 第一刀域 + 统一模型来源候选）：
    GET    /api/models/catalog        列已知模型 + 各自磁盘状态 + 当前下载状态
    GET    /api/models/path-defaults  当前 Settings 算出的 4 个模型字段绝对路径
    POST   /api/models/download       启动后台下载，返回 status key
    POST   /api/models/anima/custom   注册一个本地 .safetensors 主模型
    DELETE /api/models/anima/custom   注销一个本地 custom 主模型
    POST   /api/models/krea2/custom   注册一个 Krea2 本地主模型
    DELETE /api/models/krea2/custom   注销一个 Krea2 本地主模型
    POST   /api/model-sources/{domain}   添加一条来源候选（下载型 / 本地文件）
    DELETE /api/model-sources/{domain}   移除一条候选（不动磁盘；选中项回退默认）
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..schemas.models import (
    AnimaCustomModelRequest,
    FamilySwitchRequest,
    ModelDownloadRequest,
    ModelSourceCandidateRequest,
)
from ... import secrets
from ...domain.errors import ValidationError
from ...domain.family_switch import switch_family
from ...services import models as model_downloader

router = APIRouter()

# 本地 custom 主模型只接受单文件 .safetensors（防误传可执行 / 目录穿越）。
ANIMA_CUSTOM_EXTS = (".safetensors",)


def _validated_custom_model_path(raw_path: str) -> str:
    raw = (raw_path or "").strip()
    if not raw:
        raise ValidationError(
            "Model path is required",
            code="model.path_required", http_status=400,
        )
    path = Path(raw).expanduser()
    if path.suffix.lower() not in ANIMA_CUSTOM_EXTS:
        raise ValidationError(
            "Select a .safetensors file",
            code="model.ext_invalid",
            details={"types": ".safetensors"}, http_status=400,
        )
    if not path.is_file():
        raise ValidationError(
            "File not found",
            code="model.not_found", details={"path": str(path)}, http_status=400,
        )
    return str(path)


def _add_custom_model(family_id: str, raw_path: str) -> dict[str, Any]:
    resolved = _validated_custom_model_path(raw_path)
    current = secrets.load()
    custom = {key: list(paths) for key, paths in current.models.custom.items()}
    paths = custom.setdefault(family_id, [])
    if resolved not in paths:
        paths.append(resolved)
        new_models = current.models.model_copy(update={"custom": custom})
        secrets.save(current.model_copy(update={"models": new_models}))
    return model_downloader.build_catalog()


def _remove_custom_model(family_id: str, raw_path: str, fallback: str) -> dict[str, Any]:
    target = (raw_path or "").strip()
    current = secrets.load()
    custom = {key: list(paths) for key, paths in current.models.custom.items()}
    custom[family_id] = [path for path in custom.get(family_id, []) if path != target]
    update: dict[str, Any] = {"custom": custom}
    if current.models.selected.get(family_id) == target:
        update["selected"] = {**current.models.selected, family_id: fallback}
    new_models = current.models.model_copy(update=update)
    secrets.save(current.model_copy(update={"models": new_models}))
    return model_downloader.build_catalog()


@router.get("/api/models/catalog")
def get_models_catalog() -> dict[str, Any]:
    """前端设置页 Models 区块用：列已知模型 + 各自磁盘状态 + 当前下载状态。"""
    return model_downloader.build_catalog()


@router.get("/api/models/path-defaults")
def get_models_path_defaults(family: str = "anima") -> dict[str, str]:
    """当前 Settings 算出的 4 个模型字段绝对路径（按 `family` query 参数解析）。

    给预设页 reset 按钮和「新建预设」初始填充用——这两个场景没有 project
    上下文，拿不到 /api/projects/{pid}/versions/{vid}/config 里的
    project_specific_defaults，所以单独开一个端点。
    """
    try:
        return model_downloader.default_paths_for_new_version(family=family)
    except ValueError as exc:
        raise ValidationError(
            f"Unknown model family: {family}",
            code="model.family_invalid", details={"reason": str(exc)},
            http_status=400,
        ) from exc


@router.post("/api/models/family-switch")
def switch_model_family(body: FamilySwitchRequest) -> dict[str, Any]:
    """训练配置切换模型族的预览计算（多模型 P4-3）。

    纯计算不落盘：重算 4 个权重路径 + 重置族风味字段（sampler / scheduler /
    timestep 等）+ 关闭目标族不支持的能力字段，返回切换后的完整 config 与
    变更清单。前端拿去弹确认对话框，用户确认后走正常保存链路。
    """
    try:
        path_defaults = model_downloader.default_paths_for_new_version(
            family=body.target)
    except ValueError as exc:
        raise ValidationError(
            f"Unknown model family: {body.target}",
            code="model.family_invalid", details={"reason": str(exc)},
            http_status=400,
        ) from exc
    new_config, changes = switch_family(body.config, body.target, path_defaults)
    return {"config": new_config, "changes": changes}


@router.delete("/api/models/asset")
def delete_model_asset(model_id: str, variant: str | None = None) -> dict[str, Any]:
    """删除一个已下载资产（下载的逆操作：用户先删除、再重新下载）。

    目标路径由服务端解析（不接受任意路径）；下载进行中 / 文件被占用时报错。
    返回删除后的完整 catalog。
    """
    try:
        model_downloader.delete_asset(model_id, variant)
    except ValueError as exc:
        raise ValidationError(
            f"Invalid model selection: {exc}",
            code="model.invalid", details={"reason": str(exc)}, http_status=400,
        ) from exc
    except RuntimeError as exc:
        raise ValidationError(
            str(exc), code="model.delete_failed", http_status=409,
        ) from exc
    return model_downloader.build_catalog()


@router.post("/api/models/download")
def start_model_download(body: ModelDownloadRequest) -> dict[str, Any]:
    """启动后台下载，立即返回 status key；前端通过 SSE
    (`model_download_changed`) 或轮询 catalog 看进度。"""
    try:
        key = model_downloader.trigger(body.model_id, body.variant)
    except ValueError as exc:
        raise ValidationError(
            f"Invalid model selection: {exc}",
            code="model.invalid", details={"reason": str(exc)}, http_status=400,
        ) from exc
    snap = model_downloader.get_status_snapshot()
    return {"key": key, "status": snap.get(key, {}).get("status", "running")}


def _assets_or_400(family: str):
    from ...services.models.families import get_assets

    try:
        return get_assets(family)
    except ValueError as exc:
        raise ValidationError(
            f"Unknown model family: {family}",
            code="model.family_invalid", details={"reason": str(exc)},
            http_status=400,
        ) from exc


@router.post("/api/models/{family}/custom")
def add_custom_model(family: str, body: AnimaCustomModelRequest) -> dict[str, Any]:
    """注册一个本地 `.safetensors` 主模型权重（去重 append），返回新 catalog。

    仅登记路径，不下载 / 不复制。校验文件存在 + 后缀白名单。注册后可在设置页
    选作该族默认主模型，驱动训练新建默认 + 测试出图（在微调权重上炼丹 / 验证）。
    路由按族参数化（多模型 P4-5）：未知族 400。
    """
    assets = _assets_or_400(family)
    return _add_custom_model(assets.family_id, body.path)


@router.delete("/api/models/{family}/custom")
def remove_custom_model(family: str, body: AnimaCustomModelRequest) -> dict[str, Any]:
    """注销一个本地 custom 主模型，返回新 catalog。

    若被删路径正是该族当前默认，把默认重置回最新官方 variant，
    避免训练 / 出图解析落到已注销的路径。
    """
    assets = _assets_or_400(family)
    return _remove_custom_model(assets.family_id, body.path, assets.latest)


# ---------------------------------------------------------------------------
# 统一模型来源候选（docs/design/model-source-unification.md §6）
# ---------------------------------------------------------------------------

# HF / MS repo id 形如 owner/name。校验从简（D3）：不做网络探测，repo 不存在
# 等到下载时报错。
_REPO_ID_RE = re.compile(r"^[\w.\-]+/[\w.\-]+$")


def _source_domains() -> set[str]:
    from ...services.models.families import FAMILY_ASSETS

    return (
        set(secrets.MODEL_SOURCE_REPO_DOMAINS)
        | {"upscaler"}
        | set(FAMILY_ASSETS.keys())
    )


def _domain_or_400(domain: str) -> str:
    if domain not in _source_domains():
        raise ValidationError(
            f"Unknown model source domain: {domain}",
            code="model_source.domain_invalid",
            details={"domain": domain}, http_status=400,
        )
    return domain


def _require_file(p: Path, exts: tuple[str, ...]) -> None:
    if p.suffix.lower() not in exts:
        raise ValidationError(
            f"Select a {' / '.join(exts)} file",
            code="file.ext_invalid", details={"types": " / ".join(exts)},
            http_status=400,
        )
    if not p.is_file():
        raise ValidationError(
            "File not found", code="model.not_found",
            details={"path": str(p)}, http_status=400,
        )


def _require_dir_with(p: Path, required: tuple[str, ...]) -> None:
    missing = [f for f in required if not (p / f).exists()]
    if not p.is_dir() or missing:
        raise ValidationError(
            "Directory is missing required model files",
            code="model_source.dir_incomplete",
            details={"path": str(p), "missing": missing}, http_status=400,
        )


def _validate_candidate(domain: str, cand: "secrets.SourceCandidate") -> None:
    """简单校验（D3）：格式 / 存在性 / 域结构；运行时报错兜底。"""
    from ...services.models.families import FAMILY_ASSETS
    from ...services.models.paths import UPSCALER_EXTS, WD14_FILES

    if cand.kind == "download":
        if not _REPO_ID_RE.match(cand.repo):
            raise ValidationError(
                "Repository ID must look like owner/name",
                code="model_source.repo_invalid",
                details={"repo": cand.repo}, http_status=400,
            )
        # 单文件资产必须带 filename + 后缀白名单；目录型资产不接受 filename
        if domain == "upscaler" or domain in FAMILY_ASSETS:
            exts = UPSCALER_EXTS if domain == "upscaler" else (".safetensors",)
            name = Path(cand.filename).name
            if not name or not name.lower().endswith(exts):
                raise ValidationError(
                    f"File name must end with {' / '.join(exts)}",
                    code="file.ext_invalid",
                    details={"types": " / ".join(exts)}, http_status=400,
                )
        elif cand.filename:
            raise ValidationError(
                "This model type downloads a whole repository (no file name)",
                code="model_source.filename_unexpected", http_status=400,
            )
        return

    # kind == "local"
    if not cand.path.strip() or not secrets.is_abs_path(cand.path):
        raise ValidationError(
            "An absolute local path is required",
            code="model_source.path_invalid",
            details={"path": cand.path}, http_status=400,
        )
    p = Path(cand.path).expanduser()
    if domain == "wd14":
        _require_dir_with(p, WD14_FILES)
    elif domain in ("eval_clip", "eval_dino"):
        _require_dir_with(p, ("config.json",))
    elif domain == "eval_ccip":
        _require_dir_with(
            p, ("model_feat.onnx", "model_metrics.onnx", "metrics.json"))
    elif domain == "upscaler":
        _require_file(p, UPSCALER_EXTS)
    elif domain == "cltagger":
        _require_file(p, (".onnx",))
        mapping = cand.extra.get("tag_mapping_path", "")
        if not mapping or not secrets.is_abs_path(mapping):
            raise ValidationError(
                "tag_mapping_path (absolute path) is required",
                code="model_source.path_invalid", http_status=400,
            )
        _require_file(Path(mapping).expanduser(), (".json",))
    else:  # 主模型族：单文件 .safetensors（同 PathPicker 注册校验）
        _require_file(p, (".safetensors",))


def _candidate_from_request(body: ModelSourceCandidateRequest) -> "secrets.SourceCandidate":
    if body.kind not in ("download", "local"):
        raise ValidationError(
            f"Unknown candidate kind: {body.kind}",
            code="model_source.kind_invalid", http_status=400,
        )
    return secrets.SourceCandidate(
        kind=body.kind,
        repo=body.repo.strip(),
        filename=Path(body.filename).name if body.filename.strip() else "",
        path=body.path.strip(),
        extra={k: str(v).strip() for k, v in body.extra.items() if str(v).strip()},
    )


def _selected_value_reset(domain: str, removed: "secrets.SourceCandidate") -> dict[str, Any]:
    """移除的候选正是当前选中 → 附带把选中值回退默认的 update partial。"""
    from ...services.models.families import FAMILY_ASSETS
    from ...services.models.paths import DEFAULT_UPSCALER

    s = secrets.load()
    removed_value = removed.repo if removed.kind == "download" else removed.path
    if domain == "wd14" and s.wd14.model_id == removed_value:
        return {"wd14": {"model_id": secrets.DEFAULT_WD14_MODELS[0]}}
    if domain == "cltagger":
        # download=fork repo（比 model_id）；local=双文件（比 model_path）
        is_current = (
            removed.kind == "download" and s.cltagger.model_id == removed.repo
        ) or (
            removed.kind == "local" and s.cltagger.model_path == removed.path
        )
        if is_current:
            fields = secrets.CLTaggerConfig.model_fields
            return {"cltagger": {
                "model_id": fields["model_id"].default,
                "model_path": fields["model_path"].default,
                "tag_mapping_path": fields["tag_mapping_path"].default,
            }}
    if domain in ("eval_clip", "eval_dino", "eval_ccip"):
        field = {
            "eval_clip": "clip_model_name",
            "eval_dino": "dino_model_name",
            "eval_ccip": "ccip_model_name",
        }[domain]
        if getattr(s.eval_metrics, field) == removed_value:
            default = secrets.EvalMetricModelsConfig.model_fields[field].default
            return {"eval_metrics": {field: default}}
    if domain == "upscaler":
        sel = s.models.selected_upscaler
        if sel and sel in (removed_value, removed.filename):
            return {"models": {"selected_upscaler": DEFAULT_UPSCALER}}
    if domain in FAMILY_ASSETS and s.models.selected.get(domain) == removed_value:
        return {"models": {"selected": {
            **s.models.selected, domain: FAMILY_ASSETS[domain].latest,
        }}}
    return {}


@router.post("/api/model-sources/{domain}")
def add_model_source(
    domain: str, body: ModelSourceCandidateRequest
) -> dict[str, Any]:
    """添加一条来源候选（去重 append），返回新 catalog。

    校验从简：repo 形如 owner/name、单文件资产的后缀白名单、本地路径存在 +
    域结构（wd14 双文件 / eval config.json 等）。不做网络探测。
    """
    _domain_or_400(domain)
    cand = _candidate_from_request(body)
    if domain == "cltagger" and cand.kind == "download":
        # fork repo 候选默认继承当前双文件相对路径（用户通常 fork 同版本；
        # 当前是 local 绝对路径时回退首个内置 preset 的路径）
        from ...services.models.paths import CLTAGGER_VERSIONS

        cfg = secrets.load().cltagger
        mp, tmp = cfg.model_path, cfg.tag_mapping_path
        if secrets.is_abs_path(mp) or secrets.is_abs_path(tmp):
            first = next(iter(CLTAGGER_VERSIONS.values()))
            mp, tmp = str(first["model_path"]), str(first["tag_mapping_path"])
        cand.extra.setdefault("model_path", mp)
        cand.extra.setdefault("tag_mapping_path", tmp)
    _validate_candidate(domain, cand)
    existing = secrets.load().model_sources.get(domain, [])
    if all(c.identity() != cand.identity() for c in existing):
        secrets.update({"model_sources": {
            domain: [c.model_dump() for c in existing] + [cand.model_dump()],
        }})
    return model_downloader.build_catalog()


@router.delete("/api/model-sources/{domain}")
def remove_model_source(
    domain: str, body: ModelSourceCandidateRequest
) -> dict[str, Any]:
    """移除一条候选（只移出列表，不动磁盘文件），返回新 catalog。

    被移除的候选正是当前选中时，选中值回退该 domain 默认（wd14 首个内置 /
    eval schema 默认 / cltagger 官方 preset / upscaler 默认 / 主模型族最新
    官方 variant）——与「注销本地主模型回退 latest」的现状语义一致。
    """
    _domain_or_400(domain)
    cand = _candidate_from_request(body)
    existing = secrets.load().model_sources.get(domain, [])
    remaining = [c for c in existing if c.identity() != cand.identity()]
    if len(remaining) != len(existing):
        partial: dict[str, Any] = {
            "model_sources": {domain: [c.model_dump() for c in remaining]},
        }
        for key, val in _selected_value_reset(domain, cand).items():
            partial[key] = val
        secrets.update(partial)
    return model_downloader.build_catalog()
