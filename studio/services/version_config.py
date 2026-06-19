"""Version 私有 config（PP6.2）。

每个 version 自己有一份 yaml 训练配置，存在
`studio_data/projects/{id}-{slug}/versions/{label}/config.yaml`。
和全局 `studio_data/presets/{name}.yaml` **完全独立** —— 用户「换预设」时
从全局复制一份进来，「保存为预设」时反向导出去；私有 config 修改不会回流到
预设池。

Schema 校验沿用 `TrainingConfig`（与 preset 同一 model）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .presets.io import _absolutize_model_paths, _tolerant_validate
from ..schema import TrainingConfig
from .projects.versions import version_dir
from .projects import projects as _projects


from studio.domain.errors import DomainError


class VersionConfigError(DomainError):
    """version 私有 config I/O 错误。

    PR-2 C3 加 DomainError base — handler 自动翻 dual-write envelope。
    """
    default_code = "version_config.error"


CONFIG_FILENAME = "config.yaml"


# ---------------------------------------------------------------------------
# 项目特定字段（PP6 spec §关键约定）
# ---------------------------------------------------------------------------

PROJECT_SPECIFIC_FIELDS: frozenset[str] = frozenset({
    "data_dir",
    "reg_data_dir",
    "output_dir",
    "output_name",
    "resume_lora",
    "resume_state",
    "trigger_word",
})


def project_specific_overrides(
    project: dict[str, Any], version: dict[str, Any]
) -> dict[str, Any]:
    """根据 project + version 算出项目特定字段的值。

    `data_dir` / `output_dir` / `output_name` 永远确定地填上；
    `reg_data_dir` 只有 reg 集存在（meta.json）才填，否则空（让 trainer 走默认）。
    `resume_lora` / `resume_state` 默认空 —— 用户要接续训练时显式 PUT 改写。
    `trigger_word` 来自 version 表（Step 4 Tagging 写入），保证 yaml 与 caption
    同源，runtime bootstrap_phase 会据此把 trigger 注入 sample_prompt。
    """
    pid = int(project["id"])
    slug = str(project["slug"])
    label = str(version["label"])
    vdir = version_dir(pid, slug, label)
    overrides: dict[str, Any] = {
        "data_dir": str(vdir / "train"),
        "output_dir": str(vdir / "output"),
        "output_name": f"{slug}_{label}",
        "resume_lora": None,
        "resume_state": None,
        "trigger_word": str(version.get("trigger_word") or ""),
    }
    reg_meta = vdir / "reg" / "meta.json"
    if reg_meta.exists():
        overrides["reg_data_dir"] = str(vdir / "reg")
    else:
        overrides["reg_data_dir"] = None
    return overrides


# ---------------------------------------------------------------------------
# 文件路径
# ---------------------------------------------------------------------------


def version_config_path(project: dict[str, Any], version: dict[str, Any]) -> Path:
    pid = int(project["id"])
    slug = str(project["slug"])
    label = str(version["label"])
    return version_dir(pid, slug, label) / CONFIG_FILENAME


def has_version_config(project: dict[str, Any], version: dict[str, Any]) -> bool:
    return version_config_path(project, version).exists()


# ---------------------------------------------------------------------------
# 读 / 写
# ---------------------------------------------------------------------------


def read_version_config(
    project: dict[str, Any], version: dict[str, Any]
) -> dict[str, Any]:
    """读 version 私有 config；不存在抛 VersionConfigError。"""
    cfg, _, _ = read_version_config_with_warnings(project, version)
    return cfg


def read_version_config_with_warnings(
    project: dict[str, Any], version: dict[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    """读 version 私有 config 同时返回容错校验产出的 (dropped, defaulted) 字段列表。

    用于 GET 端点把 compat 信息透传给前端（顶部 banner 提示）。InfoNoise 老 config
    互斥被 _tolerant_validate 自动关 InfoNoise 时，"infonoise_enabled" 会出现在
    defaulted 里。
    """
    p = version_config_path(project, version)
    if not p.exists():
        raise VersionConfigError(
            "Training configuration is not set for this version",
            code="version.config_missing",
        )
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise VersionConfigError(
            "Training configuration is invalid",
            code="version.config_invalid",
        )
    cfg, dropped, defaulted = _tolerant_validate(raw)
    return _absolutize_model_paths(cfg.model_dump(mode="python")), dropped, defaulted


def write_version_config(
    project: dict[str, Any], version: dict[str, Any], data: dict[str, Any],
    *, force_project_overrides: bool = True,
) -> Path:
    """写 version 私有 config。

    `force_project_overrides=True`（默认）：用 `project_specific_overrides`
    强制覆盖 PROJECT_SPECIFIC_FIELDS，防止用户绕过前端 disabled 改路径。
    """
    payload = dict(data)
    if force_project_overrides:
        payload.update(project_specific_overrides(project, version))
    cfg, _, _ = _tolerant_validate(payload)
    dumped = cfg.model_dump(mode="python")
    p = version_config_path(project, version)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(
            dumped, allow_unicode=True, sort_keys=False, default_flow_style=False
        ),
        encoding="utf-8",
    )
    return p


def delete_version_config(
    project: dict[str, Any], version: dict[str, Any]
) -> bool:
    """删除 version 私有 config。已删返回 True，本来就没有返回 False。"""
    p = version_config_path(project, version)
    if p.exists():
        p.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------


def get_project_and_version(
    conn, project_id: int, version_id: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """便捷：从 db 读 project + version；版本不属当前项目时抛 VersionConfigError。"""
    from ..services.projects import versions as _versions
    p = _projects.get_project(conn, project_id)
    if not p:
        raise VersionConfigError(
            "Project not found", code="project.not_found",
            details={"id": project_id}, http_status=404,
        )
    v = _versions.get_version(conn, version_id)
    if not v or v["project_id"] != project_id:
        raise VersionConfigError(
            "Version not found", code="version.not_found",
            details={"id": version_id}, http_status=404,
        )
    return p, v
