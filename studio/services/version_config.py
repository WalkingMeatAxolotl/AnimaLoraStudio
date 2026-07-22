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
from ..domain.config_prune import prune_inactive_fields
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
})


def initial_project_field_values(
    project: dict[str, Any], version: dict[str, Any]
) -> dict[str, Any]:
    """根据 project + version 算出项目特定字段的**初值**。

    只在创建 config 时写入一次（fork 预设 / 复制 version / bundle 导入）；此后这些
    字段归用户所有，消费期（入队 / spawn）绝不回填 —— 见
    `docs/design/version-config-ownership.md`。

    `data_dir` / `output_dir` / `output_name` 按项目结构确定地填上。
    `reg_data_dir` **无条件**填路径，不看 `reg/meta.json` 存不存在：runtime 对空目录
    与不存在的目录完全容错（warning + skip），而先建 config、后生成 reg 集是常见顺序，
    条件填充会让后建的 reg 集永远不生效。
    `resume_lora` / `resume_state` 填空 —— 用户要接续训练时自己 PUT 改写。
    """
    pid = int(project["id"])
    slug = str(project["slug"])
    label = str(version["label"])
    vdir = version_dir(pid, slug, label)
    return {
        "data_dir": str(vdir / "train"),
        "reg_data_dir": str(vdir / "reg"),
        "output_dir": str(vdir / "output"),
        "output_name": f"{slug}_{label}",
        "resume_lora": None,
        "resume_state": None,
    }


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


#: 创建 version 时初值取自全局设置（`auto_sync_paths=ON`）的 4 个模型路径字段。
#: 只是**初值**：写进 config 后归用户所有，Train 页可自由编辑，全局设置之后再变也
#: 不会回头改写已有 version（保重现性）。页面在值与全局当前设置不一致时给
#: 「恢复默认」链接，让对齐是用户的一次显式动作。
GLOBAL_MODEL_PATH_FIELDS = (
    "transformer_path", "vae_path", "text_encoder_path", "t5_tokenizer_path",
)


def write_version_config(
    project: dict[str, Any], version: dict[str, Any], data: dict[str, Any],
    *, initialize_project_fields: bool = False,
) -> Path:
    """写 version 私有 config。

    `initialize_project_fields=True` 只在**创建** config 时传（fork 预设 / 复制
    version / bundle 导入）：用 `initial_project_field_values` 填 PROJECT_SPECIFIC_FIELDS
    的初值。默认 False —— 编辑与消费期一律不回填，config 里的值就是用户的值。
    """
    payload = dict(data)
    if initialize_project_fields:
        payload.update(initial_project_field_values(project, version))
    cfg, _, _ = _tolerant_validate(payload)
    # 落盘前裁掉 show_when 为假的字段（UI 不可见 = 不生效），读取时 pydantic
    # 会把缺失字段补回 schema 默认值，GET 返回给前端的仍是完整 config。
    dumped = prune_inactive_fields(cfg.model_dump(mode="python"))
    p = version_config_path(project, version)
    p.parent.mkdir(parents=True, exist_ok=True)
    # 序列化出口统一 render_config_yaml —— 预览端点与落盘同一条路径(R4)
    from .presets.io import render_config_yaml

    p.write_text(render_config_yaml(dumped), encoding="utf-8")
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
