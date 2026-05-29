"""预设文件 I/O —— 用 pydantic 验证、用 PyYAML 落盘。

存储位置：`studio_data/presets/{name}.yaml`
名字白名单：`[A-Za-z0-9_-]+`，防止路径穿越和 Windows 非法字符。

历史：PP0 之前叫 configs_io / studio_data/configs/。`configs_io` 现在是
本模块的薄壳。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ...paths import REPO_ROOT, USER_PRESETS_DIR
from ...schema import TrainingConfig

NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# 全局模型路径字段。yaml 写盘 + UI 显示一律绝对路径；读老 yaml 时若是相对
# 路径，由 _absolutize_model_paths 兜底转绝对（忠于历史 CWD=REPO_ROOT 的解析
# 语义），下游可以无脑假定 4 字段是绝对路径。
_MODEL_PATH_FIELDS = (
    "transformer_path",
    "vae_path",
    "text_encoder_path",
    "t5_tokenizer_path",
)


from studio.domain.errors import DomainError


class PresetError(DomainError):
    """预设 I/O 错误。

    PR-2 C3 加 DomainError base — handler 自动翻 dual-write envelope。
    现有 raise PresetError("xxx") 形态不变；http_status / code 由 router 或
    C4/C5 精细化时按情况覆盖（now 用 default = 400 / preset.error）。
    """
    default_code = "preset.error"


_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _absolutize_model_paths(data: dict[str, Any]) -> dict[str, Any]:
    """规范化 4 个模型字段：相对路径 → REPO_ROOT 绝对；分隔符统一 POSIX `/`。

    - 老 yaml 里相对路径（schema fallback）→ 转为基于 REPO_ROOT 的绝对路径
      （忠于历史 supervisor cwd=REPO_ROOT 的解析语义）
    - Windows 上 `str(Path)` 给反斜杠（`G:\\foo`），PathPicker 给 POSIX
      （`G:/foo`），混存会让同一字段在不同来源下视觉不一致；统一 `as_posix()`
      让 yaml 落盘 + UI 显示一律 `/`
    - 跨平台 bundle import：Windows 盘符路径（`G:/...`）在 POSIX 上
      `Path.is_absolute()` 返 False，会被误当相对路径拼到 REPO_ROOT 下变成
      `<repo>/G:/...`。这里额外用正则识别盘符前缀视作绝对，避免静默 mangle。
      （路径在异机器上仍然不可解析，但保持原样让 UI/日志能定位到原始来源。）
    不动 yaml 文件；下次保存自然落规范化后的形式。
    """
    for f in _MODEL_PATH_FIELDS:
        v = data.get(f)
        if isinstance(v, str) and v:
            if _WIN_DRIVE_RE.match(v):
                data[f] = v.replace("\\", "/")
                continue
            p = Path(v)
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            data[f] = p.as_posix()
    return data


def _validate_name(name: str) -> None:
    if not NAME_PATTERN.fullmatch(name):
        raise PresetError(f"非法预设名: {name!r}（只允许字母/数字/下划线/连字符）")


def _preset_path(name: str, base: Path | None = None) -> Path:
    _validate_name(name)
    return (base or USER_PRESETS_DIR) / f"{name}.yaml"


def preset_path(name: str, base: Path | None = None) -> Path:
    """公开版 `_preset_path`，给端到端文件下载用（server 不要碰 _ 私有 helper）。"""
    return _preset_path(name, base)


def parse_preset_bytes(raw: bytes, filename: str) -> tuple[dict[str, Any], str]:
    """解析 .yaml/.yml/.json 上传内容 + pydantic 校验，返回 (config_dict, suggested_name)。

    不写盘 —— caller 决定最终落盘名字（前端 confirm flow 让用户改名再保存）。
    yaml.safe_load 是 JSON 的 superset，所以 .json 文件也能直接吃。
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PresetError(f"文件不是 UTF-8: {exc}") from exc
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise PresetError(f"YAML/JSON 解析失败: {exc}") from exc
    if not isinstance(data, dict):
        raise PresetError("预设格式错误（顶层不是 mapping）")
    try:
        cfg = TrainingConfig.model_validate(data)
    except ValidationError as exc:
        raise PresetError(f"预设校验失败: {exc}") from exc
    stem = re.sub(r"\.(ya?ml|json)$", "", filename, flags=re.I)
    suggested = re.sub(r"[^A-Za-z0-9_-]+", "-", stem).strip("-") or "imported"
    return _absolutize_model_paths(cfg.model_dump(mode="python")), suggested


def list_presets(base: Path | None = None) -> list[dict[str, Any]]:
    """返回 `[{name, path, updated_at}]`，按修改时间倒序。"""
    base = base or USER_PRESETS_DIR
    if not base.exists():
        return []
    items: list[dict[str, Any]] = []
    for p in base.glob("*.yaml"):
        items.append({
            "name": p.stem,
            "path": str(p),
            "updated_at": p.stat().st_mtime,
        })
    items.sort(key=lambda x: x["updated_at"], reverse=True)
    return items


def read_preset(name: str, base: Path | None = None) -> dict[str, Any]:
    """读取并校验预设；返回校验后的 dict（未知字段会被 forbid）。"""
    path = _preset_path(name, base)
    if not path.exists():
        raise PresetError(f"预设不存在: {name}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise PresetError(f"预设格式错误（顶层不是 mapping）: {name}")
    try:
        cfg = TrainingConfig.model_validate(raw)
    except ValidationError as exc:
        raise PresetError(f"预设校验失败: {exc}") from exc
    return _absolutize_model_paths(cfg.model_dump(mode="python"))


def write_preset(name: str, data: dict[str, Any], base: Path | None = None) -> Path:
    """先校验后写盘；任何未知字段或类型不匹配都会拒绝。

    保存前 normalize 4 个模型字段：相对路径 → 绝对（基于 REPO_ROOT）。
    保证 yaml 落盘统一绝对路径，避免老格式（相对）和新格式（绝对）混存。
    """
    path = _preset_path(name, base)
    try:
        cfg = TrainingConfig.model_validate(data)
    except ValidationError as exc:
        raise PresetError(f"预设校验失败: {exc}") from exc
    dumped = _absolutize_model_paths(cfg.model_dump(mode="python"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dumped, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return path


def delete_preset(name: str, base: Path | None = None) -> None:
    path = _preset_path(name, base)
    if not path.exists():
        raise PresetError(f"预设不存在: {name}")
    path.unlink()


def duplicate_preset(src: str, dst: str, base: Path | None = None) -> Path:
    src_path = _preset_path(src, base)
    dst_path = _preset_path(dst, base)
    if not src_path.exists():
        raise PresetError(f"源预设不存在: {src}")
    if dst_path.exists():
        raise PresetError(f"目标已存在: {dst}")
    dst_path.write_bytes(src_path.read_bytes())
    return dst_path
