"""主模型族的「用户注册候选」解析 + 底模架构探测（两族共用）。

用户放进来的第三方 / 本地主模型有两条通道（docs/design/model-source-unification.md）：
  - ``kind=local``：PathPicker 注册的绝对路径（``models.custom[family]`` 兼容面同源）
  - ``kind=download``：repo + filename 的下载候选，落盘在 ``{root}/diffusion_models/{basename}``

此前 ``catalog_sections().custom`` / ``path_choices()`` 只读 local 一条，下载型第三方
底模能在设置页选中出图，却不进训练页下拉与底模下拉。这里统一成一处。

底模架构（层数 / 参数量）用 ``studio.services.inference.checkpoint_arch`` 只读 header
探测，按 (path, size, mtime) 缓存；非 Anima 结构（krea2）返回 None。
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Optional

from ...inference.checkpoint_arch import inspect_anima_checkpoint


def registered_main_paths(
    root: Path, models_cfg: Any, family_id: str, source_cfg: Any = None,
) -> list[Path]:
    """用户注册的主模型候选路径（local + download 落盘），按注册顺序去重；不检查存在性。

    ``source_cfg`` = ``secrets.model_sources``（dict[domain, list[SourceCandidate]]），
    由调用方显式传入（registry / catalog 已接线）；None = 只看 local 兼容面——
    这里**不**自行读全局 secrets，避免纯函数偷读用户配置（测试也不可控）。
    """
    out: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        key = str(p)
        if key in seen:
            return
        seen.add(key)
        out.append(p)

    for registered in (getattr(models_cfg, "custom", None) or {}).get(family_id, []) or []:
        _add(Path(str(registered)).expanduser())
    for cand in (source_cfg or {}).get(family_id, []) or []:
        kind = getattr(cand, "kind", None)
        if kind == "download" and getattr(cand, "filename", None):
            _add(root / "diffusion_models" / Path(str(cand.filename)).name)
        elif kind == "local" and getattr(cand, "path", None):
            _add(Path(str(cand.path)).expanduser())
    return out


@functools.lru_cache(maxsize=512)
def _probe(path: str, size: int, mtime: float) -> Optional[dict[str, Any]]:
    try:
        arch = inspect_anima_checkpoint(path)
    except Exception:  # noqa: BLE001 — 非 Anima 结构 / 坏文件：不展示架构
        return None
    return {
        "num_blocks": arch.num_blocks,
        "model_channels": arch.model_channels,
        "param_count": arch.param_count,
    }


def arch_summary(path: Path) -> Optional[dict[str, Any]]:
    """底模架构摘要（层数 / 通道 / 参数量），文件不存在或不是 Anima 结构 → None。"""
    try:
        st = path.stat()
    except OSError:
        return None
    return _probe(str(path), st.st_size, st.st_mtime)


def clear_arch_cache() -> None:
    _probe.cache_clear()


__all__ = ["arch_summary", "clear_arch_cache", "registered_main_paths"]
