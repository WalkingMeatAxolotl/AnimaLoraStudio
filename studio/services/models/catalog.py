"""模型 catalog —— 扫盘组装"哪些模型已下载、目标路径、大小"（PR-3.8 拆出 4-way 第 4 个）。

build_catalog 是 /api/models/catalog 端点的核心，前端 ModelsPage 用它展示安装状态。
依赖 paths.py 的常量 + target 函数；不调下载（只读盘）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ... import secrets
from .. import eval_registry
from .downloader import get_status_snapshot
from .families import FAMILY_ASSETS
from .paths import (
    CLTAGGER_VERSIONS,
    DEFAULT_UPSCALER,
    TAEFLUX_FILES,
    TAEFLUX_REPO,
    UPSCALER_EXTS,
    UPSCALER_VARIANTS,
    WD14_FILES,
    ccip_model_dir,
    cltagger_required_files,
    cltagger_target_root,
    eval_model_target_dir,
    models_root,
    selected_upscaler,
    taeflux_dir,
    upscaler_dir,
    upscaler_target,
    wd14_target_dir,
)

# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------


# CLIP / DINO 常见模型的下载大小预估（bytes，约值）。下载前给用户一个体量参考；
# 未知 model_id 不显示预估。已下载后 UI 用实际目录大小。
_EVAL_SIZE_ESTIMATES = {
    "openai/clip-vit-base-patch32": 605_000_000,
    "openai/clip-vit-base-patch16": 599_000_000,
    "openai/clip-vit-large-patch14": 1_710_000_000,
    "facebook/dinov2-small": 88_000_000,
    "facebook/dinov2-base": 346_000_000,
    "facebook/dinov2-large": 1_220_000_000,
    "facebook/dinov2-giant": 4_600_000_000,
    # CCIP（deepghs/ccip_onnx 变体）：只下 model_feat.onnx(~150MB)+metric+threshold。
    "ccip-caformer-24-randaug-pruned": 152_000_000,
    "ccip-caformer_b36-24": 384_000_000,
}

# CCIP 变体的 3 个必备文件（齐全才算已下载）。
_CCIP_FILES = ("model_feat.onnx", "model_metrics.onnx", "metrics.json")


def _file_status(p: Path) -> dict[str, Any]:
    try:
        st = p.stat()
        return {"exists": True, "size": st.st_size, "mtime": st.st_mtime}
    except OSError:
        return {"exists": False, "size": 0, "mtime": 0.0}


def build_catalog(root: Optional[Path] = None) -> dict[str, Any]:
    """扫盘组装 catalog 给前端展示。

    每项含 `id` / `name` / `description` / 目标路径 / 已下载状态。
    Anima 主模型多版本时返回 `variants[]`，每个独立 status。
    `downloads` 字段返回当前活跃下载 status。
    """
    r = root or models_root()

    # 模型族区块经 FAMILY_ASSETS registry 遍历（多模型 PR-4）；单族时输出与
    # 旧实现逐字节一致，前端零改动
    models_cfg = secrets.load().models
    family_sections: dict[str, Any] = {}
    for _assets in FAMILY_ASSETS.values():
        family_sections.update(_assets.catalog_sections(r, models_cfg))

    _secrets = secrets.load()
    cl_cfg = _secrets.cltagger
    wd14_cfg = _secrets.wd14
    eval_cfg = _secrets.eval_metrics
    src_cfg = _secrets.download_sources
    source_cfg = _secrets.model_sources

    # CLIP / DINO eval 指标模型：各一行 variant，整目录有 config.json 即"已下载"。
    eval_variants = []
    for kind, mid in (("clip", eval_cfg.clip_model_name), ("dino", eval_cfg.dino_model_name)):
        target = eval_model_target_dir(r, kind, mid)
        exists = (target / "config.json").exists()
        size = (
            sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            if target.exists() else 0
        )
        eval_variants.append({
            "kind": kind,
            "model_id": mid,
            "target_path": str(target),
            "exists": exists,
            "size": size,
            "size_estimate": _EVAL_SIZE_ESTIMATES.get(mid, 0),
        })
    # CCIP（anime 角色身份）：3 个文件齐全才算已下载（无 config.json）。
    ccip_mid = eval_cfg.ccip_model_name
    ccip_dir = ccip_model_dir(r, ccip_mid)
    eval_variants.append({
        "kind": "ccip",
        "model_id": ccip_mid,
        "target_path": str(ccip_dir),
        "exists": all((ccip_dir / f).exists() for f in _CCIP_FILES),
        "size": (
            sum(f.stat().st_size for f in ccip_dir.rglob("*") if f.is_file())
            if ccip_dir.exists() else 0
        ),
        "size_estimate": _EVAL_SIZE_ESTIMATES.get(ccip_mid, 0),
    })

    # WD14 候选每个 model_id 一行：两文件全在才算"已下载"。
    wd14_variants = []
    for mid in wd14_cfg.model_ids:
        target = wd14_target_dir(r, mid)
        files = [{"name": f, **_file_status(target / f)} for f in WD14_FILES]
        all_exist = all(f["exists"] for f in files)
        total_size = sum(f["size"] for f in files)
        wd14_variants.append({
            "model_id": mid,
            "is_current": mid == wd14_cfg.model_id,
            "target_path": str(target),
            "exists": all_exist,
            "size": total_size,
            "files": files,
        })

    # CLTagger 版本预设（每个 variant 可以来自不同 HF repo）。
    cl_root = cltagger_target_root(r, cl_cfg.model_id)
    cl_variants = []
    for label, preset in CLTAGGER_VERSIONS.items():
        mid = preset["model_id"]
        mp = preset["model_path"]
        tmp = preset["tag_mapping_path"]
        variant_root = cltagger_target_root(r, mid)
        version_dir = variant_root / Path(mp).parent
        files = [
            {"name": f, **_file_status(variant_root / f)}
            for f in cltagger_required_files(mp, tmp)
        ]
        all_exist = all(f["exists"] for f in files)
        total_size = sum(f["size"] for f in files)
        cl_variants.append({
            "label": label,
            "model_id": mid,
            "model_path": mp,
            "tag_mapping_path": tmp,
            "description": preset.get("description", ""),
            # target_path = repo 本地根；version_dir = 该版本子目录（UI 提示文件落点）。
            "target_path": str(variant_root),
            "version_dir": str(version_dir),
            "is_current": (
                cl_cfg.model_id == mid
                and cl_cfg.model_path == mp
                and cl_cfg.tag_mapping_path == tmp
            ),
            "exists": all_exist,
            "size": total_size,
            "files": files,
        })

    # ── 统一来源候选行（docs/design/model-source-unification.md §6）────────
    #
    # 每 domain 一个平铺列表：内置 preset + 用户候选（download / local），能力
    # 位（removable / deletable）由这里拼好，前端泛化候选卡不再各自判断。
    # 本键当前覆盖 wd14 / eval_*；upscaler / 主模型族 / cltagger 在各自区块
    # 迁移时并入。

    def _source_row(
        *, kind: str, value: str, download_id: Optional[str],
        exists: bool, size: int, is_current: bool,
        label: Optional[str] = None, files: Optional[list] = None,
        size_estimate: int = 0, extra: Optional[dict] = None,
        download_variant: Optional[str] = None,
        status_key: Optional[str] = None,
        description: str = "",
        removable: Optional[bool] = None,
        candidate: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        variant = download_variant or value
        return {
            # 用户候选的原始存储记录（DELETE /api/model-sources 的身份键）；
            # preset / scanned 行无。
            "candidate": candidate,
            "kind": kind,               # preset | download | local | scanned
            "value": value,             # 写进选中值字段的值（repo id / 绝对路径 / label）
            "label": label or value,
            "description": description,
            # 触发下载：POST /api/models/download {model_id: download_id,
            # variant: download_variant}；status key 默认拼接、可显式覆盖
            # （upscaler custom 的 key 形如 upscaler:custom:{filename}）。local 无。
            "download_id": download_id,
            "download_variant": variant if download_id else None,
            "status_key": (
                status_key if status_key is not None
                else (f"{download_id}:{variant}" if download_id else None)
            ),
            "exists": exists,
            "size": size,
            "files": files,
            "size_estimate": size_estimate,
            "is_current": is_current,
            # 内置不可移除（保护默认）；扫盘行不在候选存储里、也不可移除
            "removable": (
                removable if removable is not None
                else kind not in ("preset", "scanned")
            ),
            "deletable": kind != "local",    # 本地文件永不从 UI 删除
            "extra": extra or {},
        }

    def _wd14_row(kind: str, value: str) -> dict[str, Any]:
        target = wd14_target_dir(r, value)
        files = [{"name": f, **_file_status(target / f)} for f in WD14_FILES]
        return _source_row(
            kind=kind, value=value,
            download_id="wd14" if kind != "local" else None,
            exists=all(f["exists"] for f in files),
            size=sum(f["size"] for f in files),
            files=files,
            is_current=value == wd14_cfg.model_id,
        )

    def _eval_row(domain: str, em_kind: str, kind: str, value: str) -> dict[str, Any]:
        if em_kind == "ccip":
            target = ccip_model_dir(r, value)
            exists = all((target / f).exists() for f in _CCIP_FILES)
        else:
            target = eval_model_target_dir(r, em_kind, value)
            exists = (target / "config.json").exists()
        size = (
            sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            if target.exists() else 0
        )
        current = {
            "eval_clip": eval_cfg.clip_model_name,
            "eval_dino": eval_cfg.dino_model_name,
            "eval_ccip": eval_cfg.ccip_model_name,
        }[domain]
        return _source_row(
            kind=kind, value=value,
            download_id=domain if kind != "local" else None,
            exists=exists, size=size,
            size_estimate=_EVAL_SIZE_ESTIMATES.get(value, 0),
            is_current=value == current,
        )

    def _user_rows(domain: str, row_fn) -> list[dict[str, Any]]:
        rows = []
        for c in source_cfg.get(domain, []):
            value = c.repo if c.kind == "download" else c.path
            row = row_fn(c.kind, value)
            row["extra"] = dict(c.extra)
            row["candidate"] = c.model_dump()
            rows.append(row)
        return rows

    model_source_rows: dict[str, list[dict[str, Any]]] = {
        "wd14": (
            [_wd14_row("preset", m) for m in secrets.DEFAULT_WD14_MODELS]
            + _user_rows("wd14", _wd14_row)
        ),
    }
    _eval_defaults = {
        "eval_clip": ("clip", secrets.EvalMetricModelsConfig.model_fields["clip_model_name"].default),
        "eval_dino": ("dino", secrets.EvalMetricModelsConfig.model_fields["dino_model_name"].default),
        "eval_ccip": ("ccip", secrets.EvalMetricModelsConfig.model_fields["ccip_model_name"].default),
    }
    for _domain, (_em_kind, _default) in _eval_defaults.items():
        model_source_rows[_domain] = (
            [_eval_row(_domain, _em_kind, "preset", str(_default))]
            + _user_rows(
                _domain,
                lambda kind, value, _d=_domain, _k=_em_kind: _eval_row(_d, _k, kind, value),
            )
        )

    # 放大器：预设 + 扫盘合并。
    # - Pass 1：UPSCALER_VARIANTS 全列（即便未下载，提供"下载"入口）
    # - Pass 2：扫 upscalers/ 目录里所有 .pth/.safetensors，把不在预设里的当
    #   custom 加进列表（用户通过自定义 repo 下载或之后扩展的上传功能落地的文件）
    selected_label = selected_upscaler()
    upscaler_variants = []
    seen_filenames: set[str] = set()
    for label, info in UPSCALER_VARIANTS.items():
        target = upscaler_target(label, r)
        seen_filenames.add(info["filename"])
        hf_repo = (info.get("hf") or (None,))[0]
        ms_repo = (info.get("ms") or (None,))[0]
        upscaler_variants.append({
            "label": label,
            "filename": info["filename"],
            "kind": "preset",
            "hf_repo": hf_repo,
            "ms_repo": ms_repo,
            "size_mb": info.get("size_mb"),
            "description": info.get("description", ""),
            "target_path": str(target),
            "is_current": label == selected_label,
            **_file_status(target),
        })
    up_dir = upscaler_dir(r)
    if up_dir.exists():
        for f in sorted(up_dir.iterdir()):
            if not f.is_file():
                continue
            if f.suffix.lower() not in UPSCALER_EXTS:
                continue
            if f.name in seen_filenames:
                continue
            upscaler_variants.append({
                "label": f.name,
                "filename": f.name,
                "kind": "custom",
                "hf_repo": None,
                "ms_repo": None,
                "size_mb": None,
                "description": "自定义/已下载",
                "target_path": str(f),
                "is_current": f.name == selected_label or f.stem == selected_label,
                **_file_status(f),
            })

    # 放大器统一行：preset + download/local 候选 + 扫盘兜底（D6）。
    # value 语义与 selected_upscaler 一致：preset=label / download·scanned=
    # 文件名 / local=绝对路径。
    up_rows: list[dict[str, Any]] = []
    for label, info in UPSCALER_VARIANTS.items():
        target = upscaler_target(label, r)
        st = _file_status(target)
        up_rows.append(_source_row(
            kind="preset", value=label,
            download_id="upscaler",
            exists=st["exists"], size=st["size"],
            is_current=label == selected_label,
            description=str(info.get("description", "")),
            size_estimate=int(info.get("size_mb") or 0) * 1_000_000,
            extra={
                "hf_repo": (info.get("hf") or ("",))[0] or "",
                "ms_repo": (info.get("ms") or ("",))[0] or "",
            },
        ))
    _custom_filenames: set[str] = set()
    for c in source_cfg.get("upscaler", []):
        if c.kind == "download":
            save_name = Path(c.filename).name
            _custom_filenames.add(save_name)
            target = upscaler_dir(r) / save_name
            st = _file_status(target)
            up_rows.append(_source_row(
                kind="download", value=save_name,
                download_id="upscaler_custom", download_variant=c.filename,
                status_key=f"upscaler:custom:{save_name}",
                exists=st["exists"], size=st["size"],
                is_current=save_name == selected_label,
                description=c.repo,
                candidate=c.model_dump(),
            ))
        else:
            p = Path(c.path)
            st = _file_status(p)
            up_rows.append(_source_row(
                kind="local", value=c.path, label=p.name, download_id=None,
                exists=st["exists"], size=st["size"],
                is_current=c.path == selected_label,
                candidate=c.model_dump(),
            ))
    for v in upscaler_variants:
        # 扫盘发现、但既非预设也未被 download 候选登记的文件：只能删除
        if v["kind"] != "custom" or v["filename"] in _custom_filenames:
            continue
        up_rows.append(_source_row(
            kind="scanned", value=v["filename"], label=v["label"],
            download_id="upscaler", download_variant=v["filename"],
            status_key=f"upscaler:custom:{v['filename']}",
            exists=bool(v.get("exists")), size=int(v.get("size") or 0),
            is_current=bool(v.get("is_current")),
        ))
    model_source_rows["upscaler"] = up_rows

    # 主模型族统一行：官方 variants（preset）+ download 候选（第三方微调，
    # value=落盘绝对路径，与 local/selected 同语义）+ local（PathPicker 注册）。
    for family_id in FAMILY_ASSETS:
        main = family_sections.get(f"{family_id}_main")
        if not main:
            continue
        selected_val = str(
            models_cfg.selected.get(family_id) or main.get("latest") or "")
        fam_rows: list[dict[str, Any]] = []
        for v in main["variants"]:
            fam_rows.append(_source_row(
                kind="preset", value=v["variant"],
                download_id=f"{family_id}_main",
                exists=bool(v.get("exists")), size=int(v.get("size") or 0),
                is_current=v["variant"] == selected_val,
                extra={"purpose": str(v.get("purpose") or "")},
            ))
        for c in source_cfg.get(family_id, []):
            if c.kind == "download":
                save_name = Path(c.filename).name
                target = r / "diffusion_models" / save_name
                st = _file_status(target)
                fam_rows.append(_source_row(
                    kind="download", value=str(target), label=save_name,
                    download_id=f"{family_id}_custom", download_variant=c.filename,
                    exists=st["exists"], size=st["size"],
                    is_current=str(target) == selected_val,
                    description=c.repo,
                    candidate=c.model_dump(),
                ))
            else:
                p = Path(c.path)
                st = _file_status(p)
                fam_rows.append(_source_row(
                    kind="local", value=c.path, label=p.name, download_id=None,
                    exists=st["exists"], size=st["size"],
                    is_current=c.path == selected_val,
                    candidate=c.model_dump(),
                ))
        model_source_rows[family_id] = fam_rows

    return {
        "models_root": str(r),
        **family_sections,
        "wd14": {
            "id": "wd14",
            "name": "WD14",
            "description": "SmilingWolf 系列 ONNX 打标",
            "repo": "SmilingWolf/*",
            "current_model_id": wd14_cfg.model_id,
            "variants": wd14_variants,
        },
        "cltagger": {
            "id": "cltagger",
            "name": "CLTagger",
            "description": "cella110n CLTagger ONNX",
            "repo": cl_cfg.model_id,
            "target_dir": str(cl_root),
            "current_model_path": cl_cfg.model_path,
            "current_tag_mapping_path": cl_cfg.tag_mapping_path,
            "variants": cl_variants,
        },
        "eval_metrics": {
            "id": "eval_metrics",
            "name": "评估指标模型",
            "description": "CLIP / DINO，用于 LoRA 训练后指标评估",
            "variants": eval_variants,
        },
        # 评估指标 registry（Settings 复选框列表用）：每个指标的 key/label/说明/默认。
        "eval_metric_catalog": eval_registry.public_catalog(),
        "upscalers": {
            "id": "upscalers",
            "name": "放大器",
            "description": "预处理阶段的 super-resolution 模型",
            "default": DEFAULT_UPSCALER,
            "current": selected_label,
            "target_dir": str(upscaler_dir(r)),
            "variants": upscaler_variants,
        },
        # 统一来源候选行（前端泛化候选卡消费；键 = domain）。
        "model_sources": model_source_rows,
        # 按类型的下载源选择：双源类型给 dropdown，固定 HF 的给单选指示。
        # current 来自 secrets.download_sources（已迁移种子）；available 决定前端
        # 渲染真 dropdown 还是 1-option 禁用框。
        "download_source_options": {
            "training": {"current": src_cfg.get("training", "huggingface"),
                         "available": ["huggingface", "modelscope"]},
            "wd14": {"current": src_cfg.get("wd14", "huggingface"),
                     "available": ["huggingface", "modelscope"]},
            "eval": {"current": src_cfg.get("eval", "huggingface"),
                     "available": ["huggingface", "modelscope"]},
            "upscaler": {"current": src_cfg.get("upscaler", "huggingface"),
                         "available": ["huggingface", "modelscope"]},
            "cltagger": {"current": "huggingface", "available": ["huggingface"]},
            "taeflux": {"current": "huggingface", "available": ["huggingface"]},
        },
        "downloads": get_status_snapshot(),
    }
