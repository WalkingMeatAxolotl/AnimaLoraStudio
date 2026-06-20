"""模型 catalog —— 扫盘组装"哪些模型已下载、目标路径、大小"（PR-3.8 拆出 4-way 第 4 个）。

build_catalog 是 /api/models/catalog 端点的核心，前端 ModelsPage 用它展示安装状态。
依赖 paths.py 的常量 + target 函数；不调下载（只读盘）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ... import secrets
from .downloader import get_status_snapshot
from .paths import (
    ANIMA_REPO,
    ANIMA_VAE_PATH,
    ANIMA_VARIANTS,
    CLTAGGER_VERSIONS,
    DEFAULT_UPSCALER,
    LATEST_ANIMA,
    QWEN_FILES,
    QWEN_REPO,
    T5_FILES,
    T5_REPO,
    TAEFLUX_FILES,
    TAEFLUX_REPO,
    UPSCALER_EXTS,
    UPSCALER_VARIANTS,
    WD14_FILES,
    anima_main_target,
    anima_vae_target,
    cltagger_target_root,
    models_root,
    qwen_dir,
    selected_upscaler,
    t5_tokenizer_dir,
    taeflux_dir,
    upscaler_dir,
    upscaler_target,
    wd14_target_dir,
)

# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------


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

    anima_variants = []
    for vname, subpath in ANIMA_VARIANTS.items():
        target = anima_main_target(r, vname)
        st = _file_status(target)
        anima_variants.append({
            "variant": vname,
            "is_latest": vname == LATEST_ANIMA,
            "target_path": str(target),
            **st,
        })

    vae_target = anima_vae_target(r)
    qwen_d = qwen_dir(r)
    t5_d = t5_tokenizer_dir(r)
    cl_cfg = secrets.load().cltagger
    wd14_cfg = secrets.load().wd14

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

    # CLTagger 版本预设（CLTAGGER_VERSIONS 写死的子目录布局）。
    cl_root = cltagger_target_root(r, cl_cfg.model_id)
    cl_variants = []
    for label, preset in CLTAGGER_VERSIONS.items():
        model_id = preset["model_id"]
        mp = preset["model_path"]
        tmp = preset["tag_mapping_path"]
        target_root = cltagger_target_root(r, model_id)
        file_names = [mp, tmp, *preset.get("extra_files", [])]
        files = [{"name": f, **_file_status(target_root / f)} for f in file_names]
        all_exist = all(f["exists"] for f in files)
        total_size = sum(f["size"] for f in files)
        cl_variants.append({
            "label": label,
            "model_id": model_id,
            "model_path": mp,
            "tag_mapping_path": tmp,
            "description": preset.get("description", ""),
            "target_dir": str(target_root),
            "is_current": (
                cl_cfg.model_id == model_id
                and cl_cfg.model_path == mp
                and cl_cfg.tag_mapping_path == tmp
            ),
            "exists": all_exist,
            "size": total_size,
            "files": files,
        })

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

    return {
        "models_root": str(r),
        "anima_main": {
            "id": "anima_main",
            "name": "Anima 主模型",
            "description": "Cosmos transformer (~4 GB)",
            "repo": ANIMA_REPO,
            "variants": anima_variants,
            "latest": LATEST_ANIMA,
        },
        "anima_vae": {
            "id": "anima_vae",
            "name": "Anima VAE",
            "description": "qwen_image_vae (~250 MB)",
            "repo": ANIMA_REPO,
            "target_path": str(vae_target),
            **_file_status(vae_target),
        },
        "qwen3": {
            "id": "qwen3",
            "name": "Qwen3-0.6B-Base",
            "description": "Text encoder (~1.2 GB)",
            "repo": QWEN_REPO,
            "target_dir": str(qwen_d),
            "files": [
                {"name": f, **_file_status(qwen_d / f)} for f in QWEN_FILES
            ],
        },
        "t5_tokenizer": {
            "id": "t5_tokenizer",
            "name": "T5 tokenizer",
            "description": "spiece.model 等 3 个 tokenizer 文件（不含权重）",
            "repo": T5_REPO,
            "target_dir": str(t5_d),
            "files": [
                {"name": f, **_file_status(t5_d / f)} for f in T5_FILES
            ],
        },
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
        "upscalers": {
            "id": "upscalers",
            "name": "放大器",
            "description": "预处理阶段的 super-resolution 模型",
            "default": DEFAULT_UPSCALER,
            "current": selected_label,
            "target_dir": str(upscaler_dir(r)),
            "variants": upscaler_variants,
        },
        "downloads": get_status_snapshot(),
    }
