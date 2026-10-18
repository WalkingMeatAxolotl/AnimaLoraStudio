"""MOSS-Image 族资产清单（MOSS 设计稿 §7）。

主权重取 Comfy-Org 重打包的单文件（bf16 / int8_tensorwise+convrot，Base / SFT 各一，
civitai 官方页挂的就是这一套）；文本编码器两种：

- int8：Comfy-Org 的 ``qwen3.5_9b_int8_convrot`` 单文件 + 官方仓库的 config / tokenizer 小文件
- bf16：官方 ``OpenMOSS-Team/MOSS-Image-Base`` 的 ``text_encoder/`` 分片 + ``tokenizer/``
  （Base 与 SFT 的 TE 逐文件相同，只存一份）

两种 TE 都落成「一个目录 = config.json + tokenizer 文件 + 权重」，训练侧
``MossTextStack`` 直接认这个布局。默认选 int8：bf16 DiT 10.2 GB / TE 16.8 GB 超出多数
消费卡的可用显存（设计稿 M2）。下载动作与设置页 UI 在后续 PR 接入。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .... import secrets
from ..paths import flux2_vae_target, models_root, safe_dir_name
from .custom_paths import registered_main_paths

FAMILY_ID = "moss"
COMFY_REPO = "Comfy-Org/MOSS-Image"
OFFICIAL_REPO = "OpenMOSS-Team/MOSS-Image-Base"
MOSS_LICENSE = "Apache-2.0"
MOSS_LICENSE_URL = "https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base"

MOSS_VARIANTS: dict[str, dict[str, Any]] = {
    "base_int8": {
        "repo": COMFY_REPO,
        "subpath": "diffusion_models/moss_image_base_int8_convrot.safetensors",
        "ms_repo": COMFY_REPO,
        "ms_subpath": "diffusion_models/moss_image_base_int8_convrot.safetensors",
        "filename": "moss-image-base-int8-convrot.safetensors",
        "purpose": "training",
        "size_estimate": 5_413_698_958,
    },
    "base_bf16": {
        "repo": COMFY_REPO,
        "subpath": "diffusion_models/moss_image_base_bf16.safetensors",
        "ms_repo": COMFY_REPO,
        "ms_subpath": "diffusion_models/moss_image_base_bf16.safetensors",
        "filename": "moss-image-base-bf16.safetensors",
        "purpose": "training",
        "size_estimate": 10_160_000_000,
    },
    "sft_int8": {
        "repo": COMFY_REPO,
        "subpath": "diffusion_models/moss_image_sft_int8_convrot.safetensors",
        "ms_repo": COMFY_REPO,
        "ms_subpath": "diffusion_models/moss_image_sft_int8_convrot.safetensors",
        "filename": "moss-image-sft-int8-convrot.safetensors",
        "purpose": "training",
        "size_estimate": 5_413_698_958,
    },
    "sft_bf16": {
        "repo": COMFY_REPO,
        "subpath": "diffusion_models/moss_image_sft_bf16.safetensors",
        "ms_repo": COMFY_REPO,
        "ms_subpath": "diffusion_models/moss_image_sft_bf16.safetensors",
        "filename": "moss-image-sft-bf16.safetensors",
        "purpose": "training",
        "size_estimate": 10_160_000_000,
    },
}
#: 默认 variant（设计稿 M3：默认训 Base；M2：int8 适配多数显卡）
LATEST_MOSS = "base_int8"

#: TE 小文件：(仓库内路径, 落盘文件名)。tokenizer/ 与 text_encoder/ 在官方仓库并列，
#: 这里平铺进同一目录。
_TE_SMALL_FILES: tuple[tuple[str, str], ...] = (
    ("text_encoder/config.json", "config.json"),
    ("tokenizer/tokenizer.json", "tokenizer.json"),
    ("tokenizer/tokenizer_config.json", "tokenizer_config.json"),
    ("tokenizer/chat_template.jinja", "chat_template.jinja"),
)
#: bf16 TE 权重分片（官方仓库 text_encoder/）
_TE_BF16_SHARDS: tuple[str, ...] = (
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
)
TE_INT8_SUBPATH = "text_encoders/qwen3.5_9b_int8_convrot.safetensors"
TE_INT8_FILE = "qwen3.5_9b_int8_convrot.safetensors"
VAE_SUBPATH = "vae/flux2-vae.safetensors"

#: TE variant（首项 = 默认）
MOSS_TE_VARIANTS = ("int8", "bf16")


def te_download_files(variant: str) -> list[dict[str, str]]:
    """一个 TE variant 的全部下载项：``{repo, subpath, filename}``（下载中心接入用）。"""
    small = [
        {"repo": OFFICIAL_REPO, "subpath": src, "filename": dst}
        for src, dst in _TE_SMALL_FILES
    ]
    if variant == "int8":
        return [{"repo": COMFY_REPO, "subpath": TE_INT8_SUBPATH, "filename": TE_INT8_FILE}, *small]
    if variant == "bf16":
        return [
            {"repo": OFFICIAL_REPO, "subpath": f"text_encoder/{name}", "filename": name}
            for name in _TE_BF16_SHARDS
        ] + small
    raise ValueError(f"unknown MOSS text encoder variant {variant!r}")


def moss_main_target(root: Path, variant: str) -> Path:
    if variant == "latest":
        variant = LATEST_MOSS
    try:
        filename = str(MOSS_VARIANTS[variant]["filename"])
    except KeyError:
        raise ValueError(f"unknown MOSS-Image variant {variant!r}") from None
    return root / "diffusion_models" / filename


def moss_te_dir(root: Path, variant: str) -> Path:
    if variant == "int8":
        return root / "text_encoders" / "moss-qwen3.5-9b-int8"
    if variant == "bf16":
        return root / "text_encoders" / safe_dir_name(f"{OFFICIAL_REPO}-text_encoder")
    raise ValueError(f"unknown MOSS text encoder variant {variant!r}")


def selected_te_variant() -> str:
    try:
        variant = secrets.load().models.selected_te.get(FAMILY_ID)
    except Exception:
        variant = None
    return str(variant) if variant in MOSS_TE_VARIANTS else MOSS_TE_VARIANTS[0]


def selected_moss_variant() -> str:
    try:
        variant = secrets.load().models.selected.get(FAMILY_ID)
    except Exception:
        variant = None
    return str(variant) if variant in MOSS_VARIANTS else LATEST_MOSS


def moss_transformer_path_for(sel: Optional[str]) -> str:
    """显式底模选择 → 主权重绝对路径（官方 variant 名 / 本地文件 / 空 = 设置页选中）。"""
    selected = (sel or "").strip()
    if not selected:
        try:
            selected = str(secrets.load().models.selected.get(FAMILY_ID) or "")
        except Exception:
            selected = ""
        if selected and selected not in MOSS_VARIANTS:
            path = Path(selected).expanduser()
            if path.is_file():
                return str(path)
        return str(moss_main_target(models_root(), selected_moss_variant()))
    if selected == "latest" or selected in MOSS_VARIANTS:
        return str(moss_main_target(models_root(), selected))
    path = Path(selected).expanduser()
    if path.is_file():
        return str(path)
    return str(moss_main_target(models_root(), selected_moss_variant()))


def default_paths_for_new_version(base_model: Optional[str] = None) -> dict[str, str]:
    root = models_root()
    return {
        "transformer_path": moss_transformer_path_for(base_model),
        "vae_path": str(flux2_vae_target(root)),
        "text_encoder_path": str(moss_te_dir(root, selected_te_variant())),
        "t5_tokenizer_path": "",
    }


def is_distilled_path(path: str) -> bool:
    """MOSS 没有官方蒸馏推理 variant（社区少步版按 custom 处理，不加白名单）。"""
    del path
    return False


def _file_status(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        return {"exists": True, "size": stat.st_size, "mtime": stat.st_mtime}
    except OSError:
        return {"exists": False, "size": 0, "mtime": 0.0}


def _te_ready(root: Path, variant: str) -> bool:
    d = moss_te_dir(root, variant)
    return all((d / item["filename"]).exists() for item in te_download_files(variant))


def catalog_sections(root: Path, models_cfg: Any, source_cfg: Any = None) -> dict[str, Any]:
    variants = []
    for name, info in MOSS_VARIANTS.items():
        target = moss_main_target(root, name)
        variants.append({
            "variant": name,
            "is_latest": name == LATEST_MOSS,
            "repo": info["repo"],
            "purpose": info["purpose"],
            "size_estimate": info["size_estimate"],
            "target_path": str(target),
            **_file_status(target),
        })
    custom = [
        {"path": str(target), "name": target.name, **_file_status(target)}
        for target in registered_main_paths(root, models_cfg, FAMILY_ID, source_cfg)
    ]
    te_selected = (getattr(models_cfg, "selected_te", None) or {}).get(FAMILY_ID)
    if te_selected not in MOSS_TE_VARIANTS:
        te_selected = MOSS_TE_VARIANTS[0]
    text_encoders = {}
    for variant in MOSS_TE_VARIANTS:
        d = moss_te_dir(root, variant)
        text_encoders[f"moss_text_encoder_{variant}"] = {
            "id": f"moss_text_encoder_{variant}",
            "name": f"MOSS-Image · Qwen3.5-9B {variant}",
            "target_dir": str(d),
            "selected": te_selected,
            "files": [
                {"name": item["filename"], **_file_status(d / item["filename"])}
                for item in te_download_files(variant)
            ],
        }
    vae = flux2_vae_target(root)
    return {
        "moss_main": {
            "id": "moss_main",
            "name": "MOSS-Image 主模型",
            "repo": COMFY_REPO,
            "variants": variants,
            "custom": custom,
            "selected": models_cfg.selected.get(FAMILY_ID) or LATEST_MOSS,
            "latest": LATEST_MOSS,
            "license": MOSS_LICENSE,
            "license_url": MOSS_LICENSE_URL,
        },
        **text_encoders,
        "moss_vae": {
            "id": "moss_vae",
            "name": "FLUX.2 VAE",
            "repo": COMFY_REPO,
            "target_path": str(vae),
            **_file_status(vae),
        },
    }


def path_choices(root: Path, models_cfg: Any, source_cfg: Any = None) -> dict[str, list[dict[str, Any]]]:
    """Train 页模型路径字段的 dropdown 候选：只列磁盘上已就绪的。"""
    transformer: list[dict[str, Any]] = []
    for name in MOSS_VARIANTS:
        target = moss_main_target(root, name)
        if target.exists():
            transformer.append({
                "label": target.name, "path": str(target), "group": "official",
                "note": "latest" if name == LATEST_MOSS else "",
            })
    official = {str(moss_main_target(root, name)) for name in MOSS_VARIANTS}
    for target in registered_main_paths(root, models_cfg, FAMILY_ID, source_cfg):
        if str(target) in official or not target.exists():
            continue
        transformer.append({"label": target.name, "path": str(target), "group": "custom", "note": ""})

    text_encoder = [
        {"label": moss_te_dir(root, v).name, "path": str(moss_te_dir(root, v)),
         "group": "official", "note": v}
        for v in MOSS_TE_VARIANTS if _te_ready(root, v)
    ]
    vae = flux2_vae_target(root)
    return {
        "transformer_path": transformer,
        "vae_path": (
            [{"label": vae.name, "path": str(vae), "group": "official", "note": ""}]
            if vae.exists() else []
        ),
        "text_encoder_path": text_encoder,
        "t5_tokenizer_path": [],
    }


class _MossAssets:
    family_id = FAMILY_ID
    display_name = "MOSS-Image"
    latest = LATEST_MOSS

    default_paths_for_new_version = staticmethod(default_paths_for_new_version)
    transformer_path_for = staticmethod(moss_transformer_path_for)
    selected_variant = staticmethod(selected_moss_variant)
    catalog_sections = staticmethod(catalog_sections)
    path_choices = staticmethod(path_choices)
    is_distilled_path = staticmethod(is_distilled_path)


ASSETS = _MossAssets()
