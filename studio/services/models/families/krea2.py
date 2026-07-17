"""Krea 2 族资产清单。

Raw / Turbo 主权重使用 Krea 官方仓库提供的单文件 checkpoint；文本编码器
使用完整 Qwen3-VL-4B-Instruct transformers 目录。Krea 2 与 Anima 共享同一
Qwen-Image VAE，因此不创建第二份 VAE 资产或磁盘副本。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .... import secrets
from ..paths import models_root, safe_dir_name
from ..paths import qwen_image_vae_target

KREA2_VARIANTS: dict[str, dict[str, Any]] = {
    "raw": {
        "repo": "krea/Krea-2-Raw",
        "subpath": "raw.safetensors",
        "ms_repo": "Comfy-Org/Krea-2",
        "ms_subpath": "diffusion_models/krea2_raw_bf16.safetensors",
        "filename": "krea2-raw-bf16.safetensors",
        "purpose": "training",
        "size_estimate": 26_300_000_000,
    },
    # Comfy-Org 官方量化管线的 fp8_scaled Raw：训练（fp8_base，权重显存
    # 25.6→13.1GB，24GB 卡可训）与推理（fp8 出图链）双用途。purpose 维持
    # training——is_distilled_path 按 purpose=inference 判 Turbo 蒸馏，
    # 本文件是 Raw（非蒸馏）不能标 inference。
    "raw_fp8": {
        "repo": "Comfy-Org/Krea-2",
        "subpath": "diffusion_models/krea2_raw_fp8_scaled.safetensors",
        "ms_repo": "Comfy-Org/Krea-2",
        "ms_subpath": "diffusion_models/krea2_raw_fp8_scaled.safetensors",
        "filename": "krea2-raw-fp8-scaled.safetensors",
        "purpose": "training",
        "size_estimate": 13_100_000_000,
    },
    "turbo": {
        "repo": "krea/Krea-2-Turbo",
        "subpath": "turbo.safetensors",
        "ms_repo": "Comfy-Org/Krea-2",
        "ms_subpath": "diffusion_models/krea2_turbo_bf16.safetensors",
        "filename": "krea2-turbo-bf16.safetensors",
        "purpose": "inference",
        "size_estimate": 26_300_000_000,
    },
}
LATEST_KREA2 = "raw"
KREA2_LICENSE = "Krea 2 Community License"
KREA2_LICENSE_URL = (
    "https://huggingface.co/krea/Krea-2-Raw/blob/main/LICENSE.pdf"
)

QWEN3_VL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_VL_FILES = [
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]


def krea2_main_target(root: Path, variant: str) -> Path:
    if variant == "latest":
        variant = LATEST_KREA2
    try:
        filename = str(KREA2_VARIANTS[variant]["filename"])
    except KeyError:
        raise ValueError(f"unknown Krea 2 variant {variant!r}") from None
    return root / "diffusion_models" / filename


def qwen3_vl_dir(root: Path) -> Path:
    """Krea 2 文本编码器目录；与 Anima 的 legacy 扁平目录隔离。"""
    return root / "text_encoders" / safe_dir_name(QWEN3_VL_REPO)


def selected_krea2_variant() -> str:
    try:
        variant = secrets.load().models.selected.get("krea2")
    except Exception:
        variant = None
    return str(variant) if variant in KREA2_VARIANTS else LATEST_KREA2


def selected_krea2_transformer_path() -> str:
    """返回设置页选中的 Krea2 官方 variant 或有效本地模型路径。"""
    try:
        selected = secrets.load().models.selected.get("krea2")
    except Exception:
        selected = None
    if selected and selected not in KREA2_VARIANTS:
        path = Path(str(selected).strip()).expanduser()
        if path.is_file():
            return str(path)
    return str(krea2_main_target(models_root(), selected_krea2_variant()))


def krea2_transformer_path_for(sel: Optional[str]) -> str:
    selected = (sel or "").strip()
    if not selected:
        return selected_krea2_transformer_path()
    if selected == "latest" or selected in KREA2_VARIANTS:
        return str(krea2_main_target(models_root(), selected))
    path = Path(selected).expanduser()
    if path.is_file():
        return str(path)
    return selected_krea2_transformer_path()


def _training_variant() -> str:
    for name, info in KREA2_VARIANTS.items():
        if info["purpose"] == "training":
            return name
    return LATEST_KREA2


def _training_default_transformer_path() -> str:
    """新建训练 version 的默认主权重：官方 variant 必须是 training 用途。

    Settings 的 selected 是训练 / 推理共用的一个选择——用户为 Generate 页
    选中 Turbo（purpose=inference，TDM 蒸馏推理模型）时，新训练 version 不
    静默跟随，落回 training variant（Raw）。用户注册的本地 custom 路径
    （社区微调等）无 purpose 元数据，尊重用户选择不加白名单；显式传
    base_model（含显式选 turbo）同样尊重，不经过本函数。
    """
    try:
        selected = secrets.load().models.selected.get("krea2")
    except Exception:
        selected = None
    if selected and selected not in KREA2_VARIANTS:
        path = Path(str(selected).strip()).expanduser()
        if path.is_file():
            return str(path)
    variant = selected if selected in KREA2_VARIANTS else LATEST_KREA2
    if KREA2_VARIANTS[variant]["purpose"] != "training":
        variant = _training_variant()
    return str(krea2_main_target(models_root(), variant))


def default_paths_for_new_version(base_model: Optional[str] = None) -> dict[str, str]:
    """返回 Krea 2 新 version 应使用的本地资产路径。"""
    root = models_root()
    transformer = (
        krea2_transformer_path_for(base_model)
        if (base_model or "").strip()
        else _training_default_transformer_path()
    )
    return {
        "transformer_path": transformer,
        "vae_path": str(qwen_image_vae_target(root)),
        "text_encoder_path": str(qwen3_vl_dir(root)),
        "t5_tokenizer_path": "",
    }


def is_distilled_path(path: str) -> bool:
    """transformer 路径是否为官方 Turbo（TDM 蒸馏推理）variant。

    Turbo 与 Raw 结构全等（430 键同形状），loader 指纹物理上无法区分——
    只能按 catalog variant 的文件名判。用户注册的 custom 权重无 purpose
    元数据，一律按非蒸馏处理（A1：不加白名单，采样参数由用户控制）。
    """
    if not path:
        return False
    name = Path(str(path)).name
    return any(
        info["purpose"] == "inference" and info["filename"] == name
        for info in KREA2_VARIANTS.values()
    )


def _file_status(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        return {"exists": True, "size": stat.st_size, "mtime": stat.st_mtime}
    except OSError:
        return {"exists": False, "size": 0, "mtime": 0.0}


def catalog_sections(root: Path, models_cfg: Any) -> dict[str, Any]:
    variants = []
    for name, info in KREA2_VARIANTS.items():
        target = krea2_main_target(root, name)
        variants.append({
            "variant": name,
            "is_latest": name == LATEST_KREA2,
            "repo": info["repo"],
            "purpose": info["purpose"],
            "size_estimate": info["size_estimate"],
            "target_path": str(target),
            **_file_status(target),
        })

    custom_models = []
    for registered_path in models_cfg.custom.get("krea2", []):
        target = Path(str(registered_path)).expanduser()
        custom_models.append({
            "path": registered_path,
            "name": target.name,
            **_file_status(target),
        })

    text_dir = qwen3_vl_dir(root)
    return {
        "krea2_main": {
            "id": "krea2_main",
            "name": "Krea 2 主模型",
            "description": (
                "Raw 训练底模（bf16 26.3 GB；fp8 13.1 GB，训练/出图权重"
                "显存减半）/ Turbo 推理底模（26.3 GB）"
            ),
            "repo": "krea/Krea-2-{Raw,Turbo}",
            "variants": variants,
            "custom": custom_models,
            "selected": models_cfg.selected.get("krea2") or LATEST_KREA2,
            "latest": LATEST_KREA2,
            "license": KREA2_LICENSE,
            "license_url": KREA2_LICENSE_URL,
        },
        "krea2_text_encoder": {
            "id": "krea2_text_encoder",
            "name": "Krea 2 · Qwen3-VL-4B-Instruct",
            "description": "自然语言文本编码器（约 8.89 GB）",
            "repo": QWEN3_VL_REPO,
            "target_dir": str(text_dir),
            "files": [
                {"name": filename, **_file_status(text_dir / filename)}
                for filename in QWEN3_VL_FILES
            ],
        },
    }


class _Krea2Assets:
    family_id = "krea2"
    display_name = "Krea 2"
    #: 注销 custom 时 selected 的回退目标（最新官方 variant key）
    latest = LATEST_KREA2

    default_paths_for_new_version = staticmethod(default_paths_for_new_version)
    transformer_path_for = staticmethod(krea2_transformer_path_for)
    selected_variant = staticmethod(selected_krea2_variant)
    catalog_sections = staticmethod(catalog_sections)
    is_distilled_path = staticmethod(is_distilled_path)


ASSETS = _Krea2Assets()
