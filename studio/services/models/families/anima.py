"""Anima 族资产清单（多模型 PR-4，自 paths.py 函数级迁入）。

权重 repo / variant 清单 / 下载 target / selected 解析 / 新建 version 默认
路径——全部是族知识。工具模型（WD14 等）与 models_root 留在 ..paths。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .... import secrets
from ..paths import models_root, qwen_image_vae_target
from .custom_paths import arch_summary, registered_main_paths

ANIMA_REPO = "circlestone-labs/Anima"


@dataclass(frozen=True)
class AnimaVariant:
    """内置下载目录里的一个 Anima 底模条目。

    只承载「去哪下载、怎么称呼」；**不承载任何加载信息**——层数等架构一律
    从文件 header 探测（studio.services.inference.checkpoint_arch），内置条目
    与用户手动放进来的文件走完全同一条路。
    """

    key: str
    subpath: str
    repo: str = ANIMA_REPO
    #: official（circlestone-labs 官方）| community（第三方续训 / 扩展版）
    group: str = "official"
    #: 显示名；空 → key
    label: str = ""
    #: 第三方条目的作者（前端拼描述用；官方为空）
    author: str = ""

    @property
    def filename(self) -> str:
        return Path(self.subpath).name


# 顺序：官方最新在前，第三方在官方之后。`find_anima_main` 的 fallback 查找按
# 本表序遍历（LATEST 优先），`build_catalog` 给 UI 的 variants 列表也复用本
# 顺序——所以官方新版本加在最前，老版本往下排，第三方条目放官方后面。
ANIMA_VARIANT_TABLE: tuple[AnimaVariant, ...] = (
    AnimaVariant("1.0", "split_files/diffusion_models/anima-base-v1.0.safetensors"),
    AnimaVariant("preview3-base", "split_files/diffusion_models/anima-preview3-base.safetensors"),
    AnimaVariant("preview2", "split_files/diffusion_models/anima-preview2.safetensors"),
    AnimaVariant("preview", "split_files/diffusion_models/anima-preview.safetensors"),
    # 第三方：Anima v1.0 的 28 层 DiT 插层扩到 40 层并续训（LLaMA Pro 式），
    # TE / VAE / key 布局全同；许可同 Anima（非商用，derivative）。作者仍在
    # 出新版：跟版本 = 改这一行的 filename。
    AnimaVariant(
        "2.9b-preview-v1", "Anima-2.9B-preview-v1.safetensors",
        repo="Gazingstars123/Anima-2.9B", group="community",
        label="Anima-2.9B preview-v1", author="Gazingstars123",
    ),
)
ANIMA_VARIANT_BY_KEY: dict[str, AnimaVariant] = {v.key: v for v in ANIMA_VARIANT_TABLE}
#: 兼容读面：variant key → repo 内子路径（downloader / CLI / 测试沿用）
ANIMA_VARIANTS: dict[str, str] = {v.key: v.subpath for v in ANIMA_VARIANT_TABLE}
LATEST_ANIMA = "1.0"
ANIMA_VAE_PATH = "split_files/vae/qwen_image_vae.safetensors"


def anima_variant_repo(variant: str) -> str:
    """variant 的下载 repo（官方 circlestone-labs/Anima；第三方各自的 repo）。"""
    if variant == "latest":
        variant = LATEST_ANIMA
    v = ANIMA_VARIANT_BY_KEY.get(variant)
    if v is None:
        raise ValueError(f"unknown variant {variant!r}")
    return v.repo

QWEN_REPO = "Qwen/Qwen3-0.6B-Base"
# 注：Qwen3 把 special tokens 直接塞进 tokenizer.json，所以 repo 里没有
# `special_tokens_map.json`（旧 Qwen 版本有，照搬就 404）。
QWEN_FILES = [
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "config.json",
]

T5_REPO = "google/t5-v1_1-xxl"
T5_FILES = [
    "spiece.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
]


def anima_main_target(root: Path, variant: str) -> Path:
    if variant == "latest":
        variant = LATEST_ANIMA
    if variant not in ANIMA_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return root / "diffusion_models" / Path(ANIMA_VARIANTS[variant]).name


def qwen_dir(root: Path) -> Path:
    return root / "text_encoders"


def t5_tokenizer_dir(root: Path) -> Path:
    return root / "t5_tokenizer"


def find_anima_main(root: Optional[Path] = None) -> Optional[Path]:
    """按 ANIMA_VARIANTS 优先级（latest 在前）找第一个磁盘上存在的主模型。

    仅做兜底（裸 CLI / yaml 缺失时）；Studio 创建 version 时优先用
    `selected_anima_path()` 拿用户在 settings 里选定的 variant。
    """
    r = root or models_root()
    order = [LATEST_ANIMA] + [v for v in ANIMA_VARIANTS if v != LATEST_ANIMA]
    for v in order:
        target = anima_main_target(r, v)
        if target.exists():
            return target
    return None


def selected_anima_variant() -> str:
    """读 `secrets.models.selected_anima`，回退 LATEST_ANIMA。"""
    try:
        v = secrets.load().models.selected_anima
    except Exception:
        v = None
    if v and v in ANIMA_VARIANTS:
        return v
    return LATEST_ANIMA


def selected_anima_transformer_path() -> str:
    """选中主模型的 transformer 绝对路径（训练新建默认 + 测试出图共用）。

    `selected_anima` 为官方 variant key → 用 `anima_main_target` 算路径；为用户
    注册的本地 custom 路径（不在 ANIMA_VARIANTS 且文件存在）→ 直接返回该路径。
    custom 路径失效（被删 / 移走）时回退到当前 variant，保证永不返回不存在的
    死路径。
    """
    try:
        sel = secrets.load().models.selected_anima
    except Exception:
        sel = None
    if sel and sel not in ANIMA_VARIANTS:
        p = Path(str(sel).strip()).expanduser()
        if p.exists():
            return str(p)
    return str(anima_main_target(models_root(), selected_anima_variant()))


def anima_transformer_path_for(sel: Optional[str]) -> str:
    """把一个显式的主模型选择解析成 transformer 绝对路径。

    `sel` 语义同 `secrets.models.selected_anima`：官方 variant key（"1.0" /
    "latest" 等）或注册的本地 custom `.safetensors` 绝对路径。空值 → 回退到
    Settings 里 `selected_anima` 的解析结果（`selected_anima_transformer_path`），
    即先验生成 / 测试出图沿用「设置页选定的底模」。custom 路径失效（被删 /
    移走）→ 回退当前 selected，绝不返回不存在的死路径。
    """
    s = (sel or "").strip()
    if not s:
        return selected_anima_transformer_path()
    if s == "latest" or s in ANIMA_VARIANTS:
        return str(anima_main_target(models_root(), s))
    p = Path(s).expanduser()
    if p.exists():
        return str(p)
    return selected_anima_transformer_path()


def default_paths_for_new_version(base_model: Optional[str] = None) -> dict[str, str]:
    """Studio 创建新 version 时用：返回 4 项路径的**绝对路径字符串**。

    根据当前 `secrets.models.root` 和 `secrets.models.selected_anima` 计算。
    用户在 settings 切了 selected_anima（官方 variant 或注册的本地 custom 路径）
    → 之后新建的 version 自动用新选择；已存在 version 的 yaml 不动（重现性）。

    `base_model` 非空时只覆盖 transformer_path（先验生成 / 测试出图按用户在
    页面上临时选定的底模出图）；vae / text_encoder / t5 仍跟随全局设置。
    """
    root = models_root()
    return {
        "transformer_path": anima_transformer_path_for(base_model),
        "vae_path": str(qwen_image_vae_target(root)),
        "text_encoder_path": str(qwen_dir(root)),
        "t5_tokenizer_path": str(t5_tokenizer_dir(root)),
    }


# ---------------------------------------------------------------------------
# catalog 区块（自 catalog.py build_catalog 迁入，输出形状不变——前端零改动）
# ---------------------------------------------------------------------------


def _file_status(p: Path) -> dict[str, Any]:
    try:
        st = p.stat()
        return {"exists": True, "size": st.st_size, "mtime": st.st_mtime}
    except OSError:
        return {"exists": False, "size": 0, "mtime": 0.0}


def catalog_sections(root: Path, models_cfg: Any, source_cfg: Any = None) -> dict[str, Any]:
    """/api/models/catalog 的 Anima 族区块（anima_main / anima_vae / qwen3 / t5_tokenizer）。

    variants / custom 行带 ``arch``（层数 / 通道 / 参数量，文件存在才有，header
    探测）——底模下拉与设置卡据此显示「N 层」，不写死任何层数。
    """
    anima_variants = []
    for v in ANIMA_VARIANT_TABLE:
        target = anima_main_target(root, v.key)
        anima_variants.append({
            "variant": v.key,
            "label": v.label or v.key,
            "group": v.group,
            "repo": v.repo,
            "author": v.author,
            "is_latest": v.key == LATEST_ANIMA,
            "target_path": str(target),
            **_file_status(target),
            "arch": arch_summary(target),
        })

    custom_anima = []
    for target in registered_main_paths(root, models_cfg, "anima", source_cfg):
        custom_anima.append({
            "path": str(target),
            "name": target.name,
            **_file_status(target),
            "arch": arch_summary(target),
        })

    vae_target = qwen_image_vae_target(root)
    qwen_d = qwen_dir(root)
    t5_d = t5_tokenizer_dir(root)
    return {
        "anima_main": {
            "id": "anima_main",
            "name": "Anima 主模型",
            "description": "Cosmos transformer",
            "repo": ANIMA_REPO,
            "variants": anima_variants,
            "custom": custom_anima,
            "selected": models_cfg.selected_anima,
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
    }


def path_choices(root: Path, models_cfg: Any, source_cfg: Any = None) -> dict[str, list[dict[str, Any]]]:
    """Train 页 4 个模型路径字段的 dropdown 候选（Anima 族）。

    **只列磁盘上已就绪的**：没下载的选了也训不起来，下载是 Settings 页的职责。
    label 一律取 basename（文件名 / 目录名），与用户在 Settings 看到的一致；
    `group`（official / community / custom）/ `note` 是给前端翻译的 id，不是
    显示文案；``arch`` 是 header 探测的层数 / 参数量（前端标「N 层」）。
    """
    transformer: list[dict[str, Any]] = []
    for v in ANIMA_VARIANT_TABLE:
        target = anima_main_target(root, v.key)
        if target.exists():
            transformer.append({
                "label": target.name,
                "path": str(target),
                "group": v.group,
                "note": "latest" if v.key == LATEST_ANIMA else "",
                "arch": arch_summary(target),
            })
    variant_paths = {str(anima_main_target(root, v.key)) for v in ANIMA_VARIANT_TABLE}
    for target in registered_main_paths(root, models_cfg, "anima", source_cfg):
        # 用户又把内置条目的落盘文件注册了一遍 → 不重复列
        if str(target) in variant_paths or not target.exists():
            continue
        transformer.append({
            "label": target.name, "path": str(target),
            "group": "custom", "note": "",
            "arch": arch_summary(target),
        })

    def _ready_dir(d: Path, files: list[str]) -> list[dict[str, Any]]:
        """目录型资产：必需文件齐全才算就绪（与 catalog 的「已下载」同口径）。"""
        if not all((d / f).exists() for f in files):
            return []
        return [{"label": d.name, "path": str(d), "group": "official", "note": ""}]

    vae = qwen_image_vae_target(root)
    return {
        "transformer_path": transformer,
        "vae_path": (
            [{"label": vae.name, "path": str(vae), "group": "official", "note": ""}]
            if vae.exists() else []
        ),
        "text_encoder_path": _ready_dir(qwen_dir(root), QWEN_FILES),
        "t5_tokenizer_path": _ready_dir(t5_tokenizer_dir(root), T5_FILES),
    }


class _AnimaAssets:
    """duck-typed 族资产对象（families/__init__.py 注册）。"""

    family_id = "anima"
    display_name = "Anima"
    #: 注销 custom 时 selected 的回退目标（最新官方 variant key）
    latest = LATEST_ANIMA

    default_paths_for_new_version = staticmethod(default_paths_for_new_version)
    transformer_path_for = staticmethod(anima_transformer_path_for)
    selected_variant = staticmethod(selected_anima_variant)
    catalog_sections = staticmethod(catalog_sections)
    path_choices = staticmethod(path_choices)
    # Anima 无蒸馏推理 variant
    is_distilled_path = staticmethod(lambda path: False)


ASSETS = _AnimaAssets()
