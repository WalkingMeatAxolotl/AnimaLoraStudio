"""模型路径常量 + 本地路径解析（PR-3.8 从 model_downloader 1068 行拆出 4-way 第 1 个）。

只做"模型在本地哪儿"的回答：常量目录、各模型类型的 target Path、用户选定的
variant 读取（selected_anima / selected_upscaler）。不做下载、不读 endpoint /
mirror（那些在 sources.py）。

注意：`download_taeflux` 等 download_* 函数都搬到 downloader.py 了；这里只留
`taeflux_dir` / `taeflux_available` 这种"是否就绪"的查询。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ... import secrets
from ...paths import REPO_ROOT

# ---------------------------------------------------------------------------
# 模型清单常量（新版本发布时改这里）
# ---------------------------------------------------------------------------

ANIMA_REPO = "circlestone-labs/Anima"
# 顺序：最新在前。`find_anima_main` 的 fallback 查找按本 dict 序遍历，
# `build_catalog` 给 UI 的 variants 列表也直接复用本顺序——所以新版本
# 加在最前，老版本往下排。
ANIMA_VARIANTS: dict[str, str] = {
    "1.0":           "split_files/diffusion_models/anima-base-v1.0.safetensors",
    "preview3-base": "split_files/diffusion_models/anima-preview3-base.safetensors",
    "preview2":      "split_files/diffusion_models/anima-preview2.safetensors",
    "preview":       "split_files/diffusion_models/anima-preview.safetensors",
}
LATEST_ANIMA = "1.0"
ANIMA_VAE_PATH = "split_files/vae/qwen_image_vae.safetensors"

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

# TAEFlux：1.6MB 的 tiny autoencoder for Flux/Anima，daemon 预览中间步用。
# 用 diffusers.AutoencoderTiny.from_pretrained 加载 → 需要同时拿 config.json
# + safetensors 两个文件。
TAEFLUX_REPO = "madebyollin/taef1"
TAEFLUX_FILES = [
    "diffusion_pytorch_model.safetensors",
    "config.json",
]

CLTAGGER_REPO = "cella110n/cl_tagger"
CLTAGGER_V2_REPO = "cella110n/cl_tagger_v2"

# CLTagger 预设。v1 在 cella110n/cl_tagger 的版本子目录下；v2 是独立 gated
# repo，但文件仍在版本子目录下。新版本出现时往这里加一行，UI 自动作为 radio 暴露。
#
# 每个 variant 显式声明 extra_files（除 model_path / tag_mapping_path 之外还需
# 一并下载 / 校验的文件），不再靠"v2 一定有同名 .data"的启发式：
#   - v2 的 onnx 权重在外部 sidecar model.onnx.data（2GB+），缺它 onnxruntime 加载
#     external data 时才黑盒炸 → 必须纳入；
#   - model_metadata.json 一并带下，作为"下载是否完整"的就绪信号。
# 将来若出现单文件（无 .data）的 v2 变体，把它的 extra_files 留空即可，不会误要 .data。
CLTAGGER_VERSIONS: dict[str, dict[str, Any]] = {
    "cl_tagger_1_02": {
        "model_id": CLTAGGER_REPO,
        "model_path": "cl_tagger_1_02/model.onnx",
        "tag_mapping_path": "cl_tagger_1_02/tag_mapping.json",
        "extra_files": [],
        "description": "CLTagger 1.02 ONNX",
    },
    "cl_tagger_v2_v2_01a": {
        "model_id": CLTAGGER_V2_REPO,
        "model_path": "v2_01a/model.onnx",
        "tag_mapping_path": "v2_01a/model_vocabulary.json",
        "extra_files": [
            "v2_01a/model.onnx.data",
            "v2_01a/model_metadata.json",
        ],
        "description": "CL Tagger v2 provisional SigLIP2 ONNX",
    },
}


def cltagger_preset_for_paths(
    model_path: str, tag_mapping_path: str
) -> Optional[dict[str, Any]]:
    """按 (model_path, tag_mapping_path) 反查匹配的预设；自定义路径返回 None。

    v1/v2 的 model_path + tag_mapping_path 两两唯一，足以定位预设（无需 model_id）。
    """
    norm_model = model_path.replace("\\", "/")
    norm_mapping = tag_mapping_path.replace("\\", "/")
    for preset in CLTAGGER_VERSIONS.values():
        if (
            preset["model_path"] == norm_model
            and preset["tag_mapping_path"] == norm_mapping
        ):
            return preset
    return None


def cltagger_canonical_file_paths(
    model_id: str,
    model_path: str,
    tag_mapping_path: str,
) -> tuple[str, str]:
    """把早期 v2 的"裸根路径"配置还原成带版本子目录的规范路径。

    早期 v2 支持曾把文件存成仓库根名（model.onnx / model_vocabulary.json）。
    这里按 model_id + 文件名在 CLTAGGER_VERSIONS 里反查回带版本子目录的路径，
    不写死版本号——以后加 v2_02 等变体时自动适配；已是版本化路径则原样返回。
    """
    normalized_model = model_path.replace("\\", "/")
    normalized_mapping = tag_mapping_path.replace("\\", "/")
    if model_id != CLTAGGER_V2_REPO:
        return model_path, tag_mapping_path
    for preset in CLTAGGER_VERSIONS.values():
        if (
            preset["model_id"] == model_id
            and Path(preset["model_path"]).name == normalized_model
            and Path(preset["tag_mapping_path"]).name == normalized_mapping
        ):
            return preset["model_path"], preset["tag_mapping_path"]
    return model_path, tag_mapping_path


def is_cltagger_v2_paths(model_path: str, tag_mapping_path: str) -> bool:
    joined = f"{model_path}/{tag_mapping_path}".replace("\\", "/").lower()
    return (
        "cl_tagger_v2" in joined
        or "cl-tagger-v2" in joined
        or Path(tag_mapping_path).name.lower() == "model_vocabulary.json"
    )


def cltagger_required_files(model_path: str, tag_mapping_path: str) -> tuple[str, ...]:
    """一个 variant 完整可用所需的全部文件（下载 + 就绪校验共用）。

    优先用预设里显式声明的 extra_files；非预设（用户自定义路径）回退到
    "v2 onnx 必带同名 .data 权重"的启发式，保证手填路径也能正确校验。
    """
    preset = cltagger_preset_for_paths(model_path, tag_mapping_path)
    if preset is not None:
        extra = list(preset.get("extra_files", []))
    elif is_cltagger_v2_paths(model_path, tag_mapping_path):
        extra = [f"{model_path}.data"]
    else:
        extra = []
    return (model_path, *extra, tag_mapping_path)

# WD14 模型常驻文件名（HF SmilingWolf/* 仓库顶层都是这两个）。
WD14_FILES = ("model.onnx", "selected_tags.csv")

# 预处理放大器预设清单。
#
# label → 元数据 dict：
#   filename      落地文件名（也是 `selected_upscaler` 持久化的 key 之一）
#   hf            (repo_id, repo_subpath) HuggingFace 源；None 表示该模型在 HF 上无稳定镜像
#   ms            (repo_id, repo_subpath) ModelScope 源；None 表示无镜像
#   size_mb       近似下载体积，前端展示用
#   description   一句话用途描述（前端展示）
#
# 路由：download_upscaler 先按 _get_download_source() 取偏好源，对应 None 时透明
# fallback 到另一个源。两个源都 None 视为非法预设。
#
# 选源参考：libfishopen/upscaler 在魔搭上聚合了一批 A1111 时代主流权重，文件名 +
# 字节大小与 HF 原仓库一致；HF 一侧则使用各上游作者的官方仓库（更权威）。
UPSCALER_VARIANTS: dict[str, dict[str, Any]] = {
    "4x-AnimeSharp": {
        "filename": "4x-AnimeSharp.pth",
        "hf": ("Kim2091/AnimeSharp", "4x-AnimeSharp.pth"),
        "ms": ("libfishopen/upscaler", "4x-AnimeSharp.pth"),
        "size_mb": 64,
        "description": "二次元线稿/扁色友好（Kim2091, ESRGAN-RRDB）",
    },
    "R-ESRGAN_4x+Anime6B": {
        "filename": "R-ESRGAN_4x+Anime6B.pth",
        "hf": None,  # 上游 RealESRGAN 仓库未直接发 .pth，先只走 MS
        "ms": ("libfishopen/upscaler", "R-ESRGAN_4x+Anime6B.pth"),
        "size_mb": 18,
        "description": "动漫专用小模型（Real-ESRGAN，A1111 默认）",
    },
    "4x_foolhardy_Remacri": {
        "filename": "4x_foolhardy_Remacri.pth",
        "hf": None,
        "ms": ("libfishopen/upscaler", "4x_foolhardy_Remacri.pth"),
        "size_mb": 64,
        "description": "写实风格（口碑模型）",
    },
    "ESRGAN_4x": {
        "filename": "ESRGAN_4x.pth",
        "hf": None,
        "ms": ("libfishopen/upscaler", "ESRGAN_4x.pth"),
        "size_mb": 64,
        "description": "通用 ESRGAN baseline",
    },
}
DEFAULT_UPSCALER = "4x-AnimeSharp"
# 允许的自定义/上传放大器扩展名（白名单防写错路径 / 误传可执行）。
UPSCALER_EXTS = (".pth", ".safetensors")
# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def safe_dir_name(model_id: str) -> str:
    """把 HF/MS repo id 转成本地目录名（替换路径分隔符为 _）。

    通用 path-sanitization 工具，曾在 tagging.onnx_base 内（PR-3.8 移到这里
    打破循环：models/paths.py ← tagging/onnx_base.py ← models/downloader.py）。
    """
    return model_id.replace("/", "_").replace("\\", "_")


def models_root() -> Path:
    """模型根目录（所有训练 / WD14 模型共用）。

    优先读 `secrets.models.root`（用户在设置页配置），未设 / 空字符串时回退
    到 `{REPO_ROOT}/models/`。解决云端机系统盘小需要把模型放数据盘的场景。

    注意目录命名：与 schema.py 里的 `transformer_path` 默认值（同 `models/`）
    + WD14 的 `models/wd14/` 对齐；HF repo 内部命名 `diffusion_models/`，本地
    扁平化时也用同名子目录。
    """
    try:
        cfg_root = secrets.load().models.root
    except Exception:
        cfg_root = None
    if cfg_root and str(cfg_root).strip():
        return Path(str(cfg_root).strip()).expanduser()
    return REPO_ROOT / "models"


def anima_main_target(root: Path, variant: str) -> Path:
    if variant == "latest":
        variant = LATEST_ANIMA
    if variant not in ANIMA_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    return root / "diffusion_models" / Path(ANIMA_VARIANTS[variant]).name


def anima_vae_target(root: Path) -> Path:
    return root / "vae" / Path(ANIMA_VAE_PATH).name


def qwen_dir(root: Path) -> Path:
    return root / "text_encoders"


def t5_tokenizer_dir(root: Path) -> Path:
    return root / "t5_tokenizer"


def taeflux_dir(root: Optional[Path] = None) -> Path:
    """TAEFlux 本地目录。daemon 用 AutoencoderTiny.from_pretrained 加载。"""
    r = root or models_root()
    return r / "taeflux"


def taeflux_available(root: Optional[Path] = None) -> bool:
    """两个文件都到位才算就绪。"""
    d = taeflux_dir(root)
    return all((d / f).exists() for f in TAEFLUX_FILES)



def wd14_target_dir(root: Path, model_id: str) -> Path:
    """WD14 单个 model_id 的本地目录。同 wd14_tagger 的 _resolve_model_dir 路径布局。"""
    return root / "wd14" / safe_dir_name(model_id)


def cltagger_target_root(root: Path, model_id: str) -> Path:
    """CLTagger repo 的本地根目录。子目录布局来自 CLTAGGER_VERSIONS。"""
    return root / "cltagger" / safe_dir_name(model_id)


def upscaler_dir(root: Optional[Path] = None) -> Path:
    """放大器权重根目录 `{models_root}/upscalers/`。"""
    r = root or models_root()
    return r / "upscalers"


def upscaler_target(label: str, root: Optional[Path] = None) -> Path:
    """单个放大器权重的目标路径。

    label 可以是：
      - 预设 key（在 UPSCALER_VARIANTS 中）→ 用预设里的 filename
      - 直接的文件名（带 .pth/.safetensors 扩展名）→ 视为自定义/已上传模型

    路径穿越保护：禁止 label 含 `/`、`\\` 或 `..`，避免落到 upscalers/ 之外。
    """
    if "/" in label or "\\" in label or ".." in label:
        raise ValueError(f"invalid upscaler label {label!r}")
    if label in UPSCALER_VARIANTS:
        fname = UPSCALER_VARIANTS[label]["filename"]
    else:
        if not label.lower().endswith(UPSCALER_EXTS):
            raise ValueError(f"unknown upscaler {label!r}")
        fname = label
    return upscaler_dir(root) / fname


def find_upscaler(label: str, root: Optional[Path] = None) -> Optional[Path]:
    """已下载返回本地路径，没下载返回 None。"""
    target = upscaler_target(label, root)
    return target if target.exists() else None


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


def selected_upscaler() -> str:
    """读 `secrets.models.selected_upscaler`，回退 DEFAULT_UPSCALER。

    返回值可能是：
      - 预设 label（在 UPSCALER_VARIANTS 中）
      - 已存在的 custom filename（带扩展名）
    都未匹配时回退 DEFAULT_UPSCALER（预设 4x-AnimeSharp）。
    """
    try:
        v = secrets.load().models.selected_upscaler
    except Exception:
        v = None
    if not v:
        return DEFAULT_UPSCALER
    if v in UPSCALER_VARIANTS:
        return v
    # custom：扫盘看文件存不存在
    if v.lower().endswith(UPSCALER_EXTS) and (upscaler_dir() / v).exists():
        return v
    return DEFAULT_UPSCALER


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
        "vae_path": str(anima_vae_target(root)),
        "text_encoder_path": str(qwen_dir(root)),
        "t5_tokenizer_path": str(t5_tokenizer_dir(root)),
    }
