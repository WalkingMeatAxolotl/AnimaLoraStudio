"""Anima 训练模型下载服务（PP7 第一刀）。

把 `tools/download_models.py` 的核心逻辑库化，让 Studio 设置页也能用。

提供：
- 模型清单常量（ANIMA_VARIANTS / VAE / Qwen3 / T5 tokenizer）
- `models_root()` — 默认 `REPO_ROOT/anima/`，后续 PP7 加 `models_root` 全局配置后会改读 secrets
- `build_catalog(root)` — 扫盘组装一份 catalog（哪些已下载、目标路径、大小）
- 同步下载 helper：`download_anima_main` / `download_anima_vae` / `download_qwen3` /
  `download_t5_tokenizer`，CLI 直接调
- 异步下载状态：`start_download_async(key, fn)` 起后台 thread，状态写到全局
  `_DOWNLOADS` dict；`get_status_snapshot()` 给端点查
- bus.publish `model_download_changed` 让前端 SSE 实时拿到状态变化

进度条暂不实现 — `hf_hub_download` 同步阻塞，难以 hook 进度。MVP 只显示
pending/running/done/failed 四态 + 完成后大小变化。
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .. import secrets
from ..event_bus import bus
from ..paths import REPO_ROOT
from .onnx_tagger_base import safe_dir_name


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

# CLTagger 子目录布局：仓库内 cl_tagger_1_02/model.onnx 等。新版本（1.03 等）
# 出现时往这里加一行；UI 自动作为 radio 选项暴露。
# label → (model_path, tag_mapping_path)
CLTAGGER_VERSIONS: dict[str, tuple[str, str]] = {
    "cl_tagger_1_02": (
        "cl_tagger_1_02/model.onnx",
        "cl_tagger_1_02/tag_mapping.json",
    ),
}

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


def download_taeflux(
    *, root: Optional[Path] = None,
    on_log: Callable[[str], None] = print,
) -> bool:
    """同步下载 TAEFlux（config + weights）到本地。任意一个文件失败则返 False。"""
    target_dir = taeflux_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for f in TAEFLUX_FILES:
        target = target_dir / f
        if not download_flat(TAEFLUX_REPO, f, target, on_log=on_log):
            ok = False
    return ok


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


def default_paths_for_new_version() -> dict[str, str]:
    """Studio 创建新 version 时用：返回 4 项路径的**绝对路径字符串**。

    根据当前 `secrets.models.root` 和 `secrets.models.selected_anima` 计算。
    用户在 settings 切了 selected_anima → 之后新建的 version 自动用新选择；
    已存在 version 的 yaml 不动（重现性）。
    """
    root = models_root()
    variant = selected_anima_variant()
    return {
        "transformer_path": str(anima_main_target(root, variant)),
        "vae_path": str(anima_vae_target(root)),
        "text_encoder_path": str(qwen_dir(root)),
        "t5_tokenizer_path": str(t5_tokenizer_dir(root)),
    }


# ---------------------------------------------------------------------------
# ModelScope 镜像源映射
# ---------------------------------------------------------------------------

# ModelScope 镜像路径常量。
# circlestone-labs 同步在 HF 和魔搭发布，repo ID 一致；
# 魔搭里 Anima repo 将主模型 / VAE / 文本编码器全部打包在 split_files/ 下，
# 文本编码器是单个 safetensors（而不是 HF 上 Qwen3 的散文件目录）。
MS_ANIMA_TEXT_ENCODER_PATH = "split_files/text_encoders/qwen_3_06b_base.safetensors"
# T5 tokenizer / TAEFlux / CLTagger 在魔搭暂无对应镜像，走 HF 回退。
# WD14：fireicewolf 在魔搭镜像了 SmilingWolf 系列，repo 命名规则为
#   SmilingWolf/{name} → fireicewolf/{name}
_MS_WD14_OWNER = "fireicewolf"
_HF_WD14_OWNER = "SmilingWolf"


def _ms_wd14_repo_id(hf_repo_id: str) -> Optional[str]:
    """把 SmilingWolf/wd-xxx 换成 fireicewolf/wd-xxx；其它 repo 返回 None。"""
    if hf_repo_id.startswith(_HF_WD14_OWNER + "/"):
        name = hf_repo_id[len(_HF_WD14_OWNER) + 1:]
        return f"{_MS_WD14_OWNER}/{name}"
    return None


# ---------------------------------------------------------------------------
# 同步下载 helper
# ---------------------------------------------------------------------------


def setup_mirror(use_mirror: bool) -> None:
    """[Legacy] 设置 HF_ENDPOINT 环境变量。

    PR-S3 之后 Studio UI 走 secrets.huggingface.endpoint per-call 传给 HF 库，
    不依赖 env var（env var 只在 huggingface_hub 模块 import 时读一次）。
    本函数仅保留给 `tools/download_models.py` CLI 早期 setup 流程兼容。
    """
    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # 关镜像不主动 unset HF_ENDPOINT — 留给上层显式管理


def _resolve_endpoint() -> Optional[str]:
    """决定本次下载用什么 HF endpoint。优先级：

    1. `HF_ENDPOINT` 环境变量（CLI 走 setup_mirror 设的，或用户手 export）
    2. `secrets.huggingface.endpoint`（Studio UI 配的）
    3. None（让 huggingface_hub 用默认 huggingface.co）

    每次下载都调一次，UI 改了配置无需重启 server。
    """
    env = os.environ.get("HF_ENDPOINT", "").strip()
    if env:
        return env
    try:
        endpoint = secrets.load().huggingface.endpoint
    except Exception:  # noqa: BLE001  secrets 损坏不应阻断下载
        return None
    return endpoint or None


def _get_download_source() -> str:
    """返回当前配置的下载源（'huggingface' 或 'modelscope'）。

    优先读 MODELSCOPE_SOURCE env var（CLI flag 用）；否则读 secrets。
    """
    env = os.environ.get("MODELSCOPE_SOURCE", "").strip()
    if env:
        return env
    try:
        return secrets.load().download_source or "huggingface"
    except Exception:  # noqa: BLE001
        return "huggingface"


def _ms_token() -> Optional[str]:
    """读 ModelScope token：环境变量优先，其次 secrets。"""
    env = os.environ.get("MODELSCOPE_API_TOKEN", "").strip()
    if env:
        return env
    try:
        t = secrets.load().modelscope.token
        return t or None
    except Exception:  # noqa: BLE001
        return None


def download_flat_ms(
    ms_repo_id: str,
    repo_subpath: str,
    target: Path,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """用 modelscope Python API 下载单个文件到 target。

    `model_file_download(local_dir=target.parent)` 会把文件落在
    `target.parent / repo_subpath`（保留 repo 内路径结构），之后复用与
    `download_flat` 完全相同的 rename + 清理空目录逻辑把文件移到 target。

    需要 ``pip install modelscope``；未安装时返回 False 并打印提示。
    token 优先读 MODELSCOPE_API_TOKEN env var，其次 secrets.modelscope.token。
    """
    if target.exists():
        on_log(f"   ✓ {target.name} 已存在，跳过")
        return True
    try:
        from modelscope.hub.file_download import model_file_download
    except ImportError:
        on_log("   ✗ 缺 modelscope（pip install modelscope）")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    token = _ms_token()
    try:
        kwargs: dict = dict(
            model_id=ms_repo_id,
            file_path=repo_subpath,
            local_dir=str(target.parent),
        )
        if token:
            kwargs["token"] = token
        model_file_download(**kwargs)
    except Exception as exc:
        on_log(f"   ✗ {target.name} (ModelScope): {exc}")
        return False
    # model_file_download 保留 repo 内路径；与 download_flat 逻辑完全一致
    src = target.parent / repo_subpath
    if src != target:
        try:
            target.unlink(missing_ok=True)
            src.rename(target)
        except OSError as exc:
            on_log(f"   ✗ rename 失败 {src} → {target}: {exc}")
            return False
        parent = src.parent
        while parent != target.parent and parent.exists():
            try:
                if any(parent.iterdir()):
                    break
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    on_log(f"   ✓ {target.name} (via ModelScope)")
    return True


def download_flat(
    repo_id: str,
    repo_subpath: str,
    target: Path,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """从 HF 下载 repo_subpath，扁平落到 target；返回 True = 已就绪。

    实现：`hf_hub_download(local_dir=target.parent)` 把 repo 内部目录建出来，
    再 rename 到 target（同卷 atomic，不重复 4 GB）。已存在直接跳过。
    """
    if target.exists():
        on_log(f"   ✓ {target.name} 已存在，跳过")
        return True
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        on_log("   ✗ 缺 huggingface_hub")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    endpoint = _resolve_endpoint()
    try:
        kwargs = dict(
            repo_id=repo_id,
            filename=repo_subpath,
            local_dir=str(target.parent),
            local_dir_use_symlinks=False,
        )
        if endpoint:
            kwargs["endpoint"] = endpoint
        hf_hub_download(**kwargs)
    except Exception as exc:
        on_log(f"   ✗ {target.name}: {exc}")
        return False
    src = target.parent / repo_subpath
    if src != target:
        try:
            target.unlink(missing_ok=True)
            src.rename(target)
        except OSError as exc:
            on_log(f"   ✗ rename 失败 {src} → {target}: {exc}")
            return False
        # 清理空中间目录
        parent = src.parent
        while parent != target.parent and parent.exists():
            try:
                if any(parent.iterdir()):
                    break
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    on_log(f"   ✓ {target.name}")
    return True


def download_anima_main(
    root: Path, variant: str, *, on_log: Callable[[str], None] = print
) -> bool:
    if variant == "latest":
        variant = LATEST_ANIMA
    if variant not in ANIMA_VARIANTS:
        on_log(f"✗ 未知 variant {variant!r}")
        return False
    target = anima_main_target(root, variant)
    subpath = ANIMA_VARIANTS[variant]
    on_log(f"\n📥 Anima 主模型 [{variant}] (~4 GB)")
    if _get_download_source() == "modelscope":
        return download_flat_ms(ANIMA_REPO, subpath, target, on_log=on_log)
    return download_flat(ANIMA_REPO, subpath, target, on_log=on_log)


def download_anima_vae(root: Path, *, on_log: Callable[[str], None] = print) -> bool:
    target = anima_vae_target(root)
    on_log("\n📥 Anima VAE (~250 MB)")
    if _get_download_source() == "modelscope":
        return download_flat_ms(ANIMA_REPO, ANIMA_VAE_PATH, target, on_log=on_log)
    return download_flat(ANIMA_REPO, ANIMA_VAE_PATH, target, on_log=on_log)


def download_qwen3(root: Path, *, on_log: Callable[[str], None] = print) -> bool:
    """下载文本编码器（Qwen3）。

    - HuggingFace 源：从 Qwen/Qwen3-0.6B-Base 下载完整目录所需的 6 个文件。
    - ModelScope 源：从 circlestone-labs/Anima 下载权重文件，另外从
      Qwen/Qwen3-0.6B-Base 补齐 tokenizer / config 文件，确保本地
      text_encoders/ 是 transformers 可直接加载的完整目录。
    """
    target_dir = qwen_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    ok = True

    if _get_download_source() == "modelscope":
        on_log(f"\n📥 Anima 文本编码器（ModelScope 权重 + HF tokenizer）→ {target_dir}")
        # 魔搭 Anima repo 里只有权重；训练脚本仍要求完整 transformers 目录。
        ok &= download_flat_ms(
            ANIMA_REPO,
            MS_ANIMA_TEXT_ENCODER_PATH,
            target_dir / "model.safetensors",
            on_log=on_log,
        )
        for f in QWEN_FILES:
            if f == "model.safetensors":
                continue
            if not download_flat(QWEN_REPO, f, target_dir / f, on_log=on_log):
                ok = False
        return ok

    on_log(f"\n📥 Qwen3-0.6B-Base (~1.2 GB) → {target_dir}")
    for f in QWEN_FILES:
        if not download_flat(QWEN_REPO, f, target_dir / f, on_log=on_log):
            ok = False
    return ok


def download_t5_tokenizer(
    root: Path, *, on_log: Callable[[str], None] = print
) -> bool:
    target_dir = t5_tokenizer_dir(root)
    on_log(f"\n📥 T5 tokenizer (3 个文件) → {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for f in T5_FILES:
        if not download_flat(T5_REPO, f, target_dir / f, on_log=on_log):
            ok = False
    return ok


def download_cltagger(
    target_root: Path,
    cfg: Optional["secrets.CLTaggerConfig"] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    cfg = cfg or secrets.load().cltagger
    on_log(f"\n📥 CLTagger → {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    ok = True
    for f in (cfg.model_path, cfg.tag_mapping_path):
        if not download_flat(cfg.model_id, f, target_root / f, on_log=on_log):
            ok = False
    return ok


def download_upscaler(
    label: str = DEFAULT_UPSCALER,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """下载放大器权重到 `{models_root}/upscalers/{filename}`。

    源选择：按 _get_download_source() 取偏好；对应源缺失时透明回退到另一个源
    （e.g. R-ESRGAN_4x+Anime6B 没有 HF 镜像 → 用户即便选了 HF 也走 MS）。
    """
    if label not in UPSCALER_VARIANTS:
        on_log(f"✗ 未知放大器 {label!r}")
        return False
    info = UPSCALER_VARIANTS[label]
    hf_src = info.get("hf")
    ms_src = info.get("ms")
    if hf_src is None and ms_src is None:
        on_log(f"✗ 放大器 {label!r} 未配置任何下载源")
        return False

    target = upscaler_target(label, root)
    size_mb = info.get("size_mb", 64)
    prefer_ms = _get_download_source() == "modelscope"
    on_log(f"\n📥 放大器 {label} (~{size_mb} MB) → {target}")

    if prefer_ms and ms_src is not None:
        return download_flat_ms(ms_src[0], ms_src[1], target, on_log=on_log)
    if hf_src is not None:
        return download_flat(hf_src[0], hf_src[1], target, on_log=on_log)
    # 偏好 HF 但 HF 缺失 → fallback MS
    on_log(f"   ⚠ HF 无镜像，回退 ModelScope")
    return download_flat_ms(ms_src[0], ms_src[1], target, on_log=on_log)  # type: ignore[index]


def download_upscaler_custom(
    source: str,
    repo_id: str,
    filename: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """自定义 repo 下载：用户指定 HF/MS 仓库 + 文件名，落到 `{upscalers}/{filename}`。

    扩展名白名单同 UPSCALER_EXTS（.pth / .safetensors）。filename 仅作落地文件名，
    repo 内子路径直接走 repo_id + filename — 大多数 upscaler repo 都把权重摆在
    根目录，需要子目录的话用户可以在 filename 里写 `subdir/foo.pth` 这种相对路径，
    但落地时会被剥成纯文件名（避免穿越）。
    """
    if source not in ("hf", "ms"):
        on_log(f"✗ 未知下载源 {source!r}（支持 hf / ms）")
        return False
    repo_subpath = filename
    save_name = Path(filename).name  # 剥目录前缀，仅保留纯文件名
    if "/" in save_name or "\\" in save_name or ".." in save_name:
        on_log(f"✗ 非法文件名 {save_name!r}")
        return False
    if not save_name.lower().endswith(UPSCALER_EXTS):
        on_log(f"✗ 仅支持 {UPSCALER_EXTS} 扩展名，收到 {save_name!r}")
        return False
    target = upscaler_dir(root) / save_name
    on_log(f"\n📥 自定义放大器 [{source}] {repo_id}/{repo_subpath} → {target}")
    if source == "ms":
        return download_flat_ms(repo_id, repo_subpath, target, on_log=on_log)
    return download_flat(repo_id, repo_subpath, target, on_log=on_log)


def download_wd14(
    model_id: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """下载 WD14 单个 model_id 的两个文件到 `{models_root}/wd14/{safe_id}/`。

    ModelScope 源：SmilingWolf/* → fireicewolf/*（fireicewolf 在魔搭镜像了全套）。
    没有 MS 映射（非 SmilingWolf 前缀）时自动回退 HF。
    """
    r = root or models_root()
    target = wd14_target_dir(r, model_id)
    target.mkdir(parents=True, exist_ok=True)
    ok = True
    if _get_download_source() == "modelscope":
        ms_repo = _ms_wd14_repo_id(model_id)
        if ms_repo:
            on_log(f"\n📥 WD14 {model_id} → {target}（via ModelScope: {ms_repo}）")
            for f in WD14_FILES:
                if not download_flat_ms(ms_repo, f, target / f, on_log=on_log):
                    ok = False
            return ok
        on_log(f"\n📥 WD14 {model_id}：无魔搭映射，回退 HuggingFace")
    else:
        on_log(f"\n📥 WD14 {model_id} → {target}")
    for f in WD14_FILES:
        if not download_flat(model_id, f, target / f, on_log=on_log):
            ok = False
    return ok


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
    for label, (mp, tmp) in CLTAGGER_VERSIONS.items():
        files = [
            {"name": mp, **_file_status(cl_root / mp)},
            {"name": tmp, **_file_status(cl_root / tmp)},
        ]
        all_exist = all(f["exists"] for f in files)
        total_size = sum(f["size"] for f in files)
        cl_variants.append({
            "label": label,
            "model_path": mp,
            "tag_mapping_path": tmp,
            "is_current": cl_cfg.model_path == mp and cl_cfg.tag_mapping_path == tmp,
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


# ---------------------------------------------------------------------------
# 异步下载状态机
# ---------------------------------------------------------------------------


@dataclass
class DownloadStatus:
    key: str
    status: str  # pending | running | done | failed
    started_at: float = 0.0
    finished_at: Optional[float] = None
    message: str = ""
    log: list[str] = field(default_factory=list)


_LOCK = threading.Lock()
_DOWNLOADS: dict[str, DownloadStatus] = {}


def get_status_snapshot() -> dict[str, dict[str, Any]]:
    """端点序列化用：浅拷贝当前所有 download status。"""
    with _LOCK:
        return {
            k: {
                "key": v.key,
                "status": v.status,
                "started_at": v.started_at,
                "finished_at": v.finished_at,
                "message": v.message,
                "log_tail": v.log[-30:],
            }
            for k, v in _DOWNLOADS.items()
        }


def start_download_async(
    key: str, fn: Callable[[Callable[[str], None]], bool]
) -> DownloadStatus:
    """启动后台 thread 跑 `fn(on_log)`；fn 返回 True=成功。

    `key` 是任务标识，重复启动同 key（仍 running）会复用现有 status。
    完成 / 失败时通过 `bus.publish` 推 `model_download_changed` SSE 事件。
    """
    with _LOCK:
        existing = _DOWNLOADS.get(key)
        if existing and existing.status == "running":
            return existing
        ds = DownloadStatus(
            key=key, status="running", started_at=time.time(), log=[]
        )
        _DOWNLOADS[key] = ds

    def _on_log(line: str) -> None:
        with _LOCK:
            ds.log.append(line)
            if len(ds.log) > 200:
                del ds.log[:-200]
        # 回显到 backend stdout —— UI ring buffer 容量 200 行；长下载早期日志会被
        # 截掉，print 让 studio_*.log / 终端保留完整流，调试 / oncall 排错时能直接 grep。
        # 锁外执行避免持锁做 I/O 拖慢其它 download tasks 写日志。
        print(line, flush=True)

    def _run() -> None:
        bus.publish({
            "type": "model_download_changed",
            "key": key,
            "status": "running",
        })
        try:
            ok = fn(_on_log)
            with _LOCK:
                ds.status = "done" if ok else "failed"
                ds.finished_at = time.time()
                if not ok:
                    ds.message = "下载失败，看 log_tail"
        except Exception as exc:
            with _LOCK:
                ds.status = "failed"
                ds.finished_at = time.time()
                ds.message = str(exc)
                ds.log.append(f"[exception] {exc}")
        bus.publish({
            "type": "model_download_changed",
            "key": key,
            "status": ds.status,
        })

    threading.Thread(
        target=_run, daemon=True, name=f"model-dl-{key}"
    ).start()
    bus.publish({
        "type": "model_download_changed",
        "key": key,
        "status": "running",
    })
    return ds


def trigger(model_id: str, variant: Optional[str] = None) -> str:
    """便于端点调用的入口：根据 model_id 选对应的 download_* 函数 + 启动异步。

    返回 status key（前端用来拼 SSE 关心的 key）。
    """
    root = models_root()
    if model_id == "anima_main":
        v = variant or "latest"
        if v == "latest":
            v = LATEST_ANIMA
        if v not in ANIMA_VARIANTS:
            raise ValueError(f"unknown anima variant {variant!r}")
        key = f"anima_main:{v}"
        start_download_async(
            key,
            lambda log: download_anima_main(root, v, on_log=log),
        )
        return key
    if model_id == "anima_vae":
        key = "anima_vae"
        start_download_async(
            key, lambda log: download_anima_vae(root, on_log=log)
        )
        return key
    if model_id == "qwen3":
        key = "qwen3"
        start_download_async(
            key, lambda log: download_qwen3(root, on_log=log)
        )
        return key
    if model_id == "t5_tokenizer":
        key = "t5_tokenizer"
        start_download_async(
            key, lambda log: download_t5_tokenizer(root, on_log=log)
        )
        return key
    if model_id == "cltagger":
        cfg = secrets.load().cltagger
        target = cltagger_target_root(root, cfg.model_id)
        # variant 可指定预设 label（覆盖 cfg 当前的 model_path），便于 UI 一键
        # 下载非"当前选中"的版本。未指定时用 cfg 当前路径。
        if variant:
            preset = CLTAGGER_VERSIONS.get(variant)
            if preset is None:
                raise ValueError(f"unknown cltagger variant {variant!r}")
            cfg = secrets.CLTaggerConfig(
                **{**cfg.model_dump(), "model_path": preset[0], "tag_mapping_path": preset[1]}
            )
            key = f"cltagger:{variant}"
        else:
            key = "cltagger"
        start_download_async(
            key, lambda log: download_cltagger(target, cfg, on_log=log)
        )
        return key
    if model_id == "wd14":
        if not variant:
            raise ValueError("wd14 需要 variant=model_id")
        key = f"wd14:{variant}"
        start_download_async(
            key, lambda log: download_wd14(variant, root, on_log=log)
        )
        return key
    if model_id == "upscaler":
        label = variant or DEFAULT_UPSCALER
        if label not in UPSCALER_VARIANTS:
            raise ValueError(f"unknown upscaler variant {variant!r}")
        key = f"upscaler:{label}"
        start_download_async(
            key, lambda log: download_upscaler(label, root, on_log=log)
        )
        return key
    raise ValueError(f"unknown model_id {model_id!r}")
