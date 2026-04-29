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


# ---------------------------------------------------------------------------
# 模型清单常量（新版本发布时改这里）
# ---------------------------------------------------------------------------

ANIMA_REPO = "circlestone-labs/Anima"
ANIMA_VARIANTS: dict[str, str] = {
    "preview":       "split_files/diffusion_models/anima-preview.safetensors",
    "preview2":      "split_files/diffusion_models/anima-preview2.safetensors",
    "preview3-base": "split_files/diffusion_models/anima-preview3-base.safetensors",
}
LATEST_ANIMA = "preview3-base"
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
# 同步下载 helper
# ---------------------------------------------------------------------------


def setup_mirror(use_mirror: bool) -> None:
    """设置 HF 镜像端点。CLI 启动一次，UI 启动一次。"""
    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # 关镜像不主动 unset HF_ENDPOINT — 留给上层显式管理


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
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=repo_subpath,
            local_dir=str(target.parent),
            local_dir_use_symlinks=False,
        )
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
    on_log(f"\n📥 Anima 主模型 [{variant}] (~4 GB)")
    return download_flat(ANIMA_REPO, ANIMA_VARIANTS[variant], target, on_log=on_log)


def download_anima_vae(root: Path, *, on_log: Callable[[str], None] = print) -> bool:
    target = anima_vae_target(root)
    on_log("\n📥 Anima VAE (~250 MB)")
    return download_flat(ANIMA_REPO, ANIMA_VAE_PATH, target, on_log=on_log)


def download_qwen3(root: Path, *, on_log: Callable[[str], None] = print) -> bool:
    target_dir = qwen_dir(root)
    on_log(f"\n📥 Qwen3-0.6B-Base (~1.2 GB) → {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    ok = True
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
    raise ValueError(f"unknown model_id {model_id!r}")
