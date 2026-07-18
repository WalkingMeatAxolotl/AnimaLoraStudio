"""逐模型高层下载流程 + 异步状态跟踪（PR-3.8 拆出 4-way 第 3 个）。

含 Anima / Krea 2 训练资产与各类工具模型的逐模型下载函数，调 sources.py 的
download_flat[_ms] 实际下载，调 paths.py / families 拿 target Path 和模型清单。

异步：DownloadStatus / start_download_async / trigger 把同步下载包成后台 thread，
向 event_bus 推 model_download_changed。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ... import secrets
from ...infrastructure.event_bus import bus
from .families.anima import (
    ANIMA_REPO,
    ANIMA_VAE_PATH,
    ANIMA_VARIANTS,
    LATEST_ANIMA,
    QWEN_FILES,
    QWEN_REPO,
    T5_FILES,
    T5_REPO,
    anima_main_target,
    qwen_dir,
    selected_anima_variant,
    t5_tokenizer_dir,
)
from .families.krea2 import (
    KREA2_VARIANTS,
    LATEST_KREA2,
    QWEN3_VL_FILES,
    QWEN3_VL_FP8_FILE,
    QWEN3_VL_FP8_REPO,
    QWEN3_VL_FP8_SMALL_FILES,
    QWEN3_VL_FP8_SUBPATH,
    QWEN3_VL_REPO,
    krea2_main_target,
    qwen3_vl_dir,
    qwen3_vl_fp8_dir,
)
from .paths import (
    CLTAGGER_VERSIONS,
    DEFAULT_UPSCALER,
    TAEFLUX_FILES,
    TAEFLUX_REPO,
    UPSCALER_EXTS,
    UPSCALER_VARIANTS,
    WD14_FILES,
    ccip_model_dir,
    cltagger_canonical_file_paths,
    cltagger_required_files,
    cltagger_target_root,
    eval_model_target_dir,
    models_root,
    qwen_image_vae_target,
    selected_upscaler,
    taeflux_dir,
    upscaler_dir,
    upscaler_target,
    wd14_target_dir,
)
from . import sources as _sources
from .sources import MS_ANIMA_TEXT_ENCODER_PATH

# 提示：跨文件调用 download_flat[_ms] / _get_download_source / _resolve_endpoint /
# _ms_wd14_repo_id 一律走 _sources.X(...) —— 这样测试 monkeypatch
# `studio.services.models.sources.X` 才会生效。若改成 `from .sources import X`
# 则会 bind 成本 module 的 local name，patch sources 模块对 downloader 内调用无效。

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
        if not _sources.download_flat(TAEFLUX_REPO, f, target, on_log=on_log):
            ok = False
    return ok


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
    if _sources._source_for("training") == "modelscope":
        return _sources.download_flat_ms(ANIMA_REPO, subpath, target, on_log=on_log)
    return _sources.download_flat(ANIMA_REPO, subpath, target, on_log=on_log)


def download_anima_vae(root: Path, *, on_log: Callable[[str], None] = print) -> bool:
    # 落点是族无关共享资产（Krea2 同用）；下载渠道走 Anima repo（文件在那儿）。
    target = qwen_image_vae_target(root)
    on_log("\n📥 Anima VAE (~250 MB)")
    if _sources._source_for("training") == "modelscope":
        return _sources.download_flat_ms(ANIMA_REPO, ANIMA_VAE_PATH, target, on_log=on_log)
    return _sources.download_flat(ANIMA_REPO, ANIMA_VAE_PATH, target, on_log=on_log)


def download_krea2_main(
    root: Path, variant: str, *, on_log: Callable[[str], None] = print
) -> bool:
    """从 HuggingFace 官方仓库或 ModelScope Comfy-Org 镜像下载 Krea2。"""
    if variant == "latest":
        variant = LATEST_KREA2
    info = KREA2_VARIANTS.get(variant)
    if info is None:
        on_log(f"✗ 未知 Krea 2 variant {variant!r}")
        return False
    target = krea2_main_target(root, variant)
    size_gb = float(info.get("size_estimate", 0)) / 1e9
    on_log(f"\n📥 Krea 2 [{variant}] (~{size_gb:.1f} GB) → {target}")
    if _sources._source_for("training") == "modelscope":
        return _sources.download_flat_ms(
            str(info["ms_repo"]), str(info["ms_subpath"]), target, on_log=on_log,
        )
    return _sources.download_flat(
        str(info["repo"]), str(info["subpath"]), target, on_log=on_log,
    )


def download_qwen3_vl(
    root: Path, *, on_log: Callable[[str], None] = print
) -> bool:
    """下载 Krea 2 使用的完整 Qwen3-VL-4B-Instruct transformers 目录。"""
    target_dir = qwen3_vl_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    use_modelscope = _sources._source_for("training") == "modelscope"
    source_label = "ModelScope" if use_modelscope else "HuggingFace"
    on_log(
        f"\n📥 Krea 2 文本编码器 Qwen3-VL-4B-Instruct "
        f"(~8.89 GB, {source_label}) → {target_dir}"
    )
    ok = True
    for filename in QWEN3_VL_FILES:
        target = target_dir / filename
        if use_modelscope:
            downloaded = _sources.download_flat_ms(
                QWEN3_VL_REPO, filename, target, on_log=on_log,
            )
        else:
            downloaded = _sources.download_flat(
                QWEN3_VL_REPO, filename, target, on_log=on_log,
            )
        if not downloaded:
            ok = False
    return ok


def download_qwen3_vl_fp8(
    root: Path, *, on_log: Callable[[str], None] = print
) -> bool:
    """下载官方 fp8_scaled 单文件 TE + config/tokenizer 小文件到独立目录。

    权重来自 Comfy-Org/Krea-2（HF 与 ModelScope 同 repo 布局）；小文件来自
    Qwen 官方 repo（单文件里没有，loader 需要 config 建结构、tokenizer
    编码）。
    """
    target_dir = qwen3_vl_fp8_dir(root)
    target_dir.mkdir(parents=True, exist_ok=True)
    use_ms = _sources._source_for("training") == "modelscope"
    on_log(f"\n📥 Krea 2 文本编码器 Qwen3-VL fp8 (~5.24 GB) → {target_dir}")
    download = _sources.download_flat_ms if use_ms else _sources.download_flat
    ok = download(
        QWEN3_VL_FP8_REPO, QWEN3_VL_FP8_SUBPATH,
        target_dir / QWEN3_VL_FP8_FILE, on_log=on_log,
    )
    for filename in QWEN3_VL_FP8_SMALL_FILES:
        if not download(
            QWEN3_VL_REPO, filename, target_dir / filename, on_log=on_log,
        ):
            ok = False
    return ok


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

    if _sources._source_for("training") == "modelscope":
        on_log(f"\n📥 Anima 文本编码器（ModelScope 权重 + HF tokenizer）→ {target_dir}")
        # 魔搭 Anima repo 里只有权重；训练脚本仍要求完整 transformers 目录。
        ok &= _sources.download_flat_ms(
            ANIMA_REPO,
            MS_ANIMA_TEXT_ENCODER_PATH,
            target_dir / "model.safetensors",
            on_log=on_log,
        )
        for f in QWEN_FILES:
            if f == "model.safetensors":
                continue
            if not _sources.download_flat(QWEN_REPO, f, target_dir / f, on_log=on_log):
                ok = False
        return ok

    on_log(f"\n📥 Qwen3-0.6B-Base (~1.2 GB) → {target_dir}")
    for f in QWEN_FILES:
        if not _sources.download_flat(QWEN_REPO, f, target_dir / f, on_log=on_log):
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
        if not _sources.download_flat(T5_REPO, f, target_dir / f, on_log=on_log):
            ok = False
    return ok


def download_cltagger(
    target_root: Path,
    cfg: Optional["secrets.CLTaggerConfig"] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    cfg = cfg or secrets.load().cltagger
    model_path, tag_mapping_path = cltagger_canonical_file_paths(
        cfg.model_id,
        cfg.model_path,
        cfg.tag_mapping_path,
    )
    on_log(f"\n📥 CLTagger → {target_root}")
    target_root.mkdir(parents=True, exist_ok=True)
    ok = True
    for f in cltagger_required_files(model_path, tag_mapping_path):
        if not _sources.download_flat(cfg.model_id, f, target_root / f, on_log=on_log):
            ok = False
    return ok


def download_upscaler(
    label: str = DEFAULT_UPSCALER,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """下载放大器权重到 `{models_root}/upscalers/{filename}`。

    源选择：按 _sources._source_for("upscaler") 取偏好；对应源缺失时透明回退到另一个源
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
    prefer_ms = _sources._source_for("upscaler") == "modelscope"
    on_log(f"\n📥 放大器 {label} (~{size_mb} MB) → {target}")

    if prefer_ms and ms_src is not None:
        return _sources.download_flat_ms(ms_src[0], ms_src[1], target, on_log=on_log)
    if hf_src is not None:
        return _sources.download_flat(hf_src[0], hf_src[1], target, on_log=on_log)
    # 偏好 HF 但 HF 缺失 → fallback MS
    on_log(f"   ⚠ HF 无镜像，回退 ModelScope")
    return _sources.download_flat_ms(ms_src[0], ms_src[1], target, on_log=on_log)  # type: ignore[index]


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
        return _sources.download_flat_ms(repo_id, repo_subpath, target, on_log=on_log)
    return _sources.download_flat(repo_id, repo_subpath, target, on_log=on_log)


def download_main_custom(
    repo_id: str,
    filename: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """统一来源候选的第三方主模型单文件下载 → `{models_root}/diffusion_models/`。

    filename 是 repo 内路径（可含子目录），落地时剥成纯文件名（与官方
    variant 同目录，文件名即身份）。走 download_flat 的「有 MS 映射用 MS、
    否则 HF」默认逻辑——自定义 repo 无映射即直连 HF。
    """
    save_name = Path(filename).name
    if not save_name.lower().endswith(".safetensors"):
        on_log(f"✗ 仅支持 .safetensors，收到 {save_name!r}")
        return False
    r = root or models_root()
    target = r / "diffusion_models" / save_name
    on_log(f"\n📥 自定义主模型 {repo_id}/{filename} → {target}")
    return _sources.download_flat(repo_id, filename, target, on_log=on_log)


def _download_candidate(domain: str, filename: str) -> "secrets.SourceCandidate":
    """按 filename 查该 domain 的下载型候选（trigger / delete 共用）。"""
    for c in secrets.load().model_sources.get(domain, []):
        if c.kind == "download" and c.filename == filename:
            return c
    raise ValueError(f"no download candidate {filename!r} for domain {domain!r}")


def _custom_family(model_id: str) -> Optional[str]:
    """`{family}_custom` 形式的 model_id → family id（未注册族返回 None）。"""
    from .families import FAMILY_ASSETS

    family = model_id[: -len("_custom")]
    return family if family in FAMILY_ASSETS else None


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
    if _sources._source_for("wd14") == "modelscope":
        ms_repo = _sources._ms_wd14_repo_id(model_id)
        if ms_repo:
            on_log(f"\n📥 WD14 {model_id} → {target}（via ModelScope: {ms_repo}）")
            for f in WD14_FILES:
                if not _sources.download_flat_ms(ms_repo, f, target / f, on_log=on_log):
                    ok = False
            return ok
        on_log(f"\n📥 WD14 {model_id}：无魔搭映射，回退 HuggingFace")
    else:
        on_log(f"\n📥 WD14 {model_id} → {target}")
    for f in WD14_FILES:
        if not _sources.download_flat(model_id, f, target / f, on_log=on_log):
            ok = False
    return ok


def download_eval_model(
    kind: str,
    model_id: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """下载 CLIP / DINO eval 模型整个 repo 到 `{models_root}/eval/{kind}/{safe_id}/`。

    源选择：eval 源选 modelscope 且有镜像映射时走 MS，否则（无映射 / 选 HF）走
    HuggingFace —— 与 wd14 同样的「有 MS 映射用 MS、否则回退 HF」逻辑。
    """
    r = root or models_root()
    target = eval_model_target_dir(r, kind, model_id)
    if _sources._source_for("eval") == "modelscope":
        ms_repo = _sources._ms_eval_repo_id(model_id)
        if ms_repo:
            on_log(f"\n📥 {kind.upper()} {model_id} → {target}（via ModelScope: {ms_repo}）")
            return _sources.download_snapshot_ms(ms_repo, target, on_log=on_log)
        on_log(f"\n📥 {kind.upper()} {model_id}：无魔搭映射，回退 HuggingFace")
    else:
        on_log(f"\n📥 {kind.upper()} {model_id} → {target}")
    return _sources.download_snapshot(model_id, target, on_log=on_log)


def ensure_eval_model(
    kind: str,
    model_id: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> Path:
    """返回 eval 模型本地目录，缺失则先下载（懒加载兜底，路径与下载卡片一致）。

    与 wd14 `_resolve_model_dir` 同模式：跑 eval 时用户若没预下载，自动下到项目
    目录，`from_pretrained` 指向它。已就绪（有 config.json）直接返回。

    若 ``model_id`` 本身是一个已存在的本地模型目录（用户在文本框直接填了路径），
    直接用它、不当 repo id 下载。
    """
    local = Path(str(model_id)).expanduser()
    if local.is_dir() and (local / "config.json").exists():
        return local
    r = root or models_root()
    target = eval_model_target_dir(r, kind, model_id)
    if (target / "config.json").exists():
        return target
    download_eval_model(kind, model_id, r, on_log=on_log)
    return target


# CCIP（anime 角色身份）：只下变体子目录里的 3 个文件，别整库拉（含 torch ckpt + png）。
CCIP_REPO = "deepghs/ccip_onnx"
CCIP_FILES = ("model_feat.onnx", "model_metrics.onnx", "metrics.json")


def download_ccip_model(
    variant: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> bool:
    """下 deepghs/ccip_onnx 指定变体的 3 个文件到 `{models_root}/eval/ccip/{variant}/`。"""
    r = root or models_root()
    eval_ccip_root = r / "eval" / "ccip"
    patterns = [f"{variant}/{f}" for f in CCIP_FILES]
    on_log(f"\n📥 CCIP {variant} → {ccip_model_dir(r, variant)}")
    return _sources.download_snapshot(
        CCIP_REPO, eval_ccip_root, allow_patterns=patterns, on_log=on_log,
    )


def ensure_ccip_model(
    variant: str,
    root: Optional[Path] = None,
    *,
    on_log: Callable[[str], None] = print,
) -> Path:
    """返回 CCIP 变体本地目录，缺 3 个文件则先下载（懒加载兜底）。"""
    r = root or models_root()
    target = ccip_model_dir(r, variant)
    if all((target / f).exists() for f in CCIP_FILES):
        return target
    download_ccip_model(variant, r, on_log=on_log)
    return target


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


def _failure_summary(log: list[str]) -> str:
    """从下载日志里提取一句可操作的失败原因（给前端 toast / message 用）。

    download_flat 把错误写成 `   ✗ ...`；gated / 授权类失败再追加 `   ↳ ...提示`
    （含 token / 申请授权指引）。优先返回带提示的那条，否则退回最后一条 ✗ 错误，
    都没有再退到通用串。避免前端只看到 badge 红了却不知为何（原因只在终端 / 折叠
    日志里）。
    """
    err = next((ln.strip() for ln in reversed(log) if ln.lstrip().startswith("✗")), "")
    hint = next((ln.strip() for ln in reversed(log) if "↳" in ln), "")
    if hint:
        return f"{err} {hint}".strip() if err else hint
    return err or "下载失败，详见下载日志"


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
                    ds.message = _failure_summary(ds.log)
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


def delete_asset(model_id: str, variant: Optional[str] = None) -> None:
    """删除一个已下载资产（下载按钮的逆操作：用户先删除、再下载）。

    目标路径全部由服务端 target 函数解析——不接受任意路径；对应 key 的
    下载进行中拒绝。覆盖下载中心全部资产 id：训练模型区（主模型 variant /
    VAE / 文本编码器 / tokenizer）、打标（wd14 / cltagger）、eval 指标
    （clip / dino / ccip）、放大器（预设 + 自定义文件名）。文件被占用
    （模型已加载 / 训练中）时 OSError 原样转可操作报错。
    """
    import shutil

    root = models_root()
    target: Path
    key = model_id
    if model_id == "anima_main":
        v = variant or ""
        if v not in ANIMA_VARIANTS:
            raise ValueError(f"unknown anima variant {variant!r}")
        key = f"anima_main:{v}"
        target = anima_main_target(root, v)
    elif model_id == "krea2_main":
        v = variant or ""
        if v not in KREA2_VARIANTS:
            raise ValueError(f"unknown Krea 2 variant {variant!r}")
        key = f"krea2_main:{v}"
        target = krea2_main_target(root, v)
    elif model_id == "anima_vae":
        target = qwen_image_vae_target(root)
    elif model_id == "qwen3":
        target = qwen_dir(root)
    elif model_id == "t5_tokenizer":
        target = t5_tokenizer_dir(root)
    elif model_id == "krea2_text_encoder":
        target = qwen3_vl_dir(root)
    elif model_id == "krea2_text_encoder_fp8":
        target = qwen3_vl_fp8_dir(root)
    elif model_id == "wd14":
        if not variant:
            raise ValueError("wd14 需要 variant=model_id")
        key = f"wd14:{variant}"
        target = wd14_target_dir(root, variant)
    elif model_id == "cltagger":
        preset = CLTAGGER_VERSIONS.get(variant or "")
        if preset is None:
            raise ValueError(f"unknown cltagger variant {variant!r}")
        key = f"cltagger:{variant}"
        # 只删该版本子目录——v1/v2 同 repo 根下可并存多版本
        target = cltagger_target_root(root, preset["model_id"]) / Path(
            preset["model_path"]
        ).parent
    elif model_id in ("eval_clip", "eval_dino"):
        if not variant:
            raise ValueError(f"{model_id} 需要 variant=model_id")
        kind = "clip" if model_id == "eval_clip" else "dino"
        key = f"{model_id}:{variant}"
        target = eval_model_target_dir(root, kind, variant)
    elif model_id == "eval_ccip":
        if not variant:
            raise ValueError("eval_ccip 需要 variant=ccip 变体名")
        key = f"eval_ccip:{variant}"
        target = ccip_model_dir(root, variant)
    elif model_id == "upscaler":
        if not variant:
            raise ValueError("upscaler 需要 variant=label")
        # 预设 label 或自定义文件名；upscaler_target 自带路径穿越校验
        key = (
            f"upscaler:{variant}"
            if variant in UPSCALER_VARIANTS
            else f"upscaler:custom:{variant}"
        )
        target = upscaler_target(variant, root)
    elif model_id == "upscaler_custom":
        # 统一来源 download 候选落盘的文件（filename 即身份）
        if not variant:
            raise ValueError("upscaler_custom 需要 variant=filename")
        save_name = Path(variant).name
        key = f"upscaler:custom:{save_name}"
        target = upscaler_dir(root) / save_name
    elif model_id.endswith("_custom") and _custom_family(model_id) is not None:
        if not variant:
            raise ValueError(f"{model_id} 需要 variant=filename")
        key = f"{model_id}:{variant}"
        target = root / "diffusion_models" / Path(variant).name
    else:
        raise ValueError(f"asset {model_id!r} does not support deletion")

    with _LOCK:
        existing = _DOWNLOADS.get(key)
        if existing and existing.status == "running":
            raise RuntimeError(f"{key} 正在下载中，无法删除")

    try:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"删除失败（文件可能被占用——模型已加载或训练中）：{exc}"
        ) from exc


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
    if model_id == "krea2_main":
        v = variant or "latest"
        if v == "latest":
            v = LATEST_KREA2
        if v not in KREA2_VARIANTS:
            raise ValueError(f"unknown Krea 2 variant {variant!r}")
        key = f"krea2_main:{v}"
        start_download_async(
            key, lambda log: download_krea2_main(root, v, on_log=log)
        )
        return key
    if model_id == "krea2_text_encoder":
        key = "krea2_text_encoder"
        start_download_async(
            key, lambda log: download_qwen3_vl(root, on_log=log)
        )
        return key
    if model_id == "krea2_text_encoder_fp8":
        key = "krea2_text_encoder_fp8"
        start_download_async(
            key, lambda log: download_qwen3_vl_fp8(root, on_log=log)
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
        # variant 可指定预设 label（覆盖 cfg 当前的 repo/path），便于 UI 一键
        # 下载非"当前选中"的版本。未指定时用 cfg 当前路径。
        if variant:
            preset = CLTAGGER_VERSIONS.get(variant)
            if preset is None:
                raise ValueError(f"unknown cltagger variant {variant!r}")
            cfg = secrets.CLTaggerConfig(
                **{
                    **cfg.model_dump(),
                    "model_id": preset["model_id"],
                    "model_path": preset["model_path"],
                    "tag_mapping_path": preset["tag_mapping_path"],
                }
            )
            key = f"cltagger:{variant}"
        else:
            key = "cltagger"
        target = cltagger_target_root(root, cfg.model_id)
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
    if model_id in ("eval_clip", "eval_dino"):
        if not variant:
            raise ValueError(f"{model_id} 需要 variant=model_id")
        kind = "clip" if model_id == "eval_clip" else "dino"
        key = f"{model_id}:{variant}"
        start_download_async(
            key, lambda log: download_eval_model(kind, variant, root, on_log=log)
        )
        return key
    if model_id == "eval_ccip":
        if not variant:
            raise ValueError("eval_ccip 需要 variant=ccip 变体名")
        key = f"eval_ccip:{variant}"
        start_download_async(
            key, lambda log: download_ccip_model(variant, root, on_log=log)
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
    if model_id == "upscaler_custom":
        # 统一来源 download 候选（variant=filename；repo 从候选记录取，
        # 源跟全局 download_sources.upscaler）。key 与扫盘行一致。
        if not variant:
            raise ValueError("upscaler_custom 需要 variant=filename")
        cand = _download_candidate("upscaler", variant)
        key = f"upscaler:custom:{Path(variant).name}"
        source = "ms" if _sources._source_for("upscaler") == "modelscope" else "hf"
        start_download_async(
            key,
            lambda log: download_upscaler_custom(
                source, cand.repo, variant, root, on_log=log),
        )
        return key
    if model_id.endswith("_custom"):
        from .families import FAMILY_ASSETS

        family = model_id[: -len("_custom")]
        if family in FAMILY_ASSETS:
            if not variant:
                raise ValueError(f"{model_id} 需要 variant=filename")
            cand = _download_candidate(family, variant)
            key = f"{model_id}:{variant}"
            start_download_async(
                key,
                lambda log: download_main_custom(
                    cand.repo, variant, root, on_log=log),
            )
            return key
    raise ValueError(f"unknown model_id {model_id!r}")
