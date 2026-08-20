"""下载源路由 + 镜像 endpoint + 低层下载原语（PR-3.8 拆出 4-way 第 2 个）。

回答两个问题：
  1. 从哪儿下？（_get_download_source / _resolve_endpoint / _ms_token）
  2. 怎么落地单个文件？（download_flat / download_flat_ms）

不持有模型路径常量（在 paths.py），不做模型特定的下载流程（在 downloader.py）。
"""
from __future__ import annotations

import logging
import os
import weakref
from pathlib import Path
from typing import Any, Callable, Optional

from ... import secrets
from studio.infrastructure.log_messages import msg
from studio.infrastructure.task_log import TaskLogLike, TaskLog, as_task_log

logger = logging.getLogger(__name__)

#: 手跑 / 无人传 on_log 时的兜底：走本模块 logger（终端 INFO 可见，不再裸 print）。
_DEFAULT_LOG = TaskLog(logger)

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


# CLIP / DINO eval 指标模型的 ModelScope 镜像映射。社区镜像组织 AI-ModelScope
# 同步了 HF 上常见的视觉/多模态模型，repo 名一般一致。没有映射的返回 None →
# 回退 HuggingFace（与 wd14 非 SmilingWolf 前缀同样的优雅回退）。MS 源仅在用户
# 主动把 eval 源切到 modelscope 时才走到；默认 HF。具体 repo 是否存在以
# ModelScope 实际为准，缺失时该模型下载失败、用户可切回 HF 源。
_MS_EVAL_REPO_IDS = {
    "openai/clip-vit-base-patch32": "AI-ModelScope/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14": "AI-ModelScope/clip-vit-large-patch14",
    "facebook/dinov2-small": "AI-ModelScope/dinov2-small",
    "facebook/dinov2-base": "AI-ModelScope/dinov2-base",
}


def _ms_eval_repo_id(hf_repo_id: str) -> Optional[str]:
    """CLIP / DINO 模型的 ModelScope 镜像 id；无映射返回 None（回退 HF）。"""
    return _MS_EVAL_REPO_IDS.get(hf_repo_id)


# ---------------------------------------------------------------------------
# 下载日志：统一模板 + 每任务计数（rewrite-c §2.5 的 T-* 模板）
# ---------------------------------------------------------------------------

#: gated / private 仓库的提示。作为失败记录的**续行**（2 空格缩进），不是独立
#: 记录 —— 同一个 repo 反复失败时按 repo 去重，不再把这段长提示刷 12 遍。
T_GATED_HF = (
    "  the repository may be gated or private: request access on huggingface.co, "
    "then set a HuggingFace token in Settings → Secrets (or the HF_TOKEN "
    "environment variable) and retry"
)
T_GATED_MS = (
    "  the repository may be private on ModelScope: request access, then set a "
    "ModelScope token in Settings → Secrets and retry"
)

#: 每个下载任务（= 一个 on_log 对象）的计数/去重状态。用弱引用键，任务对象被
#: 回收时状态自动消失，不需要显式清理。
_TASK_STATE: "weakref.WeakKeyDictionary[Any, dict[str, Any]]" = (
    weakref.WeakKeyDictionary()
)


def _state(on_log: TaskLogLike) -> dict[str, Any]:
    try:
        st = _TASK_STATE.get(on_log)
    except TypeError:  # 不可弱引用的回调（极少数测试桩）→ 退化成无状态
        return {
            "attempted": 0, "present": 0, "failed": 0, "first_error": "",
            "dep_seen": set(), "gated_seen": set(),
        }
    if st is None:
        st = {
            "attempted": 0, "present": 0, "failed": 0, "first_error": "",
            "dep_seen": set(), "gated_seen": set(),
        }
        _TASK_STATE[on_log] = st
    return st


def finish_download_task(on_log: TaskLogLike) -> None:
    """一次下载任务收尾：把逐文件的「已存在」「失败」计数收成汇总行。

    `download_flat*` 的逐文件「已存在，跳过」是 DEBUG（重跑时最坏十几条
    「什么都没发生」），这条汇总保住「为什么秒完成」的叙事；失败汇总按 R10
    记 ERROR，也是 `_failure_summary` 取的那条。
    """
    try:
        st = _TASK_STATE.pop(on_log, None)
    except TypeError:  # 不可弱引用的回调 → 从来没攒过状态
        return
    if not st:
        return
    if st["present"]:
        on_log(msg(
            "modelsrc.present_summary", n=st["present"], total=st["attempted"],
        ))
    if st["failed"] > 1:
        as_task_log(on_log).error(
            "download failed for %d/%d files; first error: %s",
            st["failed"], st["attempted"], st["first_error"],
        )


def _note_present(on_log: TaskLogLike, name: str, source: str) -> None:
    st = _state(on_log)
    st["attempted"] += 1
    st["present"] += 1
    as_task_log(on_log).debug(
        "file already present; skipped: name=%s source=%s", name, source,
    )


def _note_dep_missing(on_log: TaskLogLike, package: str) -> None:
    """可选依赖没装 → 下载彻底跑不了。同一任务里只说一次（R8）。"""
    st = _state(on_log)
    log = as_task_log(on_log)
    if package in st["dep_seen"]:
        log.debug("%s is still not installed; skipped", package)
        return
    st["dep_seen"].add(package)
    log.error(
        "%s is not installed; the download cannot run — pip install %s",
        package, package,
    )


def _note_failure(
    on_log: TaskLogLike, name: str, source: str, exc: object,
    *, repo: str = "", gated_hint: str = "", throttle: bool = True,
) -> None:
    """单个文件/目录下载失败：首条全文 ERROR，之后逐条 DEBUG，收尾汇总。

    gated 提示作为**同一条记录的续行**跟着首条走，按 repo 去重。
    """
    st = _state(on_log)
    st["attempted"] += 1
    st["failed"] += 1
    text = (
        "download failed: name=%s source=%s err=%s; the model is incomplete "
        "and the download is marked failed"
    )
    hint = ""
    if gated_hint and repo not in st["gated_seen"]:
        st["gated_seen"].add(repo)
        hint = "\n" + gated_hint
    log = as_task_log(on_log)
    if not throttle or st["failed"] == 1:
        if st["failed"] == 1:
            st["first_error"] = f"{name}: {exc}"
        log.error(text + hint, name, source, exc)
    else:
        log.debug(text, name, source, exc)


def _note_rename_failure(
    on_log: TaskLogLike, src: object, dst: object, exc: object,
) -> None:
    """文件已下完但没落位 —— 下次要重下几 GB，这个影响必须写进文本。"""
    st = _state(on_log)
    st["failed"] += 1
    if not st["first_error"]:
        st["first_error"] = f"{dst}: {exc}"
    as_task_log(on_log).error(
        "moving the downloaded file into place failed: src=%s dst=%s err=%s; "
        "the file will be downloaded again on the next attempt", src, dst, exc,
    )


def _note_done(on_log: TaskLogLike, name: str, source: str) -> None:
    st = _state(on_log)
    st["attempted"] += 1
    on_log(msg("modelsrc.file_done", name=name, source=source))


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


def _source_for(type_key: str) -> str:
    """某下载类型（training / wd14 / upscaler）当前选的源。

    MODELSCOPE_SOURCE env 仍作全局强制覆盖（CLI flag / CI）；否则读
    secrets.download_sources[type_key]，缺省 / 非法值回落 huggingface。
    固定 HF 的类型（cltagger / t5 / taeflux）不走这里。
    """
    env = os.environ.get("MODELSCOPE_SOURCE", "").strip().lower()
    if env in ("huggingface", "modelscope"):
        return env
    try:
        src = secrets.load().download_sources.get(type_key, "huggingface")
    except Exception:  # noqa: BLE001
        return "huggingface"
    return src if src in ("huggingface", "modelscope") else "huggingface"


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


def _hf_token() -> Optional[str]:
    """读 HF token：环境变量优先，其次 secrets.huggingface.token。

    gated / private 仓库（如 cl_tagger_v2）下载需要；公开仓库不填也能下。
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        env = os.environ.get(var, "").strip()
        if env:
            return env
    try:
        t = secrets.load().huggingface.token
        return t or None
    except Exception:  # noqa: BLE001  secrets 损坏不应阻断下载
        return None


def _is_gated_auth_error(exc: BaseException) -> bool:
    """粗判下载异常是否源于 gated/private 授权失败，用于追加可操作提示。"""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    return (
        "gated" in name
        or "gated" in msg
        or "401" in msg
        or "403" in msg
        or "unauthorized" in msg
        or "private or gated" in msg
        or "authentication" in msg
    )


def download_flat_ms(
    ms_repo_id: str,
    repo_subpath: str,
    target: Path,
    *,
    on_log: TaskLogLike = _DEFAULT_LOG,
) -> bool:
    """用 modelscope Python API 下载单个文件到 target。

    `model_file_download(local_dir=target.parent)` 会把文件落在
    `target.parent / repo_subpath`（保留 repo 内路径结构），之后复用与
    `download_flat` 完全相同的 rename + 清理空目录逻辑把文件移到 target。

    需要 ``pip install modelscope``；未安装时返回 False 并打印提示。
    token 优先读 MODELSCOPE_API_TOKEN env var，其次 secrets.modelscope.token。
    """
    if target.exists():
        _note_present(on_log, target.name, "ms")
        return True
    try:
        from modelscope.hub.file_download import model_file_download
    except ImportError:
        _note_dep_missing(on_log, "modelscope")
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
        _note_failure(
            on_log, target.name, "ms", exc, repo=ms_repo_id,
            gated_hint=T_GATED_MS if _is_gated_auth_error(exc) else "",
        )
        return False
    # model_file_download 保留 repo 内路径；与 download_flat 逻辑完全一致
    src = target.parent / repo_subpath
    if src != target:
        try:
            target.unlink(missing_ok=True)
            src.rename(target)
        except OSError as exc:
            _note_rename_failure(on_log, src, target, exc)
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
    _note_done(on_log, target.name, "ModelScope")
    return True


def download_flat(
    repo_id: str,
    repo_subpath: str,
    target: Path,
    *,
    on_log: TaskLogLike = _DEFAULT_LOG,
) -> bool:
    """从 HF 下载 repo_subpath，扁平落到 target；返回 True = 已就绪。

    实现：`hf_hub_download(local_dir=target.parent)` 把 repo 内部目录建出来，
    再 rename 到 target（同卷 atomic，不重复 4 GB）。已存在直接跳过。
    """
    if target.exists():
        _note_present(on_log, target.name, "hf")
        return True
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        _note_dep_missing(on_log, "huggingface_hub")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    endpoint = _resolve_endpoint()
    token = _hf_token()
    try:
        kwargs = dict(
            repo_id=repo_id,
            filename=repo_subpath,
            local_dir=str(target.parent),
            local_dir_use_symlinks=False,
        )
        if endpoint:
            kwargs["endpoint"] = endpoint
        if token:
            kwargs["token"] = token
        hf_hub_download(**kwargs)
    except Exception as exc:
        _note_failure(
            on_log, target.name, "hf", exc, repo=repo_id,
            gated_hint=T_GATED_HF if _is_gated_auth_error(exc) else "",
        )
        return False
    src = target.parent / repo_subpath
    if src != target:
        try:
            target.unlink(missing_ok=True)
            src.rename(target)
        except OSError as exc:
            _note_rename_failure(on_log, src, target, exc)
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
    _note_done(on_log, target.name, "HuggingFace")
    return True


def download_snapshot(
    repo_id: str,
    target_dir: Path,
    *,
    allow_patterns: Optional[list[str]] = None,
    on_log: TaskLogLike = _DEFAULT_LOG,
) -> bool:
    """从 HF 把整个 repo 下到 target_dir（多文件 transformers 模型用）。

    与 download_flat（单文件 + 扁平 rename）不同，这里保留 repo 目录结构整目录
    落地，from_pretrained 直接指向 target_dir。已就绪（有 config.json）则跳过。
    """
    if (target_dir / "config.json").exists():
        # 整目录粒度（量 = 1）故意保 INFO —— 与逐文件那条的 DEBUG 差异是
        # 有意的，别当成不一致改回去。
        on_log(msg(
            "modelsrc.dir_present", name=target_dir.name, source="HuggingFace",
        ))
        return True
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _note_dep_missing(on_log, "huggingface_hub")
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    endpoint = _resolve_endpoint()
    token = _hf_token()
    try:
        kwargs: dict = dict(repo_id=repo_id, local_dir=str(target_dir))
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        if endpoint:
            kwargs["endpoint"] = endpoint
        if token:
            kwargs["token"] = token
        snapshot_download(**kwargs)
    except Exception as exc:
        # snapshot 粒度量 = 1，不需要节流
        _note_failure(
            on_log, target_dir.name, "hf", exc, repo=repo_id,
            gated_hint=T_GATED_HF if _is_gated_auth_error(exc) else "",
            throttle=False,
        )
        return False
    on_log(msg(
        "modelsrc.dir_done", name=target_dir.name, source="HuggingFace",
    ))
    return True


def download_snapshot_ms(
    ms_repo_id: str,
    target_dir: Path,
    *,
    on_log: TaskLogLike = _DEFAULT_LOG,
) -> bool:
    """从 ModelScope 把整个 repo 下到 target_dir（多文件模型用）。

    需要 ``pip install modelscope``；未安装返回 False。已就绪则跳过。
    """
    if (target_dir / "config.json").exists():
        on_log(msg(
            "modelsrc.dir_present", name=target_dir.name, source="ModelScope",
        ))
        return True
    try:
        from modelscope import snapshot_download as ms_snapshot
    except ImportError:
        _note_dep_missing(on_log, "modelscope")
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    token = _ms_token()
    try:
        kwargs: dict = dict(model_id=ms_repo_id, local_dir=str(target_dir))
        if token:
            kwargs["token"] = token
        ms_snapshot(**kwargs)
    except Exception as exc:
        _note_failure(
            on_log, target_dir.name, "ms", exc, repo=ms_repo_id,
            gated_hint=T_GATED_MS if _is_gated_auth_error(exc) else "",
            throttle=False,
        )
        return False
    on_log(msg(
        "modelsrc.dir_done", name=target_dir.name, source="ModelScope",
    ))
    return True


