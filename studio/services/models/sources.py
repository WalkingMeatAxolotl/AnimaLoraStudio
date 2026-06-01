"""下载源路由 + 镜像 endpoint + 低层下载原语（PR-3.8 拆出 4-way 第 2 个）。

回答两个问题：
  1. 从哪儿下？（_get_download_source / _resolve_endpoint / _ms_token）
  2. 怎么落地单个文件？（download_flat / download_flat_ms）

不持有模型路径常量（在 paths.py），不做模型特定的下载流程（在 downloader.py）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from ... import secrets

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


