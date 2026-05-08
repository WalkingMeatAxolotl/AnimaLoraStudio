"""跨进程 pip 安装队列：让 launcher 进程接手 server 进程不能完成的安装。

为什么：
- Windows 文件锁：已 import 的 C extension `.pyd` 不能被 pip 替换
  （`[WinError 5] 拒绝访问 torch\\_C.cp311-win_amd64.pyd`）
- onnxruntime_setup 早就走「pip uninstall + install + restart_required=True」绕这道坎，
  但它的安装路径是用户主动卸装重装；当前进程里 onnxruntime 不一定 import 过
- torch 不一样：server 启动顺路就 import 了（flash_attention_setup.detect_env、各
  service 间接 import），同进程 pip uninstall **必然撞文件锁**

设计：
- server 收到重装请求 → 不真跑 pip，写 marker `studio_data/.pending-pip-install.json` →
  返回 `pending: true`，UI 提示用户重启
- cli.py cmd_run / cmd_dev 启动期 → 先 `apply_pending()` → 装好再起 server
- 失败重试：pip 失败时 marker 不清，下次启动再试一次
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, Optional

from ..paths import STUDIO_DATA

logger = logging.getLogger(__name__)

# studio_data/ 是 gitignore 的，跨重启保留
PENDING_MARKER = STUDIO_DATA / ".pending-pip-install.json"


def register_torch_reinstall(target: str) -> None:
    """注册 torch 重装请求；返回前 marker 已落盘。"""
    STUDIO_DATA.mkdir(parents=True, exist_ok=True)
    PENDING_MARKER.write_text(
        json.dumps({"kind": "torch", "target": target}, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("[pending_install] 已注册 torch reinstall: target=%s", target)


def read_pending() -> Optional[dict[str, Any]]:
    """读 marker；不存在 / 解析失败均返回 None。"""
    if not PENDING_MARKER.exists():
        return None
    try:
        return json.loads(PENDING_MARKER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[pending_install] marker 解析失败: %s", exc)
        return None


def clear_pending() -> None:
    if PENDING_MARKER.exists():
        try:
            PENDING_MARKER.unlink()
        except OSError as exc:
            logger.warning("[pending_install] 删除 marker 失败: %s", exc)


def apply_pending() -> None:
    """启动期处理 pending 请求；必须在任何 `import torch` 之前调。

    成功 → 清 marker；失败 → 保留 marker（下次启动再试一次）。错误打印到 stderr，
    不抛异常，让 launcher 继续起 server（用户可以在 UI 里看到旧 torch 仍在用）。
    """
    pending = read_pending()
    if not pending:
        return

    kind = pending.get("kind")
    if kind == "torch":
        target = pending.get("target", "auto")
        print(f"[studio] 检测到 pending torch 重装请求 (target={target})，开始安装...")
        # 延迟 import：torch_setup -> onnxruntime_setup 链触发的副作用全留到此刻
        from . import torch_setup  # noqa: PLC0415
        try:
            res = torch_setup.reinstall(target)
        except RuntimeError as exc:
            print(f"[studio] torch 重装失败: {exc}", file=sys.stderr)
            print("[studio] marker 保留，下次启动会重试", file=sys.stderr)
            return
        print(f"[studio] torch 重装完成: {res.get('version')} ({res.get('tag')})")
    else:
        print(
            f"[studio] 警告：未知 pending install kind {kind!r}，忽略并清除",
            file=sys.stderr,
        )

    clear_pending()
