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
from typing import Any, Optional

from ...paths import STUDIO_DATA

from ...infrastructure.log_messages import msg

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
    logger.info(msg("install.torch_reinstall_registered", target=target))


def read_pending() -> Optional[dict[str, Any]]:
    """读 marker；不存在 / 解析失败均返回 None。"""
    if not PENDING_MARKER.exists():
        return None
    try:
        return json.loads(PENDING_MARKER.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "[pending_install] marker file could not be parsed: %s; ignored", exc,
        )
        return None


def clear_pending() -> None:
    if PENDING_MARKER.exists():
        try:
            PENDING_MARKER.unlink()
        except OSError as exc:
            logger.warning(
                "[pending_install] deleting the marker failed: %s; the install "
                "is retried on the next start", exc,
            )


def apply_pending() -> None:
    """启动期处理 pending 请求；必须在任何 `import torch` 之前调。

    成功 → 清 marker；失败 → 保留 marker（下次启动再试一次）。错误走 logger
    （warning / error 级），不抛异常，让 launcher 继续起 server（用户可以在 UI
    里看到旧 torch 仍在用）。
    """
    pending = read_pending()
    if not pending:
        return

    kind = pending.get("kind")
    if kind == "torch":
        target = pending.get("target", "auto")
        # 一段提示 = 一条多行记录（续行 2 空格），不是三条独立记录
        logger.info(msg(
            "install.torch_reinstall_start", target=target, marker=PENDING_MARKER,
        ))
        # 延迟 import：torch_setup -> onnxruntime_setup 链触发的副作用全留到此刻
        from . import torch as torch_setup  # noqa: PLC0415
        try:
            res = torch_setup.reinstall(target, stream=True)
        except KeyboardInterrupt:
            logger.warning(
                "[pending_install] torch reinstall interrupted by user; the "
                "marker is kept and retried on the next start\n"
                "  to skip permanently, delete the marker file: %s",
                PENDING_MARKER,
            )
            return  # 不 clear_pending，下次启动继续尝试
        except RuntimeError as exc:
            logger.exception(
                "[pending_install] torch reinstall failed: %s\n"
                "  the marker is kept and retried on the next start\n"
                "  to skip permanently, delete the marker file: %s",
                exc, PENDING_MARKER,
            )
            return
        logger.info(msg(
            "install.torch_reinstall_done",
            version=res.get("version"), tag=res.get("tag"),
        ))
    else:
        logger.warning(
            "[pending_install] unknown pending install kind %r; ignored and "
            "cleared", kind,
        )

    clear_pending()
