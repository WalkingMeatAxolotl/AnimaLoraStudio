"""`anima-studio` / `python -m studio.server` uvicorn 启动入口（PR-5 从 server.py 抽出）。

uvicorn 启动字符串仍指 `studio.server:app` —— 老 server.py 内 130 个
route decorator 在 import 时全部注册到 `api.app.app`，server.py 顶部
`from .api.app import app` re-export 同一对象。
"""
from __future__ import annotations


def main() -> None:
    import argparse

    # 计算显卡选择（#491）：必须在 torch/uvicorn 首次 import CUDA 相关代码
    # 之前注入 env。launcher（cli.py）已注入过时这里是幂等 no-op；直跑
    # `python -m studio.server` 时这里是唯一注入点。失败不挡启动。
    try:
        from ..services.runtime.gpu_select import apply_gpu_selection_env

        apply_gpu_selection_env()
    except Exception:  # noqa: BLE001
        pass

    import uvicorn

    parser = argparse.ArgumentParser(description="AnimaStudio daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--reload", action="store_true", help="dev mode (auto-reload on edit)"
    )
    args = parser.parse_args()

    # ADR 0012：SPA 入口在根路径 /（不再用 /studio 子路径）。
    # URL 由 cli.py 的 `cli.backend_started` 行打印（带 ts/级别，符合行契约），
    # 这里不再重复一条裸 print banner。
    uvicorn.run(
        "studio.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        # 浏览器开着时 /api/events 的 SSE 长连接不会主动断，graceful shutdown
        # 默认无限等 →「Waiting for connections to close」卡死；且 py3.12+ 的
        # Server.wait_closed() 等全部活跃连接，二次 Ctrl+C 的 force_exit 也
        # 解不开（transport 不被强关）。给 graceful 一个上限：超时后 uvicorn
        # cancel 剩余连接 task → 连接关闭 → lifespan 正常收尾（supervisor /
        # daemon 优雅停）。
        timeout_graceful_shutdown=3,
    )
