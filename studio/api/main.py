"""`anima-studio` / `python -m studio.server` uvicorn 启动入口（PR-5 从 server.py 抽出）。

uvicorn 启动字符串仍指 `studio.server:app` —— 老 server.py 内 130 个
route decorator 在 import 时全部注册到 `api.app.app`，server.py 顶部
`from .api.app import app` re-export 同一对象。
"""
from __future__ import annotations


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="AnimaStudio daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--reload", action="store_true", help="dev mode (auto-reload on edit)"
    )
    args = parser.parse_args()

    # ADR 0012：SPA 入口在根路径 /（不再用 /studio 子路径）。
    print(f"[AnimaStudio] http://{args.host}:{args.port}/")
    uvicorn.run(
        "studio.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
