"""PR-1 安全网 — route 数量与 decorator 计数的粗粒度不变量。

snapshot 是精细网（任何字符变化都触发），本文件是粗粒度二道防线：
- 数量落在合理区间
- server.py 源里的 @app.<verb> 装饰器数 == APIRoute 数

如果某个未来 PR 把 router 拆走了，snapshot 测试会同时挂；但本测试也会挂，
能在 reviewer 视野里多出一条提醒：APIRoute 数量变了。
"""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.routing import APIRoute

from studio.server import app

SERVER_PY = Path(__file__).parent.parent / "studio" / "server.py"


def test_route_count_in_sane_range() -> None:
    n = len(app.routes)
    assert 100 <= n <= 250, f"len(app.routes) = {n}，超出合理区间 [100, 250]"


def test_decorator_count_matches_api_routes() -> None:
    src = SERVER_PY.read_text(encoding="utf-8")
    # 匹配 @app.get(...) / @app.post(...) / ... 以及 @app.api_route(...)
    decorator_count = len(
        re.findall(
            r"^@app\.(get|post|put|delete|patch|api_route)\b",
            src,
            flags=re.MULTILINE,
        )
    )
    api_route_count = sum(1 for r in app.routes if isinstance(r, APIRoute))
    assert decorator_count == api_route_count, (
        f"server.py 里 @app.* 装饰器 {decorator_count} 个，"
        f"但 app.routes 里 APIRoute 实例 {api_route_count} 个 —— "
        f"差额可能来自 include_router 或 router 注册时丢了一个"
    )
