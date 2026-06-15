"""共享 route 内省 helper —— 兼容 FastAPI 0.137+ 把 include_router 包装成
`_IncludedRouter` 的内部表示。

行为变化（不是应用 bug，只影响 introspection 测试）：

  0.136 及之前：
    app.include_router(sub) → sub 的每个 APIRoute 直接 append 到 app.routes
  0.137 起：
    app.include_router(sub) → app.routes 多 1 个 `_IncludedRouter(original_router=sub)`
    wrapper；要拿底层 APIRoute 必须走 `wrapper.original_router.routes` 递归

HTTP routing dispatch 在两版上都正常（已用 TestClient 验证）。变化只影响
直接遍历 `app.routes` 拿 APIRoute 实例的代码 —— 主要是测试。
"""
from __future__ import annotations

from typing import Any, Iterator


def iter_leaf_routes(routes: list[Any]) -> Iterator[Any]:
    """递归展开 `_IncludedRouter` wrapper，产出 APIRoute / Mount / etc 叶子。

    跨 fastapi 版本兼容：
    - 0.136：顶层即叶子（else 分支直接 yield）
    - 0.137+：遇 wrapper 走 `original_router.routes` 再下钻
    - 嵌套 include_router 也能正确展开
    """
    for r in routes:
        if hasattr(r, "original_router") and hasattr(r.original_router, "routes"):
            yield from iter_leaf_routes(r.original_router.routes)
        else:
            yield r
