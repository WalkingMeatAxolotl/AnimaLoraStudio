"""裸 print 白名单锁（docs/design/logging-target-state.md 不变量 1「一个格式」）。

日志内容一律走 logger（同一 formatter → run.log / studio.log 每行有 ts/level/来源）。
裸 `print` 只剩三类合法用途，逐一登记在下面的白名单里：
  - stdout 协议行（`__EVENT__:` / daemon line-JSON）
  - tty 交互（rich/plain 进度行、input() 配套提示、独立 CLI 工具 `__main__` 的结果输出）
  - CLI 启动 banner

新增 print 会让本测试红：要么改 logger，要么在这里登记并写明属于哪一类。
用 AST 扫，键 = (仓库相对路径, 最近的函数名或 "<module>" / "<__main__>")。
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("runtime", "utils", "studio")
SKIP_PARTS = {"web", "__pycache__", "node_modules"}

# 路径 → 允许出现 print 的上下文集合。上下文 = 最近的 def 名；模块级用 "<module>"；
# `if __name__ == "__main__":` 块内用 "<__main__>"。
WHITELIST: dict[str, set[str]] = {
    # tty 交互：plain 模式 `\r` 进度行 / ctx.emit 的 tty 分支
    "runtime/training/loop.py": {"run"},
    "runtime/training/context.py": {"emit"},
    # tty 交互：input() 配套提示
    "runtime/training/cli.py": {"_ask_int", "_ask_float"},
    # 独立 CLI 工具的结果输出
    "utils/caption_utils.py": {"<__main__>"},
    # stdout 协议行
    "studio/workers/preprocess_worker.py": {"emit_event"},
}


def _context_of(path: list[ast.AST]) -> str:
    for node in reversed(path):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
        if isinstance(node, ast.If) and _is_main_guard(node.test):
            return "<__main__>"
    return "<module>"


def _is_main_guard(test: ast.AST) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators)
    )


def _iter_prints(tree: ast.AST):
    stack: list[ast.AST] = []

    def visit(node: ast.AST):
        stack.append(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            yield node.lineno, _context_of(stack[:-1])
        for child in ast.iter_child_nodes(node):
            yield from visit(child)
        stack.pop()

    yield from visit(tree)


def _py_files():
    for root in SCAN_ROOTS:
        for p in (REPO / root).rglob("*.py"):
            if SKIP_PARTS & set(p.parts):
                continue
            yield p


def test_bare_print_only_in_whitelisted_contexts() -> None:
    offenders: list[str] = []
    for p in _py_files():
        rel = p.relative_to(REPO).as_posix()
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:  # pragma: no cover
            offenders.append(f"{rel}: 无法解析 ({e})")
            continue
        allowed = WHITELIST.get(rel, set())
        for lineno, ctx in _iter_prints(tree):
            if ctx not in allowed:
                offenders.append(f"{rel}:{lineno} in {ctx}")
    assert not offenders, (
        "发现白名单外的裸 print（日志内容请走 logger；协议行/tty 交互/banner 请在 "
        "tests/test_print_whitelist.py WHITELIST 登记）:\n  " + "\n  ".join(offenders)
    )


def test_whitelist_entries_still_exist() -> None:
    """白名单不长草：登记的文件必须存在且该上下文里确实还有 print。"""
    stale: list[str] = []
    for rel, ctxs in WHITELIST.items():
        p = REPO / rel
        if not p.exists():
            stale.append(f"{rel}: 文件不存在")
            continue
        found = {ctx for _, ctx in _iter_prints(ast.parse(p.read_text(encoding="utf-8")))}
        for ctx in ctxs - found:
            stale.append(f"{rel}: 上下文 {ctx} 已无 print，可从白名单删除")
    assert not stale, "\n".join(stale)
