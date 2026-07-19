"""find_diffusion_pipe_root shim 行为回归（多模型 PR-2a 后）。

exec-load 与外部 diffusion-pipe checkout 兼容已退役：函数收缩为常量 shim，
返回 `modeling/anima/`（sister 契约 7 名之一，名字与返回语义保留）。
本测试沿用 AST 抽函数独立执行的手法，避免 import 整个 model_loading
（torch 依赖太重），保持单测轻量稳定。
"""
from __future__ import annotations

import ast
import os
from pathlib import Path


SRC_REL = Path("runtime") / "training" / "model_loading.py"
REPO_ROOT = Path(__file__).resolve().parent.parent


class _RecordingLogger:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else str(msg))


def _make_fn(logger):
    src_path = REPO_ROOT / SRC_REL
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "find_diffusion_pipe_root":
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {"Path": Path, "os": os, "logger": logger, "__file__": str(src_path)}
            exec(compile(mod, str(src_path), "exec"), ns)
            return ns["find_diffusion_pipe_root"]
    raise RuntimeError(f"find_diffusion_pipe_root not found in {SRC_REL}")


def test_returns_modeling_anima_dir(monkeypatch):
    monkeypatch.delenv("DIFFUSION_PIPE_ROOT", raising=False)
    logger = _RecordingLogger()
    fn = _make_fn(logger)
    root = fn()
    assert root == REPO_ROOT / "modeling" / "anima"
    assert (root / "anima_modeling.py").exists()
    assert (root / "cosmos_predict2_modeling.py").exists()
    assert logger.warnings == []


def test_diffusion_pipe_root_env_is_ignored_with_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("DIFFUSION_PIPE_ROOT", str(tmp_path))
    logger = _RecordingLogger()
    fn = _make_fn(logger)
    root = fn()
    # 环境变量不再改变返回值，只打一次弃用 warning
    assert root == REPO_ROOT / "modeling" / "anima"
    assert len(logger.warnings) == 1
    assert "DIFFUSION_PIPE_ROOT" in logger.warnings[0]
