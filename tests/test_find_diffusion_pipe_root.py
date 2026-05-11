"""find_diffusion_pipe_root 路径解析回归 —— anima_train 在 runtime/ 子目录下时也能找到 ../models/。

anima_train.py 在仓库根的子目录（runtime/，历史上叫过 scripts/）下，
`Path(__file__).parent` 变成 runtime/，原候选 `Path(__file__).parent / "models"`
找不到 repo_root/models/。线上首次跑训练立即 RuntimeError。

本测试用 AST 抽函数源码独立执行，避免 import 整个 anima_train（torch +
anima 依赖太重），保持单测轻量稳定。
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


def _exec_fn() -> tuple[object, dict]:
    """从 runtime/anima_train.py 抽 find_diffusion_pipe_root，独立编译。"""
    src_path = Path(__file__).resolve().parent.parent / "runtime" / "anima_train.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "find_diffusion_pipe_root":
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {"Path": Path, "os": os}
            return mod, ns
    raise RuntimeError("find_diffusion_pipe_root not found in anima_train.py")


def _run(file_loc: Path, env_root: str | None = None) -> Path:
    """在 ns 里执行函数，返回 find_diffusion_pipe_root() 的结果。"""
    mod, ns = _exec_fn()
    ns["__file__"] = str(file_loc)
    if env_root is None:
        # 隔离 env：忽略实际机器上的 DIFFUSION_PIPE_ROOT
        os_env_backup = os.environ.pop("DIFFUSION_PIPE_ROOT", None)
        try:
            exec(compile(mod, "<test>", "exec"), ns)
            return ns["find_diffusion_pipe_root"]()
        finally:
            if os_env_backup is not None:
                os.environ["DIFFUSION_PIPE_ROOT"] = os_env_backup
    else:
        os.environ["DIFFUSION_PIPE_ROOT"] = env_root
        try:
            exec(compile(mod, "<test>", "exec"), ns)
            return ns["find_diffusion_pipe_root"]()
        finally:
            os.environ.pop("DIFFUSION_PIPE_ROOT", None)


def test_finds_repo_root_models_when_train_in_subdir(tmp_path: Path) -> None:
    """标准 layout：repo_root/runtime/anima_train.py + repo_root/models/anima_modeling.py。"""
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    fake_train = runtime_dir / "anima_train.py"
    fake_train.write_text("# placeholder")

    models = tmp_path / "models"
    models.mkdir()
    (models / "anima_modeling.py").write_text("")

    found = _run(fake_train)
    assert found == models, f"expected {models}, got {found}"


def test_finds_script_sibling_models_directory(tmp_path: Path) -> None:
    """脚本同目录的 models/（CLI 用户把脚本和模型代码放一起，候选 1）。"""
    fake_train = tmp_path / "anima_train.py"
    fake_train.write_text("# placeholder")

    sibling_models = tmp_path / "models"
    sibling_models.mkdir()
    (sibling_models / "anima_modeling.py").write_text("")

    found = _run(fake_train)
    assert found == sibling_models


def test_env_var_override_when_no_layout_match(tmp_path: Path) -> None:
    """DIFFUSION_PIPE_ROOT 直接指向 anima_modeling.py 所在目录。"""
    fake_train = tmp_path / "runtime" / "anima_train.py"
    fake_train.parent.mkdir()
    fake_train.write_text("# placeholder")

    custom = tmp_path / "custom_pipe"
    custom.mkdir()
    (custom / "anima_modeling.py").write_text("")

    found = _run(fake_train, env_root=str(custom))
    assert found == custom


def test_raises_when_nothing_found(tmp_path: Path) -> None:
    """所有候选都没命中 → RuntimeError。"""
    fake_train = tmp_path / "runtime" / "anima_train.py"
    fake_train.parent.mkdir()
    fake_train.write_text("# placeholder")

    with pytest.raises(RuntimeError, match="anima_modeling.py"):
        _run(fake_train)
