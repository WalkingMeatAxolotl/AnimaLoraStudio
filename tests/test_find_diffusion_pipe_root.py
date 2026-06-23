"""find_diffusion_pipe_root 路径解析回归 —— 训练模块在 runtime/training/ 子目录下时也能找到 ../models/。

函数位于 `runtime/training/model_loading.py`（ADR 0003 PR-A 从 anima_train.py 搬入）。
本测试用 AST 抽函数源码独立执行，避免 import 整个 model_loading（torch + anima
依赖太重），保持单测轻量稳定。
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


SRC_REL = Path("runtime") / "training" / "model_loading.py"


def _exec_fn() -> tuple[object, dict]:
    """从 runtime/training/model_loading.py 抽 find_diffusion_pipe_root，独立编译。"""
    src_path = Path(__file__).resolve().parent.parent / SRC_REL
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "find_diffusion_pipe_root":
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {"Path": Path, "os": os}
            return mod, ns
    raise RuntimeError(f"find_diffusion_pipe_root not found in {SRC_REL}")


def _run(file_loc: Path, env_root: str | None = None) -> Path:
    """在 ns 里执行函数，返回 find_diffusion_pipe_root() 的结果。

    file_loc 应当是 fake 的 runtime/training/model_loading.py 路径——
    函数内部按 __file__.parent.parent.parent 反推 repo_root。
    """
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


def _make_fake_module(tmp_path: Path) -> Path:
    """造一个 tmp_path/runtime/training/model_loading.py 占位文件。"""
    training_dir = tmp_path / "runtime" / "training"
    training_dir.mkdir(parents=True)
    fake = training_dir / "model_loading.py"
    fake.write_text("# placeholder")
    return fake


def test_finds_repo_root_models_when_train_in_subdir(tmp_path: Path) -> None:
    """标准 layout：repo_root/runtime/training/model_loading.py + repo_root/models/anima_modeling.py。"""
    fake_module = _make_fake_module(tmp_path)

    models = tmp_path / "models"
    models.mkdir()
    (models / "anima_modeling.py").write_text("")

    found = _run(fake_module)
    assert found == models, f"expected {models}, got {found}"


def test_prefers_repo_root_modeling_over_models(tmp_path: Path) -> None:
    """新布局：模型代码在 repo_root/modeling/；同时存在旧 models/ 时 modeling/ 优先。"""
    fake_module = _make_fake_module(tmp_path)

    modeling = tmp_path / "modeling"
    modeling.mkdir()
    (modeling / "anima_modeling.py").write_text("")
    # 旧 models/ 也有同名文件（外部 checkout / 遗留）——modeling/ 应排在前面命中
    models = tmp_path / "models"
    models.mkdir()
    (models / "anima_modeling.py").write_text("")

    found = _run(fake_module)
    assert found == modeling, f"expected {modeling}, got {found}"


def test_finds_runtime_sibling_models_directory(tmp_path: Path) -> None:
    """runtime/ 同级的 models/（候选 2：runtime_dir / 'models'）。"""
    fake_module = _make_fake_module(tmp_path)

    runtime_models = tmp_path / "runtime" / "models"
    runtime_models.mkdir()
    (runtime_models / "anima_modeling.py").write_text("")

    found = _run(fake_module)
    assert found == runtime_models


def test_env_var_override_when_no_layout_match(tmp_path: Path) -> None:
    """DIFFUSION_PIPE_ROOT 直接指向 anima_modeling.py 所在目录。"""
    fake_module = _make_fake_module(tmp_path)

    custom = tmp_path / "custom_pipe"
    custom.mkdir()
    (custom / "anima_modeling.py").write_text("")

    found = _run(fake_module, env_root=str(custom))
    assert found == custom


def test_raises_when_nothing_found(tmp_path: Path) -> None:
    """所有候选都没命中 → RuntimeError。"""
    fake_module = _make_fake_module(tmp_path)

    with pytest.raises(RuntimeError, match="anima_modeling.py"):
        _run(fake_module)
