"""grad_accum 尾组处理回归（runtime/training/loop.py::_accumulation_step）。

修复：epoch 最后一个 batch 即使凑不满 grad_accum 也 step（对齐 kohya-ss / HF
Trainer）—— 否则尾部 `len % ga` 个 batch 的梯度被丢（单 epoch）或泄漏进下一 epoch
第一个 step（多 epoch）；不满的尾组按实际 micro-batch 数归一。

用 AST 抽函数独立执行（_accumulation_step 是纯函数、无依赖），避免 import 整个
training 栈（torch + 模型）。同 test_find_diffusion_pipe_root 的轻量做法。
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "runtime" / "training" / "loop.py"


def _load_fn():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_accumulation_step":
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {}
            exec(compile(mod, "<test>", "exec"), ns)
            return ns["_accumulation_step"]
    raise RuntimeError("_accumulation_step not found in loop.py")


_accumulation_step = _load_fn()


def _plan(dl_len, ga):
    return [_accumulation_step(i, dl_len, ga) for i in range(dl_len)]


def _step_idxs(plan):
    return [i for i, (_, end) in enumerate(plan) if end]


def test_divisible_unchanged():
    # 8 batch / ga4：两组满，在 idx3/idx7 step，group_size 恒 4（行为不变）
    plan = _plan(8, 4)
    assert _step_idxs(plan) == [3, 7]
    assert all(gs == 4 for gs, _ in plan)


def test_tail_steps_and_normalizes_by_actual_size():
    # 7 batch / ga4：idx3 step(满组4) + idx6 step(尾组3) —— 尾批不再被丢
    plan = _plan(7, 4)
    assert _step_idxs(plan) == [3, 6]
    assert [gs for gs, _ in plan] == [4, 4, 4, 4, 3, 3, 3]


def test_fewer_batches_than_ga_still_trains():
    # 3 batch / ga4：旧 floor=0 步（完全不训练！）；现 1 步、group_size=3
    plan = _plan(3, 4)
    assert _step_idxs(plan) == [2]
    assert all(gs == 3 for gs, _ in plan)


def test_ga_one_steps_every_batch():
    plan = _plan(5, 1)
    assert all(end for _, end in plan)
    assert all(gs == 1 for gs, _ in plan)


def test_no_len_falls_back_to_old_behavior():
    # dl_len=None：恒 grad_accum，仅整除处 step（不知末批，无法尾处理）
    assert _accumulation_step(3, None, 4) == (4, True)
    assert _accumulation_step(4, None, 4) == (4, False)
    assert _accumulation_step(6, None, 4) == (4, False)  # 尾批但无 len → 旧行为不 step
